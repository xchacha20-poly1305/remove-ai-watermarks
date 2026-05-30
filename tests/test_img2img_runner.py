"""Unit tests for the MPS->CPU fallback orchestration (no GPU/model required).

``img2img_runner`` has no torch import at module top -- the pipeline is
injected as a plain callable -- so the fallback control flow is fully
mockable. This guards the exact behavior hit in production on Apple Silicon:
a native-resolution SDXL run that OOMs on MPS must transparently retry on CPU,
while any non-MPS error must propagate unchanged.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from remove_ai_watermarks.noai import img2img_runner
from remove_ai_watermarks.noai.img2img_runner import (
    run_differential_with_mps_fallback,
    run_img2img,
    run_img2img_with_mps_fallback,
)

_MPS_OOM = "MPS backend out of memory (MPS allocated: 17.21 GiB, max allowed: 20.13 GiB)"
# The exact message a real Arc raises (torch.OutOfMemoryError, a RuntimeError
# subclass), captured on an Intel Arc Graphics (28.55 GiB).
_XPU_OOM = (
    "XPU out of memory. Tried to allocate 59.61 GiB. GPU 0 has a total "
    "capacity of 28.55 GiB of which 12.01 GiB is free."
)


def _result(image: object) -> Mock:
    """A stand-in for a diffusers pipeline output object (has .images)."""
    out = Mock()
    out.images = [image]
    return out


class TestMpsFallback:
    def test_mps_error_reloads_on_cpu_and_retries(self, monkeypatch: pytest.MonkeyPatch):
        sentinel = object()
        inner = Mock(side_effect=[RuntimeError(_MPS_OOM), sentinel])
        monkeypatch.setattr(img2img_runner, "run_img2img", inner)
        load_pipeline = Mock(return_value="gpu_pipe")
        reload_on_cpu = Mock(return_value="cpu_pipe")

        img, device = run_img2img_with_mps_fallback(
            load_pipeline, object(), 0.05, 50, 7.5, "gen", "mps", lambda _m: None, reload_on_cpu=reload_on_cpu
        )

        assert (img, device) == (sentinel, "cpu")
        reload_on_cpu.assert_called_once()
        assert inner.call_count == 2
        # Retry must use the reloaded CPU pipeline, device "cpu", and drop the
        # MPS generator (generator=None) so CPU runs deterministically.
        retry_args = inner.call_args_list[1].args
        assert retry_args[0] == "cpu_pipe"
        assert retry_args[5] is None  # generator
        assert retry_args[6] == "cpu"  # device

    def test_happy_path_returns_original_device_without_reload(self, monkeypatch: pytest.MonkeyPatch):
        sentinel = object()
        monkeypatch.setattr(img2img_runner, "run_img2img", Mock(return_value=sentinel))
        reload_on_cpu = Mock()

        img, device = run_img2img_with_mps_fallback(
            Mock(return_value="gpu_pipe"),
            object(),
            0.05,
            50,
            7.5,
            "gen",
            "mps",
            lambda _m: None,
            reload_on_cpu=reload_on_cpu,
        )

        assert (img, device) == (sentinel, "mps")
        reload_on_cpu.assert_not_called()

    def test_non_mps_runtime_error_propagates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(img2img_runner, "run_img2img", Mock(side_effect=RuntimeError("CUDA out of memory")))
        reload_on_cpu = Mock()

        with pytest.raises(RuntimeError, match="CUDA"):
            run_img2img_with_mps_fallback(
                Mock(return_value="gpu_pipe"),
                object(),
                0.05,
                50,
                7.5,
                "gen",
                "mps",
                lambda _m: None,
                reload_on_cpu=reload_on_cpu,
            )
        reload_on_cpu.assert_not_called()

    def test_mps_error_on_non_mps_device_propagates(self, monkeypatch: pytest.MonkeyPatch):
        # An "mps"-worded error while running on cpu must NOT trigger the reload.
        monkeypatch.setattr(img2img_runner, "run_img2img", Mock(side_effect=RuntimeError(_MPS_OOM)))
        reload_on_cpu = Mock()

        with pytest.raises(RuntimeError, match="MPS backend"):
            run_img2img_with_mps_fallback(
                Mock(return_value="cpu_pipe"),
                object(),
                0.05,
                50,
                7.5,
                None,
                "cpu",
                lambda _m: None,
                reload_on_cpu=reload_on_cpu,
            )
        reload_on_cpu.assert_not_called()


class TestXpuFallback:
    """The xpu backend (#24 follow-up) shares the GPU->CPU OOM fallback, but is
    OOM-scoped: an XPU OOM transparently retries on CPU, while a non-OOM xpu
    fault must propagate rather than silently degrade to CPU."""

    def test_xpu_oom_reloads_on_cpu_and_retries(self, monkeypatch: pytest.MonkeyPatch):
        sentinel = object()
        inner = Mock(side_effect=[RuntimeError(_XPU_OOM), sentinel])
        monkeypatch.setattr(img2img_runner, "run_img2img", inner)
        clear = Mock()
        monkeypatch.setattr(img2img_runner, "_try_clear_device_cache", clear)
        reload_on_cpu = Mock(return_value="cpu_pipe")

        img, device = run_img2img_with_mps_fallback(
            Mock(return_value="gpu_pipe"),
            object(),
            0.05,
            50,
            7.5,
            "gen",
            "xpu",
            lambda _m: None,
            reload_on_cpu=reload_on_cpu,
        )

        assert (img, device) == (sentinel, "cpu")
        clear.assert_called_once_with("xpu")  # cache cleared on the right backend
        reload_on_cpu.assert_called_once()
        assert inner.call_count == 2
        retry_args = inner.call_args_list[1].args
        assert retry_args[0] == "cpu_pipe"
        assert retry_args[5] is None  # generator dropped for deterministic CPU run
        assert retry_args[6] == "cpu"  # device

    def test_xpu_non_oom_error_propagates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(img2img_runner, "run_img2img", Mock(side_effect=RuntimeError("XPU kernel build failed")))
        reload_on_cpu = Mock()

        with pytest.raises(RuntimeError, match="kernel build"):
            run_img2img_with_mps_fallback(
                Mock(return_value="gpu_pipe"),
                object(),
                0.05,
                50,
                7.5,
                "gen",
                "xpu",
                lambda _m: None,
                reload_on_cpu=reload_on_cpu,
            )
        reload_on_cpu.assert_not_called()


class TestClearDeviceCache:
    """_try_clear_device_cache routes empty_cache() to the named backend."""

    def test_clears_named_backend(self, monkeypatch: pytest.MonkeyPatch):
        import sys

        fake_torch = MagicMock()
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        img2img_runner._try_clear_device_cache("xpu")
        fake_torch.xpu.empty_cache.assert_called_once_with()

    def test_missing_backend_is_noop(self, monkeypatch: pytest.MonkeyPatch):
        import sys

        fake_torch = MagicMock(spec=[])  # getattr(torch, "xpu", None) -> None
        monkeypatch.setitem(sys.modules, "torch", fake_torch)
        img2img_runner._try_clear_device_cache("xpu")  # must not raise


class TestDifferentialMpsFallback:
    """The protect-text (Differential Diffusion) path shares the MPS->CPU
    fallback contract; mock ``run_differential`` so no torch/model is needed."""

    def test_mps_error_reloads_on_cpu_and_retries(self, monkeypatch: pytest.MonkeyPatch):
        sentinel = object()
        inner = Mock(side_effect=[RuntimeError(_MPS_OOM), sentinel])
        monkeypatch.setattr(img2img_runner, "run_differential", inner)
        reload_on_cpu = Mock(return_value="cpu_pipe")

        img, device = run_differential_with_mps_fallback(
            load_pipeline=Mock(return_value="gpu_pipe"),
            image=object(),
            change_map=object(),
            strength=0.05,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator="gen",
            device="mps",
            set_progress=lambda _m: None,
            reload_on_cpu=reload_on_cpu,
        )

        assert (img, device) == (sentinel, "cpu")
        reload_on_cpu.assert_called_once()
        assert inner.call_count == 2
        # Retry uses the reloaded CPU pipeline, device "cpu", and drops the MPS
        # generator (generator=None) for deterministic CPU execution.
        retry_args = inner.call_args_list[1].args
        assert retry_args[0] == "cpu_pipe"
        assert retry_args[6] is None  # generator
        assert retry_args[7] == "cpu"  # device

    def test_xpu_oom_reloads_on_cpu_and_retries(self, monkeypatch: pytest.MonkeyPatch):
        sentinel = object()
        inner = Mock(side_effect=[RuntimeError(_XPU_OOM), sentinel])
        monkeypatch.setattr(img2img_runner, "run_differential", inner)
        clear = Mock()
        monkeypatch.setattr(img2img_runner, "_try_clear_device_cache", clear)
        reload_on_cpu = Mock(return_value="cpu_pipe")

        img, device = run_differential_with_mps_fallback(
            load_pipeline=Mock(return_value="gpu_pipe"),
            image=object(),
            change_map=object(),
            strength=0.05,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator="gen",
            device="xpu",
            set_progress=lambda _m: None,
            reload_on_cpu=reload_on_cpu,
        )

        assert (img, device) == (sentinel, "cpu")
        clear.assert_called_once_with("xpu")
        reload_on_cpu.assert_called_once()
        retry_args = inner.call_args_list[1].args
        assert retry_args[6] is None  # generator dropped
        assert retry_args[7] == "cpu"  # device

    def test_happy_path_returns_original_device_without_reload(self, monkeypatch: pytest.MonkeyPatch):
        sentinel = object()
        monkeypatch.setattr(img2img_runner, "run_differential", Mock(return_value=sentinel))
        reload_on_cpu = Mock()

        img, device = run_differential_with_mps_fallback(
            load_pipeline=Mock(return_value="gpu_pipe"),
            image=object(),
            change_map=object(),
            strength=0.05,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator="gen",
            device="mps",
            set_progress=lambda _m: None,
            reload_on_cpu=reload_on_cpu,
        )

        assert (img, device) == (sentinel, "mps")
        reload_on_cpu.assert_not_called()

    def test_non_mps_runtime_error_propagates(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(img2img_runner, "run_differential", Mock(side_effect=RuntimeError("CUDA out of memory")))
        reload_on_cpu = Mock()

        with pytest.raises(RuntimeError, match="CUDA"):
            run_differential_with_mps_fallback(
                load_pipeline=Mock(return_value="gpu_pipe"),
                image=object(),
                change_map=object(),
                strength=0.05,
                num_inference_steps=50,
                guidance_scale=7.5,
                generator="gen",
                device="mps",
                set_progress=lambda _m: None,
                reload_on_cpu=reload_on_cpu,
            )
        reload_on_cpu.assert_not_called()


class TestRunImg2Img:
    def test_returns_first_image_from_pipeline_result(self):
        sentinel = object()
        pipeline = Mock(return_value=_result(sentinel))

        out = run_img2img(pipeline, object(), 0.05, 50, 7.5, None, "cpu", lambda _m: None)

        assert out is sentinel

    def test_typeerror_on_callback_retries_without_callback(self):
        # Older diffusers reject the progress callback kwarg with TypeError;
        # run_img2img must retry once without it rather than fail.
        sentinel = object()
        pipeline = Mock(side_effect=[TypeError("unexpected keyword 'callback'"), _result(sentinel)])

        out = run_img2img(pipeline, object(), 0.05, 50, 7.5, None, "cpu", lambda _m: None)

        assert out is sentinel
        assert pipeline.call_count == 2
        # First attempt passes the progress callback; the retry omits it.
        assert "callback" in pipeline.call_args_list[0].kwargs
        assert "callback" not in pipeline.call_args_list[1].kwargs
