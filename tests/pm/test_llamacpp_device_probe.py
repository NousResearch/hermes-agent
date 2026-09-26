"""Regression tests for #123677: a dlopen'd llama.cpp GPU backend that ggml's
loader silently skips (HIP archive missing hipblas.dll) must fail verify(),
not install as a "hip" engine that actually runs on CPU."""

import os
import sys
from pathlib import Path

import pytest

from pm.packages import LlamaCppCpu, LlamaCppCuda, LlamaCppHip
from pm.store import current_target


@pytest.fixture()
def hip_entry(tmp_path, monkeypatch):
    """An entry whose llama-server is a stub printing the --list-devices
    output the test asks for (from a file passed through the env)."""
    def make(devices_output: str) -> Path:
        marker = tmp_path / "stub-devices.txt"
        marker.write_text(devices_output)
        if sys.platform == "win32":
            pytest.skip("stub engine is a shebang script")
        entry = tmp_path / "entry"
        entry.mkdir()
        server = entry / "llama-server"
        server.write_text("#!/bin/sh\ncat \"$STUB_DEVICES_FILE\"\n")
        server.chmod(0o755)
        monkeypatch.setenv("STUB_DEVICES_FILE", str(marker))
        return entry
    return make


def test_hip_verify_fails_when_no_device(tmp_path, hip_entry):
    """--version exits 0 but --list-devices says (none): the exact
    silent-CPU shape from #123677. verify() must return a reason naming
    the backend and the device evidence."""
    entry = hip_entry("Available devices:\n  (none)\n")
    reason = LlamaCppHip().verify(entry, current_target())
    assert reason
    assert "hip" in reason.lower()
    assert "no devices" in reason.lower()


def test_hip_verify_passes_with_device(tmp_path, hip_entry):
    entry = hip_entry("Available devices:\n  ROCm0: AMD Radeon (108 GB)\n")
    assert LlamaCppHip().verify(entry, current_target()) == ""


def test_cpu_backend_is_never_device_probed():
    """The device probe is a GPU-backend requirement; the CPU package must
    not grow one."""
    assert LlamaCppCpu()._device_probe_argv() is None


def test_gpu_backends_declare_the_device_probe():
    assert LlamaCppCuda()._device_probe_argv() == ["--list-devices"]
    assert LlamaCppHip()._device_probe_argv() == ["--list-devices"]


def test_device_enumeration_parsing():
    assert LlamaCppHip._has_device("Available devices:\n  (none)\n") is False
    assert LlamaCppHip._has_device("Available devices:\n  ROCm0: x\n") is True
    assert LlamaCppHip._has_device("") is False
    assert LlamaCppHip._has_device("(none)\n") is False
