from __future__ import annotations

import subprocess
from pathlib import Path

from hermes_cli.local_runtime.devices import _cli_fallback_devices
from hermes_cli.local_runtime.hardware import _DEVICE_LINE_RE


def test_device_line_re_matches_vulkan_and_rocm():
    line_vulkan = "  Vulkan0: AMD Radeon RX 9060 XT (8144 MiB, 7283 MiB free)"
    m = _DEVICE_LINE_RE.search(line_vulkan)
    assert m is not None
    assert m.group(1) == "8144"

    line_cuda = "  CUDA0: NVIDIA GeForce RTX 4090 (24564 MiB, 22000 MiB free)"
    m_cuda = _DEVICE_LINE_RE.search(line_cuda)
    assert m_cuda is not None
    assert m_cuda.group(1) == "24564"


def test_cli_fallback_devices_parses_executable_output(tmp_path, monkeypatch):
    bin_name = "llama-server.exe" if Path("a.exe").suffix == ".exe" else "llama-server"
    fake_exe = tmp_path / bin_name
    fake_exe.write_text("#!/bin/sh\n")

    fake_output = (
        "Available devices:\n"
        "  Vulkan0: AMD Radeon RX 9060 XT (8144 MiB, 7283 MiB free)\n"
    )

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout=fake_output, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    devices = _cli_fallback_devices(tmp_path, "vulkan")
    assert len(devices) == 1
    assert devices[0]["description"] == "AMD Radeon RX 9060 XT"
    assert devices[0]["total"] == 8144 * 1024 * 1024
    assert devices[0]["free"] == 7283 * 1024 * 1024
