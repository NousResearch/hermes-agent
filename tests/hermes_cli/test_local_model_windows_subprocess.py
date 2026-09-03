"""Windowless subprocess contracts for local-model GPU polling."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace


_CREATE_NO_WINDOW = 0x08000000


def test_vram_probe_hides_short_lived_smi_console(monkeypatch):
    import hermes_cli.local_runtime.hardware as hardware

    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "8192, 4096\n", "")

    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(
        hardware, "windows_hide_flags", lambda: _CREATE_NO_WINDOW, raising=False
    )
    monkeypatch.setattr(hardware.subprocess, "run", fake_run)

    assert hardware._nvidia_vram() == (8192 << 20, 4096 << 20)
    assert captured["creationflags"] == _CREATE_NO_WINDOW


def test_hardware_route_hides_live_gpu_status_console(monkeypatch):
    import hermes_cli.local_runtime.hardware as hardware
    import hermes_cli.web_routers.local_models as local_models

    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, "Example GPU, 37, 2048\n", "")

    budget = SimpleNamespace(
        uma=False,
        total_device_bytes=8192 << 20,
        usable_vram_bytes=6144 << 20,
    )
    monkeypatch.setattr(hardware, "probe_budget", lambda: budget)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: (8192 << 20, 4096 << 20))
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (32 << 30, 16 << 30))
    monkeypatch.setattr(hardware, "_nvidia_smi_path", lambda: "nvidia-smi")
    monkeypatch.setattr(
        local_models, "windows_hide_flags", lambda: _CREATE_NO_WINDOW, raising=False
    )
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = local_models.local_models_hardware()

    assert result["gpu_name"] == "Example GPU"
    assert result["gpu_util_percent"] == 37
    assert result["vram_used_bytes"] == 2048 << 20
    assert captured["creationflags"] == _CREATE_NO_WINDOW
