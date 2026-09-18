import json
import subprocess
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import binaries, devices, hardware


@pytest.mark.parametrize("result", [
    SimpleNamespace(returncode=1, stdout=""),
    SimpleNamespace(returncode=0, stdout="not JSON"),
    SimpleNamespace(returncode=0, stdout="{}"),
    SimpleNamespace(returncode=0, stdout='[{"name":"Vulkan0","type":1,"total":-1,"free":0}]'),
    subprocess.TimeoutExpired("probe", 15),
])
def test_failed_native_probe_never_grants_system_ram_as_vram(monkeypatch, tmp_path, result):
    install = tmp_path / "b123" / "vulkan"
    monkeypatch.setattr(binaries, "server_binary", lambda p: p / "llama-server")
    def run(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(devices.subprocess, "run", run)
    budget = hardware.probe_budget(install_dir=install)
    assert budget.usable_vram_bytes == 0
    assert not budget.uma
    assert budget.device is None


def test_missing_executable_has_no_gpu_capacity(tmp_path):
    install = tmp_path / "vulkan"
    install.mkdir()
    (install / "manifest.json").write_text('{"verified_version":"test"}', encoding="utf-8")
    assert devices.probe_devices(install) == []
    budget = hardware.probe_budget(install_dir=install)
    assert budget.total_device_bytes == budget.usable_vram_bytes == 0
    assert not budget.uma


def test_probe_uses_selected_build_and_refreshes_memory(monkeypatch, tmp_path):
    selected = tmp_path / "b123" / "vulkan"
    other = tmp_path / "b456" / "cpu"
    for path in (selected, other):
        path.mkdir(parents=True)
    monkeypatch.setattr(binaries, "server_binary", lambda path: path / "llama-server")
    replies = [7 << 30, 5 << 30]
    def run(argv, **kwargs):
        assert str(selected) in argv
        assert argv[-1] == "vulkan"
        assert "-I" in argv
        assert kwargs["timeout"] == 15
        return SimpleNamespace(returncode=0, stdout=json.dumps([
            {"name": "Vulkan0", "type": 1, "total": 8 << 30, "free": replies.pop(0)},
            {"name": "CPU", "type": 0, "total": 64 << 30, "free": 50 << 30},
        ]))
    monkeypatch.setattr(devices.subprocess, "run", run)
    before = devices.probe_devices(selected)
    after = devices.probe_devices(selected)
    assert len(before) == len(after) == 1
    assert before[0]["total"] == after[0]["total"]
    assert before[0]["free"] > after[0]["free"]
