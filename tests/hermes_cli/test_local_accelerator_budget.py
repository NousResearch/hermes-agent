from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import hardware, presets
from hermes_cli.local_runtime.estimator import ModelProfile


@pytest.mark.parametrize("integrated_first", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
def test_accelerator_budget_and_preset_share_topology(monkeypatch, tmp_path, integrated_first, discrete):
    install = tmp_path / "b123" / "vulkan"
    integrated = {"name": "Vulkan1", "description": "AMD Radeon Graphics", "type": 2,
                  "total": 32 << 30, "free": 30 << 30}
    dedicated = {"name": "Vulkan0", "description": "AMD Radeon RX", "type": 1,
                 "total": 16 << 30, "free": 10 << 30}
    devices = [integrated, dedicated] if integrated_first else [dedicated, integrated]
    if not discrete:
        devices = [integrated]
    monkeypatch.setattr(hardware, "_accelerator_devices", lambda path: devices)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 << 30, 40 << 30))
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    for planning in (False, True):
        budget = hardware.probe_budget(planning=planning, install_dir=install)
        assert budget.uma is not discrete
        assert budget.device == (dedicated if discrete else integrated)["name"]
        if discrete:
            assert budget.total_device_bytes == dedicated["total"]
            assert budget.usable_vram_bytes == (dedicated["total"] if planning else dedicated["free"]) - (2 << 30)
            assert budget.ram_available_bytes == ((64 if planning else 40) << 30)
        else:
            assert budget.total_device_bytes == 64 << 30
            assert budget.usable_vram_bytes == int(((64 if planning else 40) << 30) * .8)
            assert budget.ram_available_bytes == 0
        monkeypatch.setattr(presets, "read_gguf_header", lambda path: SimpleNamespace(sampling_defaults={}))
        monkeypatch.setattr(presets, "profile_from_gguf", lambda header: ModelProfile("small", 1 << 30, 0, 65536, []))
        entry = presets.preset_for_model(tmp_path / "small.gguf", budget, set())
        assert entry.keys["device"] == budget.device


@pytest.mark.parametrize("backend", ["vulkan", "hip"])
def test_uninstalled_accelerator_does_not_become_ram_budget(monkeypatch, tmp_path, backend):
    from hermes_cli.config import load_config, save_config
    config = load_config()
    config["local_runtime"]["backend"] = backend
    save_config(config)
    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 << 30, 40 << 30))
    budget = hardware.probe_budget(planning=True)
    assert not budget.uma
    assert budget.total_device_bytes == budget.usable_vram_bytes == 0


@pytest.mark.parametrize("backend", ["cpu", "metal"])
def test_ram_backend_does_not_probe_installed_accelerators(monkeypatch, tmp_path, backend):
    def unexpected(*args):
        raise AssertionError("CPU budget must not probe a GPU")
    monkeypatch.setattr(hardware, "_accelerator_devices", unexpected)
    monkeypatch.setattr(hardware, "_nvidia_vram", unexpected)
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (64 << 30, 40 << 30))
    budget = hardware.probe_budget(planning=True, install_dir=tmp_path / "b123" / backend)
    assert budget.uma and budget.device is None
    assert budget.total_device_bytes == 64 << 30
