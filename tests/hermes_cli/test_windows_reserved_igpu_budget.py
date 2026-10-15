"""Windows BIOS-reserved AMD memory is outside OS RAM, not an ordinary shared iGPU pool.

Regression for #102593. The backend's heap total is not an additional physical memory pool.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from hermes_cli.local_runtime import devices, hardware as hw
from hermes_cli.local_runtime.devices import GGML_DEVICE_IGPU
from hermes_platform.host import gpu_adapters
from hermes_platform.host.gpu_adapters import Adapter

pytestmark = pytest.mark.platforms("windows")
GIB = 1 << 30
AMD = Adapter("AMD Radeon(TM) 8060S Graphics", 0x1002, 96 * GIB, 16 * GIB, False)


@pytest.fixture
def machine(monkeypatch, tmp_path):
    device = {
        "description": AMD.description, "type": GGML_DEVICE_IGPU,
        "total": 114 * GIB, "free": 108 * GIB,
    }
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (32 * GIB, 20 * GIB))
    monkeypatch.setattr(hw, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hw, "_device_pool_view", lambda: None)
    monkeypatch.setattr(hw, "_accelerator_cache", {})
    monkeypatch.setattr(hw, "_configured_engine",
                        lambda: SimpleNamespace(backend="vulkan", binary=tmp_path / "llama-server.exe"))
    monkeypatch.setattr(devices, "probe_devices", lambda directory, backend: [dict(device)])
    monkeypatch.setattr(hw, "_gpu_engine_runs_the_model", lambda: True)
    monkeypatch.setattr(hw, "gpu_class", lambda: "amd")
    # DXGI can enumerate this same adapter twice, including over Remote Desktop.
    monkeypatch.setattr(gpu_adapters, "windows_gpu_adapters", lambda: (AMD, AMD))
    return device


def test_reserved_pool_admits_large_weights_without_double_counting_ram(machine):
    budget = hw.probe_budget(planning=True)

    assert budget.uma and not budget.lazy_reads
    assert budget.total_device_bytes == 112 * GIB
    assert budget.ram_available_bytes == 0
    assert budget.usable_vram_bytes == (
        96 * GIB - int(96 * GIB * hw._MARGIN_FRACTION)
        + int(16 * GIB * (1 - hw._UMA_HEADROOM_FRACTION))
    )
    assert budget.usable_vram_bytes > 94 * GIB
    assert budget.gpu_name == AMD.description


def test_live_budget_tracks_other_users_and_released_server_memory(machine, monkeypatch):
    capacity = hw.probe_budget(planning=True)
    machine["free"] = 70 * GIB

    live = hw.probe_budget()
    assert live.usable_vram_bytes == (112 - (114 - 70) - 1) * GIB
    assert hw.launch_budget(capacity).usable_vram_bytes == live.usable_vram_bytes
    own = hw.launch_budget(capacity, own_bytes=40 * GIB)
    assert own.usable_vram_bytes == capacity.usable_vram_bytes

    monkeypatch.setattr(hw, "_ram_bytes", lambda: (32 * GIB, 2 * GIB))
    low_ram = hw.launch_budget(capacity)
    assert low_ram.usable_vram_bytes == (96 + 2 - (114 - 70) - 1) * GIB
    machine["free"] = 0
    assert hw.launch_budget(capacity).usable_vram_bytes == 0


@pytest.mark.parametrize("kind", ["unknown", "small-carve", "wrong-adapter", "no-engine"])
def test_uncertain_or_ordinary_igpus_do_not_gain_reserved_memory(machine, monkeypatch, kind):
    if kind == "unknown":
        monkeypatch.setattr(gpu_adapters, "windows_gpu_adapters", lambda: ())
    elif kind == "small-carve":
        adapter = replace(AMD, dedicated_bytes=1 * GIB)
        monkeypatch.setattr(gpu_adapters, "windows_gpu_adapters", lambda: (adapter,))
    elif kind == "wrong-adapter":
        machine["description"] = "Another integrated GPU"
    else:
        monkeypatch.setattr(hw, "_accelerator_device", lambda **_: None)

    budget = hw.probe_budget(planning=True)
    assert budget.total_device_bytes <= 32 * GIB
    assert budget.usable_vram_bytes <= int(32 * GIB * 0.8)


def test_reserved_launch_requires_valid_fresh_free_memory(machine, monkeypatch):
    capacity = hw.probe_budget(planning=True)
    monkeypatch.setattr(hw, "_accelerator_device", lambda **_: None)
    assert hw.launch_budget(capacity) is None

    monkeypatch.setattr(hw, "_accelerator_device", lambda **_: machine)
    del machine["free"]
    assert hw.launch_budget(capacity) is None


@pytest.mark.parametrize("reading", ["launch", "live"])
def test_failed_fresh_probe_preserves_reserved_capacity(machine, monkeypatch, reading):
    capacity = hw.probe_budget(planning=True)
    assert capacity.total_device_bytes == 112 * GIB
    monkeypatch.setattr(devices, "probe_devices", lambda directory, backend: [])

    if reading == "launch":
        assert hw.launch_budget(capacity) is None
    else:
        assert hw.probe_budget().total_device_bytes <= 32 * GIB

    assert hw._accelerator_device(fresh=True) is None
    assert hw.probe_budget(planning=True) == capacity
