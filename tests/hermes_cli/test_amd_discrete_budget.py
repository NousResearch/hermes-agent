from __future__ import annotations

from pathlib import Path
import struct

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30


def test_discrete_amd_uses_vram_budget_instead_of_host_ram(monkeypatch):
    total = 24 * GIB
    free = 22 * GIB
    ram_total = int(62.4 * GIB)

    monkeypatch.setattr(hw, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda *_: None)
    monkeypatch.setattr(hw, "_amd_vram", lambda: (total, free))
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (ram_total, 48 * GIB))

    budget = hw.probe_budget(planning=True)

    assert budget.uma is False
    assert budget.total_device_bytes == total
    assert budget.usable_vram_bytes == total - max(
        hw._MARGIN_FLOOR, int(total * hw._MARGIN_FRACTION)
    )
    assert budget.ram_available_bytes == ram_total


def test_amd_probe_excludes_fusion_devices(tmp_path, monkeypatch):
    drm = tmp_path / "drm"
    devices: list[Path] = []
    for card, total, used in (("card0", 8 * GIB, GIB), ("card1", 24 * GIB, 2 * GIB)):
        device = drm / card / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text("0x1002\n", encoding="utf-8")
        (device / "mem_info_vram_total").write_text(str(total), encoding="utf-8")
        (device / "mem_info_vram_used").write_text(str(used), encoding="utf-8")
        devices.append(device)

    monkeypatch.setattr(
        hw,
        "_amd_device_integrated",
        lambda device: device.parent.name == "card0",
    )

    assert hw._amd_vram_from_devices(devices) == (24 * GIB, 22 * GIB)

    response = bytearray(hw._AMDGPU_INFO_BUFFER_SIZE)
    struct.pack_into("<I", response, hw._AMDGPU_DEVICE_ID_OFFSET, 0x744C)
    struct.pack_into(
        "<Q",
        response,
        hw._AMDGPU_IDS_FLAGS_OFFSET,
        hw._AMDGPU_IDS_FLAGS_FUSION,
    )
    assert hw._amd_info_integrated(response, 0x744C) is True
    assert hw._amd_info_integrated(response, 0x7448) is None
    struct.pack_into("<Q", response, hw._AMDGPU_IDS_FLAGS_OFFSET, 0)
    assert hw._amd_info_integrated(response, 0x744C) is False
