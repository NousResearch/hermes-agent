from __future__ import annotations

import ctypes
import os
from pathlib import Path
import struct

import pytest

import hermes_cli.local_runtime.hardware as hw

GIB = 1 << 30


def test_discrete_amd_uses_vram_budget_instead_of_host_ram(monkeypatch):
    vram_total = 25_753_026_560
    vram_used = 1_494_384_640
    ram_total = 67_004_100_608

    monkeypatch.setattr(hw, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hw, "_unified_pool_bytes", lambda *_: None)
    monkeypatch.setattr(
        hw,
        "_amd_vram",
        lambda: (vram_total, vram_total - vram_used),
        raising=False,
    )
    monkeypatch.setattr(hw, "_ram_bytes", lambda: (ram_total, 48 * GIB))

    budget = hw.probe_budget(planning=True)

    margin = max(hw._MARGIN_FLOOR, int(vram_total * hw._MARGIN_FRACTION))
    assert budget.uma is False
    assert budget.total_device_bytes == vram_total
    assert budget.usable_vram_bytes == vram_total - margin
    assert budget.ram_available_bytes == ram_total


@pytest.mark.linux_only
def test_amd_probe_uses_driver_classification_and_discrete_vram(tmp_path, monkeypatch):
    devices = []
    responses = {}
    for card, render, device_id, flags, total, used in (
        ("card0", "renderD128", 0x15BF, 1, 96 * GIB, GIB),
        ("card1", "renderD129", 0x744C, 0, 24 * GIB, 2 * GIB),
    ):
        device = tmp_path / card / "device"
        (device / "drm").mkdir(parents=True)
        (device / "drm" / render).touch()
        (device / "vendor").write_text("0x1002\n", encoding="utf-8")
        (device / "device").write_text(f"0x{device_id:x}\n", encoding="utf-8")
        (device / "mem_info_vram_total").write_text(str(total), encoding="utf-8")
        (device / "mem_info_vram_used").write_text(str(used), encoding="utf-8")
        devices.append(device)
        responses[render] = (device_id, flags)

    descriptors = {}
    closed = []

    def fake_open(path, flags):
        render = Path(path).name
        assert flags == os.O_RDWR | os.O_CLOEXEC
        descriptor = len(descriptors) + 100
        descriptors[descriptor] = render
        return descriptor

    class FakeCommandWrite:
        argtypes = None
        restype = None

        def __call__(self, descriptor, command, request_pointer, request_size):
            assert command == hw._DRM_AMDGPU_INFO
            assert request_size == 32
            request = request_pointer._obj
            assert request.query == hw._AMDGPU_INFO_DEV_INFO
            device_id, flags = responses[descriptors[descriptor]]
            response = bytearray(hw._AMDGPU_INFO_BUFFER_SIZE)
            struct.pack_into("=I", response, hw._AMDGPU_DEVICE_ID_OFFSET, device_id)
            struct.pack_into("=Q", response, hw._AMDGPU_IDS_FLAGS_OFFSET, flags)
            ctypes.memmove(request.return_pointer, bytes(response), len(response))
            return 0

    class FakeLibdrm:
        drmCommandWrite = FakeCommandWrite()

    monkeypatch.setattr(hw.os, "open", fake_open)
    monkeypatch.setattr(hw.os, "close", closed.append)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: FakeLibdrm())

    assert hw._amd_vram_from_devices(devices) == (24 * GIB, 22 * GIB)
    assert closed == [100, 101]
