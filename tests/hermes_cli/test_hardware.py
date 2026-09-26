"""Hardware budget contracts for discrete Intel Xe devices."""

from __future__ import annotations

import struct

import pytest


_GIB = 1 << 30
_REGION_FMT = "<HHIQQQQ6Q"


def _region(mem_class: int, total: int, used: int) -> bytes:
    return struct.pack(
        _REGION_FMT,
        mem_class,
        0,
        4096,
        total,
        used,
        total,
        used,
        *(0 for _ in range(6)),
    )


def test_xe_memory_query_parses_vram_and_clamps_used():
    from hermes_cli.local_runtime.hardware import _parse_xe_mem_regions

    payload = struct.pack("<II", 3, 0)
    payload += _region(0, 128 * _GIB, 0)
    payload += _region(1, 32 * _GIB, 4 * _GIB)
    payload += _region(1, 8 * _GIB, 12 * _GIB)

    assert _parse_xe_mem_regions(payload) == (40 * _GIB, 24 * _GIB)


@pytest.mark.linux_only
def test_probe_budget_uses_intel_vram_when_query_is_available(monkeypatch):
    import hermes_cli.local_runtime.hardware as hardware

    monkeypatch.setattr(hardware, "_nvidia_vram", lambda: None)
    monkeypatch.setattr(hardware, "_intel_vram", lambda: (32 * _GIB, 24 * _GIB))
    monkeypatch.setattr(hardware, "_ram_bytes", lambda: (128 * _GIB, 96 * _GIB))

    budget = hardware.probe_budget()

    margin = max(2 * _GIB, int(32 * _GIB * 0.09))
    assert budget.total_device_bytes == 32 * _GIB
    assert budget.usable_vram_bytes == 24 * _GIB - margin
    assert budget.ram_available_bytes == 96 * _GIB
    assert budget.uma is False
