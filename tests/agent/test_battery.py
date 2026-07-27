"""Behavior tests for the status-bar battery helper (agent/battery.py)."""

from __future__ import annotations

import sys
import types

import pytest

from agent import battery as battery_mod
from agent.battery import (
    BatteryStatus,
    battery_category,
    battery_glyph,
    format_battery,
    read_battery,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    battery_mod.clear_cache()
    yield
    battery_mod.clear_cache()


def _fake_psutil(percent, plugged):
    """Install a fake psutil module whose sensors_battery returns a reading."""
    mod = types.ModuleType("psutil")
    reading = types.SimpleNamespace(percent=percent, power_plugged=plugged)
    mod.sensors_battery = lambda: reading  # type: ignore[attr-defined]
    return mod


def test_read_battery_no_psutil(monkeypatch):
    # Force the import inside read_battery to fail.
    monkeypatch.setitem(sys.modules, "psutil", None)
    status = read_battery(use_cache=False)
    assert status.available is False
    assert status.percent is None


def test_read_battery_no_battery(monkeypatch):
    mod = types.ModuleType("psutil")
    mod.sensors_battery = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", mod)

    status = read_battery(use_cache=False)
    assert status.available is False


def test_read_battery_reads_and_clamps(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(87.6, False))
    status = read_battery(use_cache=False)
    assert status.available is True
    assert status.percent == 88  # rounded
    assert status.plugged is False


def test_read_battery_clamps_out_of_range(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(150, True))
    status = read_battery(use_cache=False)
    assert status.percent == 100
    assert status.plugged is True


def test_read_battery_caches(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(50, False))
    first = read_battery(use_cache=True)
    assert first.percent == 50

    # Swap the reading; a cached call must still return the first value.
    monkeypatch.setitem(sys.modules, "psutil", _fake_psutil(10, True))
    cached = read_battery(use_cache=True)
    assert cached.percent == 50

    # Bypassing the cache picks up the new reading.
    fresh = read_battery(use_cache=False)
    assert fresh.percent == 10


@pytest.mark.parametrize(
    "percent,plugged,expected",
    [
        (100, False, "good"),
        (51, False, "good"),
        (50, False, "warn"),
        (21, False, "warn"),
        (20, False, "bad"),
        (11, False, "bad"),
        (10, False, "critical"),
        (1, False, "critical"),
        # On AC power the level never reads as low.
        (5, True, "good"),
    ],
)
def test_battery_category_thresholds(percent, plugged, expected):
    status = BatteryStatus(available=True, percent=percent, plugged=plugged)
    assert battery_category(status) == expected


def test_battery_category_unavailable_is_dim():
    assert battery_category(BatteryStatus(available=False)) == "dim"
    assert battery_category(BatteryStatus(available=True, percent=None)) == "dim"


def test_format_and_glyph():
    on_battery = BatteryStatus(available=True, percent=82, plugged=False)
    charging = BatteryStatus(available=True, percent=82, plugged=True)

    assert battery_glyph(on_battery) == "\U0001f50b"  # 🔋
    assert battery_glyph(charging) == "\u26a1"  # ⚡
    assert format_battery(on_battery) == "\U0001f50b 82%"
    assert format_battery(charging) == "\u26a1 82%"
    assert format_battery(BatteryStatus(available=False)) == ""
