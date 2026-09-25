"""Behavior tests for the live telemetry frame (agent/system_metrics.py)."""

from __future__ import annotations

import time
import types

import pytest

from agent import system_metrics
from agent.system_metrics import _rate, _volumes, read_system_metrics


@pytest.fixture(autouse=True)
def _fresh_sampler():
    system_metrics.reset()
    yield
    system_metrics.reset()


def test_first_frame_has_no_window_and_the_next_frame_covers_the_gap():
    first = read_system_metrics(use_cache=False)
    assert first["interval_s"] is None
    assert first["disk"]["read_bps"] is None and first["net"]["rx_bps"] is None
    assert first["gpus"] == [] and first["cpu"]["clusters"] == []  # SoC delta would span milliseconds

    time.sleep(0.3)
    second = read_system_metrics(use_cache=False)
    assert second["interval_s"] > 0
    assert second["ts"] > first["ts"]
    for rate in (second["net"]["rx_bps"], second["net"]["tx_bps"]):
        assert rate is None or rate >= 0


def test_frame_sections_agree_with_each_other():
    read_system_metrics(use_cache=False)
    frame = read_system_metrics(use_cache=False)
    cpu, mem = frame["cpu"], frame["memory"]
    assert len(cpu["per_core"]) == cpu["count_logical"]
    assert 0 <= cpu["percent"] <= 100
    assert mem["used"] <= mem["total"] and mem["available"] <= mem["total"]
    for vol in frame["disk"]["volumes"]:
        assert vol["used"] <= vol["total"]
        assert vol["percent"] == pytest.approx(100 * vol["used"] / vol["total"], abs=0.1)
    assert frame["process"]["rss"] > 0


def test_pollers_inside_the_ttl_share_one_window():
    first = read_system_metrics()
    assert read_system_metrics() is first
    assert read_system_metrics(use_cache=False) is not first


def test_counter_reset_yields_no_rate():
    assert _rate(100, 400, 1.0) is None
    assert _rate(400, None, 1.0) is None
    assert _rate(400, 100, 0.0) is None
    assert _rate(400, 100, 2.0) == 150


def test_container_mounts_collapse_to_one_row_with_container_usage():
    gib = 1 << 30
    usage = {
        "/": types.SimpleNamespace(total=100 * gib, used=10 * gib, free=40 * gib),
        "/System/Volumes/Data": types.SimpleNamespace(total=100 * gib, used=50 * gib, free=40 * gib),
        "/Volumes/External": types.SimpleNamespace(total=500 * gib, used=5 * gib, free=495 * gib),
        "/System/Volumes/xarts": types.SimpleNamespace(total=gib // 2, used=0, free=gib // 2),
    }
    fake = types.SimpleNamespace(
        disk_partitions=lambda all=False: [types.SimpleNamespace(mountpoint=m, fstype="apfs") for m in usage],
        disk_usage=lambda m: usage[m],
    )
    rows = _volumes(fake)
    assert [r["mount"] for r in rows] == ["/Volumes/External", "/"]
    root = rows[1]
    assert root["used"] == 60 * gib  # total - shared free, not the "/" volume's own 10 GiB
    assert root["percent"] == pytest.approx(60.0)
