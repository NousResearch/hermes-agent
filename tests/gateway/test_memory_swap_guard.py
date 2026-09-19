"""Swap-aware memory-pressure guard (#116333).

classify_pressure read MemAvailable only, so a host deep into swap could
still read "ok" while the kernel OOM-killed workers. These pins cover the
swap escalation: optional swap params, escalate-never-downgrade, and
zero/missing swap skipping the tier.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gateway.lifecycle_ledger import sample_memory
from gateway.memory_status import classify_pressure, collect_memory_status
from gateway.shutdown_watchdog import get_loop_heartbeat_path
from hermes_cli import kanban_db_dispatch as kbd

GIB = 1024 * 1024  # KiB per GiB
_NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)

# MemAvailable 50% of total: firmly "ok" on the MemAvailable tiers alone.
OK_MEM = (512 * 1024, 1024 * 1024)


def test_swap_half_escalates_ok_to_elevated() -> None:
    assert classify_pressure(*OK_MEM, swap_used_kib=60, swap_total_kib=100) == "elevated"


def test_swap_80pct_escalates_ok_to_critical() -> None:
    assert classify_pressure(*OK_MEM, swap_used_kib=80, swap_total_kib=100) == "critical"


def test_swap_escalates_but_never_downgrades() -> None:
    mem_critical = (32 * 1024, 8 * 1024 * 1024)
    assert classify_pressure(*mem_critical) == "critical"
    assert classify_pressure(*mem_critical, swap_used_kib=0, swap_total_kib=100) == "critical"
    mem_elevated = (100 * 1024, 1024 * 1024)
    assert classify_pressure(*mem_elevated, swap_used_kib=90, swap_total_kib=100) == "critical"


def test_missing_swap_leaves_verdict_unchanged() -> None:
    assert classify_pressure(*OK_MEM) == "ok"
    assert classify_pressure(*OK_MEM, swap_used_kib=None, swap_total_kib=None) == "ok"


def test_zero_swap_total_skips_swap_tier() -> None:
    # Swapless host: no division, no verdict change.
    assert classify_pressure(*OK_MEM, swap_used_kib=0, swap_total_kib=0) == "ok"


def test_sample_memory_reports_swap_total(monkeypatch) -> None:
    from gateway import lifecycle_ledger as ll

    def fake_proc_fields(path: str, wanted: dict) -> dict:
        if path == "/proc/self/status":
            return {"rss_kib": 1234}
        if path == "/proc/meminfo":
            return {
                "mem_total_kib": 8 * GIB,
                "mem_available_kib": 4 * GIB,
                "SwapTotal": 4 * GIB,
                "SwapFree": GIB,
            }
        raise AssertionError(path)

    monkeypatch.setattr(ll, "_proc_fields", fake_proc_fields)
    sample = sample_memory()
    assert sample["swap_total_kib"] == 4 * GIB
    assert sample["swap_used_kib"] == 3 * GIB


def test_memory_pressure_level_escalates_on_swap() -> None:
    sample = {
        "mem_available_kib": GIB // 2,
        "mem_total_kib": GIB,
        "swap_used_kib": 64,
        "swap_total_kib": 100,
    }
    assert kbd._memory_pressure_level(sample) == "elevated"


def test_collect_memory_status_escalates_on_swap(tmp_path: Path) -> None:
    path = get_loop_heartbeat_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "pid": 12345,
            "updated_at": _NOW.isoformat(),
            "monotonic": 1.0,
            "mem": {
                "rss_kib": 400 * 1024,
                "mem_total_kib": 1024 * 1024,
                "mem_available_kib": 512 * 1024,
                "swap_used_kib": 70,
                "swap_total_kib": 100,
            },
        }),
        encoding="utf-8",
    )
    assert collect_memory_status(tmp_path, now=_NOW)["pressure"] == "elevated"
