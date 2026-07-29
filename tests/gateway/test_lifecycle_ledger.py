"""Tests for gateway.lifecycle_ledger — unclean-shutdown detection (NS-608).

The ledger is a tiny sentinel state machine:
``record_startup`` claims ``state/gateway.lifecycle.json`` as
``phase=running``; every exit path calls ``mark_exited``; the next boot's
``record_startup``/``detect_unclean_exit`` reports a still-``running``
sentinel from a dead process as an unclean death (SIGKILL / OOM / VM loss)
and enriches the report with the last heartbeat's memory sample.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from gateway.lifecycle_ledger import (
    detect_unclean_exit,
    get_lifecycle_sentinel_path,
    mark_exited,
    read_prior_exit_label,
    record_startup,
    sample_memory,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max on Linux; never alive


def _write_sentinel(home: Path, payload: dict) -> Path:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _read_sentinel(home: Path) -> dict:
    return json.loads(get_lifecycle_sentinel_path(home).read_text(encoding="utf-8"))


def _write_heartbeat(home: Path, payload: dict) -> Path:
    path = home / "state" / "gateway.heartbeat"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _exit_diag_records(home: Path) -> list[dict]:
    path = home / "logs" / "gateway-exit-diag.log"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# sample_memory
# ---------------------------------------------------------------------------


def test_sample_memory_never_raises() -> None:
    sample = sample_memory()
    assert isinstance(sample, dict)


@pytest.mark.skipif(sys.platform != "linux", reason="/proc is Linux-only")
def test_sample_memory_has_expected_keys_on_linux() -> None:
    sample = sample_memory()
    assert sample.get("rss_kib", 0) > 0
    assert sample.get("mem_total_kib", 0) > 0
    assert "mem_available_kib" in sample


# ---------------------------------------------------------------------------
# First boot / clean lifecycle
# ---------------------------------------------------------------------------


def test_first_boot_reports_nothing_and_claims_sentinel(tmp_path: Path) -> None:
    assert record_startup(home=tmp_path) is None
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["phase"] == "running"
    assert sentinel["pid"] == os.getpid()
    assert "start_time" in sentinel


def test_clean_exit_then_boot_reports_nothing(tmp_path: Path) -> None:
    record_startup(home=tmp_path)
    mark_exited(0, reason="graceful_shutdown", home=tmp_path)

    sentinel = _read_sentinel(tmp_path)
    assert sentinel["phase"] == "exited"
    assert sentinel["exit_code"] == 0
    assert sentinel["exit_reason"] == "graceful_shutdown"

    assert record_startup(home=tmp_path) is None
    assert _exit_diag_records(tmp_path) == []


def test_mark_exited_records_watchdog_reason(tmp_path: Path) -> None:
    record_startup(home=tmp_path)
    mark_exited(70, reason="loop_liveness_watchdog", home=tmp_path)
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["exit_reason"] == "loop_liveness_watchdog"
    assert sentinel["exit_code"] == 70


# ---------------------------------------------------------------------------
# Unclean-death detection
# ---------------------------------------------------------------------------


def test_running_sentinel_from_dead_pid_is_unclean(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {
        "phase": "running",
        "pid": _DEAD_PID,
        "start_time": 1000.0,
        "started_at": "2026-07-11T04:30:00+00:00",
    })

    evidence = detect_unclean_exit(home=tmp_path)
    assert evidence is not None
    assert evidence["prior_pid"] == _DEAD_PID
    assert evidence["prior_started_at"] == "2026-07-11T04:30:00+00:00"


def test_record_startup_persists_unclean_report_and_reclaims(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {
        "phase": "running",
        "pid": _DEAD_PID,
        "start_time": 1000.0,
        "started_at": "2026-07-11T04:30:00+00:00",
    })

    evidence = record_startup(home=tmp_path)
    assert evidence is not None

    records = _exit_diag_records(tmp_path)
    assert len(records) == 1
    assert records[0]["tag"] == "gateway.previous_unclean_exit"
    assert records[0]["prior_pid"] == _DEAD_PID
    assert records[0]["pid"] == os.getpid()

    # Sentinel reclaimed for the new life.
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["phase"] == "running"
    assert sentinel["pid"] == os.getpid()


def test_unclean_report_includes_last_heartbeat_memory(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {
        "phase": "running", "pid": _DEAD_PID, "start_time": 1000.0,
    })
    _write_heartbeat(tmp_path, {
        "pid": _DEAD_PID,
        "updated_at": "2026-07-12T19:33:00+00:00",
        "mem": {
            "rss_kib": 900_000,
            "mem_total_kib": 2_015_136,
            "mem_available_kib": 40_000,  # ~2% available → OOM suspicion
            "swap_used_kib": 900_000,
        },
    })

    evidence = detect_unclean_exit(home=tmp_path)
    assert evidence is not None
    assert evidence["last_heartbeat_at"] == "2026-07-12T19:33:00+00:00"
    assert evidence["last_heartbeat_mem"]["mem_available_kib"] == 40_000
    assert evidence["suspected_oom"] is True


def test_healthy_memory_heartbeat_does_not_suspect_oom(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {
        "phase": "running", "pid": _DEAD_PID, "start_time": 1000.0,
    })
    _write_heartbeat(tmp_path, {
        "pid": _DEAD_PID,
        "updated_at": "2026-07-12T19:33:00+00:00",
        "mem": {
            "mem_total_kib": 2_015_136,
            "mem_available_kib": 1_000_000,
        },
    })

    evidence = detect_unclean_exit(home=tmp_path)
    assert evidence is not None
    assert "suspected_oom" not in evidence


def test_live_owner_is_not_reported_as_unclean(tmp_path: Path) -> None:
    """A live matching PID means a --replace takeover is in flight, not a
    death — the detector must stay quiet (start_time omitted → assume alive)."""
    _write_sentinel(tmp_path, {
        "phase": "running",
        "pid": os.getpid(),
    })
    assert detect_unclean_exit(home=tmp_path) is None


def test_corrupt_sentinel_is_ignored(tmp_path: Path) -> None:
    path = get_lifecycle_sentinel_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert detect_unclean_exit(home=tmp_path) is None
    assert record_startup(home=tmp_path) is None
    # And the boot still claims a fresh sentinel.
    assert _read_sentinel(tmp_path)["phase"] == "running"


# ---------------------------------------------------------------------------
# Takeover ownership guard on mark_exited
# ---------------------------------------------------------------------------


def test_old_life_cannot_clobber_new_owner_sentinel(tmp_path: Path) -> None:
    """--replace: the replacement claims the sentinel while the old process
    is mid-teardown; the old life's mark_exited must be a no-op."""
    _write_sentinel(tmp_path, {
        "phase": "running",
        "pid": os.getpid() + 1,  # someone else owns it
        "start_time": 2000.0,
    })
    mark_exited(0, reason="graceful_shutdown", home=tmp_path)
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["phase"] == "running"
    assert sentinel["pid"] == os.getpid() + 1


def test_mark_exited_without_prior_sentinel_writes_exited(tmp_path: Path) -> None:
    mark_exited(1, reason="graceful_shutdown", home=tmp_path)
    assert _read_sentinel(tmp_path)["phase"] == "exited"


def test_mark_exited_leaves_pid_none_sentinel_alone(tmp_path: Path) -> None:
    """A sentinel with pid=None has unknown ownership — mark_exited must not
    clobber it with a clean-exit claim it cannot prove is its own."""
    _write_sentinel(tmp_path, {"phase": "running", "pid": None, "start_time": 2000.0})
    mark_exited(0, reason="graceful_shutdown", home=tmp_path)
    sentinel = _read_sentinel(tmp_path)
    assert sentinel["phase"] == "running"
    assert sentinel["pid"] is None


def test_mark_exited_rewrites_own_sentinel(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {
        "phase": "running", "pid": os.getpid(), "start_time": 2000.0,
    })
    mark_exited(0, reason="graceful_shutdown", home=tmp_path)
    assert _read_sentinel(tmp_path)["phase"] == "exited"


# ---------------------------------------------------------------------------
# read_prior_exit_label (container-boot annotation)
# ---------------------------------------------------------------------------


def test_prior_exit_label_unknown_when_no_sentinel(tmp_path: Path) -> None:
    assert read_prior_exit_label(tmp_path) == "unknown"


def test_prior_exit_label_clean_after_exit(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {"phase": "exited", "pid": 123, "exit_code": 0})
    assert read_prior_exit_label(tmp_path) == "clean"


def test_prior_exit_label_unclean_when_still_running(tmp_path: Path) -> None:
    _write_sentinel(tmp_path, {"phase": "running", "pid": _DEAD_PID})
    assert read_prior_exit_label(tmp_path) == "unclean"


def test_prior_exit_label_survives_corrupt_sentinel(tmp_path: Path) -> None:
    path = get_lifecycle_sentinel_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("garbage", encoding="utf-8")
    assert read_prior_exit_label(tmp_path) == "unknown"
