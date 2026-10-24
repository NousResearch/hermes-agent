"""Private TEST-FIRST proposal for the six jobs/occurrence file-writer entrypoints.

Each operation gets its own real EX lease and its own release-positive assertion.  The jobs
ContextVar is deliberately B while the spawned child starts with Hermes/config home A, proving
that the actual jobs writer lease follows B rather than a destination-only expectation.
"""
from __future__ import annotations

import json
from pathlib import Path

from tests.cron.writer_test_support import WriterChild


def _homes(tmp_path: Path) -> tuple[Path, Path]:
    home_a = tmp_path / "profile-a"
    home_b = tmp_path / "profile-b"
    home_a.mkdir()
    home_b.mkdir()
    return home_a, home_b


def _last_jsonl(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows
    return rows[-1]


def test_each_jobs_marker_telemetry_and_occurrence_writer_waits_for_b_ex(
    tmp_path: Path,
) -> None:
    home_a, home_b = _homes(tmp_path)
    store_b = {"store_home": str(home_b)}

    with WriterChild(home_a) as child:
        setup = child.run_unfenced("seed_job", {**store_b, "name": "opaque B setup"})
        assert setup["id"]

        # Group 1: each marker entrypoint is dispatched separately while the real B EX is held.
        child.run_under_exclusive("ticker_heartbeat", store_b, lease_home=home_b)
        child.run_under_exclusive("ticker_error", store_b, lease_home=home_b)
        ticker_error_path = home_b / "cron" / "ticker_last_error"
        ticker_error_lines = ticker_error_path.read_text(encoding="utf-8").splitlines()
        assert ticker_error_lines and ticker_error_lines[-1] == "opaque ticker failure"
        child.run_under_exclusive("ticker_clear", store_b, lease_home=home_b)
        assert not ticker_error_path.exists()

        # Group 2: only the two producer-defined telemetry filenames are used.
        child.run_under_exclusive("telemetry_recovery", store_b, lease_home=home_b)
        child.run_under_exclusive("telemetry_timezone", store_b, lease_home=home_b)

        # Group 3: the real catch-up counter read-modify-write is a separate case.
        child.run_under_exclusive("occurrence", store_b, lease_home=home_b)

    cron_b = home_b / "cron"
    cron_a = home_a / "cron"
    heartbeat = (cron_b / "ticker_heartbeat").read_text(encoding="utf-8").split()
    assert len(heartbeat) >= 2 and heartbeat[1].isdigit()
    assert float((cron_b / "ticker_last_success").read_text(encoding="utf-8").strip()) > 0
    assert not (cron_b / "ticker_last_error").exists()
    assert (cron_b / "catch_up_occurrences").read_text(encoding="utf-8").strip() == "1"

    recovery = _last_jsonl(cron_b / "persisted_error_recoveries.jsonl")
    assert recovery["job_id"] == "opaque-recovery-job"
    assert recovery["name"] == "opaque recovery job"
    assert recovery["previous_next_run_at"] == "opaque previous next run"
    assert isinstance(recovery["rearmed_at"], str) and recovery["rearmed_at"]

    timezone = _last_jsonl(cron_b / "timezone_migration_catchups.jsonl")
    assert timezone["job_id"] == "opaque-timezone-job"
    assert timezone["name"] == "opaque timezone job"
    assert timezone["expr"] == "every 1h"
    assert timezone["stored_next_run_at"] == "2026-10-17T05:00:00+00:00"
    assert timezone["normalized_next_run_at"] == "2026-10-17T06:00:00+00:00"
    assert isinstance(timezone["fired_at"], str) and timezone["fired_at"]

    assert (cron_b / "jobs.json").is_file()
    assert not (cron_a / "ticker_heartbeat").exists()
    assert not (cron_a / "ticker_last_success").exists()
    assert not (cron_a / "persisted_error_recoveries.jsonl").exists()
    assert not (cron_a / "timezone_migration_catchups.jsonl").exists()
    assert not (cron_a / "catch_up_occurrences").exists()
