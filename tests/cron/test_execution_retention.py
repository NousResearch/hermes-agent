"""Cron outcome retention is a time floor, not a volume-dependent window."""

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest


@pytest.mark.parametrize("status", ["completed", "failed", "unknown"])
@pytest.mark.parametrize("age_seconds", [30 * 86400 - 1, 30 * 86400, 30 * 86400 + 1])
def test_retention_boundary_preserves_full_outcome(
    tmp_path, monkeypatch, status, age_seconds
):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    now = datetime(2026, 10, 18, 12, tzinfo=UTC)
    finished = now - timedelta(seconds=age_seconds)
    monkeypatch.setattr(executions, "_hermes_now", lambda: finished)
    row = executions.create_execution("boundary", source="builtin")
    executions.mark_execution_running(row["id"])
    if status == "unknown":
        assert executions.give_up_execution(row["id"], reason="worker interrupted")
    else:
        executions.finish_execution(
            row["id"], success=status == "completed", error="provider error"
        )
    original = executions.get_execution(row["id"])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now)

    with executions._transaction() as conn:
        executions._prune_unlocked(conn)

    retained = executions.get_execution(row["id"])
    if age_seconds > 30 * 86400:
        assert retained is None
    else:
        assert retained == original
        assert retained["status"] == status
        assert retained["started_at"] and retained["finished_at"]
        if status != "completed":
            assert retained["error"]


def test_high_volume_retains_recent_outcomes_and_inflight(tmp_path, monkeypatch):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    now = datetime(2026, 10, 18, 12, tzinfo=UTC)
    monkeypatch.setattr(executions, "_hermes_now", lambda: now - timedelta(days=31))
    old = executions.create_execution("old", source="builtin")
    executions.finish_execution(old["id"], success=True)
    inflight = executions.create_execution("live", source="builtin")
    executions.mark_execution_running(inflight["id"])
    monkeypatch.setattr(executions, "_hermes_now", lambda: now)
    recent = now - timedelta(days=29)
    stamp = recent.astimezone(timezone(timedelta(hours=-7))).isoformat()
    count = executions.MAX_TERMINAL_EXECUTIONS + 1
    with executions._transaction() as conn:
        conn.executemany(
            """INSERT INTO executions
               (id, job_id, source, process_id, pid, status,
                claimed_at, started_at, finished_at, error)
               VALUES (?, 'busy-job', 'builtin', 'worker', 1, 'failed', ?, ?, ?, ?)""",
            [
                (f"recent-{index}", stamp, stamp, stamp, "provider error")
                for index in range(count)
            ],
        )
    trigger = executions.create_execution("trigger", source="builtin")

    executions.finish_execution(trigger["id"], success=True)

    with executions._transaction() as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM executions WHERE job_id='busy-job' AND error='provider error'"
        ).fetchone()[0] == count
    assert executions.get_execution(old["id"]) is None
    assert executions.get_execution(inflight["id"])["status"] == "running"
    assert executions.get_execution(trigger["id"])["status"] == "completed"


@pytest.mark.parametrize("month,day", [(3, 20), (11, 15)])
@pytest.mark.parametrize("age_seconds", [30 * 86400 - 1, 30 * 86400, 30 * 86400 + 1])
def test_retention_measures_elapsed_days_across_dst(
    tmp_path, monkeypatch, month, day, age_seconds
):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "executions.db")
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 0)
    zone = ZoneInfo("America/Los_Angeles")
    now = datetime(2026, month, day, 12, tzinfo=zone)
    finished = (now.astimezone(UTC) - timedelta(seconds=age_seconds)).astimezone(zone)
    monkeypatch.setattr(executions, "_hermes_now", lambda: finished)
    row = executions.create_execution("dst-boundary", source="builtin")
    executions.mark_execution_running(row["id"])
    executions.finish_execution(row["id"], success=False, error="provider error")
    original = executions.get_execution(row["id"])
    assert original is not None
    monkeypatch.setattr(executions, "_hermes_now", lambda: now)

    with executions._transaction() as conn:
        executions._prune_unlocked(conn)

    if age_seconds > 30 * 86400:
        assert executions.get_execution(row["id"]) is None
    else:
        assert executions.get_execution(row["id"]) == original
