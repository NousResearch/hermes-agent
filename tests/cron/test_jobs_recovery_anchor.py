"""Recovery of a recurring job with a missing ``next_run_at`` must anchor on
``last_run_at``, not on ``now`` — otherwise the currently-due window is
silently skipped (a weekly job recovered around its slot waits a full week).
"""
from datetime import datetime, timezone

from cron.jobs import get_due_jobs, get_job, save_jobs

# Reuse the shared cron test fixture.
from tests.cron.test_jobs import tmp_cron_dir  # noqa: F401


def _job(job_id: str, schedule: dict, last_run_at):
    return {
        "id": job_id,
        "name": job_id,
        "prompt": "check the feeds",
        "schedule": schedule,
        "schedule_display": schedule.get("display", ""),
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": "2026-03-01T09:00:00+00:00",
        "next_run_at": None,
        "last_run_at": last_run_at,
        "last_status": "ok" if last_run_at else None,
        "last_error": None,
        "deliver": "local",
        "origin": None,
    }


def test_recurring_recovery_anchors_on_last_run(tmp_cron_dir, monkeypatch):  # noqa: F811
    """Weekly Monday-09:00 job, last ran a week ago Monday, next_run_at lost,
    recovery happens Wednesday: the missed Monday slot must fire now (single
    catch-up via the stale-grace path), not wait until NEXT Monday."""
    # Wednesday 2026-03-18 12:00 UTC; the missed slot was Monday 03-16 09:00.
    now = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

    save_jobs([
        _job(
            "weekly-anchor",
            {"kind": "cron", "expr": "0 9 * * 1", "display": "every Monday 09:00"},
            last_run_at="2026-03-09T09:00:05+00:00",
        )
    ])

    due_ids = [job["id"] for job in get_due_jobs()]

    assert due_ids == ["weekly-anchor"], (
        "recovery anchored on `now` skips the missed Monday slot entirely"
    )
    # The stale-grace path fast-forwards past the backlog: after the catch-up
    # dispatch the persisted next_run_at points at a FUTURE occurrence.
    recovered = get_job("weekly-anchor")
    nxt = datetime.fromisoformat(recovered["next_run_at"])
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=timezone.utc)
    assert nxt > now


def test_recurring_recovery_without_history_still_anchors_on_now(tmp_cron_dir, monkeypatch):  # noqa: F811
    """A job that has NEVER run has no missed window to catch up — recovery
    keeps the existing behaviour and schedules the next future occurrence."""
    now = datetime(2026, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: now)

    save_jobs([
        _job(
            "weekly-fresh",
            {"kind": "cron", "expr": "0 9 * * 1", "display": "every Monday 09:00"},
            last_run_at=None,
        )
    ])

    assert get_due_jobs() == []
    recovered = get_job("weekly-fresh")
    nxt = datetime.fromisoformat(recovered["next_run_at"])
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=timezone.utc)
    assert nxt > now
