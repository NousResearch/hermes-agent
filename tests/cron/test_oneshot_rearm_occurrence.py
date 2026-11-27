"""An explicit one-shot re-arm creates a fresh occurrence without deleting history."""

from datetime import datetime, timedelta, timezone

import pytest

from cron import executions, jobs
from cron.scheduler_provider import InProcessCronScheduler


def test_rearm_can_reuse_a_completed_instant_once(monkeypatch):
    now = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(executions, "_hermes_now", lambda: now)
    slot = now + timedelta(minutes=1)
    job = jobs.create_job("local fixture", slot.isoformat())
    provider = InProcessCronScheduler()
    assert jobs.get_due_jobs() == []

    now = slot
    assert [row["id"] for row in jobs.get_due_jobs()] == [job["id"]]
    first = provider.claim_fire(job["id"])
    assert first is not None
    assert jobs.claim_dispatch(job["id"])
    executions.finish_execution(first["execution_id"], success=True)
    jobs.mark_job_run(
        job["id"], success=True, expected_fire_owner=first["fire_claim"]["by"]
    )
    assert jobs.get_due_jobs() == []

    now += timedelta(seconds=10)
    # --at accepts this same absolute instant in another offset, inside the grace window.
    rearmed = jobs.rearm_oneshot(
        job["id"], slot.astimezone(timezone(timedelta(hours=-4))).isoformat()
    )
    assert rearmed["repeat"]["completed"] == 0
    assert [row["id"] for row in jobs.get_due_jobs()] == [job["id"]]
    second = provider.claim_fire(job["id"])
    assert second is not None
    assert jobs.claim_dispatch(job["id"])
    executions.finish_execution(second["execution_id"], success=True)
    jobs.mark_job_run(
        job["id"], success=True, expected_fire_owner=second["fire_claim"]["by"]
    )
    assert jobs.get_job(job["id"])["state"] == "completed"
    assert jobs.get_job(job["id"])["repeat"]["completed"] == 1
    assert jobs.get_due_jobs() == []
    assert provider.claim_fire(job["id"]) is None
    assert executions.get_execution(first["execution_id"])["status"] == "completed"
    assert executions.get_execution(second["execution_id"])["status"] == "completed"

    # Explicit re-arm still refuses a time beyond the one-shot grace window.
    now = slot + timedelta(seconds=jobs.ONESHOT_GRACE_SECONDS + 1)
    before = jobs.load_jobs()
    with pytest.raises(ValueError, match="past"):
        jobs.rearm_oneshot(job["id"], slot.isoformat())
    assert jobs.load_jobs() == before


def test_completed_scheduled_instant_still_blocks_snapshot_replay(monkeypatch):
    now = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    monkeypatch.setattr(executions, "_hermes_now", lambda: now)
    job = jobs.create_job("rollback fixture", now.isoformat())
    snapshot = jobs.load_jobs()
    assert [row["id"] for row in jobs.get_due_jobs()] == [job["id"]]
    provider = InProcessCronScheduler()
    claimed = provider.claim_fire(job["id"])
    assert claimed is not None
    assert jobs.claim_dispatch(job["id"])
    executions.finish_execution(claimed["execution_id"], success=True)
    jobs.mark_job_run(
        job["id"], success=True, expected_fire_owner=claimed["fire_claim"]["by"]
    )

    jobs.save_jobs(snapshot)
    assert jobs.get_due_jobs() == []
    assert provider.claim_fire(job["id"]) is None
    assert executions.get_execution(claimed["execution_id"])["status"] == "completed"
