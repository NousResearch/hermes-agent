"""A fire fence this process already holds is BUSY, not lost.

Regression guard for the fleet-wide "Interrupted by shutdown before terminal completion" class:
the per-job side-effect fence is held across network delivery, so any delivery slower than
``_JOBS_LOCK_TIMEOUT_SECONDS`` (every bot-chat turn, a slow platform send) made the run's own
heartbeat block on its own lock, time out, fail closed, and abort a live run as an ownership loss.
The heartbeat must distinguish "we still own the claim" from "another owner took it".
"""
import threading
import time

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json/lock files don't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _claimed_job():
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="fence-contention")
    assert claim_job_for_fire(job["id"], return_job=True)
    return job["id"], get_job(job["id"])["fire_claim"]["by"]


def test_heartbeat_survives_a_fence_held_by_this_process(temp_home, monkeypatch):
    """Delivery side effect in flight (fence held here) must NOT read as a lost claim."""
    from cron import jobs

    # Keep the red-on-base path fast: without the fix the heartbeat blocks for the fence
    # timeout before failing closed.
    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.5)
    job_id, owner = _claimed_job()

    holding = threading.Event()
    release = threading.Event()

    def _hold_fence():
        with jobs.fire_claim_fence(job_id, expected_owner=owner):
            holding.set()
            release.wait(5)

    holder = threading.Thread(target=_hold_fence, daemon=True)
    holder.start()
    assert holding.wait(5), "fence holder never acquired the fence"

    started = time.monotonic()
    try:
        still_ours = jobs.heartbeat_fire_claim(job_id, expected_owner=owner)
    finally:
        release.set()
        holder.join(5)
    elapsed = time.monotonic() - started

    assert still_ours is True, "a fence held by this process is busy, not a lost claim"
    # The fix reads the stored claim instead of queueing behind our own lock.
    assert elapsed < 0.3, f"heartbeat blocked behind own fence for {elapsed:.2f}s"


def test_heartbeat_still_reports_loss_after_a_real_takeover(temp_home):
    """A claim whose stored owner changed is still lost — the fix must not mask takeovers."""
    from cron import jobs

    job_id, owner = _claimed_job()
    with jobs._jobs_lock():
        records = jobs.load_jobs()
        for record in records:
            if record.get("id") == job_id:
                record["fire_claim"] = {"at": record["fire_claim"]["at"], "by": "other-machine:abc"}
        jobs.save_jobs(records)

    assert jobs.heartbeat_fire_claim(job_id, expected_owner=owner) is False
