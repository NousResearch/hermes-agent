"""The external-worker waiter must release a run's claims when the worker dies out-of-band.

When the restart-safe cron worker is OOM-killed, the waiter recovers the execution to
``unknown`` and returns True *without* ``mark_job_run``. The fire_claim stamped by the (live)
gateway survives and ``_claim_is_live`` keeps reporting it live for the full 300 s TTL, so every
tick in that window dies on ``claim_job_for_fire`` -> "Fire claim lost; execution was not
started." — the measured schedule hole after the DK2 OOM (#OOM-recovery).
"""
from __future__ import annotations

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _make_claimed_job():
    from cron.jobs import claim_job_for_fire, create_job, get_job

    job = create_job(prompt="x", schedule="every 5m", name="dk2-cycle")
    jid = job["id"]
    assert claim_job_for_fire(jid) is True
    claim = get_job(jid)["fire_claim"]
    assert isinstance(claim, dict)
    return jid


def test_waiter_releases_live_own_fire_claim_on_worker_death(temp_home):
    """A claim stamped by THIS machine (the gateway) is dropped once its worker is dead."""
    import cron.scheduler as scheduler
    from cron.jobs import get_job

    jid = _make_claimed_job()
    assert get_job(jid)["fire_claim"] is not None

    assert scheduler._release_finished_run_fire_claim(jid) is True
    assert get_job(jid)["fire_claim"] is None

    # The job can be claimed again immediately — no 300 s "Fire claim lost" window.
    from cron.jobs import claim_job_for_fire

    assert claim_job_for_fire(jid) is True


def test_waiter_does_not_touch_a_foreign_live_claim(temp_home, monkeypatch):
    """A claim owned by another host must never be cleared (fail safe)."""
    import cron.jobs as jobs
    import cron.scheduler as scheduler

    jid = _make_claimed_job()
    # Re-stamp the claim as a foreign machine.
    from cron.jobs import _with_job, get_job, save_jobs

    def _foreign(store, _i, job):
        job["fire_claim"] = {"at": job["fire_claim"]["at"], "by": "other-host:1234"}
        save_jobs(store)
        return True

    assert _with_job(jid, _foreign, False) is True
    assert get_job(jid)["fire_claim"]["by"] == "other-host:1234"

    assert scheduler._release_finished_run_fire_claim(jid) is False
    assert get_job(jid)["fire_claim"]["by"] == "other-host:1234"


def test_release_is_noop_without_job_id(temp_home):
    """No job id (older callers) — the helper never raises and clears nothing."""
    import cron.scheduler as scheduler

    assert scheduler._release_finished_run_fire_claim is not None
    # Direct call with a bogus id must be a safe no-op.
    assert scheduler._release_finished_run_fire_claim("does-not-exist") is False
