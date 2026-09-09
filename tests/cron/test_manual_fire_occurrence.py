"""Manual (non-tick) fires must not consume the next scheduled occurrence.

Regression context (2026-09-08, live install): after the scheduled 19:00 run of
job d88fde170fb4, a manual run at 19:53 claimed the occurrence instant at
next_run_at (= tomorrow 19:00) and the executions ledger stamped that future
instant completed. From then on (a) every later manual fire was refused with
"Job is already being fired by the scheduler; not run again." and (b) the next
scheduled tick dedupe-skipped its real delivery and merely advanced
next_run_at.

The ticker never fires early: ``_evaluate_due_job`` returns False while
next_run_dt > now. So any fire claim arriving BEFORE the stored next
occurrence cannot be the tick that owns it — it is a manual / dashboard /
webhook fire and must stay occurrence-free (its ledger row carries
scheduled_instant=NULL). Ticks that fire on time or late (catch-up) keep the
exact at-most-once occurrence identity unchanged.

Clock is faked at ``cron.jobs._hermes_now`` (the only binding the claim, mark
and due-decision paths read); ledger claimed_at stamps use the real clock,
which no assertion depends on.
"""
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _patch_now(monkeypatch, dt):
    import cron.jobs as jobs_mod

    monkeypatch.setattr(jobs_mod, "_hermes_now", lambda: dt)


def _simulate_last_scheduled_run(job_id, monkeypatch):
    """One on-time scheduled fire completed: occurrence-bound ledger row for the
    stored next_run_at, then mark_job_run re-arms to the following occurrence."""
    from cron import executions, jobs
    from cron.occurrences import scheduled_instant as canonical_instant

    n0 = datetime.fromisoformat(jobs.get_job(job_id)["next_run_at"])
    _patch_now(monkeypatch, n0 + timedelta(seconds=5))
    instant = canonical_instant(jobs.get_job(job_id)["next_run_at"])
    row = executions.create_execution(job_id, source="builtin", scheduled_instant=instant)
    executions.finish_execution(row["id"], success=True)
    jobs.mark_job_run(job_id, success=True)
    from cron.occurrences import completed_occurrence as _completed

    assert _completed(jobs.get_job(job_id), instant) is True
    return instant


def _midcycle_now(job_id):
    """Clock between the completed run and the next occurrence: 23h before the
    stored next_run_at. compute_next_run from this instant still yields the
    stored occurrence (the live 19:53 shape)."""
    import cron.jobs as jobs

    return datetime.fromisoformat(jobs.get_job(job_id)["next_run_at"]) - timedelta(hours=23)


def _manual_fire_and_complete(job_id):
    """The production manual-run sequence — claim, run_one_job's ledger row
    (source='direct', scheduled_instant=job['_scheduled_instant']), terminal
    finish, mark_job_run — without the agent/session machinery."""
    from cron import executions
    from cron.jobs import claim_job_for_fire, mark_job_run

    claimed = claim_job_for_fire(job_id, return_job=True)
    assert isinstance(claimed, dict)
    row = executions.create_execution(
        job_id, source="direct", scheduled_instant=claimed["_scheduled_instant"])
    executions.finish_execution(row["id"], success=True)
    mark_job_run(job_id, success=True)
    return claimed


def test_manual_fire_between_scheduled_runs_does_not_consume_next_occurrence(temp_home, monkeypatch):
    """A mid-cycle manual run must stay occurrence-free: its ledger row carries
    scheduled_instant=NULL, the next occurrence stays deliverable, and the next
    scheduled tick actually fires instead of dedupe-skipping."""
    from cron import executions
    from cron.jobs import _DueScan, _evaluate_due_job, create_job, get_job
    from cron.occurrences import completed_occurrence

    job = create_job(prompt="x", schedule="0 19 * * *", name="evening")
    jid = job["id"]
    _simulate_last_scheduled_run(jid, monkeypatch)
    next_occurrence = get_job(jid)["next_run_at"]

    _patch_now(monkeypatch, _midcycle_now(jid))

    claimed = _manual_fire_and_complete(jid)
    assert claimed["_scheduled_instant"] is None

    rows = executions.list_executions(job_id=jid)
    completed = [r for r in rows if r["status"] == "completed"]
    assert len(completed) == 2  # the on-time scheduled run + the manual run
    manual_rows = [r for r in completed if r["source"] != "builtin"]
    assert manual_rows and all(r["scheduled_instant"] is None for r in manual_rows)

    assert completed_occurrence(get_job(jid), next_occurrence) is False
    assert get_job(jid)["next_run_at"] == next_occurrence

    # The next scheduled tick (tomorrow 19:00) must deliver, not dedupe-skip.
    scan = _DueScan([get_job(jid)], datetime.fromisoformat(next_occurrence))
    assert _evaluate_due_job(get_job(jid), scan, 3600.0) is True

    # Symptom (a): a SECOND manual fire must still win instead of being refused
    # as "Job is already being fired by the scheduler; not run again."
    from cron.jobs import claim_job_for_fire

    assert claim_job_for_fire(jid) is True


def test_fire_webhook_before_next_occurrence_is_occurrence_free(temp_home, monkeypatch):
    """Dashboard / NAS webhook fires (provider.claim_fire) arriving before the
    next occurrence must not consume it either."""
    from cron import executions, jobs
    from cron.scheduler_provider import InProcessCronScheduler

    job = jobs.create_job(prompt="x", schedule="0 19 * * *", name="evening")
    jid = job["id"]
    _simulate_last_scheduled_run(jid, monkeypatch)
    _patch_now(monkeypatch, _midcycle_now(jid))

    claimed = InProcessCronScheduler().claim_fire(jid)
    assert claimed is not None
    assert claimed["_scheduled_instant"] is None
    row = executions.get_execution(claimed["execution_id"])
    assert row is not None
    assert row["scheduled_instant"] is None


def test_ticker_keeps_occurrence_binding_when_firing_on_time(temp_home, monkeypatch):
    """Invariant the fix must not weaken: a scheduled tick firing AT the stored
    occurrence binds that exact instant (at-most-once per occurrence)."""
    from cron import jobs
    from cron.jobs import claim_job_for_fire, create_job, get_job
    from cron.occurrences import scheduled_instant as canonical_instant

    job = create_job(prompt="x", schedule="0 19 * * *", name="evening")
    jid = job["id"]
    next_run = get_job(jid)["next_run_at"]
    instant = canonical_instant(next_run)
    _patch_now(monkeypatch, datetime.fromisoformat(next_run) + timedelta(seconds=5))

    claimed = claim_job_for_fire(jid, return_job=True)
    assert isinstance(claimed, dict)
    assert claimed["_scheduled_instant"] == instant


def test_ticker_keeps_occurrence_binding_when_firing_late_in_catchup(temp_home, monkeypatch):
    """A scheduled tick firing LATE (within the catch-up window, after the
    stored instant) keeps the occurrence binding — the instant was genuinely due."""
    from cron import jobs
    from cron.jobs import claim_job_for_fire, create_job, get_job
    from cron.occurrences import scheduled_instant as canonical_instant

    job = create_job(prompt="x", schedule="0 19 * * *", name="evening")
    jid = job["id"]
    next_run = get_job(jid)["next_run_at"]
    instant = canonical_instant(next_run)
    _patch_now(monkeypatch, datetime.fromisoformat(next_run) + timedelta(minutes=90))

    claimed = claim_job_for_fire(jid, return_job=True)
    assert isinstance(claimed, dict)
    assert claimed["_scheduled_instant"] == instant


def test_forced_fire_stays_occurrence_free(temp_home, monkeypatch):
    """force=True (Trigger-now / paused-job fire) is a manual fire by definition
    and must never bind an occurrence (guards test_scheduled_occurrence's
    "manual force is not a completion of the pending scheduled slot")."""
    from cron import jobs
    from cron.jobs import claim_job_for_fire, create_job, get_job, pause_job

    job = create_job(prompt="x", schedule="0 19 * * *", name="paused-manual")
    pause_job(job["id"])

    claimed = claim_job_for_fire(job["id"], force=True, return_job=True)
    assert isinstance(claimed, dict)
    assert claimed["_scheduled_instant"] is None
