"""Manual "Run now" triggers must survive the timezone migration-repair pass.

Regression tests for #78516. ``trigger_job`` sets ``next_run_at`` to now so the
next scheduler tick fires the job. The migration-repair branch in
``get_due_jobs`` recomputes ``next_run_at`` from the cron expression whenever a
due cron job's stored offset differs from the current offset and its stored
wall clock still reads as future — which silently discards the pending manual
run.

The offsets differ in practice because the trigger and the scheduler tick can
observe different offsets: the dashboard/CLI writer and the gateway are separate
processes, and a config timezone edit, host timezone change, or DST boundary
between the two makes the stored offset stale by exactly the amount the repair
branch keys on.

These exercise the real store against a temp HERMES_HOME (no mocks) per the
E2E-over-mocks discipline for file-touching code.
"""
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _daily_job(name="briefing"):
    from cron.jobs import create_job

    return create_job(
        name=name,
        schedule="0 8 * * *",
        prompt="say hi",
    )


def _force_stored_offset(job_id, hours):
    """Rewrite next_run_at (and any marker) to a different UTC offset.

    Simulates the stored value having been written under a different timezone
    than the one the scheduler now observes.
    """
    from cron.jobs import get_job, load_jobs, save_jobs

    job = get_job(job_id)
    stored = datetime.fromisoformat(job["next_run_at"])
    shifted = stored.astimezone(timezone(timedelta(hours=hours)))
    jobs = load_jobs()
    for raw in jobs:
        if raw["id"] == job_id:
            raw["next_run_at"] = shifted.isoformat()
            if raw.get("manual_trigger_at"):
                raw["manual_trigger_at"] = shifted.isoformat()
    save_jobs(jobs)
    return shifted


class TestManualTriggerSurvives:
    def test_trigger_records_a_marker(self, temp_home):
        from cron.jobs import get_job, trigger_job

        job = _daily_job()
        trigger_job(job["id"])
        stored = get_job(job["id"])
        assert stored.get("manual_trigger_at") == stored["next_run_at"], (
            "trigger_job must mark the pending run as human-requested"
        )

    def test_triggered_job_is_due_despite_offset_mismatch(self, temp_home):
        """The reported bug: the manual run must not be recomputed away."""
        from cron.jobs import get_due_jobs, trigger_job

        job = _daily_job()
        trigger_job(job["id"])
        # Stored value now carries an offset east of the local one, which is
        # what makes _stored_wall_clock_is_future() read True.
        _force_stored_offset(job["id"], hours=+10)

        due = get_due_jobs()
        assert any(j["id"] == job["id"] for j in due), (
            "manual trigger was swallowed by the migration-repair branch"
        )

    def test_triggered_job_next_run_not_recomputed(self, temp_home):
        from cron.jobs import get_due_jobs, get_job, trigger_job

        job = _daily_job()
        trigger_job(job["id"])
        shifted = _force_stored_offset(job["id"], hours=+10)

        get_due_jobs()
        after = get_job(job["id"])
        assert datetime.fromisoformat(after["next_run_at"]) == shifted, (
            "next_run_at was rewritten, discarding the pending manual run"
        )


class TestMigrationRepairStillWorks:
    """The fix must not disable the repair the branch exists for (#28934)."""

    def test_untriggered_job_still_repaired(self, temp_home):
        from cron.jobs import get_due_jobs

        job = _daily_job()
        _force_stored_offset(job["id"], hours=+10)

        # With no manual marker, a future stored wall clock is a migration
        # artifact rather than a pending run: the job must not fire early.
        assert not any(j["id"] == job["id"] for j in get_due_jobs()), (
            "a non-triggered job with a stale offset must not fire early"
        )

    def test_marker_does_not_persist_across_runs(self, temp_home):
        """A stale marker must not keep suppressing repair forever."""
        from cron.jobs import get_job, load_jobs, save_jobs, trigger_job
        from cron.jobs import _is_manual_trigger

        job = _daily_job()
        trigger_job(job["id"])
        stored = get_job(job["id"])
        assert _is_manual_trigger(stored, stored["next_run_at"])

        # Simulate the job having run: next_run_at moves to the next occurrence
        # while the marker keeps its old value.
        jobs = load_jobs()
        for raw in jobs:
            if raw["id"] == job["id"]:
                raw["next_run_at"] = "2099-01-01T08:00:00+00:00"
        save_jobs(jobs)

        moved = get_job(job["id"])
        assert not _is_manual_trigger(moved, moved["next_run_at"]), (
            "marker must stop matching once next_run_at is recomputed"
        )


class TestMarkerHelper:
    def test_absent_marker_is_not_a_trigger(self):
        from cron.jobs import _is_manual_trigger

        assert not _is_manual_trigger({}, "2026-08-04T10:00:00+00:00")

    def test_empty_marker_is_not_a_trigger(self):
        from cron.jobs import _is_manual_trigger

        assert not _is_manual_trigger(
            {"manual_trigger_at": ""}, "2026-08-04T10:00:00+00:00"
        )

    def test_mismatched_marker_is_not_a_trigger(self):
        from cron.jobs import _is_manual_trigger

        assert not _is_manual_trigger(
            {"manual_trigger_at": "2026-08-04T09:00:00+00:00"},
            "2026-08-04T10:00:00+00:00",
        )
