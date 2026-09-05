"""Behavioral coverage for terminal cron jobs and explicit one-shot re-arm."""

from datetime import datetime, timedelta, timezone
import copy
from unittest import mock

import pytest

from cron.jobs import (
    advance_next_run,
    advance_next_runs,
    claim_job_for_fire,
    create_job,
    get_due_jobs,
    get_job,
    load_jobs,
    mark_job_run,
    rearm_oneshot,
    resume_job,
    save_jobs,
    trigger_job,
    update_job,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_completed_oneshot_trigger_is_refused_and_disk_record_is_unchanged(tmp_cron_dir):
    job = create_job("done", "in 30m", name="done", repeat=1)
    mark_job_run(job["id"], success=True)
    before = copy.deepcopy(load_jobs())

    with pytest.raises(ValueError, match="terminal"):
        trigger_job(job["id"])

    assert load_jobs() == before
    assert get_job(job["id"]) == before[0]


def test_exhausted_recurring_job_trigger_is_refused(tmp_cron_dir):
    job = create_job("done", "every 1h", repeat=1)
    mark_job_run(job["id"], success=True)

    with pytest.raises(ValueError, match="terminal"):
        trigger_job(job["id"])


def test_wedged_claimed_oneshot_remains_triggerable(tmp_cron_dir):
    now = datetime.now(timezone.utc)
    job = create_job("wedged", "in 30m", repeat=2)
    record = get_job(job["id"])
    record.update({
        "run_claim": {"at": now.isoformat(), "by": "dead-worker"},
        "state": "scheduled",
        "enabled": True,
        "next_run_at": (now - timedelta(minutes=1)).isoformat(),
    })
    save_jobs([record])

    triggered = trigger_job(job["id"])
    assert triggered["state"] == "scheduled"
    assert triggered["enabled"] is True


def test_paused_job_run_override_remains_allowed(tmp_cron_dir):
    job = create_job("paused", "every 1h")
    from cron.jobs import pause_job

    pause_job(job["id"])
    triggered = trigger_job(job["id"])
    assert triggered["state"] == "scheduled"
    assert triggered["enabled"] is True


def test_terminal_jobs_are_not_due_or_advanced(tmp_cron_dir):
    job = create_job("done", "every 1h", repeat=1)
    mark_job_run(job["id"], success=True)
    before = copy.deepcopy(load_jobs())

    assert get_due_jobs() == []
    assert advance_next_run(job["id"]) is False
    assert load_jobs() == before


def test_terminal_refusal_survives_reload(tmp_cron_dir):
    job = create_job("done", "in 30m", repeat=1)
    mark_job_run(job["id"], success=True)
    before = copy.deepcopy(load_jobs())
    assert get_job(job["id"])["state"] == "completed"

    with pytest.raises(ValueError):
        trigger_job(job["id"])
    assert load_jobs() == before


def test_update_cannot_reactivate_terminal_record(tmp_cron_dir):
    job = create_job("done", "in 30m", repeat=1)
    mark_job_run(job["id"], success=True)
    with pytest.raises(ValueError, match="terminal"):
        update_job(job["id"], {"enabled": True})
    with pytest.raises(ValueError, match="terminal"):
        update_job(job["id"], {"schedule": "every 1h"})


def test_rearm_completed_oneshot_restores_schedule_and_preserves_history(tmp_cron_dir):
    from cron.jobs import rearm_oneshot

    job = create_job("done", "in 30m", repeat=3)
    mark_job_run(job["id"], success=True)
    finished = get_job(job["id"])
    run_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

    rearmed = rearm_oneshot(job["id"], run_at)
    assert rearmed["schedule"]["kind"] == "once"
    assert rearmed["repeat"]["times"] == 3
    assert rearmed["repeat"]["completed"] == 0
    assert rearmed["state"] == "scheduled"
    assert rearmed["enabled"] is True
    assert rearmed["next_run_at"] == rearmed["schedule"]["run_at"]
    assert rearmed["last_run_at"] == finished["last_run_at"]
    assert rearmed["last_status"] == finished["last_status"]


def test_rearm_refuses_recurring_and_live_claim(tmp_cron_dir):
    from cron.jobs import rearm_oneshot

    recurring = create_job("recurring", "every 1h")
    future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    with pytest.raises(ValueError, match="one-shot"):
        rearm_oneshot(recurring["id"], future)

    oneshot = create_job("claimed", "in 30m")
    record = get_job(oneshot["id"])
    record["run_claim"] = {"at": datetime.now(timezone.utc).isoformat(), "by": "live"}
    save_jobs([record])
    with pytest.raises(ValueError, match="claim"):
        rearm_oneshot(oneshot["id"], future)


class TestRecurringJobStuckInErrorStateIsRecoverable:
    """A recurring (cron/interval) job that could not compute its next
    occurrence is marked ``state=error`` but left ``enabled=True`` — issue
    #16265's invariant that recurring jobs must never be silently disabled.

    ``is_terminal_job()`` previously treated ``state=error`` identically to
    ``state=completed`` at every call site, which blocked BOTH the due-scan's
    own ``next_run_at`` self-heal AND every manual recovery path
    (``resume_job``, ``claim_job_for_fire``, ``advance_next_runs``) — wedging
    the job forever with no exit except deleting and recreating it. These
    tests pin the fix: an error-state recurring job stays recoverable through
    every one of those paths, while a genuinely terminal ``state=completed``
    job (covered by the tests above) remains blocked through all of them.
    """

    @staticmethod
    def _force_error_state(job_id):
        """Reproduce the exact state _mark_job_run_locked produces when
        compute_next_run() fails for a recurring job (e.g. croniter
        missing): state=error, enabled stays True, next_run_at=None."""
        with mock.patch("cron.jobs.compute_next_run", return_value=None):
            mark_job_run(job_id, success=True)

    def test_due_scan_self_heals_next_run_at(self, tmp_cron_dir):
        job = create_job("recurring", "every 5m")
        self._force_error_state(job["id"])
        stuck = get_job(job["id"])
        assert stuck["state"] == "error"
        assert stuck["enabled"] is True
        assert stuck["next_run_at"] is None

        # compute_next_run works again on the next tick (the transient
        # issue resolved) — the due-scan must recompute next_run_at even
        # though the persisted record is still state=error.
        get_due_jobs()

        healed = get_job(job["id"])
        assert healed["next_run_at"] is not None, (
            "due-scan self-heal never reached an error-state recurring job"
        )

    def test_resume_job_recovers_error_state(self, tmp_cron_dir):
        job = create_job("recurring", "every 5m")
        self._force_error_state(job["id"])

        resumed = resume_job(job["id"])

        assert resumed is not None, "resume_job must not raise on state=error"
        assert resumed["state"] == "scheduled"
        assert resumed["enabled"] is True
        assert resumed["next_run_at"] is not None

    def test_claim_job_for_fire_recovers_error_state(self, tmp_cron_dir):
        job = create_job("recurring", "every 5m")
        self._force_error_state(job["id"])

        claimed = claim_job_for_fire(job["id"], return_job=True)

        assert isinstance(claimed, dict), (
            "claim_job_for_fire must not refuse an error-state recurring "
            "job — it still has future occurrences"
        )
        assert claimed["fire_claim"] is not None

    def test_advance_next_runs_recovers_error_state(self, tmp_cron_dir):
        job = create_job("recurring", "every 5m")
        self._force_error_state(job["id"])

        advanced = advance_next_runs([job["id"]])

        assert advanced == 1, (
            "advance_next_runs must still pre-advance an error-state "
            "recurring job's next_run_at before it fires (crash-safety)"
        )

    def test_pause_job_now_works_on_error_state(self, tmp_cron_dir):
        from cron.jobs import pause_job

        job = create_job("recurring", "every 5m")
        self._force_error_state(job["id"])

        paused = pause_job(job["id"])

        assert paused is not None, "pause_job must not raise on state=error"
        assert paused["state"] == "paused"
        assert paused["enabled"] is False

    def test_completed_oneshot_still_blocked_on_every_path(self, tmp_cron_dir):
        """Control: the fix must not weaken protection for a genuinely
        terminal state=completed job — only state=error recurring jobs are
        exempted."""
        job = create_job("done", "in 30m", repeat=1)
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"])["state"] == "completed"

        with pytest.raises(ValueError, match="terminal"):
            resume_job(job["id"])
        assert claim_job_for_fire(job["id"], return_job=True) is False
        assert advance_next_runs([job["id"]]) == 0
        assert get_due_jobs() == []
