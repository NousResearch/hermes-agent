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




def test_update_cannot_reactivate_terminal_record(tmp_cron_dir):
    job = create_job("done", "in 30m", repeat=1)
    mark_job_run(job["id"], success=True)
    with pytest.raises(ValueError, match="terminal"):
        update_job(job["id"], {"enabled": True})
    with pytest.raises(ValueError, match="terminal"):
        update_job(job["id"], {"schedule": "every 1h"})


def test_rearm_completed_oneshot_restores_schedule_and_preserves_history(tmp_cron_dir):

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


class TestSpentRecurringJobRevival:
    """A recurring job whose finite ``repeat.times`` budget ran out lands in
    ``state=completed`` via ``mark_job_run``'s limit branch — but unlike a
    spent one-shot it still has future occurrences. Issue #125872: every
    documented recovery door refused and pointed at another refusing door
    (``update_job`` blocked terminal activation while advising
    ``resume --run-now/--at``, which ``rearm_oneshot`` rejects for recurring
    schedules), leaving delete-and-recreate (dropping run history) as the
    only exit. These tests pin the fix: plain ``resume_job`` restarts the
    series — budget reset to 0/N, schedule and run history preserved —
    through a sanctioned dedicated write path, while ``update_job`` stays
    blocked so no *automatic* path can revive a spent budget."""

    @staticmethod
    def _spend(job_id, times):
        for _ in range(times):
            mark_job_run(job_id, success=True)

    def test_plain_resume_restarts_series_and_preserves_history(self, tmp_cron_dir):
        job = create_job("burn-in", "every 1h", name="burn-in", repeat=2)
        self._spend(job["id"], 2)
        finished = get_job(job["id"])
        assert finished["state"] == "completed"
        assert finished["enabled"] is False
        assert finished["repeat"] == {"times": 2, "completed": 2}

        resumed = resume_job(job["id"])

        assert resumed is not None, "resume_job must not refuse a spent recurring job"
        assert resumed["state"] == "scheduled"
        assert resumed["enabled"] is True
        assert resumed["next_run_at"] is not None
        assert resumed["repeat"]["completed"] == 0
        assert resumed["repeat"]["times"] == 2, "the user's stated budget must survive"
        assert resumed["last_run_at"] == finished["last_run_at"], "run history survives"
        assert resumed["last_status"] == finished["last_status"]

    def test_restarted_budget_governs_again(self, tmp_cron_dir):
        job = create_job("series", "every 1h", repeat=2)
        self._spend(job["id"], 2)
        resume_job(job["id"])

        mark_job_run(job["id"], success=True)

        once = get_job(job["id"])
        assert once["state"] == "scheduled", "one run of a restarted 2-budget job retires nothing"
        assert once["repeat"]["completed"] == 1

        mark_job_run(job["id"], success=True)
        spent_again = get_job(job["id"])
        assert spent_again["state"] == "completed"
        assert spent_again["repeat"]["completed"] == 2
        # ...and the second series is revivable the same way.
        assert resume_job(job["id"])["state"] == "scheduled"

    def test_restarted_job_is_claimable_again(self, tmp_cron_dir):
        job = create_job("watch", "every 1h", repeat=1)
        self._spend(job["id"], 1)
        assert claim_job_for_fire(job["id"], return_job=True) is False

        resume_job(job["id"])

        claimed = claim_job_for_fire(job["id"], return_job=True)
        assert isinstance(claimed, dict), "a restarted job must re-enter the fire path"
        assert claimed["fire_claim"] is not None

    def test_resume_refuses_live_run_and_fire_claims(self, tmp_cron_dir):
        job = create_job("held", "every 1h", repeat=1)
        self._spend(job["id"], 1)
        now_iso = datetime.now(timezone.utc).isoformat()

        for claim_field in ("run_claim", "fire_claim"):
            record = get_job(job["id"])
            record[claim_field] = {"at": now_iso, "by": "live-worker"}
            save_jobs([record])
            before = copy.deepcopy(load_jobs())

            with pytest.raises(ValueError, match="live (run|fire) claim"):
                resume_job(job["id"])

            assert load_jobs() == before, f"a live {claim_field} must block the restart cleanly"

    def test_resume_refuses_when_no_next_occurrence_computable(self, tmp_cron_dir):
        job = create_job("broken", "every 5m", repeat=1)
        self._spend(job["id"], 1)
        before = copy.deepcopy(load_jobs())

        with mock.patch("cron.jobs.compute_next_run", return_value=None):
            with pytest.raises(ValueError, match="no next occurrence"):
                resume_job(job["id"])

        after = load_jobs()
        assert after == before, "a refused restart must not half-apply on disk"
        assert after[0]["state"] == "completed"

    def test_update_job_remains_blocked_with_restart_hint(self, tmp_cron_dir):
        job = create_job("burn-in", "every 1h", repeat=1)
        self._spend(job["id"], 1)

        with pytest.raises(ValueError, match="plain 'cron resume'"):
            update_job(job["id"], {"enabled": True})

        assert get_job(job["id"])["state"] == "completed", "guard itself must not weaken"

    def test_trigger_refusal_points_at_plain_resume(self, tmp_cron_dir):
        job = create_job("burn-in", "every 1h", repeat=1)
        self._spend(job["id"], 1)

        with pytest.raises(ValueError, match="Restart the series with plain 'hermes cron resume"):
            trigger_job(job["id"])

    def test_spent_oneshot_keeps_old_refusal_and_rearm(self, tmp_cron_dir):
        job = create_job("done", "in 30m", repeat=1)
        self._spend(job["id"], 1)

        with pytest.raises(ValueError, match="terminal"):
            trigger_job(job["id"])
        with pytest.raises(ValueError, match="terminal"):
            resume_job(job["id"])
        run_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        rearmed = rearm_oneshot(job["id"], run_at)
        assert rearmed["state"] == "scheduled"

    def test_paused_recurring_resume_does_not_touch_budget(self, tmp_cron_dir):
        from cron.jobs import pause_job

        job = create_job("paused-series", "every 1h", repeat=5)
        mark_job_run(job["id"], success=True)
        pause_job(job["id"])

        resumed = resume_job(job["id"])

        assert resumed["state"] == "scheduled"
        assert resumed["repeat"]["completed"] == 1, (
            "the budget reset belongs to the spent-restart path only"
        )

    def test_rearm_still_refuses_spent_recurring(self, tmp_cron_dir):
        """Re-arm remains one-shot-only by design; its advice ("use plain resume
        or cron run") is truthful now that plain resume restarts the series."""
        job = create_job("burn-in", "every 1h", repeat=1)
        self._spend(job["id"], 1)
        future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()

        with pytest.raises(ValueError, match="one-shot"):
            rearm_oneshot(job["id"], future)

        assert get_job(job["id"])["state"] == "completed"
