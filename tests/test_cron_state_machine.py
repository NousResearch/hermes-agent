"""
Property-based tests for the Hermes cron job state machine.

Uses hypothesis.stateful.RuleBasedStateMachine to randomly exercise cron job
operations and check invariants after every step.  Standalone unit tests cover
exhaustive transition tables, initial states, roundtrips, and terminal states.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Set

import pytest
from hypothesis import given, settings, HealthCheck, Phase
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
    precondition,
)

import cron.jobs
from cron.jobs import (
    create_job,
    pause_job,
    resume_job,
    trigger_job,
    remove_job,
    mark_job_run,
    get_job,
    list_jobs,
    load_jobs,
    save_jobs,
    JOB_STATES,
    VALID_TRANSITIONS,
    is_valid_transition,
    job_is_active,
    JOBS_FILE,
)


# =============================================================================
# Fixtures for temp-dir isolation
# =============================================================================

@pytest.fixture(autouse=True)
def _isolate_cron_dir(tmp_path, monkeypatch):
    """Redirect cron.jobs paths to a temp directory for test isolation."""
    cron_dir = tmp_path / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("cron.jobs.CRON_DIR", cron_dir)
    monkeypatch.setattr("cron.jobs.JOBS_FILE", cron_dir / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", cron_dir / "output")


def _patch_cron_paths(tmp_dir: Path):
    """Directly patch cron.jobs module globals for non-fixture contexts
    (e.g. inside hypothesis stateful machines where fixtures aren't available)."""
    cron_dir = tmp_dir / "cron"
    cron_dir.mkdir(parents=True, exist_ok=True)
    cron.jobs.CRON_DIR = cron_dir
    cron.jobs.JOBS_FILE = cron_dir / "jobs.json"
    cron.jobs.OUTPUT_DIR = cron_dir / "output"


# =============================================================================
# Hypothesis Stateful Machine
# =============================================================================

class CronJobStateMachine(RuleBasedStateMachine):
    """Randomly exercise cron job operations and verify invariants hold."""

    def __init__(self):
        super().__init__()
        # Use a fresh temp directory for each test case
        self._tmp_dir = Path(tempfile.mkdtemp())
        _patch_cron_paths(self._tmp_dir)
        self.tracked_ids: Set[str] = set()
        self.paused_ids: Set[str] = set()
        self.completed_ids: Set[str] = set()
        self.failed_ids: Set[str] = set()
        self._job_counter = 0

    def teardown(self):
        """Clean up temp directory."""
        import shutil
        try:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
        except Exception:
            pass

    # -- Rules -----------------------------------------------------------------

    @rule()
    def create_a_job(self):
        """Create a new recurring job (every 1h schedule so it stays active)."""
        self._job_counter += 1
        job = create_job(
            prompt=f"Test task #{self._job_counter}",
            schedule="every 1h",
            name=f"StateMachine Job {self._job_counter}",
        )
        assert job["state"] == "scheduled"
        assert job["id"] not in self.tracked_ids, "Duplicate job ID generated"
        self.tracked_ids.add(job["id"])

    @precondition(lambda self: len(self.tracked_ids - self.paused_ids - self.completed_ids) > 0)
    @rule()
    def pause_a_job(self):
        """Pause a random active (non-paused, non-completed) job."""
        active_ids = self.tracked_ids - self.paused_ids - self.completed_ids
        if not active_ids:
            return
        job_id = sorted(active_ids)[0]  # deterministic pick
        result = pause_job(job_id, reason="stateful test pause")
        if result is not None:
            assert result["state"] == "paused"
            self.paused_ids.add(job_id)

    @precondition(lambda self: len(self.paused_ids) > 0)
    @rule()
    def resume_a_job(self):
        """Resume a random paused job."""
        if not self.paused_ids:
            return
        job_id = sorted(self.paused_ids)[0]
        result = resume_job(job_id)
        if result is not None:
            assert result["state"] == "scheduled"
            self.paused_ids.discard(job_id)

    @precondition(lambda self: len(self.tracked_ids - self.completed_ids) > 0)
    @rule()
    def mark_success(self):
        """Simulate a successful run on a random job."""
        eligible = self.tracked_ids - self.completed_ids
        if not eligible:
            return
        job_id = sorted(eligible)[0]
        job_before = get_job(job_id)
        if job_before is None:
            # Job was removed (e.g. repeat limit hit) — clean up tracking
            self.tracked_ids.discard(job_id)
            self.paused_ids.discard(job_id)
            return
        completed_before = (job_before.get("repeat") or {}).get("completed", 0)

        mark_job_run(job_id, success=True)

        job_after = get_job(job_id)
        if job_after is None:
            # Job was removed because repeat limit was reached
            self.tracked_ids.discard(job_id)
            self.paused_ids.discard(job_id)
        else:
            # Repeat counter should have incremented
            completed_after = (job_after.get("repeat") or {}).get("completed", 0)
            assert completed_after == completed_before + 1
            if job_after["state"] == "completed":
                self.completed_ids.add(job_id)
                self.paused_ids.discard(job_id)

    @precondition(lambda self: len(self.tracked_ids - self.completed_ids) > 0)
    @rule()
    def mark_failure(self):
        """Simulate a failed run on a random job."""
        eligible = self.tracked_ids - self.completed_ids
        if not eligible:
            return
        job_id = sorted(eligible)[0]
        job_before = get_job(job_id)
        if job_before is None:
            self.tracked_ids.discard(job_id)
            self.paused_ids.discard(job_id)
            return

        mark_job_run(job_id, success=False, error="simulated failure")

        job_after = get_job(job_id)
        if job_after is None:
            self.tracked_ids.discard(job_id)
            self.paused_ids.discard(job_id)
        else:
            assert job_after["last_status"] == "error"
            assert job_after["last_error"] == "simulated failure"

    @precondition(lambda self: len(self.tracked_ids) > 0)
    @rule()
    def remove_a_job(self):
        """Remove a random job."""
        if not self.tracked_ids:
            return
        job_id = sorted(self.tracked_ids)[0]
        removed = remove_job(job_id)
        if removed:
            self.tracked_ids.discard(job_id)
            self.paused_ids.discard(job_id)
            self.completed_ids.discard(job_id)
            self.failed_ids.discard(job_id)

    @precondition(lambda self: len(self.tracked_ids - self.paused_ids - self.completed_ids) > 0)
    @rule()
    def trigger_a_job(self):
        """Trigger immediate execution scheduling on a random active job."""
        active_ids = self.tracked_ids - self.paused_ids - self.completed_ids
        if not active_ids:
            return
        job_id = sorted(active_ids)[0]
        result = trigger_job(job_id)
        if result is not None:
            assert result["state"] == "scheduled"

    # -- Invariants (checked after every step) ----------------------------------

    @invariant()
    def all_states_are_valid(self):
        """Every stored job must have a state in JOB_STATES."""
        for job in load_jobs():
            assert job.get("state", "scheduled") in JOB_STATES, (
                f"Job '{job.get('id')}' has invalid state '{job.get('state')}'"
            )

    @invariant()
    def no_duplicate_ids(self):
        """No two stored jobs should share the same ID."""
        jobs = load_jobs()
        ids = [j["id"] for j in jobs]
        assert len(ids) == len(set(ids)), (
            f"Duplicate job IDs found: {[i for i in ids if ids.count(i) > 1]}"
        )

    @invariant()
    def completed_jobs_have_no_next_run(self):
        """Completed jobs must not have a next_run_at value."""
        for job in load_jobs():
            if job.get("state") == "completed":
                assert job.get("next_run_at") is None, (
                    f"Completed job '{job.get('id')}' still has "
                    f"next_run_at={job.get('next_run_at')}"
                )

    @invariant()
    def repeat_counters_consistent(self):
        """Repeat completed count must not exceed repeat times limit."""
        for job in load_jobs():
            repeat = job.get("repeat")
            if isinstance(repeat, dict):
                times = repeat.get("times")
                completed = repeat.get("completed", 0)
                if times is not None:
                    assert completed <= times, (
                        f"Job '{job.get('id')}' has completed={completed} "
                        f"but limit is {times}"
                    )

    @invariant()
    def tracked_ids_match_storage(self):
        """Every tracked ID should exist in storage or have been removed."""
        stored_ids = {j["id"] for j in load_jobs()}
        # Every stored ID should be in our tracking (or we missed it)
        for sid in stored_ids:
            assert sid in self.tracked_ids, (
                f"Stored job '{sid}' not in tracked set"
            )


# Run the state machine with hypothesis settings tuned for reasonable runtime
TestCronStateMachine = CronJobStateMachine.TestCase
TestCronStateMachine.settings = settings(
    max_examples=30,
    stateful_step_count=20,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    phases=[Phase.generate],  # skip shrinking for speed in CI
)


# =============================================================================
# Standalone Tests
# =============================================================================

class TestValidTransitionsExhaustive:
    """Verify every (from, to) pair against VALID_TRANSITIONS."""

    def test_all_valid_transitions_accepted(self):
        """Every pair listed in VALID_TRANSITIONS should be accepted."""
        for from_state, targets in VALID_TRANSITIONS.items():
            for to_state in targets:
                assert is_valid_transition(from_state, to_state), (
                    f"Expected {from_state} -> {to_state} to be valid"
                )

    def test_all_invalid_transitions_rejected(self):
        """Every pair NOT listed in VALID_TRANSITIONS should be rejected."""
        for from_state in JOB_STATES:
            for to_state in JOB_STATES:
                if to_state in VALID_TRANSITIONS.get(from_state, set()):
                    continue
                assert not is_valid_transition(from_state, to_state), (
                    f"Expected {from_state} -> {to_state} to be INVALID"
                )

    def test_completed_is_terminal(self):
        """'completed' state should have no outgoing transitions."""
        assert VALID_TRANSITIONS["completed"] == set()
        for target in JOB_STATES:
            assert not is_valid_transition("completed", target), (
                f"completed -> {target} should be invalid (terminal state)"
            )

    def test_every_state_has_entry_in_table(self):
        """Every JOB_STATE must appear as a key in VALID_TRANSITIONS."""
        for state in JOB_STATES:
            assert state in VALID_TRANSITIONS, (
                f"State '{state}' missing from VALID_TRANSITIONS table"
            )

    def test_unknown_state_rejected(self):
        """Transitions from an unknown state should be rejected."""
        assert not is_valid_transition("bogus", "scheduled")
        assert not is_valid_transition("scheduled", "bogus")


class TestCreateJobInitialState:
    """New jobs always start in 'scheduled' state."""

    def test_recurring_job_starts_scheduled(self):
        job = create_job(prompt="Test", schedule="every 1h")
        assert job["state"] == "scheduled"

    def test_one_shot_job_starts_scheduled(self):
        job = create_job(prompt="Test", schedule="30m")
        assert job["state"] == "scheduled"

    def test_initial_repeat_completed_is_zero(self):
        job = create_job(prompt="Test", schedule="every 1h", repeat=5)
        assert job["repeat"]["completed"] == 0
        assert job["repeat"]["times"] == 5

    def test_initial_last_run_at_is_none(self):
        job = create_job(prompt="Test", schedule="every 2h")
        assert job["last_run_at"] is None

    def test_initial_last_status_is_none(self):
        job = create_job(prompt="Test", schedule="every 2h")
        assert job["last_status"] is None

    def test_job_has_next_run_at(self):
        job = create_job(prompt="Test", schedule="every 1h")
        assert job["next_run_at"] is not None


class TestPauseResumeRoundtrip:
    """Pause then resume should return to 'scheduled'."""

    def test_pause_resume_returns_to_scheduled(self):
        job = create_job(prompt="Roundtrip test", schedule="every 1h")
        job_id = job["id"]

        paused = pause_job(job_id, reason="test")
        assert paused["state"] == "paused"
        assert paused["paused_at"] is not None
        assert paused["paused_reason"] == "test"

        resumed = resume_job(job_id)
        assert resumed["state"] == "scheduled"
        assert resumed["paused_at"] is None
        assert resumed["paused_reason"] is None

    def test_pause_resume_preserves_id(self):
        job = create_job(prompt="ID preservation", schedule="every 1h")
        job_id = job["id"]

        pause_job(job_id)
        resumed = resume_job(job_id)
        assert resumed["id"] == job_id

    def test_pause_resume_restores_next_run(self):
        job = create_job(prompt="Next run test", schedule="every 1h")
        job_id = job["id"]

        pause_job(job_id)
        resumed = resume_job(job_id)
        assert resumed["next_run_at"] is not None

    def test_double_pause_idempotent(self):
        job = create_job(prompt="Double pause", schedule="every 1h")
        job_id = job["id"]

        pause_job(job_id)
        second = pause_job(job_id)
        # Should still be paused
        assert second["state"] == "paused"

    def test_resume_non_paused_is_safe(self):
        """resume_job on a non-paused job should still work (sets to scheduled)."""
        job = create_job(prompt="Not paused", schedule="every 1h")
        job_id = job["id"]

        result = resume_job(job_id)
        assert result["state"] == "scheduled"


class TestMarkRunDecrementsRepeat:
    """mark_job_run increments the repeat counter correctly."""

    def test_single_run_increments_counter(self):
        job = create_job(prompt="Counter test", schedule="every 1h", repeat=5)
        job_id = job["id"]

        mark_job_run(job_id, success=True)
        updated = get_job(job_id)
        assert updated["repeat"]["completed"] == 1

    def test_multiple_runs_increment(self):
        job = create_job(prompt="Multi run", schedule="every 1h", repeat=10)
        job_id = job["id"]

        for i in range(3):
            mark_job_run(job_id, success=True)

        updated = get_job(job_id)
        assert updated["repeat"]["completed"] == 3

    def test_reaching_limit_removes_job(self):
        """When repeat limit is reached, the job is removed from storage."""
        job = create_job(prompt="Finite job", schedule="every 1h", repeat=2)
        job_id = job["id"]

        mark_job_run(job_id, success=True)
        assert get_job(job_id) is not None  # still alive

        mark_job_run(job_id, success=True)
        assert get_job(job_id) is None  # removed after reaching limit

    def test_unlimited_repeat_never_removes(self):
        """Jobs with repeat.times=None run forever."""
        job = create_job(prompt="Forever job", schedule="every 1h")
        job_id = job["id"]

        for _ in range(5):
            mark_job_run(job_id, success=True)

        updated = get_job(job_id)
        assert updated is not None
        assert updated["repeat"]["completed"] == 5
        assert updated["repeat"]["times"] is None

    def test_failed_run_still_increments(self):
        job = create_job(prompt="Fail counter", schedule="every 1h", repeat=5)
        job_id = job["id"]

        mark_job_run(job_id, success=False, error="oops")
        updated = get_job(job_id)
        assert updated["repeat"]["completed"] == 1
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "oops"


class TestCompletedIsTerminal:
    """Completed jobs cannot transition to any other state."""

    def test_completed_transition_table_is_empty(self):
        assert VALID_TRANSITIONS["completed"] == set()

    def test_is_valid_transition_rejects_all_from_completed(self):
        for target in JOB_STATES:
            assert not is_valid_transition("completed", target)

    def test_completed_job_is_not_active(self):
        """job_is_active should return False for completed jobs."""
        assert not job_is_active({"state": "completed"})

    def test_paused_job_is_not_active(self):
        assert not job_is_active({"state": "paused"})

    def test_failed_job_is_not_active(self):
        assert not job_is_active({"state": "failed"})

    def test_scheduled_job_is_active(self):
        assert job_is_active({"state": "scheduled"})

    def test_running_job_is_active(self):
        assert job_is_active({"state": "running"})


class TestJobIsActive:
    """Verify job_is_active for all states."""

    @pytest.mark.parametrize("state,expected", [
        ("scheduled", True),
        ("running", True),
        ("paused", False),
        ("completed", False),
        ("failed", False),
    ])
    def test_active_status_by_state(self, state, expected):
        assert job_is_active({"state": state}) == expected

    def test_missing_state_defaults_to_active(self):
        """A job dict without an explicit state defaults to scheduled (active)."""
        assert job_is_active({})


class TestSaveLoadRoundtrip:
    """Jobs survive a save/load cycle."""

    def test_save_then_load_preserves_jobs(self):
        job1 = create_job(prompt="Job A", schedule="every 1h")
        job2 = create_job(prompt="Job B", schedule="every 2h")

        loaded = load_jobs()
        ids = {j["id"] for j in loaded}
        assert job1["id"] in ids
        assert job2["id"] in ids

    def test_empty_load_when_no_file(self, tmp_path, monkeypatch):
        fresh_dir = tmp_path / "fresh_cron"
        fresh_dir.mkdir()
        monkeypatch.setattr("cron.jobs.CRON_DIR", fresh_dir)
        monkeypatch.setattr("cron.jobs.JOBS_FILE", fresh_dir / "jobs.json")
        monkeypatch.setattr("cron.jobs.OUTPUT_DIR", fresh_dir / "output")

        assert load_jobs() == []
