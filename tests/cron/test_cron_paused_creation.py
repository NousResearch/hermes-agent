"""Atomic disabled (paused) job creation.

A job created with paused=True is persisted disabled inside create_job's single
locked write — enabled=false, state=paused, pause markers set, no
next_run_at — so there is no create-then-pause window where the scheduler can
pick it up. Ordinary creation stays enabled/scheduled byte-for-byte.
"""

import pytest

from cron.jobs import (
    create_job,
    effective_job_state,
    get_due_jobs,
    get_job,
    is_job_runnable,
    load_jobs,
    resume_job,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Isolate the cron store (same pattern as tests/cron/test_jobs.py)."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path / "cron"


class TestPausedCreation:
    def test_paused_creation_persists_disabled_state(self, tmp_cron_dir):
        job = create_job(
            prompt="canary",
            schedule="every 1h",
            paused=True,
            paused_reason="canary — awaiting operator approval",
        )
        assert job["enabled"] is False
        assert job["state"] == "paused"
        assert job["paused_at"] is not None
        assert job["paused_reason"] == "canary — awaiting operator approval"
        # No first trigger armed: nothing for the due scan or a provider to act on.
        assert job["next_run_at"] is None

    def test_paused_creation_is_single_atomic_record(self, tmp_cron_dir):
        job = create_job(prompt="canary", schedule="every 1h", paused=True)
        stored = load_jobs()
        assert len(stored) == 1
        assert stored[0]["id"] == job["id"]
        assert stored[0]["enabled"] is False

    def test_paused_job_is_not_runnable_and_never_due(self, tmp_cron_dir):
        job = create_job(prompt="canary", schedule="every 1m", paused=True)
        assert is_job_runnable(job) is False
        assert effective_job_state(job) == "paused"
        # The store-level scan (basis of every scheduler tick) must skip it.
        assert get_due_jobs() == []
        # And the re-read record keeps the same contract.
        refreshed = get_job(job["id"])
        assert refreshed["enabled"] is False
        assert get_due_jobs() == []

    def test_paused_job_cannot_execute_before_resume(self, tmp_cron_dir, monkeypatch):
        job = create_job(prompt="canary", schedule="every 1m", paused=True)
        fired = []
        monkeypatch.setattr("cron.jobs.mark_job_run", lambda *a, **k: fired.append(a))
        assert get_due_jobs() == []
        assert fired == []

    def test_resume_rearms_paused_creation(self, tmp_cron_dir):
        job = create_job(prompt="canary", schedule="every 1h", paused=True)
        resumed = resume_job(job["id"])
        assert resumed["enabled"] is True
        assert resumed["state"] == "scheduled"
        assert resumed["paused_at"] is None
        assert resumed["paused_reason"] is None
        assert resumed["next_run_at"] is not None
        assert is_job_runnable(resumed) is True

    def test_ordinary_creation_remains_enabled_and_scheduled(self, tmp_cron_dir):
        job = create_job(prompt="hello", schedule="every 1h")
        assert job["enabled"] is True
        assert job["state"] == "scheduled"
        assert job["paused_at"] is None
        assert job["paused_reason"] is None
        assert job["next_run_at"] is not None
        assert is_job_runnable(job) is True

    def test_invalid_combinations_fail_before_persistence(self, tmp_cron_dir):
        with pytest.raises(ValueError):
            create_job(prompt="x", schedule="every 1h", paused_reason="orphan")
        with pytest.raises(ValueError):
            create_job(prompt="x", schedule="every 1h", paused="yes")
        assert load_jobs() == []

    def test_paused_one_shot_still_rejects_past_run_at(self, tmp_cron_dir):
        # The one-shot past-grace rejection is a schedule contract, not a liveness one:
        # a canary must not silently store a fire time that can never happen after resume.
        with pytest.raises(ValueError):
            create_job(prompt="x", schedule="2001-01-01T00:00:00", paused=True)
        assert load_jobs() == []
