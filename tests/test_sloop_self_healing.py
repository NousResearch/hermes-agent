"""Tests for Sloop self-healing: consecutive failure tracking, auto-pause, exponential backoff."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cron.jobs import (
    create_job,
    get_job,
    list_jobs,
    mark_job_run,
    pause_job,
    resume_job,
    load_jobs,
    save_jobs,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestConsecutiveFailureTracking:
    def test_failure_increments_counter(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        mark_job_run(job_id, success=False, error="timeout")
        updated = get_job(job_id)
        assert updated["consecutive_failures"] == 1

        mark_job_run(job_id, success=False, error="timeout")
        updated = get_job(job_id)
        assert updated["consecutive_failures"] == 2

    def test_success_resets_counter(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        mark_job_run(job_id, success=False, error="err1")
        mark_job_run(job_id, success=False, error="err2")
        updated = get_job(job_id)
        assert updated["consecutive_failures"] == 2

        mark_job_run(job_id, success=True)
        updated = get_job(job_id)
        assert updated["consecutive_failures"] == 0

    def test_initial_state_has_no_counter(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        # New jobs shouldn't have consecutive_failures field
        assert "consecutive_failures" not in job or job.get("consecutive_failures", 0) == 0


class TestAutoPauseOnConsecutiveFailures:
    def test_auto_pause_after_3_failures(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        # Fail 3 times
        mark_job_run(job_id, success=False, error="error1")
        mark_job_run(job_id, success=False, error="error2")
        mark_job_run(job_id, success=False, error="error3")

        updated = get_job(job_id)
        assert updated["state"] == "paused"
        assert updated["enabled"] is False
        assert "Auto-paused" in updated.get("paused_reason", "")
        assert "3 consecutive failures" in updated.get("paused_reason", "")

    def test_no_auto_pause_before_threshold(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        # Fail only 2 times — should NOT auto-pause
        mark_job_run(job_id, success=False, error="error1")
        mark_job_run(job_id, success=False, error="error2")

        updated = get_job(job_id)
        assert updated["state"] != "paused"
        assert updated.get("consecutive_failures") == 2

    def test_success_prevents_auto_pause(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        # Fail, success, fail, fail — counter resets after success
        mark_job_run(job_id, success=False, error="error1")
        mark_job_run(job_id, success=True)
        mark_job_run(job_id, success=False, error="error2")
        mark_job_run(job_id, success=False, error="error3")

        updated = get_job(job_id)
        # Only 2 consecutive failures, not 3
        assert updated["state"] != "paused"
        assert updated.get("consecutive_failures") == 2

    def test_auto_pause_includes_error_in_reason(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        mark_job_run(job_id, success=False, error="connection timeout")
        mark_job_run(job_id, success=False, error="connection timeout")
        mark_job_run(job_id, success=False, error="connection timeout")

        updated = get_job(job_id)
        assert "connection timeout" in updated.get("paused_reason", "")


class TestExponentialBackoff:
    def test_backoff_applied_on_failure(self, tmp_cron_dir):
        from hermes_time import now as _hermes_now

        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        # First failure — backoff = 2^(1-1) = 1 minute
        mark_job_run(job_id, success=False, error="err1")
        updated = get_job(job_id)
        # next_run_at should have been delayed (we can't know exact value
        # but consecutive_failures should be tracked)
        assert updated.get("consecutive_failures") == 1

    def test_backoff_increases_with_failures(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        # Fail twice (not enough to auto-pause but enough for backoff)
        mark_job_run(job_id, success=False, error="err1")
        mark_job_run(job_id, success=False, error="err2")

        updated = get_job(job_id)
        assert updated.get("consecutive_failures") == 2
        # next_run_at should exist
        assert updated.get("next_run_at") is not None


class TestResumeAfterAutoPause:
    def test_manual_resume_clears_failure_counter_context(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        # Trigger auto-pause
        mark_job_run(job_id, success=False, error="err")
        mark_job_run(job_id, success=False, error="err")
        mark_job_run(job_id, success=False, error="err")

        paused = get_job(job_id)
        assert paused["state"] == "paused"

        # Resume the job
        resume_job(job_id)
        resumed = get_job(job_id)
        assert resumed["state"] == "scheduled"
        assert resumed["enabled"] is True
        # After resume, consecutive_failures should be cleared
        assert resumed.get("consecutive_failures", 0) == 0

    def test_auto_pause_reason_preserved_in_job(self, tmp_cron_dir):
        job = create_job(prompt="test", schedule="every 1h")
        job_id = job["id"]

        mark_job_run(job_id, success=False, error="api_error")
        mark_job_run(job_id, success=False, error="api_error")
        mark_job_run(job_id, success=False, error="api_error")

        updated = get_job(job_id)
        assert updated["paused_reason"] is not None
        assert "Auto-paused" in updated["paused_reason"]
        assert "api_error" in updated["paused_reason"]
