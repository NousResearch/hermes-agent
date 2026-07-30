"""Tests for rejecting past-timestamp one-shot jobs at creation/update time.

Regression test for: one-shot jobs with a schedule target that has already
passed (beyond the grace window) are silently stored with next_run_at=null,
never fire, and produce no warning. This happened when an LLM agent or user
supplied an absolute ISO timestamp without accounting for timezone offsets.

The fix raises ValueError in create_job() and update_job() when
compute_next_run() returns None for a one-shot schedule.
"""

import pytest
from datetime import datetime, timedelta, timezone

from cron.jobs import create_job, update_job, compute_next_run


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestRejectPastOneshotOnCreate:
    """create_job must raise ValueError for one-shot schedules whose target
    time has already passed the grace window."""

    def test_past_iso_timestamp_rejected(self, tmp_cron_dir):
        """An ISO timestamp far in the past should be rejected."""
        past_schedule = "2020-01-01T09:00:00"
        with pytest.raises(ValueError, match="has already passed"):
            create_job(prompt="test", schedule=past_schedule, name="past-job")

    def test_recently_past_timestamp_within_grace_accepted(self, tmp_cron_dir):
        """A timestamp just barely past (within grace window) should still be
        accepted — the grace window exists precisely for sub-minute clock skew."""
        recent_past = (datetime.now(timezone.utc) - timedelta(seconds=30))
        # Within 120s grace — should succeed
        job = create_job(
            prompt="test",
            schedule=recent_past.strftime("%Y-%m-%dT%H:%M:%S"),
            name="grace-job",
        )
        assert job["next_run_at"] is not None

    def test_future_iso_timestamp_accepted(self, tmp_cron_dir):
        """A far-future timestamp should be accepted normally."""
        future_schedule = "2099-12-31T23:59:00"
        job = create_job(prompt="test", schedule=future_schedule, name="future-job")
        assert job["next_run_at"] is not None

    def test_relative_offset_accepted(self, tmp_cron_dir):
        """Relative offsets like '30m' compute from now, so they always work."""
        job = create_job(prompt="test", schedule="5m", name="relative-job")
        assert job["next_run_at"] is not None

    def test_error_message_mentions_relative_offset(self, tmp_cron_dir):
        """The error should guide the user toward the fix."""
        with pytest.raises(ValueError) as exc_info:
            create_job(prompt="test", schedule="2020-01-01T09:00:00")
        msg = str(exc_info.value)
        assert "relative offset" in msg or "30m" in msg


class TestRejectPastOneshotOnUpdate:
    """update_job must also reject past one-shot schedules."""

    def test_update_to_past_oneshot_rejected(self, tmp_cron_dir):
        """Updating a job's schedule to a past one-shot should raise."""
        job = create_job(prompt="test", schedule="1h", name="original")
        with pytest.raises(ValueError, match="has already passed"):
            update_job(job["id"], {"schedule": "2020-01-01T09:00:00"})

    def test_update_to_future_oneshot_accepted(self, tmp_cron_dir):
        """Updating to a future one-shot should work fine."""
        job = create_job(prompt="test", schedule="1h", name="original")
        updated = update_job(job["id"], {"schedule": "2099-12-31T23:59:00"})
        assert updated is not None
        assert updated["next_run_at"] is not None


class TestComputeNextRunReturnsNoneForPastOneshot:
    """Document the underlying behavior that the guard relies on:
    compute_next_run returns None for past one-shots."""

    def test_past_oneshot_returns_none(self):
        schedule = {"kind": "once", "run_at": "2020-01-01T09:00:00+00:00"}
        result = compute_next_run(schedule)
        assert result is None
