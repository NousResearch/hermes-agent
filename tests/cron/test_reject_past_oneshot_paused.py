"""Tests for rejecting past one-shot schedules on update of paused jobs.

Regression: updating a *paused* job's schedule to a past one-shot used to
succeed silently (the validation gate was inside the `if not paused` block),
storing next_run_at=None and creating a ghost job that would never fire.
"""

import datetime as _dt
from unittest.mock import patch

import pytest

from cron.jobs import update_job, create_job


def _make_paused_job():
    """Create a recurring job and pause it."""
    now = _dt.datetime(2026, 8, 11, 12, 0, 0)
    ts = now.timestamp()
    with patch("time.time", return_value=ts):
        job = create_job(
            schedule="every 1h",
            prompt="test prompt",
        )
        job = update_job(job["id"], {"state": "paused"})
    return job


class TestPausedJobPastOneshotUpdate:
    """Validate that paused jobs can't be given past one-shot schedules."""

    def test_update_paused_job_to_past_oneshot_rejected(self, tmp_cron_dir):
        """Paused job with past one-shot schedule must be rejected."""
        job = _make_paused_job()
        now = _dt.datetime(2026, 8, 11, 13, 0, 0)
        ts = now.timestamp()
        # 1 hour in the past — well beyond 120s grace
        past_time = (now - _dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

        with pytest.raises(ValueError, match="cannot be scheduled"):
            with patch("time.time", return_value=ts):
                update_job(job["id"], {"schedule": past_time})

    def test_update_paused_job_to_past_oneshot_just_expired(self, tmp_cron_dir):
        """Paused job with a one-shot that just expired (beyond grace) — rejected."""
        job = _make_paused_job()
        now = _dt.datetime(2026, 8, 11, 13, 0, 5)
        ts = now.timestamp()
        # 3 minutes in the past — beyond 120s grace
        past_time = (now - _dt.timedelta(minutes=3)).strftime("%Y-%m-%dT%H:%M:%S")

        with pytest.raises(ValueError, match="cannot be scheduled"):
            with patch("time.time", return_value=ts):
                update_job(job["id"], {"schedule": past_time})

    def test_update_paused_job_to_recurring_does_not_raise(self, tmp_cron_dir):
        """Paused job with recurring schedule update must NOT raise ValueError."""
        job = _make_paused_job()
        # Should succeed without any past-oneshot ValueError
        with patch("time.time", return_value=_dt.datetime(2026, 8, 11, 13, 0, 0).timestamp()):
            updated = update_job(job["id"], {"schedule": "every 30m"})

        # Just verify it returned successfully — the key assertion is
        # that no ValueError was raised for a valid recurring schedule.
        assert updated is not None

    def test_update_active_job_to_past_oneshot_still_rejected(self, tmp_cron_dir):
        """Active (non-paused) jobs with past one-shots still rejected."""
        now = _dt.datetime(2026, 8, 11, 12, 0, 0)
        ts = now.timestamp()
        with patch("time.time", return_value=ts):
            job = create_job(schedule="every 1h", prompt="test prompt")

        later = _dt.datetime(2026, 8, 11, 13, 0, 0)
        past_time = (later - _dt.timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")

        with pytest.raises(ValueError, match="cannot be scheduled"):
            with patch("time.time", return_value=later.timestamp()):
                update_job(job["id"], {"schedule": past_time})
