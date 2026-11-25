"""Tests for the sticky ``last_failure`` stamp (#118354).

A failed cron run is recorded correctly (``last_status="error"``,
``failure_streak`` climbing), but the next successful run resets both in the
same pass — so a monitor sampling ``jobs.json`` after the fact sees a
permanently green job and a relapse is invisible. ``last_failure`` now keeps
the most recent failure (``{at, detail}``) on the job record and a later
success does NOT clear it; the recency window is the consumer's call.
"""

from datetime import datetime

import pytest

from cron.jobs import create_job, get_job, mark_job_run


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestLastFailureStamp:
    def test_failed_run_stamps_last_failure(self, tmp_cron_dir):
        job = create_job(prompt="Watch the fleet", schedule="every 1h")
        assert mark_job_run(job["id"], success=False, error="Script exited with code 1") is True

        stamp = get_job(job["id"])["last_failure"]
        assert isinstance(stamp, dict)
        assert stamp["detail"] == "Script exited with code 1"
        # Timestamp parses as ISO.
        datetime.fromisoformat(stamp["at"])

    def test_success_resets_status_but_keeps_stamp(self, tmp_cron_dir):
        """The heart of #118354: the healing success must not erase the failure."""
        job = create_job(prompt="Watch the fleet", schedule="every 1h")
        assert mark_job_run(job["id"], success=False, error="Script exited with code 1") is True
        failed = get_job(job["id"])
        assert failed["last_status"] == "error"
        assert failed["failure_streak"] == 1

        assert mark_job_run(job["id"], success=True) is True
        healed = get_job(job["id"])
        assert healed["last_status"] == "ok"
        assert healed["failure_streak"] == 0
        assert healed["last_failure"]["detail"] == "Script exited with code 1"

    def test_latest_failure_wins(self, tmp_cron_dir):
        job = create_job(prompt="Watch the fleet", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="first failure")
        mark_job_run(job["id"], success=True)
        mark_job_run(job["id"], success=False, error="second failure")
        assert get_job(job["id"])["last_failure"]["detail"] == "second failure"

    def test_success_without_prior_failure_leaves_no_stamp(self, tmp_cron_dir):
        job = create_job(prompt="Watch the fleet", schedule="every 1h")
        assert mark_job_run(job["id"], success=True) is True
        assert get_job(job["id"]).get("last_failure") is None

    def test_failure_without_error_text_uses_status(self, tmp_cron_dir):
        job = create_job(prompt="Watch the fleet", schedule="every 1h")
        assert mark_job_run(job["id"], success=False, status="blocked_config") is True
        assert get_job(job["id"])["last_failure"]["detail"] == "blocked_config"

    def test_delivery_failure_keeps_its_own_channel(self, tmp_cron_dir):
        """Agent succeeded but delivery failed: success-path (no stamp), the sticky
        ``last_delivery_error`` already covers that case."""
        job = create_job(prompt="Watch the fleet", schedule="every 1h")
        assert mark_job_run(
            job["id"], success=True, delivery_error="webhook unreachable") is True
        record = get_job(job["id"])
        assert record.get("last_failure") is None
        assert record["last_delivery_error"] == "webhook unreachable"
