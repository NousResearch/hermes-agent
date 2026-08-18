"""Tests for the last_fire_error stamp (missed scheduled fires made visible).

When the hosted fire path (NAS -> dashboard -> loopback forward to the
gateway api_server) cannot reach the gateway, no execution row is ever
created — the miss used to be invisible outside gui.log. The dashboard fire
webhook now stamps ``last_fire_error`` on the job record via
``note_fire_forward_failure`` so `cronjob list`, `hermes cron list`, and the
dashboard surface it, and ``mark_job_run`` clears the stamp on the next
successful run so it always describes CURRENT auto-fire health.
"""

import pytest

from cron.jobs import (
    create_job,
    get_job,
    mark_job_run,
    note_fire_forward_failure,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestNoteFireForwardFailure:
    def test_stamps_last_fire_error(self, tmp_cron_dir):
        job = create_job(prompt="Daily invoice triage", schedule="every 1h")
        assert note_fire_forward_failure(job["id"], "gateway unreachable") is True

        stamped = get_job(job["id"])
        err = stamped["last_fire_error"]
        assert isinstance(err, dict)
        assert err["detail"] == "gateway unreachable"
        # Timestamp parses as ISO.
        from datetime import datetime
        datetime.fromisoformat(err["at"])

    def test_unknown_job_returns_false(self, tmp_cron_dir):
        assert note_fire_forward_failure("nope", "gateway unreachable") is False

    def test_repeated_failures_overwrite_latest_wins(self, tmp_cron_dir):
        job = create_job(prompt="Daily invoice triage", schedule="every 1h")
        note_fire_forward_failure(job["id"], "first miss")
        note_fire_forward_failure(job["id"], "second miss")
        err = get_job(job["id"])["last_fire_error"]
        assert err["detail"] == "second miss"

    def test_detail_truncated_to_500(self, tmp_cron_dir):
        job = create_job(prompt="Daily invoice triage", schedule="every 1h")
        note_fire_forward_failure(job["id"], "x" * 2000)
        err = get_job(job["id"])["last_fire_error"]
        assert len(err["detail"]) == 500

    def test_successful_run_clears_stamp(self, tmp_cron_dir):
        """The stamp describes CURRENT auto-fire health — a run that made it
        through the fire path proves the hand-off works again."""
        job = create_job(prompt="Daily invoice triage", schedule="every 1h")
        note_fire_forward_failure(job["id"], "gateway unreachable")
        assert get_job(job["id"])["last_fire_error"] is not None

        assert mark_job_run(job["id"], success=True) is True
        assert get_job(job["id"]).get("last_fire_error") is None

    def test_failed_run_keeps_stamp(self, tmp_cron_dir):
        """An agent-level failure is not proof the fire hand-off healed —
        only success clears (mirrors preflight_alerted / drift_alerted)."""
        job = create_job(prompt="Daily invoice triage", schedule="every 1h")
        note_fire_forward_failure(job["id"], "gateway unreachable")

        assert mark_job_run(job["id"], success=False, error="boom") is True
        err = get_job(job["id"]).get("last_fire_error")
        assert isinstance(err, dict)
        assert err["detail"] == "gateway unreachable"


class TestFormatJobSurfacesFireError:
    def test_cronjob_list_carries_last_fire_error(self, tmp_cron_dir):
        from tools.cronjob_tools import _format_job

        job = create_job(prompt="Daily invoice triage", schedule="every 1h")
        note_fire_forward_failure(job["id"], "gateway unreachable")
        formatted = _format_job(get_job(job["id"]))
        assert formatted["last_fire_error"]["detail"] == "gateway unreachable"
