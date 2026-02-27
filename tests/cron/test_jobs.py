"""Tests for cron/jobs.py — schedule parsing, job CRUD, and due-job detection."""

import json
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from cron.jobs import (
    parse_duration,
    parse_schedule,
    compute_next_run,
    create_job,
    load_jobs,
    save_jobs,
    get_job,
    list_jobs,
    remove_job,
    mark_job_run,
    get_due_jobs,
    save_job_output,
)


# =========================================================================
# parse_duration
# =========================================================================

class TestParseDuration:
    def test_minutes(self):
        assert parse_duration("30m") == 30
        assert parse_duration("1min") == 1
        assert parse_duration("5mins") == 5
        assert parse_duration("10minute") == 10
        assert parse_duration("120minutes") == 120

    def test_hours(self):
        assert parse_duration("2h") == 120
        assert parse_duration("1hr") == 60
        assert parse_duration("3hrs") == 180
        assert parse_duration("1hour") == 60
        assert parse_duration("24hours") == 1440

    def test_days(self):
        assert parse_duration("1d") == 1440
        assert parse_duration("7day") == 7 * 1440
        assert parse_duration("2days") == 2 * 1440

    def test_whitespace_tolerance(self):
        assert parse_duration("  30m  ") == 30
        assert parse_duration("2 h") == 120

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_duration("abc")
        with pytest.raises(ValueError):
            parse_duration("30x")
        with pytest.raises(ValueError):
            parse_duration("")
        with pytest.raises(ValueError):
            parse_duration("m30")


# =========================================================================
# parse_schedule
# =========================================================================

class TestParseSchedule:
    def test_duration_becomes_once(self):
        result = parse_schedule("30m")
        assert result["kind"] == "once"
        assert "run_at" in result
        # run_at should be ~30 minutes from now
        run_at = datetime.fromisoformat(result["run_at"])
        assert run_at > datetime.now()
        assert run_at < datetime.now() + timedelta(minutes=31)

    def test_every_becomes_interval(self):
        result = parse_schedule("every 2h")
        assert result["kind"] == "interval"
        assert result["minutes"] == 120

    def test_every_case_insensitive(self):
        result = parse_schedule("Every 30m")
        assert result["kind"] == "interval"
        assert result["minutes"] == 30

    def test_cron_expression(self):
        pytest.importorskip("croniter")
        result = parse_schedule("0 9 * * *")
        assert result["kind"] == "cron"
        assert result["expr"] == "0 9 * * *"

    def test_iso_timestamp(self):
        result = parse_schedule("2030-01-15T14:00:00")
        assert result["kind"] == "once"
        assert "2030-01-15" in result["run_at"]

    def test_invalid_schedule_raises(self):
        with pytest.raises(ValueError):
            parse_schedule("not_a_schedule")

    def test_invalid_cron_raises(self):
        pytest.importorskip("croniter")
        with pytest.raises(ValueError):
            parse_schedule("99 99 99 99 99")


# =========================================================================
# compute_next_run
# =========================================================================

class TestComputeNextRun:
    def test_once_future_returns_time(self):
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        schedule = {"kind": "once", "run_at": future}
        assert compute_next_run(schedule) == future

    def test_once_past_returns_none(self):
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        schedule = {"kind": "once", "run_at": past}
        assert compute_next_run(schedule) is None

    def test_interval_first_run(self):
        schedule = {"kind": "interval", "minutes": 60}
        result = compute_next_run(schedule)
        next_dt = datetime.fromisoformat(result)
        # Should be ~60 minutes from now
        assert next_dt > datetime.now() + timedelta(minutes=59)

    def test_interval_subsequent_run(self):
        schedule = {"kind": "interval", "minutes": 30}
        last = datetime.now().isoformat()
        result = compute_next_run(schedule, last_run_at=last)
        next_dt = datetime.fromisoformat(result)
        # Should be ~30 minutes from last run
        assert next_dt > datetime.now() + timedelta(minutes=29)

    def test_cron_returns_future(self):
        pytest.importorskip("croniter")
        schedule = {"kind": "cron", "expr": "* * * * *"}  # every minute
        result = compute_next_run(schedule)
        assert result is not None
        next_dt = datetime.fromisoformat(result)
        assert next_dt > datetime.now()

    def test_unknown_kind_returns_none(self):
        assert compute_next_run({"kind": "unknown"}) is None


# =========================================================================
# Job CRUD (with tmp file storage)
# =========================================================================

@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestJobCRUD:
    def test_create_and_get(self, tmp_cron_dir):
        job = create_job(prompt="Check server status", schedule="30m")
        assert job["id"]
        assert job["prompt"] == "Check server status"
        assert job["enabled"] is True
        assert job["schedule"]["kind"] == "once"

        fetched = get_job(job["id"])
        assert fetched is not None
        assert fetched["prompt"] == "Check server status"

    def test_list_jobs(self, tmp_cron_dir):
        create_job(prompt="Job 1", schedule="every 1h")
        create_job(prompt="Job 2", schedule="every 2h")
        jobs = list_jobs()
        assert len(jobs) == 2

    def test_remove_job(self, tmp_cron_dir):
        job = create_job(prompt="Temp job", schedule="30m")
        assert remove_job(job["id"]) is True
        assert get_job(job["id"]) is None

    def test_remove_nonexistent_returns_false(self, tmp_cron_dir):
        assert remove_job("nonexistent") is False

    def test_auto_repeat_for_once(self, tmp_cron_dir):
        job = create_job(prompt="One-shot", schedule="1h")
        assert job["repeat"]["times"] == 1

    def test_interval_no_auto_repeat(self, tmp_cron_dir):
        job = create_job(prompt="Recurring", schedule="every 1h")
        assert job["repeat"]["times"] is None

    def test_default_delivery_origin(self, tmp_cron_dir):
        job = create_job(
            prompt="Test", schedule="30m",
            origin={"platform": "telegram", "chat_id": "123"},
        )
        assert job["deliver"] == "origin"

    def test_default_delivery_local_no_origin(self, tmp_cron_dir):
        job = create_job(prompt="Test", schedule="30m")
        assert job["deliver"] == "local"


class TestMarkJobRun:
    def test_increments_completed(self, tmp_cron_dir):
        job = create_job(prompt="Test", schedule="every 1h")
        mark_job_run(job["id"], success=True)
        updated = get_job(job["id"])
        assert updated["repeat"]["completed"] == 1
        assert updated["last_status"] == "ok"

    def test_repeat_limit_removes_job(self, tmp_cron_dir):
        job = create_job(prompt="Once", schedule="30m", repeat=1)
        mark_job_run(job["id"], success=True)
        # Job should be removed after hitting repeat limit
        assert get_job(job["id"]) is None

    def test_error_status(self, tmp_cron_dir):
        job = create_job(prompt="Fail", schedule="every 1h")
        mark_job_run(job["id"], success=False, error="timeout")
        updated = get_job(job["id"])
        assert updated["last_status"] == "error"
        assert updated["last_error"] == "timeout"


class TestGetDueJobs:
    def test_past_due_returned(self, tmp_cron_dir):
        job = create_job(prompt="Due now", schedule="every 1h")
        # Force next_run_at to the past
        jobs = load_jobs()
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 1
        assert due[0]["id"] == job["id"]

    def test_future_not_returned(self, tmp_cron_dir):
        create_job(prompt="Not yet", schedule="every 1h")
        due = get_due_jobs()
        assert len(due) == 0

    def test_disabled_not_returned(self, tmp_cron_dir):
        job = create_job(prompt="Disabled", schedule="every 1h")
        jobs = load_jobs()
        jobs[0]["enabled"] = False
        jobs[0]["next_run_at"] = (datetime.now() - timedelta(minutes=5)).isoformat()
        save_jobs(jobs)

        due = get_due_jobs()
        assert len(due) == 0


class TestSaveJobOutput:
    def test_creates_output_file(self, tmp_cron_dir):
        output_file = save_job_output("test123", "# Results\nEverything ok.")
        assert output_file.exists()
        assert output_file.read_text() == "# Results\nEverything ok."
        assert "test123" in str(output_file)
