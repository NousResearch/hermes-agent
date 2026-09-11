"""CLI rendering of missed-run dispatch visibility (#99879).

`hermes cron list` shows a Dispatch line (scheduled vs actual, lateness,
disposition) and `hermes cron status` calls out jobs whose last dispatch
was a late/missed-fire catch-up.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import create_job, load_jobs, save_jobs
from hermes_cli.cron import (
    _dispatch_display,
    _format_lateness,
    _print_active_jobs_summary,
    cron_list,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _stamp_last_dispatch(job_id, stamp):
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job["last_dispatch"] = stamp
    save_jobs(jobs)


def _catch_up_stamp(late_seconds=1860.0, kind="catch_up"):
    scheduled = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)
    return {
        "scheduled_at": scheduled.isoformat(),
        "dispatched_at": (scheduled + timedelta(seconds=late_seconds)).isoformat(),
        "lateness_seconds": late_seconds,
        "kind": kind,
    }


class TestCronListDispatchLine:
    def test_catch_up_dispatch_rendered(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.cron._warn_if_gateway_not_running", lambda: None
        )
        job = create_job(prompt="daily report", schedule="0 9 * * *")
        _stamp_last_dispatch(job["id"], _catch_up_stamp())

        cron_list()

        out = capsys.readouterr().out
        assert "Dispatch:" in out
        assert "catch-up after missed fire" in out
        assert "2026-09-01T09:00:00+00:00" in out  # scheduled instant
        assert "31m late" in out

    def test_on_time_dispatch_rendered_quietly(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.cron._warn_if_gateway_not_running", lambda: None
        )
        job = create_job(prompt="daily report", schedule="0 9 * * *")
        _stamp_last_dispatch(job["id"], _catch_up_stamp(30.0, kind="on_time"))

        cron_list()

        out = capsys.readouterr().out
        assert "Dispatch:" in out
        assert "on time" in out
        assert "catch-up" not in out

    def test_no_stamp_no_dispatch_line(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.cron._warn_if_gateway_not_running", lambda: None
        )
        create_job(prompt="daily report", schedule="0 9 * * *")

        cron_list()

        assert "Dispatch:" not in capsys.readouterr().out


class TestStatusLateJobsCallout:
    def test_late_jobs_called_out(self, capsys):
        jobs = [
            {
                "id": "abc123",
                "name": "daily 9am",
                "next_run_at": "2026-09-02T09:00:00+00:00",
                "last_dispatch": _catch_up_stamp(),
            },
            {
                "id": "def456",
                "name": "hourly ping",
                "next_run_at": "2026-09-01T10:00:00+00:00",
                "last_dispatch": _catch_up_stamp(10.0, kind="on_time"),
            },
        ]

        _print_active_jobs_summary(jobs)

        out = capsys.readouterr().out
        assert "1 job(s) last fired late (missed-fire catch-up)" in out
        assert "abc123" in out
        assert "31m late" in out
        # On-time job is not in the callout.
        assert "def456" not in out

    def test_no_late_jobs_no_callout(self, capsys):
        jobs = [
            {
                "id": "def456",
                "name": "hourly ping",
                "next_run_at": "2026-09-01T10:00:00+00:00",
                "last_dispatch": _catch_up_stamp(10.0, kind="on_time"),
            }
        ]

        _print_active_jobs_summary(jobs)

        assert "fired late" not in capsys.readouterr().out


class TestDisplayHelpers:
    def test_format_lateness(self):
        assert _format_lateness(45) == "45s"
        assert _format_lateness(1860) == "31m"
        assert _format_lateness(9000) == "2h 30m"
        assert _format_lateness(90000) == "1d 1h"
        assert _format_lateness("bogus") == "?"

    def test_dispatch_display_malformed_returns_none(self):
        assert _dispatch_display(None) is None
        assert _dispatch_display("late") is None
        assert _dispatch_display({}) is None
        assert _dispatch_display({"scheduled_at": "x"}) is None

    def test_dispatch_display_late_kind(self):
        line = _dispatch_display(_catch_up_stamp(600.0, kind="late"))
        assert "late" in line
        assert "10m" in line
