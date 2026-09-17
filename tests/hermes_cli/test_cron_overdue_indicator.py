"""`hermes cron list`/`hermes cron status` flag an overdue ``next_run_at`` (#114309).

A stalled ticker (dead thread, gateway process down, every tick failing) stops
advancing a job's ``next_run_at`` once it comes due, so the field freezes on a
past timestamp forever. Before this fix, `cron list` and the "Next run" line in
`cron status` printed that stale value with no visual difference from a healthy
future-scheduled job — a user skimming the output had no way to tell scheduling
had silently stopped without cross-referencing the current time by hand.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import create_job, load_jobs, save_jobs
from hermes_cli.cron import _next_run_display, _print_active_jobs_summary, cron_list


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _freeze_now(monkeypatch, when):
    monkeypatch.setattr("hermes_time.now", lambda: when)


def _stamp_next_run(job_id, iso_stamp):
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job["next_run_at"] = iso_stamp
    save_jobs(jobs)


class TestNextRunDisplayHelper:
    def test_future_timestamp_shown_plain(self, monkeypatch):
        _freeze_now(monkeypatch, datetime(2026, 1, 1, tzinfo=timezone.utc))
        stamp = "2026-01-01T01:00:00+00:00"
        assert _next_run_display(stamp) == stamp

    def test_past_timestamp_flagged_overdue(self, monkeypatch):
        _freeze_now(monkeypatch, datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc))
        stamp = "2026-01-01T00:00:00+00:00"
        rendered = _next_run_display(stamp)
        assert stamp in rendered
        assert "OVERDUE" in rendered
        assert "1h" in rendered  # exactly one hour late

    def test_missing_or_placeholder_passthrough(self, monkeypatch):
        _freeze_now(monkeypatch, datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert _next_run_display(None) == "?"
        assert _next_run_display("") == "?"
        assert _next_run_display("?") == "?"

    def test_malformed_timestamp_passthrough_no_crash(self, monkeypatch):
        _freeze_now(monkeypatch, datetime(2026, 1, 1, tzinfo=timezone.utc))
        assert _next_run_display("not-a-date") == "not-a-date"

    def test_naive_timestamp_compared_in_configured_zone(self, monkeypatch):
        # Legacy records may lack a tz offset; must not crash comparing aware vs naive.
        _freeze_now(monkeypatch, datetime(2026, 1, 1, 2, 0, 0, tzinfo=timezone.utc))
        rendered = _next_run_display("2026-01-01T00:00:00")
        assert "OVERDUE" in rendered


class TestCronListOverdueIndicator:
    def test_overdue_job_flagged_in_list(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)
        job = create_job(prompt="daily report", schedule="0 9 * * *")
        past = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        _stamp_next_run(job["id"], past)

        cron_list()

        out = capsys.readouterr().out
        assert "OVERDUE" in out

    def test_future_job_not_flagged_in_list(self, tmp_cron_dir, capsys, monkeypatch):
        monkeypatch.setattr("hermes_cli.cron._warn_if_gateway_not_running", lambda: None)
        job = create_job(prompt="daily report", schedule="0 9 * * *")
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        _stamp_next_run(job["id"], future)

        cron_list()

        assert "OVERDUE" not in capsys.readouterr().out


class TestStatusSummaryOverdueIndicator:
    def test_status_summary_flags_overdue_next_run(self, capsys, monkeypatch):
        _freeze_now(monkeypatch, datetime(2026, 1, 2, tzinfo=timezone.utc))
        jobs = [
            {
                "id": "abc123",
                "name": "daily 9am",
                "next_run_at": "2026-01-01T09:00:00+00:00",  # a full day overdue
            },
        ]

        _print_active_jobs_summary(jobs)

        out = capsys.readouterr().out
        assert "Next run: 2026-01-01T09:00:00+00:00" in out
        assert "OVERDUE" in out

    def test_status_summary_no_flag_for_future_run(self, capsys, monkeypatch):
        _freeze_now(monkeypatch, datetime(2026, 1, 1, tzinfo=timezone.utc))
        jobs = [
            {
                "id": "abc123",
                "name": "daily 9am",
                "next_run_at": "2026-01-02T09:00:00+00:00",
            },
        ]

        _print_active_jobs_summary(jobs)

        assert "OVERDUE" not in capsys.readouterr().out
