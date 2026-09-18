"""Cron status orders persisted next runs by instant, including a DST fold."""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from cron import jobs as job_store
from hermes_cli.cron import _print_active_jobs_summary


@pytest.mark.parametrize("later,earlier", [
    ("2026-11-01T01:15:00-05:00", "2026-11-01T01:45:00-04:00"),
    ("2026-11-02T09:00:00-05:00", "2026-11-02T08:00:00-05:00"),
    ("2026-11-02T08:00:00-05:00", "2026-11-02T12:30:00+00:00"),
])
def test_status_earliest_persisted_instant(later, earlier, monkeypatch, capsys):
    # Only freeze the clock: creation, persistence, normalization and listing are real.
    now = datetime(2026, 10, 31, tzinfo=ZoneInfo("America/New_York"))
    monkeypatch.setattr(job_store, "_hermes_now", lambda: now)
    late_job = job_store.create_job(prompt="later", schedule=later)
    early_job = job_store.create_job(prompt="earlier", schedule=earlier)
    jobs = job_store.list_jobs()
    assert {job["id"] for job in jobs} == {late_job["id"], early_job["id"]}
    expected = next(job["next_run_at"] for job in jobs if job["id"] == early_job["id"])

    _print_active_jobs_summary(jobs)

    out = capsys.readouterr().out
    assert "2 active job(s)" in out
    assert f"Next run: {expected}\n" in out
    assert job_store.list_jobs() == jobs  # Rendering never reschedules a job.


@pytest.mark.parametrize("jobs,expected", [
    ([], "  No active jobs\n"),
    ([{}, {"next_run_at": None}, {"next_run_at": ""}], "  3 active job(s)\n"),
    ([{"next_run_at": "bad"}, {"next_run_at": 123}], "  2 active job(s)\n"),
    ([{"next_run_at": "bad"}, {"next_run_at": "2026-11-01T06:15:00Z"}],
     "  2 active job(s)\n  Next run: 2026-11-01T06:15:00Z\n"),
    ([{"next_run_at": "2026-11-01T06:15:00Z"},
      {"next_run_at": "2026-11-01T01:15:00-05:00"}],
     "  2 active job(s)\n  Next run: 2026-11-01T06:15:00Z\n"),
])
def test_status_missing_invalid_and_equivalent_instants(jobs, expected, capsys):
    _print_active_jobs_summary(jobs)
    assert capsys.readouterr().out == expected
