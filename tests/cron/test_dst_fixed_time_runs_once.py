"""A fixed-time cron job runs once on the DST fall-back night, however long its run takes.

America/New_York leaves daylight time on Sunday 1 November 2026 (02:00 EDT -> 01:00 EST), so
the wall clocks 01:00-01:59 happen twice. ``mark_job_run`` recomputes ``next_run_at`` from the
run's END. A ``30 1 * * *`` run that took 45 minutes ended at 01:15 EST, and the strictly-after
guard then picked 01:30 EST, the same wall clock in the second pass: the job ran twice that
night. ``59 1 * * *`` ran twice even with a one-minute run. Following cronie, only a job with
``*`` leading its minute or hour field runs both times through the repeated hour.
"""

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("croniter")

import hermes_time
from cron import jobs
from cron.jobs import compute_next_run

NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture
def new_york(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    hermes_time.reset_cache()
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    yield
    hermes_time.reset_cache()


def _fires_over_fall_back_night(monkeypatch, expr: str, run_minutes: int) -> list[str]:
    """Tick every real minute from 00:00 EDT to 04:00 EST through the scheduler's own cycle:
    due scan, fire claim (advances next_run_at), and ``mark_job_run`` when the run ends."""
    clock = [datetime(2026, 11, 1, 4, 0, tzinfo=UTC)]  # 00:00 EDT
    monkeypatch.setattr(jobs, "_hermes_now", lambda: clock[0].astimezone(NEW_YORK))
    job = jobs.create_job("fixture", expr, deliver="local")
    fires, busy_until = [], None
    while clock[0] < datetime(2026, 11, 1, 9, 0, tzinfo=UTC):  # 04:00 EST
        if busy_until is not None and clock[0] >= busy_until:
            assert jobs.mark_job_run(job["id"], True)
            busy_until = None
        if busy_until is None:
            for due in jobs.get_due_jobs():
                if jobs.claim_job_for_fire(due["id"]):
                    fires.append(clock[0].astimezone(NEW_YORK).isoformat())
                    busy_until = clock[0] + timedelta(minutes=run_minutes)
        clock[0] += timedelta(minutes=1)  # UTC arithmetic: real minutes, both passes
    return fires


@pytest.mark.parametrize(("expr", "run_minutes", "wall"), [
    ("30 1 * * *", 45, "01:30"),  # ended 01:15 EST: was re-scheduled for 01:30 EST
    ("59 1 * * *", 1, "01:59"),   # ended 01:00 EST: was re-scheduled for 01:59 EST
    ("30 1 * * *", 1, "01:30"),   # ended inside the first pass: always ran once
])
def test_fixed_time_job_runs_once_in_the_repeated_hour(new_york, monkeypatch, expr, run_minutes, wall):
    fires = _fires_over_fall_back_night(monkeypatch, expr, run_minutes)
    assert fires == [f"2026-11-01T{wall}:00-04:00"]


def test_base_in_second_pass_schedules_the_next_day(new_york):
    base = datetime(2026, 11, 1, 1, 15, tzinfo=NEW_YORK, fold=1)  # 01:15 EST, after 01:30 EDT ran
    nxt = compute_next_run({"kind": "cron", "expr": "30 1 * * *"}, last_run_at=base.isoformat())
    assert nxt == "2026-11-02T01:30:00-05:00"


@pytest.mark.parametrize(("expr", "expected"), [
    ("*/15 * * * *", "2026-11-01T01:30:00-05:00"),  # minute wildcard: keeps its cadence
    ("30 * * * *", "2026-11-01T01:30:00-05:00"),    # hour wildcard: runs both times through
    ("@hourly", "2026-11-01T02:00:00-05:00"),
])
def test_wildcard_jobs_still_run_in_the_second_pass(new_york, expr, expected):
    base = datetime(2026, 11, 1, 1, 15, tzinfo=NEW_YORK, fold=1)
    assert compute_next_run({"kind": "cron", "expr": expr}, last_run_at=base.isoformat()) == expected
