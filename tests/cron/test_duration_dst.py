"""Duration schedules measure elapsed time, including across UTC offset changes."""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from cron import jobs


@pytest.fixture(params=[
    ("America/Toronto", "2026-03-08T01:30:00", 0),
    ("America/Toronto", "2026-11-01T00:30:00", 0),
    ("America/Toronto", "2026-11-01T01:30:00", 1),
    ("Australia/Lord_Howe", "2026-04-05T01:45:00", 0),
    ("Australia/Lord_Howe", "2026-10-04T01:45:00", 0),
    ("UTC", "2026-03-08T01:30:00", 0),
])
def start(request):
    zone, wall_time, fold = request.param
    return datetime.fromisoformat(wall_time).replace(tzinfo=ZoneInfo(zone), fold=fold)


def test_one_shot_delay_preserves_elapsed_duration(monkeypatch, start):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: start)

    schedule = jobs.parse_schedule("in 2h")
    run_at = datetime.fromisoformat(schedule["run_at"])

    assert run_at.astimezone(timezone.utc) - start.astimezone(timezone.utc) == timedelta(hours=2)
    assert run_at.utcoffset() == run_at.astimezone(start.tzinfo).utcoffset()


@pytest.mark.parametrize("resume", [False, True])
def test_interval_preserves_elapsed_duration(monkeypatch, start, resume):
    now = start + timedelta(days=2) if resume else start
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)

    schedule = jobs.parse_schedule("every 2h")
    run_at = datetime.fromisoformat(jobs.compute_next_run(
        schedule, last_run_at=start.isoformat() if resume else None,
    ))

    assert run_at.astimezone(timezone.utc) - start.astimezone(timezone.utc) == timedelta(hours=2)
    assert run_at.utcoffset() == run_at.astimezone(start.tzinfo).utcoffset()
