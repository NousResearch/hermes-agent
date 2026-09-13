"""Regression for due-scan ordering across the repeated DST hour.

Duration schedules store an absolute instant.  During Toronto's fall-back
transition, Python's direct comparisons of two aware datetimes that share a
``ZoneInfo`` object use wall-clock ordering and can report the second-fold
target as due during the first fold.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from cron import jobs


TORONTO = ZoneInfo("America/Toronto")


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_TIMEZONE", "America/Toronto")
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    jobs._cron_cadence_cache.clear()
    return tmp_path


def test_duration_target_is_not_due_in_first_fall_back_fold(cron_store, monkeypatch):
    """A 00:30 EDT + 2h target is 06:30Z and must wait for the second fold."""
    created = datetime(2026, 11, 1, 0, 30, tzinfo=TORONTO, fold=0)
    target = jobs.compute_next_run(
        {"kind": "interval", "minutes": 120}, last_run_at=created.isoformat())
    assert target is not None

    job = {
        "id": "fold-interval",
        "name": "fold interval",
        "prompt": "check",
        "schedule": {"kind": "interval", "minutes": 120},
        "next_run_at": target,
        "last_run_at": created.isoformat(),
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
    }
    jobs.save_jobs([job])

    first_fold = datetime(2026, 11, 1, 1, 45, tzinfo=TORONTO, fold=0)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: first_fold)
    assert jobs.get_due_jobs() == []

    second_fold = datetime(2026, 11, 1, 1, 45, tzinfo=TORONTO, fold=1)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: second_fold)
    assert [item["id"] for item in jobs.get_due_jobs()] == ["fold-interval"]
