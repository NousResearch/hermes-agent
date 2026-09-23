"""Due-scan comparisons must use absolute instants across a repeated DST hour."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import hermes_time
import pytest

from cron import jobs


NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture
def dst_cron_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_TIMEZONE", "America/New_York")
    hermes_time.reset_cache()
    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    yield tmp_path
    hermes_time.reset_cache()


def _instant(hour, minute):
    return datetime(2026, 11, 1, hour, minute, tzinfo=timezone.utc).astimezone(NEW_YORK)


def _job(next_run_at, *, kind="interval", expr=None):
    schedule = {"kind": kind, "minutes": 60}
    if expr is not None:
        schedule["expr"] = expr
    return {
        "id": "dst-job",
        "name": "dst-job",
        "prompt": "fixture",
        "schedule": schedule,
        "next_run_at": next_run_at.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
    }


def _due_at(monkeypatch, now, scheduled):
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    jobs.save_jobs([_job(scheduled)])
    return [row["id"] for row in jobs.get_due_jobs()]


def test_fold_one_interval_is_not_due_during_fold_zero(dst_cron_store, monkeypatch):
    now = _instant(5, 1)  # 01:01 EDT, fold=0
    scheduled = _instant(6, 0)  # 01:00 EST, fold=1
    assert now.fold == 0
    assert scheduled.fold == 1
    assert scheduled.timestamp() - now.timestamp() == 59 * 60

    assert _due_at(monkeypatch, now, scheduled) == []


def test_interval_runs_when_due_in_either_fold(dst_cron_store, monkeypatch):
    first_fold_now = _instant(5, 1)
    first_fold_slot = _instant(5, 0)
    second_fold_now = _instant(6, 1)
    second_fold_slot = _instant(6, 0)

    assert first_fold_now.fold == first_fold_slot.fold == 0
    assert second_fold_now.fold == second_fold_slot.fold == 1
    assert _due_at(monkeypatch, first_fold_now, first_fold_slot) == ["dst-job"]
    assert _due_at(monkeypatch, second_fold_now, second_fold_slot) == ["dst-job"]


def test_cron_future_occurrence_is_not_due_in_prior_fold(dst_cron_store, monkeypatch):
    now = _instant(5, 31)  # 01:31 EDT, fold=0
    scheduled = _instant(6, 30)  # 01:30 EST, fold=1
    assert scheduled.timestamp() > now.timestamp()

    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    jobs.save_jobs([_job(scheduled, kind="cron", expr="30 1 * * *")])
    assert jobs.get_due_jobs() == []


def test_hourly_interval_does_not_recur_each_minute_during_fallback(dst_cron_store, monkeypatch):
    """A one-hour cadence remains hourly across both occurrences of local 01:00."""
    from datetime import timedelta

    start = datetime(2026, 11, 1, 4, 0, tzinfo=timezone.utc)  # 00:00 EDT
    initial_now = (start - timedelta(hours=1)).astimezone(NEW_YORK)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: initial_now)
    job = jobs.create_job(
        prompt="fixture", schedule="every 60m", model="fixture", deliver="local")
    fired_at = []

    for elapsed_minutes in range(181):
        now = (start + timedelta(minutes=elapsed_minutes)).astimezone(NEW_YORK)
        monkeypatch.setattr(jobs, "_hermes_now", lambda now=now: now)
        due = jobs.get_due_jobs()
        if due:
            fired_at.append(now.timestamp())
            assert [row["id"] for row in due] == [job["id"]]
            assert jobs.claim_job_for_fire(job["id"]) is True
            assert jobs.mark_job_run(job["id"], success=True) is True

    assert fired_at == [
        (start + timedelta(hours=offset)).timestamp() for offset in range(4)
    ]
