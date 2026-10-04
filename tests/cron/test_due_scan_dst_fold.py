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


def _job(next_run_at):
    return {
        "id": "dst-job",
        "name": "dst-job",
        "prompt": "fixture",
        "schedule": {"kind": "interval", "minutes": 60},
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


def test_fold_hour_timers_measure_real_elapsed_time(dst_cron_store, monkeypatch):
    """Claim ages and re-run delays count real seconds across the repeated hour."""
    from cron import unreachable_retry

    now = _instant(6, 2)  # 01:02 EST, fold=1
    claimed = _instant(5, 58)  # 01:58 EDT, fold=0: four real minutes earlier
    assert jobs._claim_is_live({"at": claimed.isoformat(), "by": "peer-host:1"}, now, 300)

    monkeypatch.setattr(unreachable_retry, "_hermes_now", lambda: now)
    job = _job(_instant(9, 0))  # natural next run 04:00 EST, well past the ladder
    assert unreachable_retry.plan_retry(job) is True
    retry_at = datetime.fromisoformat(job["next_run_at"])
    assert retry_at.timestamp() - now.timestamp() == unreachable_retry.RETRY_DELAYS_SECONDS[0]
