"""Missed-run dispatch visibility (#99879).

The catch-up machinery already ran late fires after gateway downtime, but
the run then looked like an ordinary on-time success: nothing recorded the
scheduled instant vs the actual dispatch time. The due-scan now persists a
``last_dispatch`` stamp (scheduled_at / dispatched_at / lateness_seconds /
kind) on every recurring dispatch so `hermes cron list` and
`hermes cron status` can surface late catch-ups.
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import (
    _classify_dispatch_lateness,
    get_due_jobs,
    load_jobs,
    save_jobs,
)

FIXED_NOW = datetime(2026, 9, 1, 9, 31, 0, tzinfo=timezone.utc)


@pytest.fixture()
def cron_store(tmp_path, monkeypatch):
    """Redirect cron storage to a temp dir and pin the clock."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    monkeypatch.setattr("cron.jobs._hermes_now", lambda: FIXED_NOW)
    return tmp_path


def _daily_job(jid, next_run_dt, **extra):
    job = {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "cron", "expr": "0 9 * * *"},
        "next_run_at": next_run_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
    }
    job.update(extra)
    return job


def _interval_job(jid, next_run_dt, **extra):
    job = {
        "id": jid,
        "name": jid,
        "prompt": "x",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": next_run_dt.isoformat(),
        "last_run_at": None,
        "enabled": True,
        "state": "scheduled",
        "repeat": {"times": None, "completed": 0},
        "deliver": "local",
    }
    job.update(extra)
    return job


class TestDueScanDispatchStamp:
    def test_catch_up_beyond_grace_stamped_and_persisted(self, cron_store):
        # Scheduled yesterday 09:00, now is today 09:31 — far beyond the
        # 2h max grace for a daily job: the catch-up path fires once now.
        scheduled = FIXED_NOW - timedelta(hours=24, minutes=31)
        save_jobs([_daily_job("daily", scheduled)])

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["daily"]
        stamp = due[0]["last_dispatch"]
        assert stamp["kind"] == "catch_up"
        assert stamp["scheduled_at"] == scheduled.isoformat()
        assert stamp["dispatched_at"] == FIXED_NOW.isoformat()
        expected_late = (FIXED_NOW - scheduled).total_seconds()
        assert stamp["lateness_seconds"] == pytest.approx(expected_late, abs=1)
        # Persisted, so a separate `hermes cron list` process can read it.
        persisted = load_jobs()[0]
        assert persisted["last_dispatch"] == stamp

    def test_late_within_grace_stamped_late(self, cron_store):
        # 31 minutes late: within the daily 2h grace window but beyond the
        # on-time ticker tolerance — the shape from issue #99879's report.
        scheduled = FIXED_NOW - timedelta(minutes=31)
        save_jobs([_daily_job("daily", scheduled)])

        due = get_due_jobs()

        stamp = due[0]["last_dispatch"]
        assert stamp["kind"] == "late"
        assert stamp["lateness_seconds"] == pytest.approx(31 * 60, abs=1)

    def test_on_time_dispatch_stamped_on_time(self, cron_store):
        # Interval schedule: no cron-expr matching guard, dispatch 30s late
        # is normal ticker cadence.
        scheduled = FIXED_NOW - timedelta(seconds=30)
        save_jobs([_interval_job("hourly", scheduled)])

        due = get_due_jobs()

        stamp = due[0]["last_dispatch"]
        assert stamp["kind"] == "on_time"
        assert stamp["lateness_seconds"] == pytest.approx(30, abs=1)

    def test_manual_trigger_not_stamped(self, cron_store):
        # A manual trigger stamps manual_run_at == next_run_at; there is no
        # scheduled instant to be late against, so no dispatch stamp.
        run_at = FIXED_NOW - timedelta(hours=5)
        save_jobs([
            _daily_job("manual", run_at, manual_run_at=run_at.isoformat())
        ])

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["manual"]
        assert "last_dispatch" not in due[0]

    def test_new_dispatch_overwrites_stale_stamp(self, cron_store):
        scheduled = FIXED_NOW - timedelta(seconds=10)
        stale = {
            "scheduled_at": "2026-08-30T09:00:00+00:00",
            "dispatched_at": "2026-08-30T11:00:00+00:00",
            "lateness_seconds": 7200.0,
            "kind": "catch_up",
        }
        save_jobs([_interval_job("hourly", scheduled, last_dispatch=stale)])

        due = get_due_jobs()

        stamp = due[0]["last_dispatch"]
        assert stamp["kind"] == "on_time"
        assert stamp["scheduled_at"] == scheduled.isoformat()


class TestClassifyDispatchLateness:
    def test_within_tolerance_is_on_time(self):
        assert _classify_dispatch_lateness(0, 7200) == "on_time"
        assert _classify_dispatch_lateness(299, 7200) == "on_time"

    def test_beyond_tolerance_within_grace_is_late(self):
        assert _classify_dispatch_lateness(301, 7200) == "late"
        assert _classify_dispatch_lateness(7200, 7200) == "late"

    def test_beyond_grace_is_catch_up(self):
        assert _classify_dispatch_lateness(7201, 7200) == "catch_up"
