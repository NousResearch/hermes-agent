"""cron.skip_missed_runs: opt-out of post-downtime catch-up firing.

Stock behavior: after the gateway is off past a job's catch-up grace window, the
due-scan collapses the missed backlog and still fires the job ONCE now — so a
multi-hour outage restarts every missed agent job at once, in parallel, which can
burn a whole day's budget in the first minute back. ``cron.skip_missed_runs: true``
fast-forwards such stale recurring jobs to their NEXT occurrence without firing.

Invariants asserted here:
- default (off): a stale recurring job still fires once now (unchanged stock path);
- on: the same stale job is excluded from the due set entirely and its next_run_at
  advances to a future occurrence;
- on + within grace: a merely-late job still fires (late dispatch is not affected).
"""

from datetime import datetime, timedelta, timezone

import pytest

from cron.jobs import get_due_jobs, load_jobs, save_jobs

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


class TestSkipMissedRuns:
    def test_default_off_stale_recurring_fires_once_now(self, cron_store):
        # 24h31m late: far beyond the 2h max grace for a daily job — stock
        # catch-up fires it once now (backlog collapsed).
        scheduled = FIXED_NOW - timedelta(hours=24, minutes=31)
        save_jobs([_daily_job("daily", scheduled)])

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["daily"]

    def test_on_stale_recurring_is_skipped_and_fast_forwarded(
        self, cron_store, monkeypatch
    ):
        monkeypatch.setattr(
            "cron.jobs._cron_config_number",
            lambda key, default, cast: cast(True)
            if key == "skip_missed_runs"
            else cast(default),
        )
        scheduled = FIXED_NOW - timedelta(hours=24, minutes=31)
        save_jobs([_daily_job("daily", scheduled)])

        due = get_due_jobs()

        # Excluded from the due set: no fire this tick.
        assert [d["id"] for d in due] == []
        # next_run_at fast-forwarded to a future occurrence (not left parked
        # in the past, so the job is not stuck due forever).
        persisted = load_jobs()[0]
        assert datetime.fromisoformat(persisted["next_run_at"]) > FIXED_NOW

    def test_on_within_grace_still_fires_late(self, cron_store, monkeypatch):
        # 31 minutes late is within the daily 2h grace window: late dispatch,
        # not a missed-run skip. skip_missed_runs must not suppress it.
        monkeypatch.setattr(
            "cron.jobs._cron_config_number",
            lambda key, default, cast: cast(True)
            if key == "skip_missed_runs"
            else cast(default),
        )
        scheduled = FIXED_NOW - timedelta(minutes=31)
        save_jobs([_daily_job("daily", scheduled)])

        due = get_due_jobs()

        assert [d["id"] for d in due] == ["daily"]
