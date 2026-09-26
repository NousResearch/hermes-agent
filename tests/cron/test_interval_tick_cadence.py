"""Cadence contract for an interval equal to the ticker period (#91901)."""
from datetime import datetime, timedelta, timezone

from cron import jobs


def test_interval_equal_to_ticker_keeps_every_due_slot(tmp_path, monkeypatch):
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(jobs, "_hermes_now", lambda: now)
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(prompt="cadence", schedule="every 1m")
        first_due = datetime.fromisoformat(job["next_run_at"])
        fired = []
        for tick in range(6):
            now = first_due + timedelta(seconds=60 * tick)
            due = jobs.get_due_jobs()
            if not due:
                continue
            fired.append(now)
            jobs.advance_next_runs([job["id"]])
            assert jobs.claim_job_for_fire(job["id"], return_job=True)
            now += timedelta(milliseconds=250)
            assert jobs.mark_job_run(job["id"], success=True)
        assert fired == [first_due + timedelta(seconds=60 * tick) for tick in range(6)]
