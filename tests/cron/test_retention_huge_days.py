"""A "keep forever" retention value (10**9 days, ``.inf``) must keep records, not crash the due scan."""
from datetime import UTC, datetime, timedelta

import pytest


@pytest.mark.parametrize("days", [10**9, 10**6, float("inf")])
def test_completed_oneshot_kept_and_due_scan_survives_huge_retention(tmp_path, monkeypatch, days):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(f"cron:\n  completed_retention_days: {'.inf' if days == float('inf') else days}\n")
    from cron import jobs

    done = jobs.create_job(prompt="x", schedule="in 30m", name="done")
    raw = jobs.load_jobs()
    rec = next(j for j in raw if j["id"] == done["id"])
    rec.update(state="completed", enabled=False,
               last_run_at=(datetime.now(UTC) - timedelta(days=400)).isoformat())
    jobs.save_jobs(raw)

    jobs.get_due_jobs()  # raised OverflowError before, so no cron job could ever fire

    assert jobs.get_job(done["id"]) is not None


def test_retention_cutoff_floors_at_the_earliest_datetime():
    from utils import retention_cutoff

    now = datetime.now(UTC)
    assert retention_cutoff(now, 7) == now - timedelta(days=7)
    assert retention_cutoff(now, 10**9) == datetime.min.replace(tzinfo=UTC)
    assert retention_cutoff(now, float("nan")) == datetime.min.replace(tzinfo=UTC)
