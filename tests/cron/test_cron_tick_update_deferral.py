"""Cron tick defers due jobs while a Hermes update holds the update lock.

Regression for #113293: a job due during ``hermes update`` must keep its slot
for the post-update tick instead of advancing into the swap window and losing it.
"""

from datetime import datetime, timedelta, timezone

import pytest

import cron.scheduler as scheduler_mod
from cron.jobs import create_job, get_job, load_jobs, save_jobs
from hermes_cli.update_lock import UpdateHolder


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _make_due_now():
    job = create_job("deferred", "every 1h")
    record = get_job(job["id"])
    assert record is not None
    record.update({
        "state": "scheduled",
        "enabled": True,
        "next_run_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    })
    save_jobs([record])
    return record


class TestTickUpdateDeferral:
    def test_due_jobs_wait_out_a_live_update(self, monkeypatch, tmp_cron_dir):
        """Live update marker: tick dispatches nothing and the slot stays due."""
        _make_due_now()
        slot_before = load_jobs()[0]["next_run_at"]

        monkeypatch.setattr(
            scheduler_mod, "read_live_update",
            lambda: UpdateHolder(pid=987654321, age_seconds=1.0),
        )

        def _must_not_dispatch(*_a, **_kw):
            raise AssertionError("no job may dispatch while an update holds the lock")

        monkeypatch.setattr(scheduler_mod, "_process_due_job", _must_not_dispatch)

        assert scheduler_mod.tick(verbose=False) == 0
        # The slot is not consumed...
        assert load_jobs()[0]["next_run_at"] == slot_before
        # ...so the post-update tick still sees the job as due.
        from cron.jobs import get_due_jobs

        assert [j["id"] for j in get_due_jobs()] == [load_jobs()[0]["id"]]

    def test_due_jobs_proceed_without_a_marker(self, monkeypatch, tmp_cron_dir):
        """No marker: the guard passes and dispatch proceeds as before."""
        _make_due_now()
        monkeypatch.setattr(scheduler_mod, "read_live_update", lambda: None)

        dispatched = []
        monkeypatch.setattr(
            scheduler_mod, "_process_due_job", lambda *a, **k: dispatched.append(a) or True
        )

        assert scheduler_mod.tick(verbose=False) == 1
        assert len(dispatched) == 1
