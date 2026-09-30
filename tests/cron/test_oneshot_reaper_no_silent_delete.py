"""LOCAL PATCH 018: a claimed-never-completed one-shot is retained, not deleted.

Job f98f9fcf2561 (2026-08-20) was a real financial deadline. Dispatch was
claimed (`repeat.completed >= times`) then the run died before `last_run_at`
was written. The reaper deleted the record, so there was no output, no error,
and no job left to inspect or rerun.

The record must stay in the store, disabled, in state `interrupted`, and the
operator must be told.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cron.jobs import claim_dispatch, get_due_jobs, load_jobs, save_jobs


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _wedged(job_id="os1"):
    return {
        "id": job_id,
        "name": "one-shot",
        "enabled": True,
        "schedule": {"kind": "once", "run_at": "2026-01-01T00:00:00+00:00"},
        "repeat": {"times": 1, "completed": 1},
        "last_run_at": None,
        "next_run_at": None,
    }


def test_already_dispatched_oneshot_is_retained_not_deleted(tmp_cron_dir):
    save_jobs([_wedged()])
    with patch("cron.scheduler._deliver_result", return_value=None) as deliver:
        assert claim_dispatch("os1") is False
    retained = load_jobs()
    assert len(retained) == 1
    assert retained[0]["id"] == "os1"
    assert retained[0]["enabled"] is False
    assert retained[0]["state"] == "interrupted"
    assert retained[0]["next_run_at"] is None
    assert retained[0]["last_run_at"] is None
    deliver.assert_called_once()


def test_claimed_never_completed_oneshot_still_in_store_after_false_claim(tmp_cron_dir):
    """Prove the load_jobs() contract the silent-delete bug violated."""
    save_jobs([_wedged("f98f9fcf2561")])
    with patch("cron.scheduler._deliver_result", return_value=None):
        claimed = claim_dispatch("f98f9fcf2561")
    assert claimed is False
    ids = [j["id"] for j in load_jobs()]
    assert "f98f9fcf2561" in ids


def test_get_due_jobs_disarms_stale_maxed_oneshot_in_place(tmp_cron_dir):
    past = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    job = _wedged()
    job["schedule"] = {"kind": "once", "run_at": past}
    save_jobs([job])
    with patch("cron.scheduler._deliver_result", return_value=None):
        due = get_due_jobs()
    assert due == []
    retained = load_jobs()
    assert len(retained) == 1
    assert retained[0]["enabled"] is False
    assert retained[0]["state"] == "interrupted"
