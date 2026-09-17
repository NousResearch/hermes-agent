"""Resuming a paused recurring job must not swallow the slot it paused over (#113603)."""
from datetime import timedelta

import pytest

from cron.jobs import (
    create_job,
    load_jobs,
    pause_job,
    resume_job,
    save_jobs,
    _hermes_now,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


class TestResumeAcrossMissedSlot:
    def test_resume_keeps_paused_over_slot_due(self, tmp_cron_dir):
        """A cron job paused BEFORE its slot and resumed AFTER it must come
        back with the past slot as ``next_run_at`` (due), not the next slot —
        the catch-up / fast-forward policy then decides, instead of the slot
        vanishing silently (#113603 hypothesis 5)."""
        now = _hermes_now()
        job = create_job(prompt="daily", schedule="0 9 * * *", deliver="local")
        # Park next_run_at in the past (the slot passed while paused).
        stored = load_jobs()
        row = next(r for r in stored if r["id"] == job["id"])
        past = (now - timedelta(hours=3)).isoformat()
        row["next_run_at"] = past
        save_jobs(stored)

        pause_job(job["id"], reason="maintenance")
        resumed = resume_job(job["id"])

        assert resumed is not None
        # Contract: the paused-over slot survives resume as the due instant.
        assert resumed["next_run_at"] == past, (
            "resume re-anchored past the missed slot — the occurrence was "
            "consumed with no ledger row (#113603)")
