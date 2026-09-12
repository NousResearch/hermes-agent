"""Resaving jobs.json with an unchanged cron expr must not skip the current run (#100030)."""

import pytest
from datetime import datetime

from cron.jobs import (
    create_job,
    get_job,
    load_jobs,
    save_jobs,
    update_job,
    _hermes_now,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    """Redirect cron storage to a temp directory."""
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def test_resave_same_cron_expr_preserves_current_fire(tmp_cron_dir):
    """A resave (same expr) keeps the stored next_run instead of recomputing from now.

    Without the fix, croniter's strictly-after semantics recompute the *next* minute,
    skipping the current minute's execution — the monthly "0 0 1 * *" variant of this
    jumped from Sep 1 to Oct 1.
    """
    job = create_job(prompt="tick", schedule="* * * * *", name="per-minute")
    current_fire = _hermes_now().replace(second=0, microsecond=0)
    jobs = load_jobs()
    assert len(jobs) == 1
    jobs[0]["next_run_at"] = current_fire.isoformat()
    save_jobs(jobs)

    updated = update_job(job["id"], {"schedule": dict(get_job(job["id"])["schedule"])})
    assert updated["next_run_at"] == current_fire.isoformat()


def test_resave_changed_expr_recomputes(tmp_cron_dir):
    """A genuine expr change must NOT preserve the stale next_run."""
    job = create_job(prompt="tick", schedule="* * * * *", name="per-minute")
    updated = update_job(job["id"], {"schedule": "0 * * * *"})
    assert updated["schedule"]["expr"] == "0 * * * *"
    assert datetime.fromisoformat(updated["next_run_at"]).minute == 0
