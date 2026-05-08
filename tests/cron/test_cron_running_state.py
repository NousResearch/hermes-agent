"""Cron run-integrity state tests."""

from __future__ import annotations

import importlib
import threading
from datetime import datetime

import pytest


@pytest.fixture
def hermes_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "cron").mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    import hermes_constants
    importlib.reload(hermes_constants)
    import cron.jobs
    importlib.reload(cron.jobs)
    import cron.scheduler
    importlib.reload(cron.scheduler)

    return home


def test_mark_job_started_records_in_progress_state_without_claiming_completion(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_started

    job = create_job(prompt="report", schedule="every 5m", deliver="local")

    assert mark_job_started(job["id"]) is True
    reloaded = get_job(job["id"])

    assert reloaded["state"] == "running"
    assert reloaded["last_started_at"]
    assert reloaded["current_run_started_at"] == reloaded["last_started_at"]
    assert reloaded["last_run_at"] is None
    assert reloaded["last_status"] is None


def test_mark_job_run_clears_in_progress_state_after_completion(hermes_env):
    from cron.jobs import create_job, get_job, mark_job_run, mark_job_started

    job = create_job(prompt="report", schedule="every 5m", deliver="local")
    mark_job_started(job["id"])

    mark_job_run(job["id"], True)
    reloaded = get_job(job["id"])

    assert reloaded["state"] == "scheduled"
    assert reloaded["last_started_at"]
    assert reloaded["current_run_started_at"] is None
    assert reloaded["last_run_at"]
    assert reloaded["last_status"] == "ok"


def test_tick_exposes_running_state_while_job_is_executing(hermes_env, monkeypatch):
    import cron.jobs as jobs
    from cron.jobs import create_job, get_job
    import cron.scheduler as scheduler

    job = create_job(prompt="report", schedule="every 5m", deliver="local")
    frozen_now = datetime.fromisoformat(job["next_run_at"])
    monkeypatch.setattr(jobs, "_hermes_now", lambda: frozen_now)
    monkeypatch.setattr(scheduler, "_hermes_now", lambda: frozen_now)

    started = threading.Event()
    release = threading.Event()

    def fake_run_job(due_job):
        started.set()
        release.wait(timeout=5)
        return True, "# output", "done", None

    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda job_id, output: hermes_env / "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *args, **kwargs: None)

    thread = threading.Thread(target=scheduler.tick, kwargs={"verbose": False})
    thread.start()
    assert started.wait(timeout=5)

    in_progress = get_job(job["id"])
    assert in_progress["state"] == "running"
    assert in_progress["current_run_started_at"]
    assert in_progress["last_run_at"] is None

    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()

    completed = get_job(job["id"])
    assert completed["state"] == "scheduled"
    assert completed["current_run_started_at"] is None
    assert completed["last_run_at"]
    assert completed["last_status"] == "ok"
