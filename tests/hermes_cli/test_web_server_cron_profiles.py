"""Regression tests for dashboard cron job profile routing."""

from concurrent.futures import ThreadPoolExecutor
import json
from queue import Empty, SimpleQueue
import threading

import pytest
from fastapi import HTTPException


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch):
    """Give profile discovery an isolated default home with one named profile."""
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"

    for home in (default_home, worker_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    return {"default": default_home, "worker_alpha": worker_home}


def _drain_queue(q):
    values = []
    while True:
        try:
            values.append(q.get_nowait())
        except Empty:
            return values




def test_fire_cron_job_scopes_store_and_runtime_home_together(
    isolated_profiles,
    monkeypatch,
):
    """A profile fire must execute and persist under the same profile home."""
    from cron import jobs as cron_jobs
    from cron import scheduler
    from hermes_cli import web_server

    from hermes_constants import (
        reset_hermes_home_override,
        set_hermes_home_override,
    )

    default_home = isolated_profiles["default"]
    worker_home = isolated_profiles["worker_alpha"]
    monkeypatch.setattr(scheduler, "_hermes_home", None)
    captured = {}

    class RecordingProvider:
        def fire_due(self, job_id, *, adapters=None, loop=None):
            captured["job_id"] = job_id
            captured["runtime_home"] = scheduler._get_hermes_home()
            captured["jobs_file"] = cron_jobs._current_cron_store().jobs_file
            return True

    monkeypatch.setattr(
        "cron.scheduler_provider.resolve_cron_scheduler",
        lambda: RecordingProvider(),
    )

    outer_token = set_hermes_home_override(default_home)
    try:
        assert web_server._fire_cron_job_for_profile("worker_alpha", "worker-job") is True
        assert captured == {
            "job_id": "worker-job",
            "runtime_home": worker_home,
            "jobs_file": worker_home / "cron" / "jobs.json",
        }
        assert scheduler._get_hermes_home() == default_home
    finally:
        reset_hermes_home_override(outer_token)


def test_profile_call_cannot_retarget_ticker_store_mid_write(
    isolated_profiles,
    monkeypatch,
):
    """A dashboard profile call must not redirect a concurrent ticker save."""
    from cron import jobs as cron_jobs
    from hermes_cli import web_server

    default_cron = isolated_profiles["default"] / "cron"
    worker_cron = isolated_profiles["worker_alpha"] / "cron"
    default_file = default_cron / "jobs.json"
    worker_file = worker_cron / "jobs.json"
    default_job = {
        "id": "default-job",
        "name": "default job",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": "2026-07-09T00:00:00+00:00",
    }
    worker_job = {
        "id": "worker-job",
        "name": "worker job",
        "schedule": {"kind": "interval", "minutes": 60},
        "next_run_at": "2026-07-09T00:00:00+00:00",
    }
    default_file.write_text(json.dumps({"jobs": [default_job]}), encoding="utf-8")
    worker_file.write_text(json.dumps({"jobs": [worker_job]}), encoding="utf-8")

    monkeypatch.setattr(cron_jobs, "CRON_DIR", default_cron)
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", default_file)
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", default_cron / "output")
    monkeypatch.setattr(
        cron_jobs,
        "compute_next_run",
        lambda _schedule, _last_run_at=None: "2026-07-10T00:00:00+00:00",
    )

    ticker_loaded = threading.Event()
    release_ticker = threading.Event()
    profile_entered = threading.Event()
    ticker_done = threading.Event()
    ticker_thread = threading.local()
    original_load_jobs = cron_jobs.load_jobs

    def blocking_load_jobs():
        loaded = original_load_jobs()
        if getattr(ticker_thread, "active", False):
            ticker_loaded.set()
            assert release_ticker.wait(5), "profile call did not enter in time"
        return loaded

    def hold_profile_call():
        profile_entered.set()
        assert ticker_done.wait(5), "ticker did not finish in time"
        return True

    def run_ticker_write():
        ticker_thread.active = True
        try:
            return cron_jobs.advance_next_run("default-job")
        finally:
            ticker_done.set()

    monkeypatch.setattr(cron_jobs, "load_jobs", blocking_load_jobs)
    monkeypatch.setattr(cron_jobs, "_hold_profile_call", hold_profile_call, raising=False)

    with ThreadPoolExecutor(max_workers=2) as pool:
        ticker_future = pool.submit(run_ticker_write)
        assert ticker_loaded.wait(5), "ticker did not load the default store"
        profile_future = pool.submit(
            web_server._call_cron_for_profile,
            "worker_alpha",
            "_hold_profile_call",
        )
        assert profile_entered.wait(5), "profile call did not retarget its store"
        release_ticker.set()
        assert ticker_future.result(timeout=5) is True
        assert profile_future.result(timeout=5) is True

    default_saved = json.loads(default_file.read_text(encoding="utf-8"))["jobs"]
    worker_saved = json.loads(worker_file.read_text(encoding="utf-8"))["jobs"]
    assert [job["id"] for job in worker_saved] == ["worker-job"]
    assert [job["id"] for job in default_saved] == ["default-job"]
    assert default_saved[0]["next_run_at"] == "2026-07-10T00:00:00+00:00"






@pytest.mark.asyncio
async def test_cron_mutation_without_profile_finds_named_profile_job(isolated_profiles):
    from hermes_cli import web_server

    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="named-profile-job",
    )

    paused = await web_server.pause_cron_job(worker_job["id"])
    assert paused["profile"] == "worker_alpha"
    assert paused["enabled"] is False

    default_jobs = await web_server.list_cron_jobs(profile="default")
    worker_jobs = await web_server.list_cron_jobs(profile="worker_alpha")

    assert default_jobs == []
    assert len(worker_jobs) == 1
    assert worker_jobs[0]["id"] == worker_job["id"]
    assert worker_jobs[0]["enabled"] is False




@pytest.mark.asyncio
async def test_dashboard_cron_rejects_missing_context_from(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as create_exc:
        await web_server.create_cron_job(
            web_server.CronJobCreate(
                prompt="process missing upstream",
                schedule="every 1h",
                context_from=["missing-job-id"],
            ),
            profile="worker_alpha",
        )

    assert create_exc.value.status_code == 400
    assert "missing-job-id" in create_exc.value.detail

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="context-update-target",
    )

    with pytest.raises(HTTPException) as update_exc:
        await web_server.update_cron_job(
            job["id"],
            web_server.CronJobUpdate(
                updates={
                    "context_from": ["missing-job-id"],
                }
            ),
            profile="worker_alpha",
        )

    assert update_exc.value.status_code == 400
    assert "missing-job-id" in update_exc.value.detail










