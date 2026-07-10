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


def test_call_cron_for_profile_routes_storage_without_mutating_globals(isolated_profiles):
    from cron import jobs as cron_jobs
    from hermes_cli import web_server

    old_cron_dir = cron_jobs.CRON_DIR
    old_jobs_file = cron_jobs.JOBS_FILE
    old_output_dir = cron_jobs.OUTPUT_DIR

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="run scheduled task",
        schedule="every 1h",
        name="worker-alpha-scan",
    )

    assert job["profile"] == "worker_alpha"
    assert job["profile_name"] == "worker_alpha"
    assert job["hermes_home"] == str(isolated_profiles["worker_alpha"])
    assert job["is_default_profile"] is False
    assert (isolated_profiles["worker_alpha"] / "cron" / "jobs.json").exists()
    assert not (isolated_profiles["default"] / "cron" / "jobs.json").exists()

    assert cron_jobs.CRON_DIR == old_cron_dir
    assert cron_jobs.JOBS_FILE == old_jobs_file
    assert cron_jobs.OUTPUT_DIR == old_output_dir


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
    assert [job["id"] for job in default_saved] == ["default-job"]
    assert default_saved[0]["next_run_at"] == "2026-07-10T00:00:00+00:00"
    assert [job["id"] for job in worker_saved] == ["worker-job"]


@pytest.mark.asyncio
async def test_list_cron_jobs_all_includes_default_and_named_profiles(isolated_profiles):
    from hermes_cli import web_server

    default_job = web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="default heartbeat",
        schedule="every 2h",
        name="default-heartbeat",
    )
    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="worker heartbeat",
        schedule="every 3h",
        name="worker-alpha-heartbeat",
    )

    jobs = await web_server.list_cron_jobs(profile="all")
    by_id = {job["id"]: job for job in jobs}

    assert set(by_id) >= {default_job["id"], worker_job["id"]}
    assert by_id[default_job["id"]]["profile"] == "default"
    assert by_id[default_job["id"]]["is_default_profile"] is True
    assert by_id[default_job["id"]]["hermes_home"] == str(isolated_profiles["default"])
    assert by_id[worker_job["id"]]["profile"] == "worker_alpha"
    assert by_id[worker_job["id"]]["is_default_profile"] is False
    assert by_id[worker_job["id"]]["hermes_home"] == str(isolated_profiles["worker_alpha"])


@pytest.mark.asyncio
async def test_list_cron_jobs_specific_profile_filters_results(isolated_profiles):
    from hermes_cli import web_server

    web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="default only",
        schedule="every 2h",
        name="default-only",
    )
    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="worker only",
        schedule="every 3h",
        name="worker-only",
    )

    jobs = await web_server.list_cron_jobs(profile="worker_alpha")

    assert [job["id"] for job in jobs] == [worker_job["id"]]
    assert jobs[0]["profile"] == "worker_alpha"


@pytest.mark.asyncio
async def test_create_cron_job_normalizes_representative_core_fields(
    isolated_profiles, tmp_path
):
    from hermes_cli import web_server

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect-status.py").write_text("print('ok')\n", encoding="utf-8")

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="summarize upstream status",
            schedule="every 1h",
            name="full-core-mapping",
            base_url="https://example.invalid/v1/",
            script=str(scripts_dir / "collect-status.py"),
            no_agent=True,
        ),
        profile="worker_alpha",
    )

    assert job["name"] == "full-core-mapping"
    assert job["base_url"] == "https://example.invalid/v1"
    assert job["script"] == "collect-status.py"
    assert job["no_agent"] is True


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
async def test_cron_profile_scan_runs_off_event_loop(isolated_profiles, monkeypatch):
    from hermes_cli import web_server

    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="thread-offload-job",
    )

    event_loop_thread = threading.get_ident()
    profile_scan_threads = SimpleQueue()
    worker_threads = SimpleQueue()
    original_profile_dicts = web_server._cron_profile_dicts
    original_find = web_server._find_cron_job_profile

    def tracking_profile_dicts():
        profile_scan_threads.put(threading.get_ident())
        return original_profile_dicts()

    def tracking_find(job_id):
        worker_threads.put(threading.get_ident())
        return original_find(job_id)

    monkeypatch.setattr(web_server, "_cron_profile_dicts", tracking_profile_dicts)
    monkeypatch.setattr(web_server, "_find_cron_job_profile", tracking_find)

    jobs = await web_server.list_cron_jobs(profile="all")
    paused = await web_server.pause_cron_job(worker_job["id"])

    assert any(job["id"] == worker_job["id"] for job in jobs)
    assert paused["profile"] == "worker_alpha"
    profile_scan_thread_ids = _drain_queue(profile_scan_threads)
    worker_thread_ids = _drain_queue(worker_threads)
    assert profile_scan_thread_ids
    assert worker_thread_ids
    assert all(thread_id != event_loop_thread for thread_id in profile_scan_thread_ids)
    assert all(thread_id != event_loop_thread for thread_id in worker_thread_ids)


@pytest.mark.asyncio
async def test_cron_dashboard_io_rejects_async_callables():
    from hermes_cli import web_server

    async def async_callable():
        return "nope"

    with pytest.raises(TypeError, match="only accepts sync callables"):
        await web_server._run_cron_dashboard_io(async_callable)



@pytest.mark.asyncio
async def test_update_cron_job_normalizes_dashboard_core_fields(isolated_profiles, tmp_path):
    from hermes_cli import web_server

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")
    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="normalizes-dashboard-fields",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "base_url": "https://example.invalid/v1/",
                "script": str(scripts_dir / "collect.py"),
                "context_from": "",
                "no_agent": True,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["base_url"] == "https://example.invalid/v1"
    assert updated["script"] == "collect.py"
    assert updated["context_from"] is None
    assert updated["no_agent"] is True


@pytest.mark.asyncio
async def test_create_cron_job_rejects_script_outside_profile_scripts(
    isolated_profiles, tmp_path
):
    from hermes_cli import web_server

    outside = tmp_path / "outside.py"
    outside.write_text("print('nope')\n", encoding="utf-8")

    with pytest.raises(HTTPException) as exc:
        await web_server.create_cron_job(
            web_server.CronJobCreate(
                schedule="every 1h",
                script=str(outside),
                no_agent=True,
            ),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "inside" in exc.value.detail


@pytest.mark.asyncio
async def test_create_cron_job_rejects_empty_agent_job(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as exc:
        await web_server.create_cron_job(
            web_server.CronJobCreate(schedule="every 1h"),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "prompt, skill, or script" in exc.value.detail


@pytest.mark.asyncio
async def test_update_cron_job_no_agent_reuses_existing_script(isolated_profiles):
    from hermes_cli import web_server

    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            schedule="every 1h",
            script=str(scripts_dir / "collect.py"),
        ),
        profile="worker_alpha",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(updates={"no_agent": True}),
        profile="worker_alpha",
    )

    assert updated["no_agent"] is True
    assert updated["script"] == "collect.py"


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


@pytest.mark.asyncio
async def test_dashboard_cron_context_from_is_profile_scoped(isolated_profiles):
    from hermes_cli import web_server

    default_job = web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="default upstream",
        schedule="every 1h",
        name="default-upstream",
    )
    worker_upstream = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="worker upstream",
        schedule="every 1h",
        name="worker-upstream",
    )

    with pytest.raises(HTTPException):
        await web_server.create_cron_job(
            web_server.CronJobCreate(
                prompt="worker downstream",
                schedule="every 1h",
                context_from=[default_job["id"]],
            ),
            profile="worker_alpha",
        )

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="worker downstream",
            schedule="every 1h",
            context_from=[worker_upstream["id"]],
        ),
        profile="worker_alpha",
    )

    assert job["context_from"] == [worker_upstream["id"]]


@pytest.mark.asyncio
async def test_update_cron_job_refreshes_snapshots_when_unpinning(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import runtime_provider, web_server

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {"provider": "worker-provider"},
    )

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="pinned-job",
        provider="fixed-provider",
        model="fixed-model",
    )

    assert job["provider_snapshot"] is None
    assert job["model_snapshot"] is None

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "provider": None,
                "model": None,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["provider"] is None
    assert updated["model"] is None
    assert updated["provider_snapshot"] == "worker-provider"
    assert updated["model_snapshot"] == "test-model"


@pytest.mark.asyncio
async def test_dashboard_cron_noop_inference_fields_keep_existing_snapshots(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import runtime_provider, web_server

    current_provider = {"name": "initial-provider"}
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {"provider": current_provider["name"]},
    )

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="dashboard-edit-job",
    )

    assert job["provider_snapshot"] == "initial-provider"
    assert job["model_snapshot"] == "test-model"

    current_provider["name"] = "changed-provider"
    (isolated_profiles["worker_alpha"] / "config.yaml").write_text(
        "model: changed-model\n",
        encoding="utf-8",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "name": "dashboard-edit-job-renamed",
                "provider": None,
                "model": None,
                "base_url": None,
                "no_agent": False,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["name"] == "dashboard-edit-job-renamed"
    assert updated["provider_snapshot"] == "initial-provider"
    assert updated["model_snapshot"] == "test-model"


@pytest.mark.asyncio
async def test_update_cron_job_clears_snapshots_for_no_agent(
    isolated_profiles,
    monkeypatch,
):
    from hermes_cli import runtime_provider, web_server

    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **kwargs: {"provider": "worker-provider"},
    )
    scripts_dir = isolated_profiles["worker_alpha"] / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "collect.py").write_text("print('ok')\n", encoding="utf-8")

    job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="agent-to-script-job",
    )

    assert job["provider_snapshot"] == "worker-provider"
    assert job["model_snapshot"] == "test-model"

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(
            updates={
                "script": str(scripts_dir / "collect.py"),
                "no_agent": True,
            }
        ),
        profile="worker_alpha",
    )

    assert updated["provider_snapshot"] is None
    assert updated["model_snapshot"] is None


@pytest.mark.asyncio
async def test_update_cron_job_rejects_id_mutation(isolated_profiles):
    """Dashboard surfaces a 400 (not a 500 or silent rename) when an
    id-mutation attempt is rejected by cron/jobs.update_job."""
    from hermes_cli import web_server

    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="managed by named profile",
        schedule="every 1h",
        name="immutable-id-job",
    )

    with pytest.raises(HTTPException) as exc:
        await web_server.update_cron_job(
            worker_job["id"],
            web_server.CronJobUpdate(updates={"id": "../escape"}),
            profile="worker_alpha",
        )

    assert exc.value.status_code == 400
    assert "id" in exc.value.detail
    worker_jobs = await web_server.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in worker_jobs] == [worker_job["id"]]


@pytest.mark.asyncio
async def test_cron_delete_with_profile_deletes_only_target_profile(isolated_profiles):
    from hermes_cli import web_server

    default_job = web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="same-ish default",
        schedule="every 1h",
        name="shared-name",
    )
    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="same-ish worker",
        schedule="every 1h",
        name="shared-name-worker",
    )

    deleted = await web_server.delete_cron_job(worker_job["id"], profile="worker_alpha")
    assert deleted == {"ok": True}

    remaining_default = await web_server.list_cron_jobs(profile="default")
    remaining_worker = await web_server.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in remaining_default] == [default_job["id"]]
    assert remaining_worker == []


@pytest.mark.asyncio
async def test_cron_profile_validation_errors(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as bad_name:
        await web_server.list_cron_jobs(profile="../bad")
    assert bad_name.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        await web_server.list_cron_jobs(profile="missing_profile")
    assert missing.value.status_code == 404
