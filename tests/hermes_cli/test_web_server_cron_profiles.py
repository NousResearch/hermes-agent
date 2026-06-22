"""Regression tests for dashboard cron job profile routing."""

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


def test_call_cron_for_profile_routes_storage_and_restores_globals(isolated_profiles):
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
async def test_list_cron_jobs_does_not_block_event_loop_when_sync_io_is_slow(
    isolated_profiles, monkeypatch
):
    """Regression test for #45072: slow synchronous sub-functions must NOT
    block the asyncio event loop.

    Before the fix, ``_cron_profile_dicts()`` and ``_call_cron_for_profile()``
    were called directly in the ``async def list_cron_jobs()`` handler, which
    freezes the event loop for the duration of every synchronous I/O call.

    After the fix, these calls are offloaded via ``asyncio.to_thread()`` so
    the event loop stays responsive.

    The test injects slow sync functions (time.sleep) and runs a concurrent
    heartbeat coroutine. If the event loop were blocked, the heartbeat would
    stall; with the fix, it continues ticking.
    """
    import asyncio
    import time as time_mod

    from hermes_cli import web_server

    SLEEP_PER_CALL = 0.2  # seconds per monkeypatched sync function
    TICK_INTERVAL = 0.01  # heartbeat granularity
    EXPECTED_TICKS = 10   # generous lower bound

    original_dicts = web_server._cron_profile_dicts
    original_call = web_server._call_cron_for_profile

    def slow_dicts():
        time_mod.sleep(SLEEP_PER_CALL)
        return original_dicts()

    def slow_call(profile, func_name, *args, **kwargs):
        time_mod.sleep(SLEEP_PER_CALL)
        return original_call(profile, func_name, *args, **kwargs)

    monkeypatch.setattr(web_server, "_cron_profile_dicts", slow_dicts)
    monkeypatch.setattr(web_server, "_call_cron_for_profile", slow_call)

    start = time_mod.monotonic()
    deadline = start + 1.0  # plenty of room for 3× slow calls

    ticks = 0

    async def heartbeat():
        nonlocal ticks
        while time_mod.monotonic() < deadline:
            ticks += 1
            await asyncio.sleep(TICK_INTERVAL)

    await asyncio.gather(
        web_server.list_cron_jobs(profile="all"),
        heartbeat(),
    )

    # If the event loop were blocked by synchronous calls, ticks would be
    # very low (~0-2). With async.offloaded to thread pool it should be
    # well above EXPECTED_TICKS.
    assert ticks > EXPECTED_TICKS, (
        f"Heartbeat only ticked {ticks} times (expected >{EXPECTED_TICKS}). "
        "This means synchronous cron/profile functions blocked the event loop."
    )

    # Also verify that the single-profile path is non-blocking
    ticks2 = 0
    start2 = time_mod.monotonic()
    deadline2 = start2 + 0.4

    async def heartbeat2():
        nonlocal ticks2
        while time_mod.monotonic() < deadline2:
            ticks2 += 1
            await asyncio.sleep(TICK_INTERVAL)

    await asyncio.gather(
        web_server.list_cron_jobs(profile="default"),
        heartbeat2(),
    )

    assert ticks2 > 5, (
        f"Single-profile heartbeat only ticked {ticks2} times "
        "(expected >5). The sync offload for single-profile path is broken."
    )


@pytest.mark.asyncio
async def test_list_cron_jobs_preserves_exceptions_from_sync_functions(
    isolated_profiles, monkeypatch
):
    """Verify that exceptions raised inside asyncio.to_thread still
    propagate correctly with the original error type and message."""
    from hermes_cli import web_server

    def broken_dicts():
        msg = "simulated sync crash"
        raise RuntimeError(msg)

    monkeypatch.setattr(web_server, "_cron_profile_dicts", broken_dicts)

    with pytest.raises(RuntimeError, match="simulated sync crash"):
        await web_server.list_cron_jobs(profile="all")


@pytest.mark.asyncio
async def test_cron_profile_validation_errors(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as bad_name:
        await web_server.list_cron_jobs(profile="../bad")
    assert bad_name.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        await web_server.list_cron_jobs(profile="missing_profile")
    assert missing.value.status_code == 404
