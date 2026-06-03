"""Regression tests for dashboard cron job profile routing."""

import json

import pytest
from fastapi import HTTPException


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch):
    """Give profile discovery an isolated default home with one named profile."""
    from cron import jobs as cron_jobs
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    worker_home = profiles_root / "worker_alpha"

    for home in (default_home, worker_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")

    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: profiles_root)
    monkeypatch.setattr(cron_jobs, "HERMES_DIR", default_home)
    monkeypatch.setattr(cron_jobs, "CRON_DIR", default_home / "cron")
    monkeypatch.setattr(cron_jobs, "JOBS_FILE", default_home / "cron" / "jobs.json")
    monkeypatch.setattr(cron_jobs, "OUTPUT_DIR", default_home / "cron" / "output")
    return {"default": default_home, "worker_alpha": worker_home}


@pytest.mark.asyncio
async def test_dashboard_create_cron_job_writes_root_store_and_stamps_profile(
    isolated_profiles,
):
    from cron import jobs as cron_jobs
    from hermes_cli import web_server

    old_cron_dir = cron_jobs.CRON_DIR
    old_jobs_file = cron_jobs.JOBS_FILE
    old_output_dir = cron_jobs.OUTPUT_DIR

    job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="run scheduled task",
            schedule="every 1h",
            name="worker-alpha-scan",
        ),
        profile="worker_alpha",
    )

    root_jobs_file = isolated_profiles["default"] / "cron" / "jobs.json"
    profile_jobs_file = isolated_profiles["worker_alpha"] / "cron" / "jobs.json"

    assert job["profile"] == "worker_alpha"
    assert job["profile_name"] == "worker_alpha"
    assert job["hermes_home"] == str(isolated_profiles["worker_alpha"])
    assert job["is_default_profile"] is False
    assert root_jobs_file.exists()
    assert not profile_jobs_file.exists()

    stored_jobs = json.loads(root_jobs_file.read_text(encoding="utf-8"))["jobs"]
    assert len(stored_jobs) == 1
    assert stored_jobs[0]["id"] == job["id"]
    assert stored_jobs[0]["profile"] == "worker_alpha"

    assert cron_jobs.CRON_DIR == old_cron_dir
    assert cron_jobs.JOBS_FILE == old_jobs_file
    assert cron_jobs.OUTPUT_DIR == old_output_dir


@pytest.mark.asyncio
async def test_list_cron_jobs_all_includes_default_and_named_profiles(isolated_profiles):
    from hermes_cli import web_server

    default_job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="default heartbeat",
            schedule="every 2h",
            name="default-heartbeat",
        ),
        profile="default",
    )
    worker_job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="worker heartbeat",
            schedule="every 3h",
            name="worker-alpha-heartbeat",
        ),
        profile="worker_alpha",
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
    assert (isolated_profiles["default"] / "cron" / "jobs.json").exists()
    assert not (isolated_profiles["worker_alpha"] / "cron" / "jobs.json").exists()


@pytest.mark.asyncio
async def test_list_cron_jobs_specific_profile_filters_results(isolated_profiles):
    from hermes_cli import web_server

    await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="default only",
            schedule="every 2h",
            name="default-only",
        ),
        profile="default",
    )
    worker_job = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="worker only",
            schedule="every 3h",
            name="worker-only",
        ),
        profile="worker_alpha",
    )

    jobs = await web_server.list_cron_jobs(profile="worker_alpha")

    assert [job["id"] for job in jobs] == [worker_job["id"]]
    assert jobs[0]["profile"] == "worker_alpha"
    root_jobs = json.loads(
        (isolated_profiles["default"] / "cron" / "jobs.json").read_text(
            encoding="utf-8"
        )
    )["jobs"]
    assert {job["profile"] for job in root_jobs} == {"default", "worker_alpha"}


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
async def test_cron_delete_with_wrong_profile_does_not_cross_profile_store(
    isolated_profiles,
):
    from hermes_cli import web_server

    worker_job = web_server._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="worker only",
        schedule="every 1h",
        name="worker-only-delete",
    )

    with pytest.raises(HTTPException) as exc:
        await web_server.delete_cron_job(worker_job["id"], profile="default")

    assert exc.value.status_code == 404
    worker_jobs = await web_server.list_cron_jobs(profile="worker_alpha")
    assert [job["id"] for job in worker_jobs] == [worker_job["id"]]


@pytest.mark.asyncio
async def test_cron_profile_validation_errors(isolated_profiles):
    from hermes_cli import web_server

    with pytest.raises(HTTPException) as bad_name:
        await web_server.list_cron_jobs(profile="../bad")
    assert bad_name.value.status_code == 400

    with pytest.raises(HTTPException) as missing:
        await web_server.list_cron_jobs(profile="missing_profile")
    assert missing.value.status_code == 404
