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
async def test_cron_update_preserves_storage_profile_and_exposes_run_profile(isolated_profiles):
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="default storage worker runtime",
        schedule="every 1h",
        name="profile-edit-job",
    )

    updated = await web_server.update_cron_job(
        job["id"],
        web_server.CronJobUpdate(updates={"profile": "worker_alpha"}),
        profile="default",
    )

    assert updated["profile"] == "default"
    assert updated["profile_name"] == "default"
    assert updated["run_profile"] == "worker_alpha"


@pytest.mark.asyncio
async def test_cron_create_accepts_dashboard_config_fields(isolated_profiles):
    from hermes_cli import web_server

    created = await web_server.create_cron_job(
        web_server.CronJobCreate(
            prompt="collect the latest status",
            schedule="every 1h",
            name="dashboard-full-config",
            deliver="local",
            repeat=3,
            skills=["research", "ops"],
            script="status_report.py",
            no_agent=False,
            workdir=str(isolated_profiles["default"]),
            profile="worker_alpha",
            model="gpt-test",
            provider="openai-codex",
            base_url="http://example.test",
            context_from=["seed-job"],
            enabled_toolsets=["terminal", "web"],
        ),
        profile="default",
    )

    assert created["profile"] == "default"
    assert created["profile_name"] == "default"
    assert created["run_profile"] == "worker_alpha"
    assert created["name"] == "dashboard-full-config"
    assert created["repeat"]["times"] == 3
    assert created["skills"] == ["research", "ops"]
    assert created["script"] == "status_report.py"
    assert created["workdir"] == str(isolated_profiles["default"])
    assert created["model"] == "gpt-test"
    assert created["provider"] == "openai-codex"
    assert created["base_url"] == "http://example.test"
    assert created["context_from"] == ["seed-job"]
    assert created["enabled_toolsets"] == ["terminal", "web"]


@pytest.mark.asyncio
async def test_cron_update_rejects_non_config_fields(isolated_profiles):
    from hermes_cli import web_server

    job = web_server._call_cron_for_profile(
        "default",
        "create_job",
        prompt="state should not be dashboard-editable",
        schedule="every 1h",
        name="state-guard",
    )

    with pytest.raises(HTTPException) as exc:
        await web_server.update_cron_job(
            job["id"],
            web_server.CronJobUpdate(updates={"state": "paused"}),
            profile="default",
        )

    assert exc.value.status_code == 400
    assert "Unsupported cron update field" in exc.value.detail


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
