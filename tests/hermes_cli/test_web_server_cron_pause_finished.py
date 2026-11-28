"""Pausing a routine that has finished its runs, through the dashboard's cron routes."""

import pytest
from fastapi import HTTPException
import hermes_cli.web_routers.cron as _rt_cron
import hermes_cli.web_server_cron as _web_server_cron


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


@pytest.mark.asyncio
async def test_pausing_a_finished_job_is_refused_with_its_reason(isolated_profiles):
    from cron import jobs as cron_jobs

    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha",
        "create_job",
        prompt="ran its only run",
        schedule="every 1h",
        name="finished-job",
    )

    def finish(jobs, index, record):
        jobs[index] = {**record, "state": "completed", "enabled": False, "next_run_at": None}
        cron_jobs.save_jobs(jobs)

    _web_server_cron._call_cron_for_profile("worker_alpha", "_with_job", job["id"], finish)

    with pytest.raises(HTTPException) as exc:
        await _rt_cron.pause_cron_job(job["id"], profile="worker_alpha")

    assert exc.value.status_code == 409
    assert "no scheduled runs left" in exc.value.detail
