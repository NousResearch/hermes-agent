"""Dashboard create/update carry the per-job ``wrap_response`` override strictly (#57647)."""

import pytest
from pydantic import ValidationError

import hermes_cli.web_models as _web_models
import hermes_cli.web_routers.cron as _rt_cron
import hermes_cli.web_server_cron as _web_server_cron


@pytest.fixture()
def isolated_profiles(tmp_path, monkeypatch):
    from hermes_cli import profiles

    default_home = tmp_path / ".hermes"
    worker_home = default_home / "profiles" / "worker_alpha"
    for home in (default_home, worker_home):
        (home / "cron").mkdir(parents=True, exist_ok=True)
        (home / "config.yaml").write_text("model: test-model\n", encoding="utf-8")
    monkeypatch.setattr(profiles, "_get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr(profiles, "_get_profiles_root", lambda: default_home / "profiles")
    return worker_home


@pytest.mark.asyncio
async def test_dashboard_update_sets_and_resets_wrap_response(isolated_profiles):
    job = _web_server_cron._call_cron_for_profile(
        "worker_alpha", "create_job", prompt="brief", schedule="every 1h", name="wrap")
    for value in (False, True, None):
        updated = await _rt_cron.update_cron_job(
            job["id"], _web_models.CronJobUpdate(updates={"wrap_response": value}), profile="worker_alpha")
        assert updated.get("wrap_response") is value


@pytest.mark.parametrize("bad", ["false", "on", 0, 1])
def test_dashboard_create_rejects_coercible_wrap_response(bad):
    with pytest.raises(ValidationError):
        _web_models.CronJobCreate(schedule="every 1h", prompt="brief", wrap_response=bad)
