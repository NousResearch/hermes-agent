"""Dashboard ``GET /api/cron/store-health``: the ``hermes cron status`` write probe, per profile."""

import asyncio
import errno

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException

import hermes_cli.web_routers.cron as _rt_cron


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
    return {"default": default_home, "worker_alpha": worker_home}


def test_writable_store_reports_healthy(isolated_profiles):
    result = asyncio.run(_rt_cron.get_cron_store_health(profile="worker_alpha"))
    assert result == {"profile": "worker_alpha", "writable": True, "unwritable": None}


def test_probe_targets_the_selected_profile_store(isolated_profiles, monkeypatch):
    from cron import store_health

    probed = []

    def fake_probe(cron_dir):
        probed.append(cron_dir)
        return OSError(errno.ENOSPC, "No space left on device", str(cron_dir))

    monkeypatch.setattr(store_health, "probe_store", fake_probe)
    result = asyncio.run(_rt_cron.get_cron_store_health(profile="worker_alpha"))

    worker_cron = isolated_profiles["worker_alpha"] / "cron"
    assert probed == [worker_cron]
    assert result["writable"] is False
    assert result["unwritable"]["store"] == str(worker_cron)
    assert result["unwritable"]["error"].startswith("ENOSPC")


def test_unknown_profile_is_404(isolated_profiles):
    with pytest.raises(HTTPException) as exc:
        asyncio.run(_rt_cron.get_cron_store_health(profile="nope"))
    assert exc.value.status_code == 404
