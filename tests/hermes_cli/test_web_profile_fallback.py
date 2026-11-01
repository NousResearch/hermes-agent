"""Profile-list recovery must preserve the canonical live-profile inventory."""

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI

from hermes_cli import profiles as profiles_mod
from hermes_cli.web_routers import profiles
from hermes_cli.web_server_profiles import _hermes_home_scope
from hermes_constants import mark_named_profile_deleted


@pytest.mark.asyncio
async def test_profile_fallback_excludes_retired_and_non_profile_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    homes = {}
    for name in ("alpha", "beta", "retired", "default", ".staging"):
        home = root / "profiles" / name
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"model:\n  default: model-{name}\n  provider: custom\n", encoding="utf-8"
        )
        homes[name] = home
    (root / "profiles" / "ghost" / "cron").mkdir(parents=True)
    mark_named_profile_deleted(homes["retired"])
    expected_names = profiles_mod.list_profile_names()
    assert expected_names == ["default", "alpha", "beta"]

    def broken_listing(**_kwargs):
        raise OSError("profile metadata unavailable")

    monkeypatch.setattr(profiles_mod, "list_profiles", broken_listing)
    monkeypatch.setattr(profiles_mod, "_check_gateway_running", lambda _home: False)
    monkeypatch.setattr(profiles_mod, "_served_by_running_multiplexer", lambda _name: False)
    app = FastAPI()
    app.include_router(profiles.router)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        for name in ("alpha", "beta", "alpha"):
            with _hermes_home_scope(homes[name]):
                response = await client.get("/api/profiles")
            assert response.status_code == 200
            rows = response.json()["profiles"]
            assert [row["name"] for row in rows] == expected_names
            for row in rows[1:]:
                assert row["path"] == str(homes[row["name"]])
                assert row["model"] == f"model-{row['name']}"
    assert not (root / "profiles" / "ghost" / "config.yaml").exists()
