"""Live dashboard API rescans keep working routes and the launch profile's ownership."""
import json
import os
import py_compile
from pathlib import Path

import httpx
import pytest
import pytest_asyncio

from hermes_cli import web_server
from hermes_cli.web_server_profiles import _config_profile_scope


def _plugin(home, name, source):
    directory = home / "plugins" / name / "dashboard"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps({"name": name, "api": "plugin_api.py"}))
    api = directory / "plugin_api.py"
    api.write_text(source)
    return api


def _api(value, endpoint="old"):
    return (
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        f"@router.get('/{endpoint}')\n"
        "def read():\n"
        f"    return {value!r}\n"
    )


@pytest_asyncio.fixture(params=["default", "alpha"])
async def live_dashboard(tmp_path, monkeypatch, request):
    root_home = tmp_path / ".hermes"
    home = root_home if request.param == "default" else root_home / "profiles" / request.param
    home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text("plugins:\n  enabled: [live, live-extra]\n")
    monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    monkeypatch.setattr(web_server.app.state, "plugin_api_mounts", {}, raising=False)
    # A real catch-all after the existing endpoints reproduces a built dashboard,
    # including in environments where no web_dist is installed.
    original = list(web_server.app.router.routes)
    schema = web_server.app.openapi_schema
    @web_server.app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        from fastapi.responses import JSONResponse
        return JSONResponse({"spa": path}, status_code=404)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_server.app), base_url="http://localhost",
        headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
    ) as client:
        yield home, client
    web_server.app.router.routes[:] = original
    web_server.app.openapi_schema = schema


@pytest.mark.asyncio
async def test_rescan_adds_and_replaces_routes_without_losing_working_generation(live_dashboard):
    home, client = live_dashboard
    rescan = "/api/dashboard/plugins/rescan"
    assert (await client.get(rescan)).status_code == 200
    api = _plugin(home, "live", _api("first"))
    _plugin(home, "live-extra", _api("neighbor"))
    assert (await client.get("/api/plugins/live/old")).status_code == 404
    assert (await client.get(rescan)).status_code == 200
    assert (await client.get("/api/plugins/live/old")).json() == "first"
    web_server.app.openapi()  # Remount must invalidate an already-generated schema.
    py_compile.compile(str(api), doraise=True)
    stamp = api.stat()
    api.write_text(_api("other", "new"))  # Same length and mtime: don't reuse a stale .pyc.
    os.utime(api, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert (await client.get(rescan)).status_code == 200
    assert (await client.get("/api/plugins/live/new")).json() == "other"
    assert (await client.get("/api/plugins/live/old")).status_code == 404
    assert (await client.get("/api/plugins/live-extra/old")).json() == "neighbor"
    assert "/api/plugins/live/new" in web_server.app.openapi()["paths"]
    assert "/api/plugins/live/old" not in web_server.app.openapi()["paths"]
    api.write_text("raise RuntimeError('broken update')\n")
    assert (await client.get(rescan)).status_code == 200
    assert (await client.get("/api/plugins/live/new")).json() == "other"
    api.write_text(_api("fixed", "new"))
    assert (await client.get(rescan)).status_code == 200
    assert (await client.get("/api/plugins/live/new")).json() == "fixed"
    (api.parent / "manifest.json").unlink()
    assert (await client.get(rescan)).status_code == 200
    assert (await client.get("/api/plugins/live/new")).status_code == 404
    assert (await client.get("/api/plugins/live-extra/old")).json() == "neighbor"


@pytest.mark.asyncio
async def test_rescan_keeps_launch_owned_imports_scoped_under_multiplex(live_dashboard):
    home, client = live_dashboard
    secondary = Path.home() / ".hermes" / "profiles" / "beta"
    secondary.mkdir(parents=True)
    (secondary / "config.yaml").write_text("plugins:\n  disabled: [live]\n")
    (home / ".env").write_text("PLUGIN_PROBE=launch\n")
    (secondary / ".env").write_text("PLUGIN_PROBE=secondary\n")
    from tui_gateway.launch_profile_policy import activate_multi_profile_hosting
    activate_multi_profile_hosting()
    source = (
        "from fastapi import APIRouter\n"
        "from hermes_constants import get_hermes_home\n"
        "from agent.secret_scope import get_secret\n"
        "owner = [str(get_hermes_home()), get_secret('PLUGIN_PROBE')]\n"
        "router = APIRouter()\n"
        "@router.get('/owner')\n"
        "def read():\n"
        "    return owner\n"
    )
    api = _plugin(home, "live", source)
    _plugin(secondary, "live", "raise AssertionError('secondary code must not be imported')\n")
    # A -> B -> A: the dashboard namespace is launch-owned, not the selected chat profile.
    for profile in (None, "beta", None):
        api.write_text(api.read_text() + "\n")
        with _config_profile_scope(profile):
            assert (await client.get("/api/dashboard/plugins/rescan")).status_code == 200
        assert (await client.get("/api/plugins/live/owner")).json() == [str(home), "launch"]
    # Removing launch consent must unload, even if a secondary enables the same name.
    (home / "config.yaml").write_text("plugins:\n  disabled: [live]\n")
    (secondary / "config.yaml").write_text("plugins:\n  enabled: [live]\n")
    with _config_profile_scope("beta"):
        assert (await client.get("/api/dashboard/plugins/rescan")).status_code == 200
    assert (await client.get("/api/plugins/live/owner")).status_code == 404
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=web_server.app), base_url="http://localhost") as anonymous:
        assert (await anonymous.get("/api/dashboard/plugins/rescan")).status_code == 401
