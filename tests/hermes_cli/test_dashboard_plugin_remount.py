"""Live dashboard API rescans keep working routes and the launch profile's ownership."""
import asyncio
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter

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
    from hermes_cli import plugins
    from starlette.datastructures import State

    monkeypatch.setattr(plugins, "get_bundled_plugins_dir", lambda: tmp_path / "bundled")
    monkeypatch.setattr(web_server, "_dashboard_plugins_cache", None)
    monkeypatch.setattr(web_server.app, "state", State(dict(web_server.app.state._state)))
    web_server.app.state.plugin_api_mounts = {}
    # Keep production routes and middleware, but isolate route publication and
    # lifecycle callbacks from other tests and the server's background services.
    monkeypatch.setattr(web_server.app, "router", APIRouter(
        routes=[route for route in web_server.app.router.routes
                if not getattr(route, "path", "").startswith("/api/plugins/")],
        dependency_overrides_provider=web_server.app,
    ))
    monkeypatch.setattr(web_server.app, "openapi_schema", None)
    monkeypatch.setattr(web_server.app, "middleware_stack", None)
    module_names = ("hermes_dashboard_plugin_live", "hermes_dashboard_plugin_live-extra")
    for name in module_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    # A catch-all reproduces a built dashboard even without web_dist installed.
    @web_server.app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        from fastapi.responses import JSONResponse
        return JSONResponse({"spa": path}, status_code=404)
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=web_server.app), base_url="http://localhost",
            headers={web_server._SESSION_HEADER_NAME: web_server._SESSION_TOKEN},
        ) as client:
            yield home, client
    finally:
        for name in module_names:
            sys.modules.pop(name, None)


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


def _boot_mount():
    from hermes_cli.web_server_dashboard import _mount_plugin_api_routes

    # Production's first mount happens before mount_spa. Preserve that ordering
    # when exercising the boot path on both base and candidate implementations.
    router = web_server.app.router
    fallbacks = [route for route in router.routes
                 if getattr(route, "path", "") in ("/{path:path}", "/{full_path:path}")]
    router.routes = [route for route in router.routes if route not in fallbacks]
    del web_server.app.state.plugin_api_mounts
    try:
        _mount_plugin_api_routes()
    finally:
        router.routes = [*router.routes, *fallbacks]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["enabled: [{bad: value}]", "disabled: [[live]]"])
async def test_malformed_policy_fails_closed_at_boot_and_rescan(live_dashboard, policy, caplog):
    home, client = live_dashboard
    api = _plugin(home, "live", _api("working"))
    config = home / "config.yaml"
    malformed = f"plugins:\n  {policy}\n"
    config.write_text(malformed)
    # A fresh interpreter exercises the unconditional web_server import-time
    # mount, not just reconciliation on an already-imported application.
    boot = subprocess.run(
        [sys.executable, "-c", (
            "import sys\n"
            "from hermes_cli.web_server import app\n"
            "assert any(getattr(r, 'path', '') == '/api/health' for r in app.routes)\n"
            "assert 'hermes_dashboard_plugin_live' not in sys.modules\n"
        )],
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert boot.returncode == 0, boot.stderr
    _boot_mount()
    assert "plugin activation policy" in caplog.text
    assert "hermes_dashboard_plugin_live" not in sys.modules
    assert (await client.get("/api/health")).status_code == 200
    assert (await client.get("/api/plugins/live/old")).status_code == 404

    config.write_text("plugins:\n  enabled: [live]\n")
    assert (await client.get("/api/dashboard/plugins/rescan")).status_code == 200
    assert (await client.get("/api/plugins/live/old")).json() == "working"
    config.write_text(malformed)
    api.write_text("raise AssertionError('unknown consent must not execute code')\n")
    response = await client.get("/api/dashboard/plugins/rescan")
    assert response.status_code == 200
    assert response.json()["policy_error"] is True
    assert response.json()["ok"] is False
    assert "hermes_dashboard_plugin_live" not in sys.modules
    assert "/api/plugins/live/old" not in web_server.app.openapi()["paths"]
    assert (await client.get("/api/health")).status_code == 200
    assert (await client.get("/api/plugins/live/old")).status_code == 404
    # Repairing policy alone must not resurrect retired routes before a rescan.
    config.write_text("plugins:\n  enabled: [live]\n")
    assert (await client.get("/api/plugins/live/old")).status_code == 404


def _lifecycle_api(kind):
    common = (
        "from contextlib import asynccontextmanager\n"
        "from fastapi import APIRouter\n"
        "from pathlib import Path\n"
        "marker = Path(__file__).with_suffix('.events')\n"
        "def record(event):\n"
        "    with marker.open('a') as stream:\n"
        "        stream.write(event + '\\n')\n"
        "record('import')\n"
        "client = None\n"
        "async def start():\n"
        "    global client\n"
        "    client = {'generation': 'boot'}\n"
        "    record('start')\n"
        "async def stop():\n"
        "    global client\n"
        "    client = None\n"
        "    record('stop')\n"
    )
    hooks = {
        "events": "router = APIRouter(on_startup=[start], on_shutdown=[stop])\n",
        "shutdown": "client = {'generation': 'boot'}\nrouter = APIRouter(on_shutdown=[stop])\n",
        "lifespan": (
            "@asynccontextmanager\n"
            "async def lifespan(app):\n"
            "    await start()\n"
            "    try:\n"
            "        yield\n"
            "    finally:\n"
            "        await stop()\n"
            "router = APIRouter(lifespan=lifespan)\n"
        ),
    }
    return common + hooks[kind] + (
        "@router.get('/resource')\n"
        "def read():\n"
        "    return client['generation']\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["events", "lifespan", "shutdown"])
async def test_lifecycle_generation_is_initialized_once_and_retained(live_dashboard, kind):
    home, client = live_dashboard
    source = _lifecycle_api(kind)
    api = _plugin(home, "live", source)
    events = api.with_suffix(".events")
    _boot_mount()
    async with web_server.app.router.lifespan_context(web_server.app):
        assert (await client.get("/api/plugins/live/resource")).json() == "boot"
        started = events.read_text()
        assert started.splitlines() == (["import"] if kind == "shutdown" else ["import", "start"])
        # Unchanged rescans neither reimport nor rerun startup.
        unchanged = await client.get("/api/dashboard/plugins/rescan")
        assert unchanged.status_code == 200
        assert events.read_text() == started
        # A handler-only edit is unsafe too: new globals would not be initialized.
        api.write_text(source.replace("return client['generation']", "return client['generation'] + '-edited'"))
        response = await client.get("/api/dashboard/plugins/rescan")
        assert (await client.get("/api/plugins/live/resource")).json() == "boot"
        assert events.read_text() == started
        assert unchanged.json()["restart_required"] == []
        assert response.json()["restart_required"] == ["live"]
        # Removing routes does not stop lifecycle services early.
        (api.parent / "manifest.json").unlink()
        await client.get("/api/dashboard/plugins/rescan")
        assert (await client.get("/api/plugins/live/resource")).status_code == 404
        assert events.read_text() == started
    assert events.read_text() == started + "stop\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["events", "lifespan", "shutdown"])
async def test_live_lifecycle_install_or_addition_requires_restart(live_dashboard, kind):
    home, client = live_dashboard
    api = _plugin(home, "live", _api("old", "resource"))
    await client.get("/api/dashboard/plugins/rescan")
    assert (await client.get("/api/plugins/live/resource")).json() == "old"
    api.write_text(_lifecycle_api(kind))
    late = _plugin(home, "live-extra", _lifecycle_api(kind))
    response = await client.get("/api/dashboard/plugins/rescan")
    assert set(response.json()["restart_required"]) == {"live", "live-extra"}
    assert (await client.get("/api/plugins/live/resource")).json() == "old"
    assert (await client.get("/api/plugins/live-extra/resource")).status_code == 404
    assert "hermes_dashboard_plugin_live-extra" not in sys.modules
    # Inspecting the new router requires import, but must not start any services.
    assert api.with_suffix(".events").read_text().splitlines() == ["import"]
    assert late.with_suffix(".events").read_text().splitlines() == ["import"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replace", "remove"])
async def test_in_flight_request_keeps_its_generation_during_rescan(live_dashboard, change):
    home, client = live_dashboard
    api = _plugin(home, "live", (
        "import asyncio\n"
        "from fastapi import APIRouter\n"
        "router = APIRouter()\n"
        "entered, release = asyncio.Event(), asyncio.Event()\n"
        "generation = 'old'\n"
        "@router.get('/held')\n"
        "async def held(wait: bool = False):\n"
        "    if wait:\n"
        "        entered.set()\n"
        "        await release.wait()\n"
        "    return generation\n"
    ))
    await client.get("/api/dashboard/plugins/rescan")
    assert (await client.get("/api/plugins/live/held")).json() == "old"
    module = sys.modules["hermes_dashboard_plugin_live"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=web_server.app), base_url="http://localhost",
    ) as anonymous:
        assert (await anonymous.get("/api/plugins/live/held")).status_code == 401
        pending = asyncio.create_task(client.get("/api/plugins/live/held?wait=true"))
        try:
            await asyncio.wait_for(module.entered.wait(), timeout=10)
            if change == "replace":
                api.write_text(_api("new", "held"))
            else:
                (api.parent / "manifest.json").unlink()
            response = await asyncio.wait_for(client.get("/api/dashboard/plugins/rescan"), timeout=10)
            assert response.status_code == 200
            following = await asyncio.wait_for(client.get("/api/plugins/live/held"), timeout=10)
            if change == "replace":
                assert following.json() == "new"
                assert (await anonymous.get("/api/plugins/live/held")).status_code == 401
            else:
                assert following.status_code == 404
            assert not pending.done()  # The old request is still waiting, not serial.
        finally:
            module.release.set()
            completed = await asyncio.wait_for(pending, timeout=10)
        assert completed.status_code == 200
        assert completed.json() == "old"
