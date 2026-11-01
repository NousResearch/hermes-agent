"""``/p/<profile>/api/plugins/...`` reaches the NAMED profile's own plugin routes.

The shared api_server listener wires ``register_platform_handler("api_server", ...)`` factories from
the launch profile's plugin manager only, so a named profile's plugin copy answered 404 ("Unknown or
unconfigured profile") under its prefix. Invariants (real plugin loader, real ``connect()``, real
prefix middleware and profile runtime scope):

* a named profile's plugin route answers from THAT profile's plugin instance (its ``profile_name``,
  its module), authenticated with that profile's ``API_SERVER_KEY`` (401 otherwise, including with
  the default key), and a session lookup reaches that profile's session store;
* the default profile's plain-path plugin routes are unchanged, and ``/p/default/api/plugins/...``
  does not grow a second copy;
* a named profile without the plugin never borrows the default's (404), an unknown prefix stays
  404, a runtime-parked profile stays 503;
* a plugin enabled after boot is wired on the next request (no frozen-router failure);
* a named profile's plugin app gets the lifecycle a root plugin app gets: ``on_startup`` before it
  serves, its own ``on_response_prepare``, and ``on_cleanup`` under the owning profile's scope at
  disconnect.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

PLUGIN = "probe-plugin"
DEFAULT_KEY = "default-api-server-key-0123456789abcdef"
WORKER_KEY = "worker-profile-api-key-0123456789abcdef"
BARE_KEY = "bare-profile-api-key-0123456789abcdef"

_PLUGIN_SRC = textwrap.dedent('''
    import json

    BASE = "/api/plugins/probe-plugin"

    def register(ctx):
        profile = ctx.profile_name

        def wire(native, adapter):
            from aiohttp import web

            async def status(request):
                denied = adapter._check_auth(request)
                if denied is not None:
                    return denied
                return web.json_response({"profile": profile, "module": __name__})

            async def context(request):
                denied = adapter._check_auth(request)
                if denied is not None:
                    return denied
                body = await request.json()
                return web.json_response({"profile": profile, "stored": body.get("k")}, status=202)

            async def register_control(request):
                denied = adapter._check_auth(request)
                if denied is not None:
                    return denied
                db = await adapter._ensure_session_db_async()
                session_id = (await request.json()).get("session_id")
                if db is None or not db.get_session(session_id):
                    return web.json_response({"error": "unknown session"}, status=403)
                return web.json_response({"profile": profile, "ticket": "t-" + session_id}, status=201)

            async def started(request):
                denied = adapter._check_auth(request)
                if denied is not None:
                    return denied
                return web.json_response({"profile": profile, "started": request.app.get("started", False)})

            async def hold(request):
                await request.app["gate"].wait()
                return web.json_response({"profile": profile, "held": True})

            async def on_startup(app):
                import asyncio
                app["started"] = True
                app["gate"] = asyncio.Event()

            async def on_prepare(request, response):
                from hermes_constants import get_hermes_home
                response.headers["X-Prepared-By"] = profile
                response.headers["X-Prepared-App-Started"] = str(request.app.get("started", False))
                response.headers["X-Prepared-Home"] = get_hermes_home().name
                response.headers["X-Prepared-Path"] = request.path

            async def on_cleanup(app):
                from hermes_constants import get_hermes_home
                with (get_hermes_home() / f"cleaned-{profile}").open("a") as fh:
                    fh.write("1")

            async def resource(app):
                from hermes_constants import get_hermes_home
                home = get_hermes_home()
                with (home / f"allocated-{profile}").open("a") as fh:
                    fh.write("1")
                yield
                with (home / f"released-{profile}").open("a") as fh:
                    fh.write("1")

            async def stall_startup(app):
                from hermes_constants import get_hermes_home
                if (get_hermes_home() / "stall-startup").exists():
                    import asyncio
                    await asyncio.Event().wait()
                if (get_hermes_home() / "fail-startup").exists():
                    raise RuntimeError("startup failed before the service existed")
                app["service"] = object()

            async def on_shutdown(app):
                app["service"]  # raises KeyError when startup failed before creating it

            native.cleanup_ctx.append(resource)
            native.on_startup.append(stall_startup)
            native.on_shutdown.append(on_shutdown)
            native.on_startup.append(on_startup)
            native.on_response_prepare.append(on_prepare)
            native.on_cleanup.append(on_cleanup)
            native.router.add_get(BASE + "/started", started)
            native.router.add_get(BASE + "/hold", hold)
            native.router.add_get(BASE + "/status", status)
            native.router.add_post(BASE + "/context", context)
            native.router.add_post(BASE + "/control/register", register_control)

        ctx.register_platform_handler("api_server", wire)
''')


def _write_home(home: Path, *, key: str | None, plugin: bool) -> None:
    home.mkdir(parents=True, exist_ok=True)
    enabled = f"plugins:\n  enabled: [{PLUGIN}]\n" if plugin else "{}\n"
    (home / "config.yaml").write_text(enabled, encoding="utf-8")
    if key:
        (home / ".env").write_text(f"API_SERVER_KEY={key}\n", encoding="utf-8")
    if plugin:
        d = home / "plugins" / PLUGIN
        d.mkdir(parents=True, exist_ok=True)
        (d / "plugin.yaml").write_text(f"name: {PLUGIN}\nversion: '0.1'\ndescription: t\n", encoding="utf-8")
        (d / "__init__.py").write_text(_PLUGIN_SRC, encoding="utf-8")


class _Sessions:
    """Session authorities keyed by home: ``get_session`` only knows ``<profile>-session``."""

    def __init__(self):
        self.asked: list[str] = []

    def for_home(self, home):
        name = Path(home).name if Path(home).parent.name == "profiles" else "default"
        asked = self.asked

        class _Db:
            @staticmethod
            def get_session(session_id):
                asked.append(name)
                return {"id": session_id} if session_id == f"{name}-session" else None

        return SimpleNamespace(db=_Db())


@pytest.fixture
def served(monkeypatch):
    root = Path(os.environ["HERMES_HOME"])
    _write_home(root, key=None, plugin=True)
    _write_home(root / "profiles" / "worker", key=WORKER_KEY, plugin=True)
    _write_home(root / "profiles" / "bare", key=BARE_KEY, plugin=False)
    from gateway.run import _profile_runtime_scope
    from hermes_cli import plugins
    plugins._reset_plugin_managers_for_tests()
    # Boot discovers the launch profile's plugins, then each served profile's under its own scope
    # (``_load_secondary_profile_config``).
    plugins.discover_plugins()
    for name in ("worker", "bare"):
        with _profile_runtime_scope(root / "profiles" / name):
            plugins.discover_plugins()
    return root


async def _connect(monkeypatch, runner_extra=None):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": DEFAULT_KEY, "host": "127.0.0.1", "port": 0}))
    sessions = _Sessions()
    adapter.gateway_runner = SimpleNamespace(**{
        "config": GatewayConfig(multiplex_profiles=True), "session_authorities": sessions,
        "_profile_adapters": {}, "session_runtime_descriptor": {}, **(runner_extra or {})})
    assert await adapter.connect()
    port = adapter._site._server.sockets[0].getsockname()[1]
    return adapter, f"http://127.0.0.1:{port}", sessions


def _auth(key):
    return {"Authorization": f"Bearer {key}"}


@pytest.mark.asyncio
async def test_named_profile_plugin_routes_answer_from_that_profiles_plugin(served, monkeypatch):
    adapter, base, sessions = await _connect(monkeypatch)
    status_path = f"/api/plugins/{PLUGIN}/status"
    try:
        async with aiohttp.ClientSession() as client:
            root = await client.get(base + status_path, headers=_auth(DEFAULT_KEY))
            assert root.status == 200
            root_body = await root.json()
            assert root_body["profile"] == "default"

            worker = await client.get(base + "/p/worker" + status_path, headers=_auth(WORKER_KEY))
            assert worker.status == 200
            worker_body = await worker.json()
            assert worker_body["profile"] == "worker"
            assert worker_body["module"] != root_body["module"]  # the worker's own plugin copy

            ctx = await client.post(base + f"/p/worker/api/plugins/{PLUGIN}/context?x=1",
                                    json={"k": "v"}, headers=_auth(WORKER_KEY))
            assert (ctx.status, await ctx.json()) == (202, {"profile": "worker", "stored": "v"})

            reg = await client.post(base + f"/p/worker/api/plugins/{PLUGIN}/control/register",
                                    json={"session_id": "worker-session"}, headers=_auth(WORKER_KEY))
            assert (reg.status, await reg.json()) == (201, {"profile": "worker", "ticket": "t-worker-session"})
            assert sessions.asked == ["worker"]  # the worker's session store, not the default's
            other = await client.post(base + f"/p/worker/api/plugins/{PLUGIN}/control/register",
                                      json={"session_id": "default-session"}, headers=_auth(WORKER_KEY))
            assert other.status == 403

            # Auth is the profile's own key: none, a wrong one, and the default's are all refused.
            for headers in ({}, _auth("wrong-key-0123456789abcdef"), _auth(DEFAULT_KEY)):
                denied = await client.get(base + "/p/worker" + status_path, headers=headers)
                assert denied.status == 401
            assert (await client.get(base + status_path, headers=_auth(WORKER_KEY))).status == 401

            # A named profile without the plugin never borrows the default's copy.
            bare = await client.get(base + "/p/bare" + status_path, headers=_auth(BARE_KEY))
            assert bare.status == 404
            unknown = await client.get(base + "/p/nobody" + status_path, headers=_auth(DEFAULT_KEY))
            assert unknown.status == 404
            # The default keeps its plain address only; /p/default/ does not grow a second copy.
            assert (await client.get(base + "/p/default" + status_path,
                                     headers=_auth(DEFAULT_KEY))).status == 404
            # Unrouted plugin path under a served profile: still the ingress 404.
            missing = await client.get(base + f"/p/worker/api/plugins/{PLUGIN}/nope", headers=_auth(WORKER_KEY))
            assert missing.status == 404
            # Native mirrors are untouched.
            caps = await client.get(base + "/p/worker/v1/capabilities", headers=_auth(WORKER_KEY))
            assert caps.status == 200
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_named_profile_plugin_app_lifecycle_matches_the_root_app(served, monkeypatch):
    adapter, base, _ = await _connect(monkeypatch)
    started = f"/api/plugins/{PLUGIN}/started"
    worker_home = served / "profiles" / "worker"
    try:
        async with aiohttp.ClientSession() as client:
            root = await client.get(base + started, headers=_auth(DEFAULT_KEY))
            assert (await root.json()) == {"profile": "default", "started": True}
            assert root.headers["X-Prepared-By"] == "default"
            worker = await client.get(base + "/p/worker" + started, headers=_auth(WORKER_KEY))
            assert (await worker.json()) == {"profile": "worker", "started": True}
            # The worker's prepare signal ran once, with its own app, home and route-relative path.
            assert worker.headers.getall("X-Prepared-By") == ["worker"]
            assert worker.headers["X-Prepared-App-Started"] == "True"
            assert worker.headers["X-Prepared-Home"] == "worker"
            assert worker.headers["X-Prepared-Path"] == started
            posted = await client.post(base + f"/p/worker/api/plugins/{PLUGIN}/context", json={"k": "e"},
                                       headers=_auth(WORKER_KEY), expect100=True)
            assert posted.status == 202
            assert not (worker_home / "cleaned-worker").exists()
    finally:
        await adapter.disconnect()
    assert (worker_home / "cleaned-worker").read_text() == "1"  # cleaned up in the worker's own home
    assert not (served / "cleaned-worker").exists()


@pytest.mark.asyncio
async def test_replaced_plugin_app_is_torn_down_after_its_inflight_requests(served, monkeypatch):
    """A re-discovery rebuilds the worker's app; the old one keeps serving the request it already
    holds and is cleaned up only once that request finishes, in the worker's own home."""
    import asyncio

    from gateway.run import _profile_runtime_scope
    from hermes_cli.plugins import discover_plugins
    from tests.waiting import async_wait_until

    adapter, base, _ = await _connect(monkeypatch)
    worker_home = served / "profiles" / "worker"
    marker = worker_home / "cleaned-worker"
    try:
        async with aiohttp.ClientSession() as client:
            held = asyncio.ensure_future(client.get(base + f"/p/worker/api/plugins/{PLUGIN}/hold",
                                                    headers=_auth(WORKER_KEY)))
            def _apps() -> dict:
                return getattr(adapter, "_profile_plugin_apps", None) or {}

            old = await async_wait_until(lambda: (e := _apps().get("worker")) and e.inflight and e,
                                         what="held request reached the worker app")
            with _profile_runtime_scope(worker_home):
                discover_plugins(force=True)
            fresh = await client.get(base + f"/p/worker/api/plugins/{PLUGIN}/started", headers=_auth(WORKER_KEY))
            assert (await fresh.json()) == {"profile": "worker", "started": True}
            assert _apps()["worker"] is not old
            assert not marker.exists()  # the old app still serves its held request
            old.app["gate"].set()
            resp = await held
            assert (await resp.json()) == {"profile": "worker", "held": True}
            # The retired app's own prepare signal still ran for the response it answered.
            assert resp.headers.getall("X-Prepared-By") == ["worker"]
            assert resp.headers["X-Prepared-Home"] == "worker"
            await async_wait_until(marker.exists, what="old app cleaned up after its last request")
            assert marker.read_text() == "1"
    finally:
        await adapter.disconnect()
    assert marker.read_text() == "11"  # the replacement app is cleaned up at disconnect too, once


@pytest.mark.asyncio
async def test_disconnect_stops_admission_before_tearing_plugin_apps_down(served, monkeypatch):
    """Once disconnect begins, no request builds a fresh app that would escape the cleanup pass."""
    from aiohttp import web

    from gateway.platforms import api_server_profile_plugins as profile_plugins
    from gateway.run import _profile_runtime_scope

    adapter, base, _ = await _connect(monkeypatch)
    async with aiohttp.ClientSession() as client:
        assert (await client.get(base + f"/p/worker/api/plugins/{PLUGIN}/status",
                                 headers=_auth(WORKER_KEY))).status == 200
    profile_plugins.begin_closing_profile_plugin_apps(adapter)
    adapter._profile_plugin_apps.clear()  # as if the cleanup pass had already taken the live app
    request = SimpleNamespace(rel_url=None)
    with _profile_runtime_scope(served / "profiles" / "worker"), pytest.raises(web.HTTPServiceUnavailable):
        await profile_plugins.dispatch_profile_plugin_route(adapter, "worker", f"api/plugins/{PLUGIN}/status",
                                                            request)
    assert adapter._profile_plugin_apps == {}
    await adapter.disconnect()


@pytest.mark.asyncio
async def test_cancelled_startup_releases_what_it_already_allocated(served, monkeypatch):
    """A build cancelled mid-startup (disconnect timing out the request) unwinds the resources its
    earlier startup steps allocated, owns nothing afterwards, and the cancellation propagates."""
    import asyncio

    from gateway.platforms import api_server_profile_plugins as profile_plugins
    from gateway.run import _profile_runtime_scope
    from tests.waiting import async_wait_until

    adapter, _, _ = await _connect(monkeypatch)
    worker_home = served / "profiles" / "worker"
    (worker_home / "stall-startup").write_text("", encoding="utf-8")
    try:
        with _profile_runtime_scope(worker_home):
            build = asyncio.ensure_future(profile_plugins._current_entry(adapter, "worker"))
        await async_wait_until((worker_home / "allocated-worker").exists, what="startup allocated")
        build.cancel()
        with pytest.raises(asyncio.CancelledError):
            await build
        assert (worker_home / "released-worker").read_text() == "1"
        assert getattr(adapter, "_profile_plugin_apps", {}) == {}
        assert not getattr(adapter, "_profile_plugin_detached", set())
    finally:
        await adapter.disconnect()
    assert (worker_home / "released-worker").read_text() == "1"  # released once, not again at disconnect


@pytest.mark.asyncio
async def test_failed_startup_releases_resources_even_when_shutdown_raises(served, monkeypatch):
    """Every request retries a failed build; each attempt must release what its startup allocated,
    even when a shutdown callback trips over state the failed startup never created."""
    adapter, base, _ = await _connect(monkeypatch)
    worker_home = served / "profiles" / "worker"
    (worker_home / "fail-startup").write_text("", encoding="utf-8")
    try:
        async with aiohttp.ClientSession() as client:
            for attempt in range(1, 4):
                resp = await client.get(base + f"/p/worker/api/plugins/{PLUGIN}/status", headers=_auth(WORKER_KEY))
                assert resp.status == 404  # no started app: the unchanged ingress answers
                assert (worker_home / "allocated-worker").read_text() == "1" * attempt
                assert (worker_home / "released-worker").read_text() == "1" * attempt
    finally:
        await adapter.disconnect()
    assert (worker_home / "released-worker").read_text() == "111"


@pytest.mark.asyncio
async def test_parked_profile_plugin_route_is_retryable_503(served, monkeypatch):
    adapter, base, _ = await _connect(
        monkeypatch, {"session_runtime_descriptor": {"parked_profiles": {"worker": "boot failed"}}})
    try:
        async with aiohttp.ClientSession() as client:
            resp = await client.get(base + f"/p/worker/api/plugins/{PLUGIN}/status", headers=_auth(WORKER_KEY))
            assert resp.status == 503
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_plugin_enabled_after_boot_is_wired_on_next_request(served, monkeypatch):
    adapter, base, _ = await _connect(monkeypatch)
    bare_home = served / "profiles" / "bare"
    path = f"/p/bare/api/plugins/{PLUGIN}/status"
    try:
        async with aiohttp.ClientSession() as client:
            assert (await client.get(base + path, headers=_auth(BARE_KEY))).status == 404
            _write_home(bare_home, key=BARE_KEY, plugin=True)
            from gateway.run import _profile_runtime_scope
            from hermes_cli.plugins import discover_plugins
            with _profile_runtime_scope(bare_home):
                discover_plugins(force=True)  # what `hermes plugins enable` + reload-plugins does
            resp = await client.get(base + path, headers=_auth(BARE_KEY))
            assert resp.status == 200
            assert (await resp.json())["profile"] == "bare"
    finally:
        await adapter.disconnect()
