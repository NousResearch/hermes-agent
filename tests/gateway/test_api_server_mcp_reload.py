"""``POST /v1/mcp/reload`` — the ``/reload-mcp`` slash command over REST.

Contract under test: the handler is API-key gated, runs the gateway's existing
``_execute_mcp_reload`` under a synthetic internal event whose ``source.profile`` is the
``/p/<profile>/`` address under multiplex (None otherwise), and maps the reload's own text
outcome to ``ok`` + HTTP status without inventing a second reload path.
"""

from __future__ import annotations

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.i18n import t
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import _CAPABILITY_ENDPOINTS, APIServerAdapter

API_KEY = "opensslrandhex32strongkey00000000"


class _Runner:
    def __init__(self, *, multiplex: bool, summary: str):
        self.config = GatewayConfig(multiplex_profiles=multiplex)
        self.summary = summary
        self.events = []

    async def _execute_mcp_reload(self, event):
        self.events.append(event)
        return self.summary


def _ok_summary() -> str:
    return t("gateway.reload_mcp.header") + "\n" + t("gateway.reload_mcp.reconnected", names="alpha")


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f"/p/{{profile}}{path}", handler)
    return app


@pytest.fixture()
def adapter(monkeypatch):
    # Per-server state after the reload comes from discovery's cached status; stub it so the
    # tests never touch a real MCP registry.
    monkeypatch.setattr("tools.mcp_tool_discovery.get_mcp_status", lambda: [
        {"name": "alpha", "transport": "http", "tools": 3, "connected": True, "disabled": False,
         "status": "connected"}])
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": API_KEY}))


def test_reload_is_advertised_and_routed(adapter):
    assert dict(_CAPABILITY_ENDPOINTS)["mcp_reload"] == ("POST", "/v1/mcp/reload")
    assert ("POST", "/v1/mcp/reload", adapter._handle_mcp_reload) in adapter._http_route_table()


@pytest.mark.asyncio
async def test_reload_requires_api_key(adapter):
    adapter.gateway_runner = _Runner(multiplex=False, summary=_ok_summary())
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", headers={"Authorization": "Bearer wrong"})
        assert resp.status == 401
    assert adapter.gateway_runner.events == []


@pytest.mark.asyncio
async def test_reload_runs_gateway_reload_as_internal_event(adapter):
    runner = _Runner(multiplex=False, summary=_ok_summary())
    adapter.gateway_runner = runner
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 200
        body = await resp.json()
    assert body["ok"] is True and body["profile"] == "default"
    assert body["summary"] == runner.summary
    assert body["servers"] == [{"name": "alpha", "status": "connected", "connected": True, "tools": 3}]
    (event,) = runner.events
    assert event.internal is True and event.text == "/reload-mcp"
    assert event.source.platform is Platform.API_SERVER
    assert event.source.profile is None  # single-profile gateway: no scoping


@pytest.mark.asyncio
async def test_reload_under_multiplex_scopes_to_the_addressed_profile(adapter, tmp_path, monkeypatch):
    """Real prefix middleware + real profile secret scope (A→B): the addressed profile's own
    ``API_SERVER_KEY`` authorizes, and ``_execute_mcp_reload`` receives that profile on the event."""
    from agent import secret_scope as ss

    worker_home = tmp_path / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    worker_key = "w" * 32
    (worker_home / ".env").write_text(f"API_SERVER_KEY={worker_key}\n", encoding="utf-8")
    runner = _Runner(multiplex=True, summary=_ok_summary())
    adapter.gateway_runner = runner
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex, profile_allowlist=None: [("default", tmp_path), ("worker", worker_home)])
    monkeypatch.setattr(
        "hermes_cli.profiles.get_profile_dir",
        lambda name: tmp_path if name == "default" else worker_home)
    ss.set_multiplex_active(True)
    try:
        async with TestClient(TestServer(_app(adapter))) as cli:
            # The default key does not open another profile's reload.
            assert (await cli.post("/p/worker/v1/mcp/reload",
                                   headers={"Authorization": f"Bearer {API_KEY}"})).status == 401
            resp = await cli.post("/p/worker/v1/mcp/reload",
                                  headers={"Authorization": f"Bearer {worker_key}"})
            assert resp.status == 200
            assert (await resp.json())["profile"] == "worker"
            # An unknown profile is refused by the prefix middleware before any reload runs.
            assert (await cli.post("/p/nobody/v1/mcp/reload",
                                   headers={"Authorization": f"Bearer {worker_key}"})).status == 404
    finally:
        ss.set_multiplex_active(False)
    (event,) = runner.events
    assert event.source.profile == "worker"


@pytest.mark.asyncio
async def test_reload_failure_text_maps_to_500_without_raising(adapter):
    runner = _Runner(multiplex=False, summary=t("gateway.reload_mcp.failed", error="boom"))
    adapter.gateway_runner = runner
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 500
        body = await resp.json()
    assert body["ok"] is False and "boom" in body["summary"]


@pytest.mark.asyncio
async def test_reload_with_a_failed_server_is_not_ok(adapter, monkeypatch):
    """Discovery swallows a per-server connect failure into a warning, so the reload text alone
    reads as success; the per-server state must fail the call."""
    monkeypatch.setattr("tools.mcp_tool_discovery.get_mcp_status", lambda: [
        {"name": "alpha", "transport": "http", "tools": 0, "connected": False, "disabled": False,
         "status": "failed", "error": "401 Unauthorized"}])
    adapter.gateway_runner = _Runner(multiplex=False, summary=_ok_summary())
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 500
        body = await resp.json()
    assert body["ok"] is False
    assert body["servers"][0]["error"] == "401 Unauthorized"


@pytest.mark.asyncio
async def test_reload_without_runner_is_503(adapter):
    adapter.gateway_runner = None
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/mcp/reload", headers={"Authorization": f"Bearer {API_KEY}"})
        assert resp.status == 503

