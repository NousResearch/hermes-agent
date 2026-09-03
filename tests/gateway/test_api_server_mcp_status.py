"""Behavior tests for the authenticated MCP connection-status API."""

from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter(api_key: str = "sk-secret") -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": api_key}))


def test_mcp_server_status_route_is_registered():
    adapter = _make_adapter()

    routes = {(method, path) for method, path, _handler in adapter._http_route_table()}

    assert ("GET", "/v1/mcp/servers") in routes


@pytest.mark.asyncio
async def test_mcp_server_status_requires_bearer_auth():
    adapter = _make_adapter()
    app = web.Application()
    app.router.add_get("/v1/mcp/servers", adapter._handle_mcp_servers)

    async with TestClient(TestServer(app)) as client:
        response = await client.get("/v1/mcp/servers")

    assert response.status == 401


@pytest.mark.asyncio
async def test_mcp_server_status_returns_runtime_snapshot(monkeypatch):
    import tools.mcp_tool as mcp_tool

    adapter = _make_adapter()
    expected = [
        {"name": "alpha", "state": "connected", "transport": "stdio", "tool_count": 2},
        {"name": "beta", "state": "failed", "error_code": "connection_failed"},
    ]
    monkeypatch.setattr(mcp_tool, "get_mcp_connection_status", lambda: expected, raising=False)
    app = web.Application()
    app.router.add_get("/v1/mcp/servers", adapter._handle_mcp_servers)

    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            "/v1/mcp/servers",
            headers={"Authorization": "Bearer sk-secret"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload == {"object": "list", "data": expected}


@pytest.mark.asyncio
async def test_mcp_server_status_fails_closed_for_multiplexed_same_name_profiles(monkeypatch):
    """A process-global same-name runtime must not be joined to either profile."""
    import tools.mcp_tool as mcp_tool

    adapter = _make_adapter()
    adapter.gateway_runner = SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=True)
    )
    leaked = [{"name": "shared", "state": "connected", "tool_count": 7}]
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_connection_status",
        lambda: leaked,
        raising=False,
    )
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/v1/mcp/servers", adapter._handle_mcp_servers)
    app.router.add_get("/p/{profile}/v1/mcp/servers", adapter._handle_mcp_servers)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            adapter,
            "_resolve_request_profile",
            lambda request: request.match_info.get("profile") or None,
        )
        scoped.setattr(adapter, "_profile_scope", lambda profile: __import__("contextlib").nullcontext())
        scoped.setattr(adapter, "_expected_api_key", lambda: "sk-secret")
        async with TestClient(TestServer(app)) as client:
            default_response = await client.get(
                "/v1/mcp/servers",
                headers={"Authorization": "Bearer sk-secret"},
            )
            worker_response = await client.get(
                "/p/worker/v1/mcp/servers",
                headers={"Authorization": "Bearer sk-secret"},
            )
            default_payload = await default_response.json()
            worker_payload = await worker_response.json()

    assert default_response.status == 503
    assert worker_response.status == 503
    assert default_payload == worker_payload == {
        "error": {
            "message": "MCP runtime status is unavailable while profile multiplexing is enabled",
            "type": "service_unavailable_error",
            "param": None,
            "code": "mcp_status_profile_isolation_unavailable",
        }
    }
    assert "shared" not in str(default_payload)
    assert "shared" not in str(worker_payload)


@pytest.mark.asyncio
async def test_mcp_server_status_returns_stable_json_on_snapshot_failure(monkeypatch):
    import tools.mcp_tool as mcp_tool

    adapter = _make_adapter()
    monkeypatch.setattr(
        mcp_tool,
        "get_mcp_connection_status",
        lambda: (_ for _ in ()).throw(RuntimeError("Bearer raw-secret")),
        raising=False,
    )
    app = web.Application()
    app.router.add_get("/v1/mcp/servers", adapter._handle_mcp_servers)

    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            "/v1/mcp/servers",
            headers={"Authorization": "Bearer sk-secret"},
        )
        payload = await response.json()

    assert response.status == 500
    assert payload == {
        "error": {
            "message": "Unable to read MCP runtime status",
            "type": "server_error",
            "code": "mcp_status_unavailable",
        }
    }
    assert "raw-secret" not in str(payload)


@pytest.mark.asyncio
async def test_capabilities_advertises_mcp_server_status():
    adapter = _make_adapter()
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)

    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            "/v1/capabilities",
            headers={"Authorization": "Bearer sk-secret"},
        )
        payload = await response.json()

    assert response.status == 200
    assert payload["features"]["mcp_server_status"] is True
    assert payload["endpoints"]["mcp_servers"] == {
        "method": "GET",
        "path": "/v1/mcp/servers",
    }
