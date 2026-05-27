"""
Tests for the /api/ws WebSocket endpoint in APIServerAdapter.

Tests cover:
- Auth rejection (wrong token, missing token)
- Auth success (correct token via ?token= query param)
- No-auth localhost (key not configured, host=127.0.0.1)
- No-auth network defense-in-depth (key not configured, host=0.0.0.0)
- gateway.ready handshake on connection
- JSON-RPC message round-trip
- Graceful disconnect / cleanup
"""

import asyncio
import json
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    TUI_GATEWAY_AVAILABLE,
    cors_middleware,
    security_headers_middleware,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", host: str = "127.0.0.1") -> APIServerAdapter:
    """Create an adapter with optional API key and host."""
    extra: dict = {"host": host}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
    return APIServerAdapter(config)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    """Create the aiohttp app with WS route registered (mirrors connect())."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    # Only add WS route if tui_gateway is available (as in production)
    if TUI_GATEWAY_AVAILABLE:
        app.router.add_get("/api/ws", adapter._handle_tui_ws)
    return app


# ---------------------------------------------------------------------------
# WebSocket Auth
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TUI_GATEWAY_AVAILABLE, reason="tui_gateway not installed")
class TestWsAuth:
    @pytest.mark.asyncio
    async def test_wrong_token_returns_403(self):
        """Connecting with an invalid ?token= must return HTTP 403 before WS upgrade."""
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.session.get(
                client.make_url("/api/ws?token=wrong-token"),
                headers={"Upgrade": "websocket", "Connection": "Upgrade"},
            )
            assert resp.status == 403
            text = await resp.text()
            assert "Invalid API key" in text
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_missing_token_returns_403(self):
        """Connecting without ?token= when a key is configured must return 403."""
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.session.get(
                client.make_url("/api/ws"),
                headers={"Upgrade": "websocket", "Connection": "Upgrade"},
            )
            assert resp.status == 403
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_correct_token_accepts_connection(self):
        """Connecting with the correct ?token= must allow the WS upgrade."""
        adapter = _make_adapter(api_key="sk-secret")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/ws?token=sk-secret")
            # Receive gateway.ready
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["method"] == "event"
            assert payload["params"]["type"] == "gateway.ready"
            await ws.close()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_localhost_no_key_allows_connection(self):
        """On localhost with no key configured, the WS should be accepted."""
        adapter = _make_adapter(api_key="", host="127.0.0.1")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/ws")
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["params"]["type"] == "gateway.ready"
            await ws.close()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_network_no_key_returns_403(self):
        """Defense-in-depth: on 0.0.0.0 with no key, the WS endpoint must refuse."""
        adapter = _make_adapter(api_key="", host="0.0.0.0")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.session.get(
                client.make_url("/api/ws"),
                headers={"Upgrade": "websocket", "Connection": "Upgrade"},
            )
            assert resp.status == 403
            text = await resp.text()
            assert "API key required" in text
        finally:
            await client.close()


# ---------------------------------------------------------------------------
# WebSocket Protocol
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not TUI_GATEWAY_AVAILABLE, reason="tui_gateway not installed")
class TestWsProtocol:
    @pytest.mark.asyncio
    async def test_gateway_ready_event_received(self):
        """First message after connect must be gateway.ready with valid skin data."""
        adapter = _make_adapter(api_key="sk-test")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/ws?token=sk-test")
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["jsonrpc"] == "2.0"
            assert payload["method"] == "event"
            assert payload["params"]["type"] == "gateway.ready"
            # resolve_skin() may return None in isolated tests, or a dict when
            # tui_gateway has loaded a default skin — accept either.
            skin = payload["params"]["payload"]["skin"]
            assert skin is None or isinstance(skin, dict)
            await ws.close()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_json_rpc_parse_error(self):
        """Sending invalid JSON must produce a parse error response."""
        adapter = _make_adapter(api_key="sk-test")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/ws?token=sk-test")
            # Consume gateway.ready
            await ws.receive()
            # Send garbage
            await ws.send_str("not json at all")
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["jsonrpc"] == "2.0"
            assert payload["error"]["code"] == -32700
            assert "parse error" in payload["error"]["message"]
            await ws.close()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_json_rpc_dispatch_unknown_method(self):
        """Sending a valid JSON-RPC for an unknown method must return -32601."""
        adapter = _make_adapter(api_key="sk-test")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/ws?token=sk-test")
            await ws.receive()  # gateway.ready
            req = {"jsonrpc": "2.0", "method": "nonexistent.method", "id": 42}
            await ws.send_str(json.dumps(req))
            msg = await ws.receive()
            assert msg.type == web.WSMsgType.TEXT
            resp = json.loads(msg.data)
            assert resp["id"] == 42
            assert resp["error"]["code"] == -32601
            assert "unknown method" in resp["error"]["message"]
            await ws.close()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_graceful_disconnect(self):
        """Client-initiated close must not raise unhandled exceptions."""
        adapter = _make_adapter(api_key="sk-test")
        app = _create_app(adapter)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            ws = await client.ws_connect("/api/ws?token=sk-test")
            await ws.receive()  # gateway.ready
            await ws.close()
            # Brief pause to let server-side cleanup run
            await asyncio.sleep(0.1)
            assert True  # If we got here, no exception leaked
        finally:
            await client.close()
