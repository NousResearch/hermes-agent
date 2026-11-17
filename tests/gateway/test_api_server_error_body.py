"""API server errors carry retry/field/request-id fields; /v1/schema describes the surface."""
import socket

import pytest
import pytest_asyncio
from aiohttp import ClientSession, web

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

_KEY = "sk-test-strong-key-0123456789"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sse_route(adapter):
    async def handler(request):
        """Flushes its head itself (with ``_sse_headers``), as every SSE route does."""
        response = web.StreamResponse(headers=adapter._sse_headers(request))
        await response.prepare(request)
        await response.write(b"data: {}\n\n")
        return response
    return handler


@pytest_asyncio.fixture
async def served():
    port = _free_port()
    adapter = APIServerAdapter(PlatformConfig(
        enabled=True, extra={"host": "127.0.0.1", "port": port, "key": _KEY,
                             "cors_origins": "https://ui.example"}))
    adapter._handle_toolsets = _sse_route(adapter)
    assert await adapter.connect() is True
    try:
        async with ClientSession(base_url=f"http://127.0.0.1:{port}") as client:
            yield adapter, client
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_errors_keep_openai_envelope_and_add_retry_and_request_fields(served, monkeypatch):
    adapter, client = served
    async with client.get("/v1/models", headers={
            "Authorization": "Bearer wrong", "X-Request-Id": "trace-42"}) as resp:
        assert resp.status == 401
        assert resp.headers["X-Request-Id"] == "trace-42"
        assert (await resp.json())["error"] == {
            "message": "Invalid gateway API key (API_SERVER_KEY)", "type": "gateway_auth_error",
            "code": "gateway_auth_failed", "retryable": False, "retry_after_ms": None,
            "field": None, "request_id": "trace-42"}

    monkeypatch.setattr(adapter, "_gateway_is_draining", lambda: True)
    async with client.post("/v1/chat/completions", json={"messages": []},
                           headers={"Authorization": f"Bearer {_KEY}"}) as resp:
        error = (await resp.json())["error"]
        assert resp.status == 503 and error["code"] == "gateway_draining"
        assert error["retryable"] is True and error["retry_after_ms"] == 1000
        assert error["request_id"] == resp.headers["X-Request-Id"] and error["request_id"]

    # A stream's head is flushed inside its handler; it carries the id too, readable by a browser.
    async with client.get("/v1/toolsets", headers={
            "X-Request-Id": "trace-sse", "Origin": "https://ui.example"}) as resp:
        assert resp.headers["X-Request-Id"] == "trace-sse"
        assert "X-Request-Id" in resp.headers["Access-Control-Expose-Headers"]
        assert await resp.read() == b"data: {}\n\n"


@pytest.mark.asyncio
async def test_schema_lists_served_routes_and_error_shape_behind_auth(served):
    _adapter, client = served
    async with client.get("/v1/schema") as resp:
        assert resp.status == 401
    async with client.get("/v1/schema", headers={"Authorization": f"Bearer {_KEY}"}) as resp:
        assert resp.status == 200
        doc = await resp.json()
    routes = {(r["method"], r["path"]) for r in doc["routes"]}
    assert {("POST", "/v1/chat/completions"), ("GET", "/v1/schema"), ("POST", "/v1/runs")} <= routes
    error_props = doc["error"]["properties"]["error"]["properties"]
    assert {"message", "code", "retryable", "retry_after_ms", "field", "request_id"} <= set(error_props)
    assert doc["error_codes"]["gateway_draining"] == {"status": 503, "retryable": True}
    # Every listed route is really served: a listed GET answers something other than 404/405.
    async with client.get("/v1/capabilities", headers={"Authorization": f"Bearer {_KEY}"}) as resp:
        assert ("GET", "/v1/capabilities") in routes and resp.status == 200
