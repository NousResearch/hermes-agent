"""A JSON body that is not an object is a client error on every POST route, never a 500."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

KEY = "sk-test-request-body-0123456789"


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "path", "body"), [
    ("POST", "/v1/chat/completions", "[1]"),
    ("POST", "/v1/chat/completions", '{"messages": ["hi"]}'),
    ("POST", "/v1/responses", '"text"'),
    ("POST", "/v1/runs", "[1]"),
    ("POST", "/api/jobs", "[1]"),
    ("PATCH", "/api/jobs/abc123abc123", "[1]"),
])
async def test_non_object_body_is_a_400(method, path, body):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": KEY}))
    app = web.Application()
    for route_method, route_path, handler in adapter._http_route_table():
        app.router.add_route(route_method, route_path, handler)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.request(method, path, data=body, headers={
            "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        assert resp.status == 400, await resp.text()
        assert "error" in await resp.json()
