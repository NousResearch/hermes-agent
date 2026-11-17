"""Named, scoped API keys (``platforms.api_server.keys``) on the real route table."""

from datetime import datetime, timedelta, timezone, UTC

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware

FULL = "full-" + "f" * 40
READER = "reader-" + "r" * 40
EXPIRED = "expired-" + "e" * 40


def _adapter(monkeypatch) -> APIServerAdapter:
    monkeypatch.setenv("API_SERVER_KEY_READER", READER)
    monkeypatch.setenv("API_SERVER_KEY_OLD", EXPIRED)
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    return APIServerAdapter(PlatformConfig(enabled=True, extra={"key": FULL, "keys": [
        {"name": "reader", "key_env": "API_SERVER_KEY_READER", "scopes": ["sessions:read"]},
        {"name": "old", "key_env": "API_SERVER_KEY_OLD", "scopes": ["sessions:read", "runs"],
         "expires": past},
        # Malformed entries are skipped with a warning; they must not stop the listener.
        {"name": "bad", "key_env": "API_SERVER_KEY_READER", "scopes": [{"sessions": "read"}]},
        {"name": "literal", "key": "x" * 40, "scopes": ["runs"]}]}))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
    return app


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.asyncio
async def test_scoped_key_reaches_only_its_scopes_and_full_key_is_unchanged(monkeypatch):
    adapter = _adapter(monkeypatch)
    async with TestClient(TestServer(_app(adapter))) as cli:
        listed = await cli.get("/api/sessions", headers=_auth(READER))
        assert listed.status == 200, await listed.text()

        denied = await cli.post("/v1/runs", json={"input": "hi"}, headers=_auth(READER))
        assert denied.status == 403
        error = (await denied.json())["error"]
        assert (error["code"], error["required_scope"]) == ("insufficient_scope", "runs")

        # A route the scope table does not classify answers only the full key.
        detailed = await cli.get("/health/detailed", headers=_auth(READER))
        assert detailed.status == 403

        assert (await cli.get("/api/sessions", headers=_auth(FULL))).status == 200
        # The full key passes auth on run dispatch (body validation answers, not the gate).
        full_run = await cli.post("/v1/runs", json={}, headers=_auth(FULL))
        assert full_run.status == 400, await full_run.text()

        expired = await cli.get("/api/sessions", headers=_auth(EXPIRED))
        assert expired.status == 401
        assert (await expired.json())["error"]["code"] == "api_key_expired"

        unknown = await cli.get("/api/sessions", headers=_auth("nope-" + "n" * 40))
        assert unknown.status == 401
