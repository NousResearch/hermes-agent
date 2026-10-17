"""Webhook deliveries on MCP event routes: the signed verification challenge is echoed without an
agent run, control envelopes update or drop the route, and an occurrence is deduplicated by its
``eventId`` rather than the per-attempt ``webhook-id``."""

import base64
import hashlib
import hmac
import json
import time
import uuid
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter

SECRET = "whsec_" + base64.b64encode(b"k" * 32).decode()
ROUTE = "mcp-ambush-0123456789ab"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    route = {"secret": SECRET, "prompt": "Event {eventId}: {data.feed_id}", "deliver": "log", "profile": "default",
             "mcp_event": {"server": "ambush", "name": "feed.emission.created", "arguments": {}}}
    (tmp_path / "webhook_subscriptions.json").write_text(json.dumps({ROUTE: route}))
    return tmp_path


def _routes(home):
    return json.loads((home / "webhook_subscriptions.json").read_text())


def _headers(body: bytes, secret: str = SECRET) -> dict:
    msg_id, timestamp = f"msg_{uuid.uuid4().hex}", str(int(time.time()))
    key = base64.b64decode(secret.removeprefix("whsec_"))
    digest = hmac.new(key, f"{msg_id}.{timestamp}.".encode() + body, hashlib.sha256).digest()
    return {"webhook-id": msg_id, "webhook-timestamp": timestamp,
            "webhook-signature": "v1," + base64.b64encode(digest).decode(),
            "X-MCP-Subscription-Id": "sub_1", "Content-Type": "application/json"}


def _app(adapter):
    app = web.Application()
    from tests.gateway.fixtures.webhook_route_authority import mount_authority
    mount_authority(app, adapter)
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


async def _post(cli, payload, secret=SECRET):
    body = json.dumps(payload).encode()
    return await cli.post(f"/webhooks/{ROUTE}", data=body, headers=_headers(body, secret))


def _adapter():
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0}))
    adapter.handle_message = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_verification_challenge_is_echoed_without_a_run(home):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await _post(cli, {"type": "verification", "challenge": "nonce-123"})
        assert resp.status == 200 and await resp.json() == {"challenge": "nonce-123"}
        forged = await _post(cli, {"type": "verification", "challenge": "nonce-123"},
                             secret="whsec_" + base64.b64encode(b"x" * 32).decode())
        assert forged.status == 401
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_occurrence_runs_once_per_event_id(home):
    adapter = _adapter()
    occurrence = {"eventId": "e-1", "name": "feed.emission.created", "timestamp": "2026-10-16T00:00:00Z",
                  "data": {"feed_id": "f-1"}, "cursor": None}
    async with TestClient(TestServer(_app(adapter))) as cli:
        first = await _post(cli, occurrence)
        retry = await _post(cli, occurrence)  # same event, new webhook-id
        assert first.status == 202
        assert retry.status == 200 and (await retry.json())["status"] == "duplicate"
    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "Event e-1: f-1"


@pytest.mark.asyncio
async def test_terminated_drops_the_route(home):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await _post(cli, {"type": "terminated", "error": {"code": -32024, "message": "Forbidden"}})
        assert resp.status == 200
    assert ROUTE not in _routes(home)
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_gap_remembers_the_fresh_cursor(home):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await _post(cli, {"type": "gap", "cursor": "c-9"})
        assert resp.status == 200
    assert _routes(home)[ROUTE]["mcp_event"]["cursor"] == "c-9"
    adapter.handle_message.assert_not_called()
