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
    (home / "mcp_event_subscriptions.json").write_text(json.dumps({ROUTE: {"cursor": "c-1"}}))
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await _post(cli, {"type": "terminated", "error": {"code": -32024, "message": "Forbidden"}})
        assert resp.status == 200
    assert ROUTE not in _routes(home)
    assert ROUTE not in json.loads((home / "mcp_event_subscriptions.json").read_text())
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_gap_remembers_the_fresh_cursor(home):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await _post(cli, {"type": "gap", "cursor": "c-9"})
        assert resp.status == 200
    assert json.loads((home / "mcp_event_subscriptions.json").read_text())[ROUTE]["cursor"] == "c-9"
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_subscription_state_changes_do_not_strand_an_admitted_run(home):
    # The durable admission pins a digest of the route config, and the reply leg refuses a route
    # that changed since. The cursor and the grant change all the time, so they must live outside it.
    from gateway.platforms.webhook_delivery import retained_destination
    from tools import mcp_tool_events as events
    routes = _routes(home)
    routes[ROUTE].update(deliver="telegram", deliver_extra={"chat_id": "42"})
    (home / "webhook_subscriptions.json").write_text(json.dumps(routes))
    adapter = _adapter()
    occurrence = {"name": "feed.emission.created", "timestamp": "2026-10-16T00:00:00Z", "data": {"feed_id": "f-1"}}
    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await _post(cli, {**occurrence, "eventId": "e-1", "cursor": "c-1"})).status == 202
        chat_id = adapter.handle_message.await_args_list[0].args[0].source.chat_id
        destination = retained_destination(adapter, chat_id)
        assert (await _post(cli, {**occurrence, "eventId": "e-2", "cursor": "c-2"})).status == 202
        assert (await _post(cli, {"type": "gap", "cursor": "c-3"})).status == 200
        events._store_grant(ROUTE, {"id": "sub_1", "refreshBefore": "2026-10-17T00:00:00Z", "cursor": "c-4"},
                            "https://hermes.example.com/webhooks/" + ROUTE)
        assert retained_destination(adapter, chat_id) == destination
        retry = await _post(cli, {**occurrence, "eventId": "e-1", "cursor": "c-4"})  # watermark moved on
        assert retry.status == 200 and (await retry.json())["status"] == "duplicate"
    assert _routes(home)[ROUTE] == routes[ROUTE]


@pytest.mark.asyncio
async def test_an_unknown_control_envelope_is_acknowledged_without_a_run(home):
    # The sketch: any body with a top-level ``type`` is a control envelope, including types a later
    # revision adds. None of them may start an agent run.
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await _post(cli, {"type": "paused", "reason": "maintenance"})
        assert resp.status == 200 and (await resp.json())["status"] == "ignored"
        odd = await _post(cli, {"type": {"not": "a string"}})
        assert odd.status == 200
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_the_event_payload_reaches_the_prompt_whole(home):
    from tools.mcp_tool_events import _event_prompt
    routes = _routes(home)
    routes[ROUTE]["prompt"] = _event_prompt("ambush", "feed.emission.created", "Explain it.")
    (home / "webhook_subscriptions.json").write_text(json.dumps(routes))
    adapter = _adapter()
    text = "b" * 10_000
    occurrence = {"name": "feed.emission.created", "timestamp": "2026-10-16T00:00:00Z", "cursor": None}
    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await _post(cli, {**occurrence, "eventId": "e-big",
                                  "data": {"text": text, "note": "{eventId}"}})).status == 202
        assert (await _post(cli, {**occurrence, "eventId": "e-huge", "data": {"text": "h" * 50_000}})).status == 202
    big, huge = (call.args[0].text for call in adapter.handle_message.await_args_list)
    assert text in big and big.endswith("Explain it.")
    assert '"note": "{eventId}"' in big  # payload text is never read as a placeholder
    assert "truncated" in huge and len(huge) < 40_000
