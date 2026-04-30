"""HTTP endpoint tests for LineAdapter — POST /line/webhook + HMAC validation."""
import json

import pytest
from aiohttp import web

from gateway.platforms.line import LineAdapter
from tests.gateway.conftest import line_sign as _sign, make_line_platform_config


@pytest.fixture
def line_adapter(monkeypatch):
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
    monkeypatch.setenv("LINE_CHANNEL_SECRET", "s")
    monkeypatch.setenv("LINE_ALLOWED_USERS", "U1")
    return LineAdapter(make_line_platform_config(token="t"))


@pytest.fixture
def app(line_adapter):
    app = web.Application()
    line_adapter.register_routes(app)
    return app


@pytest.mark.asyncio
async def test_post_with_valid_signature_returns_200(aiohttp_client, app):
    client = await aiohttp_client(app)
    body = json.dumps({"events": [], "destination": "U1"}).encode()
    sig = _sign("s", body)
    resp = await client.post("/line/webhook", data=body, headers={"X-Line-Signature": sig})
    assert resp.status == 200


@pytest.mark.asyncio
async def test_post_with_invalid_signature_returns_401(aiohttp_client, app):
    client = await aiohttp_client(app)
    body = json.dumps({"events": []}).encode()
    resp = await client.post("/line/webhook", data=body, headers={"X-Line-Signature": "bad"})
    assert resp.status == 401


@pytest.mark.asyncio
async def test_post_with_missing_signature_returns_401(aiohttp_client, app):
    client = await aiohttp_client(app)
    body = json.dumps({"events": []}).encode()
    resp = await client.post("/line/webhook", data=body)
    assert resp.status == 401


@pytest.mark.asyncio
async def test_get_returns_405(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/line/webhook")
    assert resp.status == 405


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(aiohttp_client, app):
    client = await aiohttp_client(app)
    resp = await client.get("/line/webhook/health")
    assert resp.status == 200
    data = await resp.json()
    assert data == {"status": "ok", "platform": "line"}


@pytest.mark.asyncio
async def test_duplicate_webhook_event_id_is_dropped(aiohttp_client, line_adapter, app):
    """Second delivery of the same webhookEventId must be silently deduplicated."""
    client = await aiohttp_client(app)
    body = json.dumps({
        "events": [{
            "type": "message",
            "webhookEventId": "wh-evt-001",
            "replyToken": "rt",
            "source": {"type": "user", "userId": "U1"},
            "timestamp": 1,
            "message": {"id": "m1", "type": "text", "text": "hello"},
        }],
        "destination": "U1",
    }).encode()
    sig = _sign("s", body)
    headers = {"X-Line-Signature": sig}

    dispatched: list[dict] = []
    original = line_adapter._dispatch_event

    async def _counting_dispatch(event):
        dispatched.append(event)
        await original(event)

    line_adapter._dispatch_event = _counting_dispatch

    await client.post("/line/webhook", data=body, headers=headers)
    await client.post("/line/webhook", data=body, headers=headers)

    assert len(dispatched) == 1, "duplicate webhookEventId must be dropped on second delivery"


@pytest.mark.asyncio
async def test_concurrent_delivery_of_same_event_id_dispatches_once(
    aiohttp_client, line_adapter, app
):
    """Concurrent re-delivery of the same webhookEventId must still dedupe.
    Documents the at-most-once contract under cooperative single-loop scheduling —
    if dedup ever moves out of the synchronous webhook path, this test will catch it."""
    import asyncio

    client = await aiohttp_client(app)
    body = json.dumps({
        "events": [{
            "type": "message",
            "webhookEventId": "wh-evt-concurrent",
            "replyToken": "rt",
            "source": {"type": "user", "userId": "U1"},
            "timestamp": 1,
            "message": {"id": "m1", "type": "text", "text": "hello"},
        }],
        "destination": "U1",
    }).encode()
    sig = _sign("s", body)
    headers = {"X-Line-Signature": sig}

    dispatched: list[dict] = []
    original = line_adapter._dispatch_event

    async def _counting_dispatch(event):
        dispatched.append(event)
        await original(event)

    line_adapter._dispatch_event = _counting_dispatch

    await asyncio.gather(
        client.post("/line/webhook", data=body, headers=headers),
        client.post("/line/webhook", data=body, headers=headers),
    )

    assert len(dispatched) == 1, (
        "concurrent webhookEventId deliveries must dedupe to a single dispatch"
    )
