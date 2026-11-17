"""A route's ``identity`` source decides which deliveries are retries of one another."""

import json

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH
from unittest.mock import AsyncMock, MagicMock


def _route(identity):
    return {"secret": _INSECURE_NO_AUTH, "deliver": "telegram", "deliver_only": True, "identity": identity,
            "deliver_extra": {"chat_id": "fake-chat"}, "prompt": "alert {data.n}"}


def _gateway(sent: list, routes: dict) -> WebhookAdapter:
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0, "routes": routes}))
    target = AsyncMock()

    async def send(chat_id, content, **_kw):
        sent.append(content)
        return SendResult(success=True)

    target.send = send
    runner = MagicMock()
    runner.adapters = {Platform.TELEGRAM: target}
    runner._authorization_adapter = lambda platform, profile=None: runner.adapters.get(platform)
    adapter.gateway_runner = runner
    return adapter


async def _post(adapter, route, body: dict, **headers):
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    async with TestClient(TestServer(app)) as client:
        # X-Request-ID changes per attempt, like a provider's per-request id.
        response = await client.post(f"/webhooks/{route}", data=json.dumps(body),
                                     headers={"Content-Type": "application/json", **headers})
        return response.status, (await response.json()).get("status")


@pytest.mark.asyncio
async def test_identity_from_a_body_pointer_or_a_named_header_dedupes_retries():
    sent = []
    adapter = _gateway(sent, {"sentry": _route("body:/data/event/event_id"), "acme": _route("header:Acme-Event-Id")})
    event = {"data": {"n": 1, "event": {"event_id": "ev-1"}}}
    assert await _post(adapter, "sentry", event, **{"X-Request-ID": "attempt-1"}) == (200, "delivered")
    # The retry has a new per-request id and a re-serialized body; the event id in the body is the same.
    retry = {"data": {"event": {"event_id": "ev-1"}, "n": 1}}
    assert await _post(adapter, "sentry", retry, **{"X-Request-ID": "attempt-2"}) == (200, "duplicate")
    assert len(sent) == 1

    first = {"data": {"n": 2}}
    assert await _post(adapter, "acme", first, **{"Acme-Event-Id": "k-1"}) == (200, "delivered")
    assert await _post(adapter, "acme", first, **{"Acme-Event-Id": "k-1"}) == (200, "duplicate")
    # The sender reused its key for a different body: refused, nothing sent.
    assert await _post(adapter, "acme", {"data": {"n": 3}}, **{"Acme-Event-Id": "k-1"}) == (409, None)
    assert len(sent) == 2


def test_an_identity_that_cannot_dedupe_is_refused_at_startup():
    for bad in ("header:X-Webhook-Timestamp", "body:data/id", "cookie:x"):
        adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"host": "127.0.0.1", "port": 0,
                                                                     "routes": {"r": _route(bad)}}))
        with pytest.raises(ValueError, match="identity"):
            adapter._validate_route("r", _route(bad))
