"""R4-SEC-002 regression tests for legacy webhook freshness."""

import asyncio
import hashlib
import hmac
import json
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _make_adapter(routes: dict) -> WebhookAdapter:
    config = PlatformConfig(
        enabled=True,
        extra={"host": "0.0.0.0", "port": 0, "routes": routes},
    )
    return WebhookAdapter(config)


def _create_app(adapter: WebhookAdapter) -> web.Application:
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    return app


def _body_signature(body: bytes, secret: str) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _generic_v2_signature(body: bytes, secret: str, timestamp: str) -> str:
    signed_content = timestamp.encode() + b"." + body
    return hmac.new(secret.encode(), signed_content, hashlib.sha256).hexdigest()


def _github_signature(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


@pytest.mark.asyncio
async def test_generic_body_hmac_replay_with_fresh_request_id_cannot_dispatch_twice():
    """Caller-controlled request IDs cannot make body-only HMAC replayable."""
    secret = "generic-freshness-secret"
    adapter = _make_adapter({"generic": {"secret": secret}})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    body = json.dumps({"event_type": "ping", "value": 1}).encode()
    signature = _body_signature(body, secret)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        headers = {"X-Webhook-Signature": signature}
        first = await client.post(
            "/webhooks/generic",
            data=body,
            headers={**headers, "X-Request-ID": "caller-id-1"},
        )
        second = await client.post(
            "/webhooks/generic",
            data=body,
            headers={**headers, "X-Request-ID": "caller-id-2"},
        )

    await asyncio.sleep(0.05)
    assert first.status == 401
    assert second.status == 401
    assert dispatched == []


@pytest.mark.asyncio
async def test_generic_v2_replay_with_fresh_request_id_is_duplicate():
    """V2 replay identity comes from the signed timestamp and body."""
    secret = "generic-v2-replay-secret"
    adapter = _make_adapter({"generic": {"secret": secret}})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    body = json.dumps({"event_type": "ping", "value": 1}).encode()
    timestamp = str(int(time.time()))
    signature = _generic_v2_signature(body, secret, timestamp)
    signed_headers = {
        "X-Webhook-Signature-V2": signature,
        "X-Webhook-Timestamp": timestamp,
    }

    async with TestClient(TestServer(_create_app(adapter))) as client:
        first = await client.post(
            "/webhooks/generic",
            data=body,
            headers={**signed_headers, "X-Request-ID": "caller-one"},
        )
        second = await client.post(
            "/webhooks/generic",
            data=body,
            headers={**signed_headers, "X-Request-ID": "caller-two"},
        )
        duplicate = await second.json()

    await asyncio.sleep(0.05)
    assert first.status == 202
    assert second.status == 200
    assert duplicate["status"] == "duplicate"
    assert len(dispatched) == 1
    assert dispatched[0].source.chat_id.startswith("webhook:generic:generic-v2:")
    assert "caller-one" not in dispatched[0].source.chat_id


@pytest.mark.asyncio
async def test_github_replay_with_changed_delivery_id_is_duplicate():
    """GitHub replay identity comes from the HMAC-authenticated body."""
    secret = "github-replay-secret"
    adapter = _make_adapter({"github": {"secret": secret}})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    body = json.dumps({"event": "push", "ref": "refs/heads/main"}).encode()
    signature = _github_signature(body, secret)
    signed_headers = {"X-Hub-Signature-256": signature}

    async with TestClient(TestServer(_create_app(adapter))) as client:
        first = await client.post(
            "/webhooks/github",
            data=body,
            headers={**signed_headers, "X-GitHub-Delivery": "github-one"},
        )
        second = await client.post(
            "/webhooks/github",
            data=body,
            headers={**signed_headers, "X-GitHub-Delivery": "github-two"},
        )
        duplicate = await second.json()

    await asyncio.sleep(0.05)
    assert first.status == 202
    assert second.status == 200
    assert duplicate["status"] == "duplicate"
    assert len(dispatched) == 1
    assert dispatched[0].source.chat_id.startswith("webhook:github:github-body:")
    assert "github-one" not in dispatched[0].source.chat_id


@pytest.mark.asyncio
async def test_linear_body_hmac_without_freshness_is_rejected_by_default():
    """Linear's body-only compatibility signature cannot authorize dispatch."""
    secret = "linear-freshness-secret"
    adapter = _make_adapter({"linear": {"secret": secret}})
    dispatched = []

    async def capture(event):
        dispatched.append(event)

    adapter.handle_message = capture
    body = json.dumps({"type": "Issue", "id": "issue-1"}).encode()
    signature = _body_signature(body, secret)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post(
            "/webhooks/linear",
            data=body,
            headers={"linear-signature": signature},
        )

    await asyncio.sleep(0.05)
    assert response.status == 401
    assert dispatched == []
