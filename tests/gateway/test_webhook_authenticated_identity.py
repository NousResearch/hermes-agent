"""Authenticated webhook identity stays bound across headers and route changes."""

import base64
import hashlib
import hmac
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from multidict import CIMultiDict

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter


def _adapter(monkeypatch, route):
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {"sample": route}}))
    monkeypatch.setattr(adapter, "_reload_dynamic_routes", lambda: None)
    # These tests isolate the HTTP signature/replay contract, not parked-profile admission.
    monkeypatch.setattr("gateway.platforms.webhook_ingress.parked_profile_refusal", lambda *a: None)
    return adapter


def _request(body, headers, before_read=None):
    async def read():
        if before_read:
            before_read()
        return body
    return SimpleNamespace(headers=CIMultiDict(headers), content_length=len(body), read=read,
                           match_info={"route_name": "sample"}, method="POST")


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["svix", "webhook"])
async def test_signed_message_id_cannot_be_overridden_by_unsigned_header(monkeypatch, prefix):
    key = b"synthetic-webhook-key"
    body = b'{"type":"sample"}'
    route = {"secret": "whsec_" + base64.b64encode(key).decode(), "prompt": "marker", "deliver_only": True}
    adapter = _adapter(monkeypatch, route)
    delivered = AsyncMock()
    monkeypatch.setattr(adapter, "_direct_deliver", delivered)
    timestamp = str(int(time.time()))

    def headers(message_id, unsigned):
        signed = message_id.encode() + b"." + timestamp.encode() + b"." + body
        signature = base64.b64encode(hmac.new(key, signed, hashlib.sha256).digest()).decode()
        return {f"{prefix}-id": message_id, f"{prefix}-timestamp": timestamp,
                f"{prefix}-signature": "v1," + signature, "X-GitHub-Delivery": unsigned}

    first = await adapter._handle_webhook(_request(body, headers("signed-1", "unsigned-1")))
    retry = await adapter._handle_webhook(_request(body, headers("signed-1", "unsigned-2")))
    fresh = await adapter._handle_webhook(_request(body, headers("signed-2", "unsigned-3")))
    assert first.status == 200
    assert retry.status == 200
    assert fresh.status == 200
    assert delivered.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", [{"enabled": False}, {"prompt": "new prompt"}])
async def test_route_change_during_body_read_refuses_old_action(monkeypatch, replacement):
    body = b'{"type":"sample"}'
    route = {"secret": "synthetic-secret", "prompt": "old prompt", "deliver_only": True}
    adapter = _adapter(monkeypatch, route)
    delivered = AsyncMock()
    monkeypatch.setattr(adapter, "_direct_deliver", delivered)
    signature = hmac.new(b"synthetic-secret", body, hashlib.sha256).hexdigest()
    headers = {"X-Hub-Signature-256": "sha256=" + signature, "X-GitHub-Delivery": "delivery-1"}

    def change_route():
        adapter._routes = {"sample": {**route, **replacement}}

    response = await adapter._handle_webhook(_request(body, headers, change_route))
    assert response.status in {403, 409}
    delivered.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [False, True])
async def test_delayed_admission_keeps_its_authenticated_route_revision(monkeypatch, changed):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from tests.gateway.fixtures.webhook_route_authority import mount_authority

    route = {"secret": "synthetic-secret", "prompt": "original prompt", "deliver": "log"}
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {"sample": route}}))
    monkeypatch.setattr(adapter, "_reload_dynamic_routes", lambda: None)
    app = web.Application()
    mount_authority(app, adapter)
    async with TestClient(TestServer(app)):
        if changed:
            adapter._routes = {"sample": {**route, "prompt": "replacement prompt"}}
        outcome = await adapter._admit_agent_run(
            {}, "original prompt", "delayed-1", time.time(), route_config=route,
            route_name="sample", profile=None, event_type="sample")
        runner = adapter._message_handler.__self__
        rows = runner.session_authority.db._read_all('SELECT payload_json FROM session_admissions')
        if changed:
            assert outcome is None
            assert rows == []
        else:
            assert outcome == "accepted"
            assert len(rows) == 1
