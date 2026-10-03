"""Threaded replies must exercise the real REST lookup, not replace it."""
import asyncio
import json

import httpx
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.bluebubbles import BlueBubblesAdapter


class Request:
    query = {"password": "secret"}
    headers = {}

    def __init__(self, **fields):
        self.payload = {"type": "new-message", "data": {
            "guid": "reply", "text": "send it",
            "handle": {"address": "peer@example.com"}, **fields,
        }}

    async def read(self):
        return json.dumps(self.payload).encode()


def adapter_for(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
    return BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://localhost:1234", "password": "secret",
        "send_read_receipts": False,
    }))


@pytest.mark.asyncio
@pytest.mark.parametrize("field, own", [
    ("isFromMe", True), ("fromMe", True), ("is_from_me", True),
    ("isFromMe", False),
])
async def test_threaded_reply_uses_real_rest_lookup(monkeypatch, field, own):
    adapter = adapter_for(monkeypatch)
    handled, requests = [], []

    async def handle(event):
        handled.append(event)

    def transport(request):
        requests.append(request)
        assert request.method == "GET"
        assert request.url.raw_path == b"/api/v1/message/origin%2Fwith%20space?password=secret"
        return httpx.Response(200, json={"data": {
            "text": "Send this, or revise?", field: own,
        }})

    monkeypatch.setattr(adapter, "handle_message", handle)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        adapter.client = client
        response = await adapter._handle_webhook(Request(
            threadOriginatorGuid="origin/with space", associatedMessageGuid="legacy"))
        await asyncio.sleep(0)
    assert response.status == 200
    assert len(requests) == len(handled) == 1
    event = handled[0]
    assert event.reply_to_message_id == "origin/with space"
    assert event.reply_to_text == "Send this, or revise?"
    assert event.reply_to_is_own_message is own


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["404", "timeout", "empty", "invalid-json"])
async def test_lookup_failure_still_acknowledges_and_dispatches(monkeypatch, failure):
    adapter = adapter_for(monkeypatch)
    handled, requests = [], []

    async def handle(event):
        handled.append(event)

    def transport(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("lookup timed out", request=request)
        if failure == "404":
            return httpx.Response(404)
        if failure == "invalid-json":
            return httpx.Response(200, text="not JSON")
        return httpx.Response(200, json={"data": None})

    monkeypatch.setattr(adapter, "handle_message", handle)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        adapter.client = client
        response = await adapter._handle_webhook(Request(threadOriginatorGuid="gone"))
        await asyncio.sleep(0)
    assert response.status == 200
    assert len(requests) == len(handled) == 1
    assert handled[0].reply_to_message_id == "gone"
    assert handled[0].reply_to_text is None
    assert handled[0].reply_to_is_own_message is False


@pytest.mark.asyncio
@pytest.mark.parametrize("fields, reply_id", [({}, None), ({"associatedMessageGuid": "legacy"}, "legacy")])
async def test_nonthreaded_message_never_looks_up_originator(monkeypatch, fields, reply_id):
    adapter = adapter_for(monkeypatch)
    handled, requests = [], []

    async def handle(event):
        handled.append(event)

    def transport(request):
        requests.append(request)
        return httpx.Response(500)

    monkeypatch.setattr(adapter, "handle_message", handle)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        adapter.client = client
        response = await adapter._handle_webhook(Request(**fields))
        await asyncio.sleep(0)
    assert response.status == 200
    assert requests == []
    assert len(handled) == 1
    assert handled[0].reply_to_message_id == reply_id
    assert handled[0].reply_to_text is None
