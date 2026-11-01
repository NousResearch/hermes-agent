"""Private API editing contracts through the real HTTP and stream paths."""

import json

import httpx
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.bluebubbles.adapter import BlueBubblesAdapter, MAX_TEXT_LENGTH
from gateway.stream_consumer import GatewayStreamConsumer


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://bluebubbles.invalid")
    monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "test-password")
    result = BlueBubblesAdapter(PlatformConfig(enabled=True, extra={
        "server_url": "http://bluebubbles.invalid", "password": "test-password",
    }))
    result._private_api_enabled = True
    result._helper_connected = True
    result.SUPPORTS_MESSAGE_EDITING = True
    return result


def _client(adapter, requests, *, fail_tail=None):
    tails = 0

    def handle(request):
        nonlocal tails
        payload = json.loads(request.content)
        requests.append((request, payload))
        if request.url.path.endswith("/edit"):
            return httpx.Response(200, json={"data": {"guid": "edited-guid"}})
        assert request.url.path == "/api/v1/message/text"
        tails += 1
        if tails == fail_tail:
            return httpx.Response(503, json={"message": "tail unavailable"})
        return httpx.Response(200, json={"data": {"guid": f"tail-{tails}"}})

    adapter.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return adapter.client


@pytest.mark.asyncio
async def test_private_api_edit_request_and_budget(adapter):
    requests = []
    async with _client(adapter, requests):
        for _ in range(adapter._EDIT_BUDGET - 1):
            result = await adapter.edit_message("iMessage;-;test", "edited-guid", "Hello")
            assert result.success
        preview = await adapter.edit_message("iMessage;-;test", "edited-guid", "No slot")
        assert preview.success
        assert len(requests) == adapter._EDIT_BUDGET - 1
        final = await adapter.edit_message("iMessage;-;test", "edited-guid", "Final", finalize=True)
        assert final.success
        assert len(requests) == adapter._EDIT_BUDGET
        request, payload = requests[-1]
        assert request.method == "POST"
        assert request.url.path == "/api/v1/message/edited-guid/edit"
        assert request.url.params["password"] == "test-password"
        assert payload == {"editedMessage": "Final", "backwardsCompatibilityMessage": "Final", "partIndex": 0}


@pytest.mark.asyncio
async def test_edit_encodes_message_guid(adapter):
    requests = []
    async with _client(adapter, requests):
        result = await adapter.edit_message("iMessage;-;test", "p:0/message?guid", "Hello")
    assert result.success
    assert requests[0][0].url.raw_path.startswith(b"/api/v1/message/p%3A0%2Fmessage%3Fguid/edit?")


@pytest.mark.asyncio
async def test_overflow_reports_every_real_bubble_and_consumer_adopts_tail(adapter):
    requests = []
    content = "first " * 800 + "\n\nsecond paragraph\n\nthird paragraph"
    async with _client(adapter, requests):
        result = await adapter.edit_message("iMessage;-;test", "original", content)
        tails = [payload["message"] for request, payload in requests if request.url.path.endswith("/text")]
        assert result.success
        assert result.continuation_message_ids == tuple(f"tail-{i + 1}" for i in range(len(tails)))
        assert result.message_id == result.continuation_message_ids[-1]
        delivered = requests[0][1]["editedMessage"] + "".join(tails)
        assert delivered == adapter.format_message(content)
        assert all(len(chunk) <= MAX_TEXT_LENGTH for chunk in tails)
        consumer = GatewayStreamConsumer(adapter, "iMessage;-;test")
        consumer._message_id = "original"
        assert await consumer._send_or_edit(content)
        assert consumer._message_id == f"tail-{len([r for r, _ in requests if r.url.path.endswith('/text')])}"
        assert consumer._last_sent_text == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_tail", [1, 2])
async def test_partial_overflow_preserves_exact_prefix_for_recovery(adapter, fail_tail):
    requests = []
    content = "first paragraph\n\n" + "word \n" * 1700
    async with _client(adapter, requests, fail_tail=fail_tail):
        consumer = GatewayStreamConsumer(adapter, "iMessage;-;test")
        consumer._message_id = "original"
        assert not await consumer._send_or_edit(content)
    assert consumer._fallback_final_send
    assert not consumer._edit_supported
    assert content.startswith(consumer._fallback_prefix)
    assert consumer._fallback_preserve_partial_messages
    successful_tails = [payload["message"] for request, payload in requests if request.url.path.endswith("/text")][:-1]
    assert consumer._fallback_prefix == requests[0][1]["editedMessage"] + "".join(successful_tails)
    assert consumer._message_id == (f"tail-{fail_tail - 1}" if fail_tail > 1 else "edited-guid")


@pytest.mark.asyncio
async def test_exhausted_final_budget_reports_fresh_message_for_adoption(adapter):
    requests = []
    adapter._edit_counts["original"] = adapter._EDIT_BUDGET
    async with _client(adapter, requests):
        consumer = GatewayStreamConsumer(adapter, "iMessage;-;test")
        consumer._message_id = "original"
        assert await consumer._send_or_edit("Final answer", finalize=True)
    assert len(requests) == 1
    assert requests[0][0].url.path == "/api/v1/message/text"
    assert consumer._message_id == "tail-1"


@pytest.mark.asyncio
async def test_edit_requires_helper_and_nonempty_text(adapter):
    requests = []
    async with _client(adapter, requests):
        assert not (await adapter.edit_message("chat", "original", "")).success
        adapter._helper_connected = False
        assert not (await adapter.edit_message("chat", "original", "Hello")).success
    assert not requests


@pytest.mark.asyncio
async def test_failed_edit_does_not_consume_budget(adapter):
    def handle(request):
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter.client = client
        result = await adapter.edit_message("chat", "original", "Hello")
    assert not result.success
    assert "original" not in adapter._edit_counts


@pytest.mark.asyncio
@pytest.mark.parametrize("private_api,helper_connected", [(False, False), (True, False), (False, True), (True, True)])
async def test_connect_advertises_editing_only_with_private_api_helper(adapter, monkeypatch, private_api, helper_connected):
    from unittest.mock import AsyncMock, MagicMock
    from aiohttp import web

    adapter._api_get = AsyncMock(side_effect=[{}, {"data": {
        "private_api": private_api, "helper_connected": helper_connected,
    }}])
    adapter._register_webhook = AsyncMock()
    runner = MagicMock()
    runner.setup = AsyncMock()
    site = MagicMock()
    site.start = AsyncMock()
    monkeypatch.setattr(web, "AppRunner", MagicMock(return_value=runner))
    monkeypatch.setattr(web, "TCPSite", MagicMock(return_value=site))
    assert await adapter.connect()
    try:
        assert adapter.SUPPORTS_MESSAGE_EDITING is (private_api and helper_connected)
    finally:
        await adapter.client.aclose()


@pytest.mark.asyncio
async def test_same_text_finalization_uses_reserved_edit_after_preview_noop(adapter):
    requests = []
    adapter._edit_counts["original"] = adapter._EDIT_BUDGET - 1
    async with _client(adapter, requests):
        consumer = GatewayStreamConsumer(adapter, "iMessage;-;test")
        consumer._message_id = "original"
        assert await consumer._send_or_edit("Complete final answer")
        assert not requests
        assert await consumer._send_or_edit("Complete final answer", finalize=True)
    assert len(requests) == 1
    assert requests[0][0].url.path == "/api/v1/message/original/edit"
    assert requests[0][1]["editedMessage"] == "Complete final answer"
