from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


class TestWhatsAppNativeFormatting:

    def test_invisible_unicode_prefixes_are_sanitized(self):
        adapter = _make_adapter()

        assert adapter.format_message("\u2060\u202ftext") == " text"


@pytest.mark.asyncio
async def test_send_location_posts_to_bridge_location_endpoint():
    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "loc-msg"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = await adapter.send_location(
        "15551234567",
        41.015,
        28.979,
        name="HQ",
        address="Example Street",
    )

    assert result.success
    assert result.message_id == "loc-msg"
    call = adapter._http_session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/send-location"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "latitude": 41.015,
        "longitude": 28.979,
        "name": "HQ",
        "address": "Example Street",
    }


@pytest.mark.asyncio
async def test_send_retry_does_not_repeat_completed_whatsapp_chunks(monkeypatch):
    adapter = _make_adapter()
    adapter.truncate_message = MagicMock(return_value=["chunk-1", "chunk-2"])
    monkeypatch.setattr("plugins.platforms.whatsapp.adapter.asyncio.sleep", AsyncMock())

    first = MagicMock(status=200)
    first.json = AsyncMock(return_value={"success": True, "messageId": "msg-1"})
    second = MagicMock(status=503)
    second.text = AsyncMock(return_value="Network is unreachable")
    adapter._http_session.post = MagicMock(side_effect=[_AsyncCM(first), _AsyncCM(second)])

    result = await adapter._send_with_retry(
        "15551234567",
        "long response",
        max_retries=2,
        base_delay=0,
    )

    assert not result.success
    assert result.partial_delivery
    assert result.raw_response == {
        "delivered_chunks": 1,
        "total_chunks": 2,
        "message_ids": ["msg-1"],
    }
    sent_chunks = [call.kwargs["json"]["message"] for call in adapter._http_session.post.call_args_list]
    assert sent_chunks == ["chunk-1", "chunk-2"]
