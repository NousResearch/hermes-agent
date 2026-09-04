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
async def test_send_forwards_metadata_mentions_on_first_chunk_only():
    adapter = _make_adapter()
    first = MagicMock(status=200)
    first.json = AsyncMock(return_value={"success": True, "messageId": "msg-1"})
    second = MagicMock(status=200)
    second.json = AsyncMock(return_value={"success": True, "messageId": "msg-2"})
    adapter._http_session.post = MagicMock(side_effect=[_AsyncCM(first), _AsyncCM(second)])

    result = await adapter.send(
        "15551234567",
        "x" * (adapter.MAX_MESSAGE_LENGTH + 100),
        metadata={"mentions": ["15550001111", "15550002222@s.whatsapp.net"]},
    )

    assert result.success
    calls = adapter._http_session.post.call_args_list
    assert calls[0].kwargs["json"]["mentions"] == [
        "15550001111@s.whatsapp.net",
        "15550002222@s.whatsapp.net",
    ]
    assert "mentions" not in calls[1].kwargs["json"]


@pytest.mark.asyncio
async def test_send_without_mentions_omits_the_field():
    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "msg-1"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = await adapter.send("15551234567", "hi")

    assert result.success
    payload = adapter._http_session.post.call_args.kwargs["json"]
    assert "mentions" not in payload
