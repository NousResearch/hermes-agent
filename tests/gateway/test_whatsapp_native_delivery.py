from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
from tests.gateway.test_whatsapp_formatting import _AsyncCM, _make_adapter


class TestWhatsAppNativeFormatting:
    def test_single_asterisk_markdown_italic_uses_whatsapp_underscore(self):
        adapter = _make_adapter()

        assert adapter.format_message("this is *italic* text") == "this is _italic_ text"
        assert adapter.format_message("- * list bullet stays literal") == "- * list bullet stays literal"

    def test_invisible_unicode_prefixes_are_sanitized(self):
        adapter = _make_adapter()

        assert adapter.format_message("\u2060\u202ftext") == " text"


@pytest.mark.asyncio
async def test_send_poll_posts_to_bridge_poll_endpoint():
    adapter = _make_adapter()
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True, "messageId": "poll-msg"})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

    result = await adapter.send_poll(
        "15551234567",
        "Proceed?",
        ["Approve", "Deny"],
    )

    assert result.success
    assert result.message_id == "poll-msg"
    call = adapter._http_session.post.call_args
    assert call.args[0] == "http://127.0.0.1:3000/send-poll"
    assert call.kwargs["json"] == {
        "chatId": "15551234567@s.whatsapp.net",
        "question": "Proceed?",
        "options": ["Approve", "Deny"],
        "selectableCount": 1,
    }


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
async def test_send_tracks_text_chunk_message_ids_in_snake_case_raw_response():
    adapter = _make_adapter()
    first = MagicMock(status=200)
    first.json = AsyncMock(return_value={"success": True, "messageId": "msg-1"})
    second = MagicMock(status=200)
    second.json = AsyncMock(return_value={"success": True, "messageId": "msg-2"})
    adapter._http_session.post = MagicMock(side_effect=[_AsyncCM(first), _AsyncCM(second)])

    result = await adapter.send("15551234567", "x" * (adapter.MAX_MESSAGE_LENGTH + 100))

    assert result.success
    assert result.message_id == "msg-2"
    assert result.continuation_message_ids == ("msg-1",)
    assert result.raw_response["message_ids"] == ["msg-1", "msg-2"]
    assert "messageIds" not in result.raw_response


@pytest.mark.asyncio
async def test_whatsapp_reply_context_is_structured_not_prerendered():
    adapter = WhatsAppAdapter(
        PlatformConfig(
            enabled=True,
            extra={"session_name": "test", "dm_policy": "allowlist", "allow_from": ["*"]},
        )
    )

    event = await adapter._build_message_event(
        {
            "body": "what do you see here?",
            "chatId": "15551234567@s.whatsapp.net",
            "chatName": "Example Chat",
            "senderId": "15551234567@s.whatsapp.net",
            "senderName": "Example User",
            "isGroup": False,
            "hasQuotedMessage": True,
            "quotedText": "the gateway should not inject reply context twice",
            "quotedMessageId": "quoted-123",
        }
    )

    assert event is not None
    assert event.text == "what do you see here?"
    assert event.reply_to_message_id == "quoted-123"
    assert event.reply_to_text == "the gateway should not inject reply context twice"
    assert not event.text.startswith("[Replying to:")
