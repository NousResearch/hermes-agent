"""Tests for WhatsApp processing lifecycle reactions."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter(*, config_extra=None, running=True, http_session=None):
    """Create a WhatsAppAdapter with test attributes (bypass __init__)."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = config_extra or {}
    adapter._bridge_port = 3000
    adapter._running = running
    adapter._http_session = http_session if http_session is not None else MagicMock()
    adapter._bridge_process = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    return adapter


def _event(*, chat_id="chat@s.whatsapp.net", message_id="msg-123", sender_id="sender@s.whatsapp.net", from_me=False):
    from gateway.platforms.base import SessionSource

    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_name="Chat",
        chat_type="dm",
        user_id=sender_id,
        user_name="Sender",
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"senderId": sender_id, "fromMe": from_me},
        message_id=message_id,
    )


class TestWhatsAppReactionConfig:
    def test_reactions_enabled_by_default(self):
        adapter = _make_adapter()

        assert adapter._reactions_enabled() is True

    @pytest.mark.parametrize("value", [False, "false", "0", "no", "off"])
    def test_reactions_can_be_disabled_by_config(self, value):
        adapter = _make_adapter(config_extra={"reactions": value})

        assert adapter._reactions_enabled() is False


class TestWhatsAppLifecycleReactions:
    @pytest.mark.asyncio
    async def test_processing_start_reacts_with_eyes(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_start(_event())

        adapter._http_session.post.assert_called_once()
        url = adapter._http_session.post.call_args.args[0]
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert url == "http://127.0.0.1:3000/react"
        assert payload == {
            "chatId": "chat@s.whatsapp.net",
            "messageId": "msg-123",
            "emoji": "👀",
            "senderId": "sender@s.whatsapp.net",
            "fromMe": False,
        }

    @pytest.mark.asyncio
    async def test_processing_complete_reacts_with_success(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["emoji"] == "✅"

    @pytest.mark.asyncio
    async def test_processing_complete_reacts_with_failure(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_complete(_event(), ProcessingOutcome.FAILURE)

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["emoji"] == "❌"

    @pytest.mark.asyncio
    async def test_cancelled_completion_is_noop(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock()

        await adapter.on_processing_complete(_event(), ProcessingOutcome.CANCELLED)

        adapter._http_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_reactions_are_noop(self):
        adapter = _make_adapter(config_extra={"reactions": False})
        adapter._http_session.post = MagicMock()

        await adapter.on_processing_start(_event())
        await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)

        adapter._http_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_message_id_is_noop(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock()

        await adapter.on_processing_start(_event(message_id=None))

        adapter._http_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_bare_phone_chat_id_is_normalized_for_bridge(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_start(_event(chat_id="1234567890", sender_id="1234567890"))

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["chatId"] == "1234567890@s.whatsapp.net"
        assert payload["senderId"] == "1234567890@s.whatsapp.net"
