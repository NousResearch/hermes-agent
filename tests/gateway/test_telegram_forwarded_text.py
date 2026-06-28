"""Tests for forwarded text message handling in TelegramAdapter.

When a user forwards a text message from another Telegram chat, the Bot API
may deliver it with ``forward_origin`` set but ``message.text`` as ``None``.
The content is accessible via ``update.effective_message.text``.  Without the
forwarded-message fallback in ``_handle_text_message``, these updates are
silently dropped — no error, no log, no response.

These tests verify the fallback paths:
1. Forwarded text with ``message.text=None`` but ``effective_message.text`` set
   is enqueued as a normal text event.
2. Forwarded media with a caption is routed to ``_handle_media_message``.
3. Forwarded updates with no text and no caption are dropped silently (no crash).
"""

import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


def _build_telegram_stubs():
    """Build minimal telegram module stubs for environments without python-telegram-bot installed."""
    telegram_mod = types.ModuleType("telegram")
    telegram_mod.Update = object
    telegram_mod.Bot = object
    telegram_mod.Message = object
    telegram_mod.InlineKeyboardButton = object
    telegram_mod.InlineKeyboardMarkup = object
    telegram_mod.LinkPreviewOptions = object

    telegram_ext_mod = types.ModuleType("telegram.ext")
    telegram_ext_mod.Application = object
    telegram_ext_mod.CommandHandler = object
    telegram_ext_mod.CallbackQueryHandler = object
    telegram_ext_mod.MessageHandler = object
    telegram_ext_mod.ContextTypes = SimpleNamespace(DEFAULT_TYPE=type(None))
    telegram_ext_mod.filters = SimpleNamespace()

    telegram_constants_mod = types.ModuleType("telegram.constants")
    telegram_constants_mod.ParseMode = SimpleNamespace(MARKDOWN_V2="MarkdownV2")
    telegram_constants_mod.ChatType = SimpleNamespace(
        GROUP="group",
        SUPERGROUP="supergroup",
        CHANNEL="channel",
        PRIVATE="private",
    )

    telegram_request_mod = types.ModuleType("telegram.request")
    telegram_request_mod.HTTPXRequest = object

    telegram_mod.ext = telegram_ext_mod
    telegram_mod.constants = telegram_constants_mod
    telegram_mod.request = telegram_request_mod

    return {
        "telegram": telegram_mod,
        "telegram.ext": telegram_ext_mod,
        "telegram.constants": telegram_constants_mod,
        "telegram.request": telegram_request_mod,
    }


@pytest.fixture
def telegram_adapter_cls(monkeypatch):
    """Import TelegramAdapter without leaking temporary telegram stubs."""
    module_name = "plugins.platforms.telegram.adapter"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        yield existing_module.TelegramAdapter
        return

    telegram_pkg = sys.modules.get("telegram")
    installed = isinstance(getattr(telegram_pkg, "__file__", None), str)
    if telegram_pkg is None:
        try:
            installed = importlib.util.find_spec("telegram") is not None
        except ValueError:
            installed = False

    if not installed:
        for name, module in _build_telegram_stubs().items():
            monkeypatch.setitem(sys.modules, name, module)

    module = importlib.import_module(module_name)
    try:
        yield module.TelegramAdapter
    finally:
        if not installed:
            sys.modules.pop(module_name, None)


def _make_adapter(telegram_adapter_cls):
    """Create a minimal adapter with auth bypassed for routing tests."""
    a = telegram_adapter_cls(PlatformConfig(enabled=True, token="***", extra={}))
    a._is_callback_user_authorized = lambda user_id, **_kw: True
    a._should_process_message = lambda msg, **_kw: True
    a._should_observe_unmentioned_group_message = lambda msg: False
    a._ensure_forum_commands = AsyncMock()
    a._clean_bot_trigger_text = lambda text: text
    a._apply_telegram_group_observe_attribution = lambda event: event
    a._cache_replied_media = AsyncMock()
    return a


def _make_forwarded_text_update(text="forwarded text content", *, msg_text=None):
    """Build an update simulating a forwarded text message.

    The key characteristic: ``message.text`` is None (so filters.TEXT doesn't
    match), but ``effective_message.text`` has the actual forwarded content.
    """
    from_user = SimpleNamespace(id=555, first_name="Chris", full_name="Chris", is_bot=False)
    chat = SimpleNamespace(id=12345, type="private", title=None, full_name=None, is_forum=False)
    forward_origin = SimpleNamespace(
        type=MessageOriginType.USER,
        sender_user=SimpleNamespace(id=999, first_name="OtherAgent", is_bot=True),
    )
    msg = SimpleNamespace(
        chat=chat,
        from_user=from_user,
        text=msg_text,  # None — this is the bug condition
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=42,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
        forward_origin=forward_origin,
    )
    # effective_message.text has the real content even when message.text is None
    eff_msg = SimpleNamespace(
        chat=chat,
        from_user=from_user,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=42,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
        forward_origin=forward_origin,
    )
    return SimpleNamespace(
        update_id=10001,
        message=msg,
        channel_post=None,
        effective_message=eff_msg,
    )


class MessageOriginType:
    """Subset of telegram.enums.MessageOriginType for stubbing."""
    USER = "user"


def _make_forwarded_caption_update(caption="check out this photo"):
    """Build an update simulating a forwarded photo with a caption."""
    from_user = SimpleNamespace(id=555, first_name="Chris", full_name="Chris", is_bot=False)
    chat = SimpleNamespace(id=12345, type="private", title=None, full_name=None, is_forum=False)
    forward_origin = SimpleNamespace(
        type=MessageOriginType.USER,
        sender_user=SimpleNamespace(id=999, first_name="OtherAgent", is_bot=True),
    )
    msg = SimpleNamespace(
        chat=chat,
        from_user=from_user,
        text=None,
        caption=caption,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=43,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
        forward_origin=forward_origin,
    )
    eff_msg = SimpleNamespace(
        chat=chat,
        from_user=from_user,
        text=None,
        caption=caption,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=43,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
        forward_origin=forward_origin,
    )
    return SimpleNamespace(
        update_id=10002,
        message=msg,
        channel_post=None,
        effective_message=eff_msg,
    )


def _make_empty_forwarded_update():
    """Build an update simulating a forwarded message with no text or caption."""
    from_user = SimpleNamespace(id=555, first_name="Chris", full_name="Chris", is_bot=False)
    chat = SimpleNamespace(id=12345, type="private", title=None, full_name=None, is_forum=False)
    forward_origin = SimpleNamespace(
        type=MessageOriginType.USER,
        sender_user=SimpleNamespace(id=999, first_name="OtherAgent", is_bot=True),
    )
    msg = SimpleNamespace(
        chat=chat,
        from_user=from_user,
        text=None,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=44,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
        forward_origin=forward_origin,
    )
    return SimpleNamespace(
        update_id=10003,
        message=msg,
        channel_post=None,
        effective_message=msg,  # effective_message also has no text
    )


class TestForwardedTextMessage:
    """Tests for the forwarded-text fallback in _handle_text_message."""

    @pytest.mark.asyncio
    async def test_forwarded_text_uses_effective_message(self, telegram_adapter_cls):
        """Forwarded text with message.text=None should use effective_message.text."""
        adapter = _make_adapter(telegram_adapter_cls)
        adapter._enqueue_text_event = MagicMock()
        update = _make_forwarded_text_update("forwarded text content")

        await adapter._handle_text_message(update, MagicMock())

        adapter._enqueue_text_event.assert_called_once()
        event = adapter._enqueue_text_event.call_args.args[0]
        assert event.text == "forwarded text content"
        assert event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_forwarded_text_preserves_chat_id(self, telegram_adapter_cls):
        """The forwarded message should keep the correct chat_id from the update."""
        adapter = _make_adapter(telegram_adapter_cls)
        adapter._enqueue_text_event = MagicMock()
        update = _make_forwarded_text_update("hello from another chat")

        await adapter._handle_text_message(update, MagicMock())

        event = adapter._enqueue_text_event.call_args.args[0]
        assert event.source.chat_id == "12345"

    @pytest.mark.asyncio
    async def test_forwarded_long_text_dispatched(self, telegram_adapter_cls):
        """A long forwarded message (like an SGLang config) should be enqueued."""
        adapter = _make_adapter(telegram_adapter_cls)
        adapter._enqueue_text_event = MagicMock()
        long_text = "model: glm-5.2\n" + "param: " + "x" * 1000
        update = _make_forwarded_text_update(long_text)

        await adapter._handle_text_message(update, MagicMock())

        adapter._enqueue_text_event.assert_called_once()
        event = adapter._enqueue_text_event.call_args.args[0]
        assert len(event.text) > 1000
        assert "glm-5.2" in event.text

    @pytest.mark.asyncio
    async def test_forwarded_media_with_caption_routes_to_media_handler(self, telegram_adapter_cls):
        """Forwarded media with caption should go to _handle_media_message, not text handler."""
        adapter = _make_adapter(telegram_adapter_cls)
        adapter._handle_media_message = AsyncMock()
        adapter._enqueue_text_event = MagicMock()
        update = _make_forwarded_caption_update(caption="check out this photo")

        await adapter._handle_text_message(update, MagicMock())

        adapter._handle_media_message.assert_awaited_once()
        adapter._enqueue_text_event.assert_not_called()

    @pytest.mark.asyncio
    async def test_forwarded_empty_message_dropped_silently(self, telegram_adapter_cls):
        """Forwarded message with no text and no caption should not crash."""
        adapter = _make_adapter(telegram_adapter_cls)
        adapter._enqueue_text_event = MagicMock()
        adapter._handle_media_message = AsyncMock()
        update = _make_empty_forwarded_update()

        await adapter._handle_text_message(update, MagicMock())

        adapter._enqueue_text_event.assert_not_called()
        adapter._handle_media_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_regular_text_still_works(self, telegram_adapter_cls):
        """Normal (non-forwarded) text messages should still be enqueued normally."""
        adapter = _make_adapter(telegram_adapter_cls)
        adapter._enqueue_text_event = MagicMock()
        from_user = SimpleNamespace(id=555, first_name="Chris", full_name="Chris", is_bot=False)
        chat = SimpleNamespace(id=12345, type="private", title=None, full_name=None, is_forum=False)
        msg = SimpleNamespace(
            chat=chat,
            from_user=from_user,
            text="hello world",
            caption=None,
            entities=[],
            caption_entities=[],
            message_thread_id=None,
            is_topic_message=False,
            message_id=50,
            reply_to_message=None,
            quote=None,
            date=None,
            forum_topic_created=None,
        )
        update = SimpleNamespace(
            update_id=10004,
            message=msg,
            channel_post=None,
            effective_message=msg,
        )

        await adapter._handle_text_message(update, MagicMock())

        adapter._enqueue_text_event.assert_called_once()
        event = adapter._enqueue_text_event.call_args.args[0]
        assert event.text == "hello world"
