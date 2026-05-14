"""Regression tests for Telegram channel_post updates.

Telegram channel broadcasts are delivered as ``Update.channel_post`` rather than
``Update.message``.  The adapter should use ``effective_message`` so channel
posts are converted into Hermes gateway events instead of being silently
ignored.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return

    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"

    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


def _make_adapter():
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_channel_message(text="channel id test @hermes_bot"):
    chat = SimpleNamespace(
        id=-1003950368353,
        type="channel",
        title="wzrd",
        full_name=None,
        is_forum=False,
    )
    return SimpleNamespace(
        chat=chat,
        from_user=None,
        text=text,
        caption=None,
        entities=[],
        caption_entities=[],
        message_thread_id=None,
        is_topic_message=False,
        message_id=11,
        reply_to_message=None,
        quote=None,
        date=None,
        forum_topic_created=None,
    )


def test_build_message_event_uses_channel_identity_for_channel_posts():
    adapter = _make_adapter()
    msg = _make_channel_message()

    event = adapter._build_message_event(msg, MessageType.TEXT, update_id=12345)

    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "-1003950368353"
    # Channel posts often have no from_user.  Preserve an identity so the
    # gateway authorization layer can allowlist the channel by numeric ID.
    assert event.source.user_id == "-1003950368353"
    assert event.source.user_name == "wzrd"
    assert event.platform_update_id == 12345


@pytest.mark.asyncio
async def test_text_handler_uses_effective_message_for_channel_post():
    adapter = _make_adapter()
    msg = _make_channel_message()
    update = SimpleNamespace(
        update_id=12345,
        message=None,
        channel_post=msg,
        effective_message=msg,
    )
    adapter._enqueue_text_event = MagicMock()

    await adapter._handle_text_message(update, MagicMock())

    adapter._enqueue_text_event.assert_called_once()
    event = adapter._enqueue_text_event.call_args.args[0]
    assert event.text == "channel id test @hermes_bot"
    assert event.source.chat_type == "channel"
    assert event.source.chat_id == "-1003950368353"
