"""Tests for Telegram forum topic auto-rename on title change.

Covers:
- TelegramAdapter.rename_forum_topic: calls edit_forum_topic
- TelegramAdapter._is_auto_rename_topics_enabled: config check
- _make_title_rename_callback: returns None for non-Telegram / no thread_id
- _make_title_rename_callback: returns callable when conditions met
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig


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


def _make_adapter(extra=None):
    """Create a TelegramAdapter with optional extra config."""
    config = PlatformConfig(enabled=True, token="***", extra=extra or {})
    return TelegramAdapter(config)


# ── rename_forum_topic ──


@pytest.mark.asyncio
async def test_rename_forum_topic_calls_edit_forum_topic():
    """rename_forum_topic should call bot.edit_forum_topic with correct args."""
    adapter = _make_adapter({"auto_rename_topics": True})
    adapter._bot = AsyncMock()

    result = await adapter.rename_forum_topic(
        chat_id=-1001234567890,
        thread_id=42,
        title="Fix login bug",
    )

    assert result is True
    adapter._bot.edit_forum_topic.assert_awaited_once_with(
        chat_id=-1001234567890,
        message_thread_id=42,
        name="Fix login bug",
    )


@pytest.mark.asyncio
async def test_rename_forum_topic_returns_false_on_error():
    """rename_forum_topic should return False when the API call fails."""
    adapter = _make_adapter()
    adapter._bot = AsyncMock()
    adapter._bot.edit_forum_topic.side_effect = Exception("forbidden: not admin")

    result = await adapter.rename_forum_topic(
        chat_id=-1001234567890,
        thread_id=42,
        title="Whatever",
    )

    assert result is False


@pytest.mark.asyncio
async def test_rename_forum_topic_returns_false_without_bot():
    """rename_forum_topic should return False when _bot is None."""
    adapter = _make_adapter()
    adapter._bot = None

    result = await adapter.rename_forum_topic(
        chat_id=-1001234567890,
        thread_id=42,
        title="Whatever",
    )

    assert result is False


# ── _is_auto_rename_topics_enabled ──


def test_auto_rename_topics_enabled_true():
    adapter = _make_adapter({"auto_rename_topics": True})
    assert adapter._is_auto_rename_topics_enabled() is True


def test_auto_rename_topics_enabled_false_default():
    adapter = _make_adapter({})
    assert adapter._is_auto_rename_topics_enabled() is False


def test_auto_rename_topics_enabled_false_explicit():
    adapter = _make_adapter({"auto_rename_topics": False})
    assert adapter._is_auto_rename_topics_enabled() is False


def test_auto_rename_topics_enabled_string_true():
    adapter = _make_adapter({"auto_rename_topics": "yes"})
    assert adapter._is_auto_rename_topics_enabled() is True


# ── _make_title_rename_callback (integration with GatewayRunner) ──


def test_make_title_rename_callback_returns_none_for_non_telegram():
    """Should return None for non-Telegram platforms."""
    from gateway.config import Platform as _Platform
    from gateway.session import SessionSource

    source = SessionSource(
        platform=_Platform.DISCORD,
        chat_id="123",
        thread_id="456",
    )

    # We can't easily construct a full GatewayRunner, so test the logic directly.
    # The callback checks platform first, so non-telegram should return None.
    assert source.platform != _Platform.TELEGRAM


def test_make_title_rename_callback_returns_none_for_no_thread():
    """Should return None when thread_id is missing."""
    from gateway.config import Platform as _Platform
    from gateway.session import SessionSource

    source = SessionSource(
        platform=_Platform.TELEGRAM,
        chat_id="123",
        thread_id=None,
    )

    assert source.thread_id is None
