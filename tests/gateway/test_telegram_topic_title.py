"""Tests for ``TelegramAdapter.update_topic_title`` — issue #16255.

Auto-generated session titles must propagate to the corresponding Telegram
forum topic name so the topic list and ``/history`` stay in sync.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Mock the python-telegram-bot package when not installed
# ---------------------------------------------------------------------------


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()


from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


@pytest.fixture()
def adapter():
    cfg = PlatformConfig(enabled=True, token="fake-token")
    a = TelegramAdapter(cfg)
    a._bot = MagicMock()
    a._bot.edit_forum_topic = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# update_topic_title
# ---------------------------------------------------------------------------


class TestUpdateTopicTitle:
    @pytest.mark.asyncio
    async def test_pushes_title_to_bot(self, adapter):
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="Fresh Title")
        adapter._bot.edit_forum_topic.assert_awaited_once_with(
            chat_id=123, message_thread_id=42, name="Fresh Title",
        )

    @pytest.mark.asyncio
    async def test_no_op_without_thread_id(self, adapter):
        await adapter.update_topic_title(chat_id="123", thread_id=None, title="X")
        adapter._bot.edit_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_for_general_topic(self, adapter):
        # The "General" topic name is owned by the chat itself, not the bot.
        await adapter.update_topic_title(
            chat_id="123",
            thread_id=TelegramAdapter._GENERAL_TOPIC_THREAD_ID,
            title="X",
        )
        adapter._bot.edit_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_when_bot_missing(self, adapter):
        adapter._bot = None
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="X")
        # No crash; nothing to assert beyond that.

    @pytest.mark.asyncio
    async def test_no_op_for_blank_title(self, adapter):
        for blank in ("", "   ", "\t"):
            await adapter.update_topic_title(chat_id="123", thread_id="42", title=blank)
        adapter._bot.edit_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncates_to_telegram_limit(self, adapter):
        # Telegram caps forum topic names at 128 chars.
        long_title = "A" * 200
        await adapter.update_topic_title(chat_id="123", thread_id="42", title=long_title)
        kwargs = adapter._bot.edit_forum_topic.await_args.kwargs
        assert len(kwargs["name"]) == 128
        assert kwargs["name"] == "A" * 128

    @pytest.mark.asyncio
    async def test_strips_whitespace(self, adapter):
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="  spaced  ")
        kwargs = adapter._bot.edit_forum_topic.await_args.kwargs
        assert kwargs["name"] == "spaced"

    @pytest.mark.asyncio
    async def test_non_numeric_thread_id_skipped(self, adapter):
        await adapter.update_topic_title(chat_id="123", thread_id="not-a-number", title="X")
        adapter._bot.edit_forum_topic.assert_not_called()

    @pytest.mark.asyncio
    async def test_bot_failure_is_swallowed(self, adapter):
        adapter._bot.edit_forum_topic.side_effect = RuntimeError("400 Bad Request")
        # Must not propagate — title sync is best-effort UX.
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="X")

    @pytest.mark.asyncio
    async def test_cache_realigned_after_rename(self, adapter):
        adapter._dm_topics = {"Old Name": 42, "Other": 7}
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="New Name")
        assert "Old Name" not in adapter._dm_topics
        assert adapter._dm_topics["New Name"] == 42
        assert adapter._dm_topics["Other"] == 7

    @pytest.mark.asyncio
    async def test_cache_filled_when_thread_unknown(self, adapter):
        adapter._dm_topics = {}
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="Brand New")
        assert adapter._dm_topics == {"Brand New": 42}

    @pytest.mark.asyncio
    async def test_cache_unchanged_when_name_unchanged(self, adapter):
        adapter._dm_topics = {"Same": 42}
        await adapter.update_topic_title(chat_id="123", thread_id="42", title="Same")
        assert adapter._dm_topics == {"Same": 42}


# ---------------------------------------------------------------------------
# Base class default — every other adapter must inherit a no-op
# ---------------------------------------------------------------------------


class TestBaseDefaultIsNoOp:
    @pytest.mark.asyncio
    async def test_base_default_returns_none(self):
        """Call the base method directly — it must remain a no-op default
        so non-Telegram adapters don't have to implement anything."""
        from gateway.platforms.base import BasePlatformAdapter

        # Bind the unbound method to None to bypass abstract-class checks.
        result = await BasePlatformAdapter.update_topic_title(
            None, chat_id="x", thread_id="1", title="anything",
        )
        assert result is None
