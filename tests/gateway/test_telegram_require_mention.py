"""Tests for Telegram require_mention feature in gateway/platforms/telegram.py.

Covers: _is_bot_mentioned, _is_reply_to_bot, _strip_bot_mention,
_should_skip_group_message — the group message filtering logic that mirrors
Discord's require_mention behavior for Telegram.
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed
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
    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from telegram.constants import ChatType  # noqa: E402
from gateway.platforms.telegram import TelegramAdapter  # noqa: E402

# Map string shortcuts to the ChatType mock constants
_CHAT_TYPE_MAP = {
    "group": ChatType.GROUP,
    "supergroup": ChatType.SUPERGROUP,
    "channel": ChatType.CHANNEL,
    "private": ChatType.PRIVATE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    adapter = TelegramAdapter(config)
    adapter._bot = MagicMock()
    adapter._bot.username = "TestBot"
    adapter._bot.id = 12345
    return adapter


def _make_message(text="hello", chat_type="group", entities=None,
                  reply_to_bot=False, bot_id=12345):
    """Create a mock Telegram Message."""
    msg = MagicMock()
    msg.text = text
    msg.chat.type = _CHAT_TYPE_MAP.get(chat_type, chat_type)
    msg.entities = entities or []

    if reply_to_bot:
        msg.reply_to_message.from_user.id = bot_id
    else:
        msg.reply_to_message = None

    return msg


def _make_mention_entity(offset, length, entity_type="mention", user=None):
    """Create a mock MessageEntity for @mentions."""
    entity = MagicMock()
    entity.type = entity_type
    entity.offset = offset
    entity.length = length
    entity.user = user
    return entity


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    return _make_adapter()


# ===========================================================================
# _is_bot_mentioned
# ===========================================================================

class TestIsBotMentioned:
    def test_mentioned_by_username(self, adapter):
        text = "@TestBot hello"
        entity = _make_mention_entity(offset=0, length=8)
        msg = _make_message(text=text, entities=[entity])
        assert adapter._is_bot_mentioned(msg) is True

    def test_mentioned_case_insensitive(self, adapter):
        text = "@testbot hello"
        entity = _make_mention_entity(offset=0, length=8)
        msg = _make_message(text=text, entities=[entity])
        assert adapter._is_bot_mentioned(msg) is True

    def test_not_mentioned(self, adapter):
        text = "hello world"
        msg = _make_message(text=text)
        assert adapter._is_bot_mentioned(msg) is False

    def test_different_bot_mentioned(self, adapter):
        text = "@OtherBot hello"
        entity = _make_mention_entity(offset=0, length=9)
        msg = _make_message(text=text, entities=[entity])
        assert adapter._is_bot_mentioned(msg) is False

    def test_text_mention(self, adapter):
        """text_mention entities include a user object instead of @username."""
        user = MagicMock()
        user.id = 12345
        entity = _make_mention_entity(offset=0, length=7, entity_type="text_mention", user=user)
        msg = _make_message(text="TestBot hello", entities=[entity])
        assert adapter._is_bot_mentioned(msg) is True

    def test_text_mention_wrong_user(self, adapter):
        user = MagicMock()
        user.id = 99999
        entity = _make_mention_entity(offset=0, length=5, entity_type="text_mention", user=user)
        msg = _make_message(text="Other hello", entities=[entity])
        assert adapter._is_bot_mentioned(msg) is False

    def test_no_bot(self, adapter):
        adapter._bot = None
        msg = _make_message(text="@TestBot hello")
        assert adapter._is_bot_mentioned(msg) is False


# ===========================================================================
# _is_reply_to_bot
# ===========================================================================

class TestIsReplyToBot:
    def test_reply_to_bot(self, adapter):
        msg = _make_message(reply_to_bot=True)
        assert adapter._is_reply_to_bot(msg) is True

    def test_not_a_reply(self, adapter):
        msg = _make_message()
        assert adapter._is_reply_to_bot(msg) is False

    def test_reply_to_other_user(self, adapter):
        msg = _make_message(reply_to_bot=True, bot_id=99999)
        assert adapter._is_reply_to_bot(msg) is False


# ===========================================================================
# _strip_bot_mention
# ===========================================================================

class TestStripBotMention:
    def test_strips_mention(self, adapter):
        assert adapter._strip_bot_mention("@TestBot hello") == "hello"

    def test_strips_case_insensitive(self, adapter):
        assert adapter._strip_bot_mention("@testbot hello") == "hello"

    def test_strips_mention_mid_text(self, adapter):
        result = adapter._strip_bot_mention("hey @TestBot do this")
        # After stripping, surrounding whitespace is collapsed by .strip()
        # but internal double-spaces may remain — the agent handles this fine
        assert "@TestBot" not in result
        assert "hey" in result and "do this" in result

    def test_no_mention_unchanged(self, adapter):
        assert adapter._strip_bot_mention("hello world") == "hello world"

    def test_no_bot(self, adapter):
        adapter._bot = None
        assert adapter._strip_bot_mention("@TestBot hello") == "@TestBot hello"


# ===========================================================================
# _should_skip_group_message
# ===========================================================================

class TestShouldSkipGroupMessage:
    def test_dm_never_skipped(self, adapter):
        msg = _make_message(chat_type="private")
        assert adapter._should_skip_group_message(msg) is False

    def test_group_without_mention_skipped(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        msg = _make_message(chat_type="group")
        assert adapter._should_skip_group_message(msg) is True

    def test_supergroup_without_mention_skipped(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        msg = _make_message(chat_type="supergroup")
        assert adapter._should_skip_group_message(msg) is True

    def test_group_with_mention_not_skipped(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        text = "@TestBot hello"
        entity = _make_mention_entity(offset=0, length=8)
        msg = _make_message(text=text, chat_type="group", entities=[entity])
        assert adapter._should_skip_group_message(msg) is False

    def test_group_with_reply_not_skipped(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        msg = _make_message(chat_type="group", reply_to_bot=True)
        assert adapter._should_skip_group_message(msg) is False

    def test_require_mention_disabled(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "false")
        msg = _make_message(chat_type="group")
        assert adapter._should_skip_group_message(msg) is False

    def test_require_mention_default_true(self, adapter, monkeypatch):
        monkeypatch.delenv("TELEGRAM_REQUIRE_MENTION", raising=False)
        msg = _make_message(chat_type="group")
        assert adapter._should_skip_group_message(msg) is True

    def test_free_response_channel(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        monkeypatch.setenv("TELEGRAM_FREE_RESPONSE_CHANNELS", "-100123,456")
        msg = _make_message(chat_type="group")
        msg.chat.id = -100123
        assert adapter._should_skip_group_message(msg) is False

    def test_not_in_free_response_channel(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        monkeypatch.setenv("TELEGRAM_FREE_RESPONSE_CHANNELS", "-100123")
        msg = _make_message(chat_type="group")
        msg.chat.id = -100999
        assert adapter._should_skip_group_message(msg) is True

    def test_trigger_word_match(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        monkeypatch.setenv("TELEGRAM_TRIGGER_WORDS", "deploy,build")
        msg = _make_message(text="can someone deploy this", chat_type="group")
        assert adapter._should_skip_group_message(msg) is False

    def test_trigger_word_case_insensitive(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        monkeypatch.setenv("TELEGRAM_TRIGGER_WORDS", "Deploy")
        msg = _make_message(text="DEPLOY now", chat_type="group")
        assert adapter._should_skip_group_message(msg) is False

    def test_trigger_word_no_match(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        monkeypatch.setenv("TELEGRAM_TRIGGER_WORDS", "deploy,build")
        msg = _make_message(text="hello everyone", chat_type="group")
        assert adapter._should_skip_group_message(msg) is True

    def test_trigger_words_empty_default(self, adapter, monkeypatch):
        monkeypatch.setenv("TELEGRAM_REQUIRE_MENTION", "true")
        monkeypatch.delenv("TELEGRAM_TRIGGER_WORDS", raising=False)
        msg = _make_message(text="deploy this", chat_type="group")
        assert adapter._should_skip_group_message(msg) is True
