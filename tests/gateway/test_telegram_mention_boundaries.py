"""Tests for Telegram bot mention word-boundary detection (bug #12545).

The old implementation used a naive substring check (`f"@{bot_username}" in text.lower()`),
which incorrectly matched partial substrings like 'foo@hermes_bot.example'.

These tests verify that the regex-based word-boundary check correctly delimits mentions.
"""
from types import SimpleNamespace

from gateway.platforms.telegram import TelegramAdapter


def _make_adapter():
    """Build a minimal TelegramAdapter with a mocked bot."""
    from gateway.config import Platform, PlatformConfig

    adapter = object.__new__(TelegramAdapter)
    adapter.platform = Platform.TELEGRAM
    adapter.config = PlatformConfig(enabled=True, token="***", extra={})
    adapter._bot = SimpleNamespace(id=999, username="hermes_bot")
    return adapter


def _group_message(text, entities=None, caption=None, caption_entities=None):
    """Produce a minimal group-message-like SimpleNamespace."""
    return SimpleNamespace(
        text=text,
        caption=caption,
        entities=entities or [],
        caption_entities=caption_entities or [],
        message_thread_id=None,
        chat=SimpleNamespace(id=-100, type="group"),
        reply_to_message=None,
    )


class TestTelegramMentionBoundaries:
    """Test that _message_mentions_bot correctly respects word boundaries."""

    def test_exact_mention_is_recognized(self):
        """'@hermes_bot' at any position should be detected."""
        adapter = _make_adapter()
        msg = _group_message("hello @hermes_bot")
        assert adapter._message_mentions_bot(msg) is True

    def test_mention_at_start_of_string(self):
        """'@hermes_bot hello' should be detected."""
        adapter = _make_adapter()
        msg = _group_message("@hermes_bot hello world")
        assert adapter._message_mentions_bot(msg) is True

    def test_mention_followed_by_punctuation(self):
        """'@hermes_bot,' should be detected."""
        adapter = _make_adapter()
        msg = _group_message("@hermes_bot, how are you?")
        assert adapter._message_mentions_bot(msg) is True

    def test_mention_in_subdomain_is_not_recognized(self):
        """'foo@hermes_bot.example' should NOT match (bug #12545)."""
        adapter = _make_adapter()
        msg = _group_message("foo@hermes_bot.example")
        assert adapter._message_mentions_bot(msg) is False

    def test_mention_in_longer_hostname_is_not_recognized(self):
        """'email me at user@hermes_bot.domain.com' should NOT match."""
        adapter = _make_adapter()
        msg = _group_message("email me at user@hermes_bot.domain.com")
        assert adapter._message_mentions_bot(msg) is False

    def test_superstring_username_is_not_recognized(self):
        """'@hermes_botx' should NOT match (different username)."""
        adapter = _make_adapter()
        msg = _group_message("@hermes_botx hello")
        assert adapter._message_mentions_bot(msg) is False

    def test_prefixed_superstring_is_not_recognized(self):
        """'foo@hermes_bot_bar' should NOT match."""
        adapter = _make_adapter()
        msg = _group_message("foo@hermes_bot_bar")
        assert adapter._message_mentions_bot(msg) is False

    def test_mention_case_insensitive(self):
        """'@HERMES_BOT' should be detected (case-insensitive)."""
        adapter = _make_adapter()
        msg = _group_message("@HERMES_BOT hello")
        assert adapter._message_mentions_bot(msg) is True

    def test_mention_mixed_case(self):
        """'@Hermes_Bot' should be detected."""
        adapter = _make_adapter()
        msg = _group_message("@Hermes_Bot hello")
        assert adapter._message_mentions_bot(msg) is True

    def test_no_mention_returns_false(self):
        """Plain text with no mention should return False."""
        adapter = _make_adapter()
        msg = _group_message("just a regular message in the group")
        assert adapter._message_mentions_bot(msg) is False

    def test_mention_in_caption(self):
        """Mention in caption should be detected."""
        adapter = _make_adapter()
        msg = _group_message(None, caption="check this out @hermes_bot")
        assert adapter._message_mentions_bot(msg) is True

    def test_subdomain_mention_in_caption_not_recognized(self):
        """'foo@hermes_bot.example' in caption should NOT match."""
        adapter = _make_adapter()
        msg = _group_message(None, caption="foo@hermes_bot.example")
        assert adapter._message_mentions_bot(msg) is False
