"""Tests for TelegramAdapter._classify_telegram_chat_type.

Pins each call site's current forum semantics so the helper preserves
intent rather than collapsing to one predicate.
"""

import sys
from pathlib import Path

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

# tests/gateway/conftest.py auto-loads and installs the telegram mock.

from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class TestClassifyTelegramChatType:
    """The shared helper must produce the same result each call site had inline."""

    # --- callback-auth semantics (thread_id only, is_forum=thread_id is not None) ---

    def test_callback_private_to_dm(self):
        assert TelegramAdapter._classify_telegram_chat_type("private") == "dm"

    def test_callback_supergroup_no_thread_is_group(self):
        assert TelegramAdapter._classify_telegram_chat_type("supergroup") == "group"

    def test_callback_supergroup_with_thread_is_forum(self):
        assert TelegramAdapter._classify_telegram_chat_type("supergroup", thread_id=42, is_forum=True) == "forum"

    def test_callback_group_type_no_thread_is_group(self):
        assert TelegramAdapter._classify_telegram_chat_type("group") == "group"

    def test_callback_group_type_with_thread_is_forum(self):
        """The helper treats group and supergroup identically (both can be
        forum-capable). This matches the post-refactor unified behavior."""
        assert TelegramAdapter._classify_telegram_chat_type("group", thread_id=42, is_forum=True) == "forum"

    # --- message-auth semantics (thread_id + is_topic/is_forum) ---

    def test_message_supergroup_thread_is_topic_is_forum(self):
        assert (
            TelegramAdapter._classify_telegram_chat_type(
                "supergroup", thread_id=42, is_topic_message=True
            )
            == "forum"
        )

    def test_message_supergroup_thread_not_topic_not_forum_is_group(self):
        assert (
            TelegramAdapter._classify_telegram_chat_type(
                "supergroup", thread_id=42, is_topic_message=False, is_forum=False
            )
            == "group"
        )

    def test_message_supergroup_thread_is_forum_no_topic(self):
        assert (
            TelegramAdapter._classify_telegram_chat_type(
                "supergroup", thread_id=42, is_forum=True
            )
            == "forum"
        )

    # --- reaction-auth semantics (is_forum only, no thread_id) ---

    def test_reaction_supergroup_is_forum(self):
        assert (
            TelegramAdapter._classify_telegram_chat_type("supergroup", is_forum=True)
            == "forum"
        )

    def test_reaction_supergroup_not_forum(self):
        assert (
            TelegramAdapter._classify_telegram_chat_type("supergroup", is_forum=False)
            == "group"
        )

    # --- channel ---

    def test_channel(self):
        assert TelegramAdapter._classify_telegram_chat_type("channel") == "channel"

    # --- edge cases ---

    def test_empty_string_defaults_to_dm(self):
        assert TelegramAdapter._classify_telegram_chat_type("") == "dm"

    def test_none_defaults_to_dm(self):
        assert TelegramAdapter._classify_telegram_chat_type(None) == "dm"

    def test_uppercase_normalized(self):
        assert TelegramAdapter._classify_telegram_chat_type("SUPERGROUP", is_forum=True) == "forum"

    def test_unknown_type_logs_debug_and_defaults_to_dm(self, caplog):
        """A future Telegram chat.type must be visible in debug logs, not
        silently misrouted as a DM."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="plugins.platforms.telegram.adapter"):
            assert TelegramAdapter._classify_telegram_chat_type("gigagroup") == "dm"
        assert "Unknown chat.type 'gigagroup'" in caplog.text

    def test_empty_type_stays_silent(self, caplog):
        """The None/'' default is a normal call shape, not an upstream change;
        it must not add debug noise."""
        import logging

        with caplog.at_level(logging.DEBUG, logger="plugins.platforms.telegram.adapter"):
            TelegramAdapter._classify_telegram_chat_type(None)
            TelegramAdapter._classify_telegram_chat_type("")
        assert "Unknown chat.type" not in caplog.text


class TestWidenedGroupForumPromotionDownstream:
    """Pin the downstream thread_id assignment for the widened group→forum
    case in _source_from_message_for_auth.

    Pre-refactor, only "supergroup" was promoted, so a plain "group" chat
    left thread_id None even with a topic + is_forum. The helper now treats
    group/supergroup identically; this locks the resulting source shape.
    """

    def _adapter(self):
        return object.__new__(TelegramAdapter)

    @staticmethod
    def _group_message(*, chat_type="group", thread_id=42, is_forum=True, is_topic_message=True):
        from types import SimpleNamespace

        return SimpleNamespace(
            from_user=SimpleNamespace(id=111, username="alice", full_name="Alice"),
            chat=SimpleNamespace(id=-100, type=chat_type, is_forum=is_forum),
            message_thread_id=thread_id,
            is_topic_message=is_topic_message,
        )

    def test_plain_group_with_topic_and_forum_assigns_thread_id(self):
        source = TelegramAdapter._source_from_message_for_auth(
            self._adapter(), self._group_message()
        )
        assert source.chat_type == "forum"
        assert source.thread_id == "42"

    def test_plain_group_without_thread_keeps_thread_id_none(self):
        source = TelegramAdapter._source_from_message_for_auth(
            self._adapter(), self._group_message(thread_id=None, is_forum=True)
        )
        # Forum supergroup General-topic shape: no thread_id stays "group"
        # even when is_forum is set (matches the original AND short-circuit).
        assert source.chat_type == "group"
        assert source.thread_id is None
