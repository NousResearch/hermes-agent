"""Tests for honcho_integration/session.py — HonchoSession and helpers."""

from datetime import datetime
from unittest.mock import MagicMock

from honcho_integration.session import (
    HonchoSession,
    HonchoSessionManager,
)


# ---------------------------------------------------------------------------
# HonchoSession dataclass
# ---------------------------------------------------------------------------


class TestHonchoSession:
    def _make_session(self):
        return HonchoSession(
            key="telegram:12345",
            user_peer_id="user-telegram-12345",
            assistant_peer_id="hermes-assistant",
            honcho_session_id="telegram-12345",
        )

    def test_initial_state(self):
        session = self._make_session()
        assert session.key == "telegram:12345"
        assert session.messages == []
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)

    def test_add_message(self):
        session = self._make_session()
        session.add_message("user", "Hello!")
        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "user"
        assert session.messages[0]["content"] == "Hello!"
        assert "timestamp" in session.messages[0]

    def test_add_message_with_kwargs(self):
        session = self._make_session()
        session.add_message("assistant", "Hi!", source="gateway")
        assert session.messages[0]["source"] == "gateway"

    def test_add_message_updates_timestamp(self):
        session = self._make_session()
        original = session.updated_at
        session.add_message("user", "test")
        assert session.updated_at >= original

    def test_get_history(self):
        session = self._make_session()
        session.add_message("user", "msg1")
        session.add_message("assistant", "msg2")
        history = session.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "msg1"}
        assert history[1] == {"role": "assistant", "content": "msg2"}

    def test_get_history_strips_extra_fields(self):
        session = self._make_session()
        session.add_message("user", "hello", extra="metadata")
        history = session.get_history()
        assert "extra" not in history[0]
        assert set(history[0].keys()) == {"role", "content"}

    def test_get_history_max_messages(self):
        session = self._make_session()
        for i in range(10):
            session.add_message("user", f"msg{i}")
        history = session.get_history(max_messages=3)
        assert len(history) == 3
        assert history[0]["content"] == "msg7"
        assert history[2]["content"] == "msg9"

    def test_get_history_max_messages_larger_than_total(self):
        session = self._make_session()
        session.add_message("user", "only one")
        history = session.get_history(max_messages=100)
        assert len(history) == 1

    def test_clear(self):
        session = self._make_session()
        session.add_message("user", "msg1")
        session.add_message("user", "msg2")
        session.clear()
        assert session.messages == []

    def test_clear_updates_timestamp(self):
        session = self._make_session()
        session.add_message("user", "msg")
        original = session.updated_at
        session.clear()
        assert session.updated_at >= original


# ---------------------------------------------------------------------------
# HonchoSessionManager._sanitize_id
# ---------------------------------------------------------------------------


class TestSanitizeId:
    def test_clean_id_unchanged(self):
        mgr = HonchoSessionManager()
        assert mgr._sanitize_id("telegram-12345") == "telegram-12345"

    def test_colons_replaced(self):
        mgr = HonchoSessionManager()
        assert mgr._sanitize_id("telegram:12345") == "telegram-12345"

    def test_special_chars_replaced(self):
        mgr = HonchoSessionManager()
        result = mgr._sanitize_id("user@chat#room!")
        assert "@" not in result
        assert "#" not in result
        assert "!" not in result

    def test_alphanumeric_preserved(self):
        mgr = HonchoSessionManager()
        assert mgr._sanitize_id("abc123_XYZ-789") == "abc123_XYZ-789"


# ---------------------------------------------------------------------------
# HonchoSessionManager._format_migration_transcript
# ---------------------------------------------------------------------------


class TestFormatMigrationTranscript:
    def test_basic_transcript(self):
        messages = [
            {"role": "user", "content": "Hello", "timestamp": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi!", "timestamp": "2026-01-01T00:01:00"},
        ]
        result = HonchoSessionManager._format_migration_transcript("telegram:123", messages)
        assert isinstance(result, bytes)
        text = result.decode("utf-8")
        assert "<prior_conversation_history>" in text
        assert "user: Hello" in text
        assert "assistant: Hi!" in text
        assert 'session_key="telegram:123"' in text
        assert 'message_count="2"' in text

    def test_empty_messages(self):
        result = HonchoSessionManager._format_migration_transcript("key", [])
        text = result.decode("utf-8")
        assert "<prior_conversation_history>" in text
        assert "</prior_conversation_history>" in text

    def test_missing_fields_handled(self):
        messages = [{"role": "user"}]  # no content, no timestamp
        result = HonchoSessionManager._format_migration_transcript("key", messages)
        text = result.decode("utf-8")
        assert "user: " in text  # empty content


# ---------------------------------------------------------------------------
# HonchoSessionManager.delete / list_sessions
# ---------------------------------------------------------------------------


class TestManagerCacheOps:
    def test_delete_cached_session(self):
        mgr = HonchoSessionManager()
        session = HonchoSession(
            key="test", user_peer_id="u", assistant_peer_id="a",
            honcho_session_id="s",
        )
        mgr._cache["test"] = session
        assert mgr.delete("test") is True
        assert "test" not in mgr._cache

    def test_delete_nonexistent_returns_false(self):
        mgr = HonchoSessionManager()
        assert mgr.delete("nonexistent") is False

    def test_list_sessions(self):
        mgr = HonchoSessionManager()
        s1 = HonchoSession(key="k1", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s1")
        s2 = HonchoSession(key="k2", user_peer_id="u", assistant_peer_id="a", honcho_session_id="s2")
        s1.add_message("user", "hi")
        mgr._cache["k1"] = s1
        mgr._cache["k2"] = s2
        sessions = mgr.list_sessions()
        assert len(sessions) == 2
        keys = {s["key"] for s in sessions}
        assert keys == {"k1", "k2"}
        s1_info = next(s for s in sessions if s["key"] == "k1")
        assert s1_info["message_count"] == 1
