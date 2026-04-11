"""Tests for plugins/memory/honcho/session.py — HonchoSession and helpers."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from plugins.memory.honcho.session import (
    HonchoSession,
    HonchoSessionManager,
)
from plugins.memory.honcho import HonchoMemoryProvider


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


class TestPeerLookupHelpers:
    def _make_cached_manager(self):
        mgr = HonchoSessionManager()
        session = HonchoSession(
            key="telegram:123",
            user_peer_id="robert",
            assistant_peer_id="hermes",
            honcho_session_id="telegram-123",
        )
        mgr._cache[session.key] = session
        return mgr, session

    def test_get_peer_card_uses_direct_peer_lookup(self):
        mgr, session = self._make_cached_manager()
        user_peer = MagicMock()
        user_peer.get_card.return_value = ["Name: Robert"]
        mgr._get_or_create_peer = MagicMock(return_value=user_peer)

        assert mgr.get_peer_card(session.key) == ["Name: Robert"]
        user_peer.get_card.assert_called_once_with()

    def test_search_context_uses_peer_context_response(self):
        mgr, session = self._make_cached_manager()
        user_peer = MagicMock()
        user_peer.context.return_value = SimpleNamespace(
            representation="Robert runs neuralancer",
            peer_card=["Location: Melbourne"],
        )
        mgr._get_or_create_peer = MagicMock(return_value=user_peer)

        result = mgr.search_context(session.key, "neuralancer")

        assert "Robert runs neuralancer" in result
        assert "- Location: Melbourne" in result
        user_peer.context.assert_called_once_with(search_query="neuralancer")

    def test_get_prefetch_context_fetches_user_and_ai_from_peer_api(self):
        mgr, session = self._make_cached_manager()
        user_peer = MagicMock()
        user_peer.context.return_value = SimpleNamespace(
            representation="User representation",
            peer_card=["Name: Robert"],
        )
        ai_peer = MagicMock()
        ai_peer.context.return_value = SimpleNamespace(
            representation="AI representation",
            peer_card=["Owner: Robert"],
        )
        mgr._get_or_create_peer = MagicMock(side_effect=[user_peer, ai_peer])

        result = mgr.get_prefetch_context(session.key)

        assert result == {
            "representation": "User representation",
            "card": "Name: Robert",
            "ai_representation": "AI representation",
            "ai_card": "Owner: Robert",
        }
        user_peer.context.assert_called_once_with()
        ai_peer.context.assert_called_once_with()

    def test_get_ai_representation_uses_peer_api(self):
        mgr, session = self._make_cached_manager()
        ai_peer = MagicMock()
        ai_peer.context.return_value = SimpleNamespace(
            representation="AI representation",
            peer_card=["Owner: Robert"],
        )
        mgr._get_or_create_peer = MagicMock(return_value=ai_peer)

        result = mgr.get_ai_representation(session.key)

        assert result == {
            "representation": "AI representation",
            "card": "Owner: Robert",
        }
        ai_peer.context.assert_called_once_with()


# ---------------------------------------------------------------------------
# Message chunking
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Provider init behavior: lazy vs eager in tools mode
# ---------------------------------------------------------------------------


class TestToolsModeInitBehavior:
    """Verify initOnSessionStart controls session init timing in tools mode."""

    def _make_provider_with_config(self, recall_mode="tools", init_on_session_start=False,
                                    peer_name=None, user_id=None):
        """Create a HonchoMemoryProvider with mocked config and dependencies."""
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(
            api_key="test-key",
            enabled=True,
            recall_mode=recall_mode,
            init_on_session_start=init_on_session_start,
            peer_name=peer_name,
        )

        provider = HonchoMemoryProvider()

        # Patch the config loading and session init to avoid real Honcho calls
        from unittest.mock import patch, MagicMock

        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        init_kwargs = {}
        if user_id:
            init_kwargs["user_id"] = user_id

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-001", **init_kwargs)

        return provider, cfg

    def test_tools_lazy_default(self):
        """tools + initOnSessionStart=false → session NOT initialized after initialize()."""
        provider, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=False,
        )
        assert provider._session_initialized is False
        assert provider._manager is None
        assert provider._lazy_init_kwargs is not None

    def test_tools_eager_init(self):
        """tools + initOnSessionStart=true → session IS initialized after initialize()."""
        provider, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
        )
        assert provider._session_initialized is True
        assert provider._manager is not None

    def test_tools_eager_prefetch_still_empty(self):
        """tools mode with eager init still returns empty from prefetch() (no auto-injection)."""
        provider, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
        )
        assert provider.prefetch("test query") == ""

    def test_tools_lazy_prefetch_empty(self):
        """tools mode with lazy init also returns empty from prefetch()."""
        provider, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=False,
        )
        assert provider.prefetch("test query") == ""

    def test_explicit_peer_name_not_overridden_by_user_id(self):
        """Explicit peerName in config must not be replaced by gateway user_id."""
        _, cfg = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
            peer_name="Kathie", user_id="8439114563",
        )
        assert cfg.peer_name == "Kathie"

    def test_user_id_used_when_no_peer_name(self):
        """Gateway user_id is used as peer_name when no explicit peerName configured."""
        _, cfg = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
            peer_name=None, user_id="8439114563",
        )
        assert cfg.peer_name == "8439114563"


class TestChunkMessage:
    def test_short_message_single_chunk(self):
        result = HonchoMemoryProvider._chunk_message("hello world", 100)
        assert result == ["hello world"]

    def test_exact_limit_single_chunk(self):
        msg = "x" * 100
        result = HonchoMemoryProvider._chunk_message(msg, 100)
        assert result == [msg]

    def test_splits_at_paragraph_boundary(self):
        msg = "first paragraph.\n\nsecond paragraph."
        # limit=30: total is 35, forces split; second chunk with prefix is 29, fits
        result = HonchoMemoryProvider._chunk_message(msg, 30)
        assert len(result) == 2
        assert result[0] == "first paragraph."
        assert result[1] == "[continued] second paragraph."

    def test_splits_at_sentence_boundary(self):
        msg = "First sentence. Second sentence. Third sentence is here."
        result = HonchoMemoryProvider._chunk_message(msg, 35)
        assert len(result) >= 2
        # First chunk should end at a sentence boundary (rstripped)
        assert result[0].rstrip().endswith(".")

    def test_splits_at_word_boundary(self):
        msg = "word " * 20  # 100 chars
        result = HonchoMemoryProvider._chunk_message(msg, 30)
        assert len(result) >= 2
        # No words should be split mid-word
        for chunk in result:
            clean = chunk.replace("[continued] ", "")
            assert not clean.startswith(" ")

    def test_continuation_prefix(self):
        msg = "a" * 200
        result = HonchoMemoryProvider._chunk_message(msg, 50)
        assert len(result) >= 2
        assert not result[0].startswith("[continued]")
        for chunk in result[1:]:
            assert chunk.startswith("[continued] ")

    def test_empty_message(self):
        result = HonchoMemoryProvider._chunk_message("", 100)
        assert result == [""]

    def test_large_message_many_chunks(self):
        msg = "word " * 10000  # 50k chars
        result = HonchoMemoryProvider._chunk_message(msg, 25000)
        assert len(result) >= 2
        for chunk in result:
            assert len(chunk) <= 25000


# ---------------------------------------------------------------------------
# Dialectic input guard
# ---------------------------------------------------------------------------


class TestDialecticInputGuard:
    def test_long_query_truncated(self):
        """Queries exceeding dialectic_max_input_chars are truncated."""
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(dialectic_max_input_chars=100)
        mgr = HonchoSessionManager(config=cfg)
        mgr._dialectic_max_input_chars = 100

        # Create a cached session so dialectic_query doesn't bail early
        session = HonchoSession(
            key="test", user_peer_id="u", assistant_peer_id="a",
            honcho_session_id="s",
        )
        mgr._cache["test"] = session

        # Mock the peer to capture the query
        mock_peer = MagicMock()
        mock_peer.chat.return_value = "answer"
        mgr._get_or_create_peer = MagicMock(return_value=mock_peer)

        long_query = "word " * 100  # 500 chars, exceeds 100 limit
        mgr.dialectic_query("test", long_query)

        # The query passed to chat() should be truncated
        actual_query = mock_peer.chat.call_args[0][0]
        assert len(actual_query) <= 100
