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
        assistant_peer = MagicMock()
        assistant_peer.get_card.return_value = ["Name: Robert"]
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)

        assert mgr.get_peer_card(session.key) == ["Name: Robert"]
        assistant_peer.get_card.assert_called_once_with(target=session.user_peer_id)

    def test_search_context_uses_assistant_perspective_with_target(self):
        mgr, session = self._make_cached_manager()
        assistant_peer = MagicMock()
        assistant_peer.context.return_value = SimpleNamespace(
            representation="Robert runs neuralancer",
            peer_card=["Location: Melbourne"],
        )
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)

        result = mgr.search_context(session.key, "neuralancer")

        assert "Robert runs neuralancer" in result
        assert "- Location: Melbourne" in result
        assistant_peer.context.assert_called_once_with(
            target=session.user_peer_id,
            search_query="neuralancer",
        )

    def test_search_context_unified_mode_uses_user_self_context(self):
        mgr, session = self._make_cached_manager()
        mgr._ai_observe_others = False
        user_peer = MagicMock()
        user_peer.context.return_value = SimpleNamespace(
            representation="Unified self context",
            peer_card=["Name: Robert"],
        )
        mgr._get_or_create_peer = MagicMock(return_value=user_peer)

        result = mgr.search_context(session.key, "self")

        assert "Unified self context" in result
        user_peer.context.assert_called_once_with(search_query="self")

    def test_search_context_accepts_explicit_ai_peer_id(self):
        mgr, session = self._make_cached_manager()
        ai_peer = MagicMock()
        ai_peer.context.return_value = SimpleNamespace(
            representation="Assistant self context",
            peer_card=["Role: Assistant"],
        )
        mgr._get_or_create_peer = MagicMock(return_value=ai_peer)

        result = mgr.search_context(session.key, "assistant", peer=session.assistant_peer_id)

        assert "Assistant self context" in result
        ai_peer.context.assert_called_once_with(
            target=session.assistant_peer_id,
            search_query="assistant",
        )

    def test_get_prefetch_context_fetches_user_and_ai_from_peer_api(self):
        mgr, session = self._make_cached_manager()
        user_peer = MagicMock()
        user_peer.context.return_value = SimpleNamespace(
            representation="User representation",
            peer_card=["Name: Robert"],
        )
        ai_peer = MagicMock()
        ai_peer.context.side_effect = lambda **kwargs: SimpleNamespace(
            representation=(
                "AI representation" if kwargs.get("target") == session.assistant_peer_id
                else "Mixed representation"
            ),
            peer_card=(
                ["Role: Assistant"] if kwargs.get("target") == session.assistant_peer_id
                else ["Name: Robert"]
            ),
        )
        mgr._get_or_create_peer = MagicMock(side_effect=[user_peer, ai_peer])

        result = mgr.get_prefetch_context(session.key)

        assert result == {
            "representation": "User representation",
            "card": "Name: Robert",
            "ai_representation": "AI representation",
            "ai_card": "Role: Assistant",
        }
        user_peer.context.assert_called_once_with(target=session.user_peer_id)
        ai_peer.context.assert_called_once_with(target=session.assistant_peer_id)

    def test_get_ai_representation_uses_peer_api(self):
        mgr, session = self._make_cached_manager()
        ai_peer = MagicMock()
        ai_peer.context.side_effect = lambda **kwargs: SimpleNamespace(
            representation=(
                "AI representation" if kwargs.get("target") == session.assistant_peer_id
                else "Mixed representation"
            ),
            peer_card=(
                ["Role: Assistant"] if kwargs.get("target") == session.assistant_peer_id
                else ["Name: Robert"]
            ),
        )
        mgr._get_or_create_peer = MagicMock(return_value=ai_peer)

        result = mgr.get_ai_representation(session.key)

        assert result == {
            "representation": "AI representation",
            "card": "Role: Assistant",
        }
        ai_peer.context.assert_called_once_with(target=session.assistant_peer_id)

    def test_create_conclusion_defaults_to_user_target(self):
        mgr, session = self._make_cached_manager()
        assistant_peer = MagicMock()
        scope = MagicMock()
        assistant_peer.conclusions_of.return_value = scope
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)

        ok = mgr.create_conclusion(session.key, "User prefers dark mode")

        assert ok is True
        assistant_peer.conclusions_of.assert_called_once_with(session.user_peer_id)
        scope.create.assert_called_once_with([{
            "content": "User prefers dark mode",
            "session_id": session.honcho_session_id,
        }])

    def test_create_conclusion_can_target_ai_peer(self):
        mgr, session = self._make_cached_manager()
        assistant_peer = MagicMock()
        scope = MagicMock()
        assistant_peer.conclusions_of.return_value = scope
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)

        ok = mgr.create_conclusion(session.key, "Assistant prefers terse summaries", peer="ai")

        assert ok is True
        assistant_peer.conclusions_of.assert_called_once_with(session.assistant_peer_id)
        scope.create.assert_called_once_with([{
            "content": "Assistant prefers terse summaries",
            "session_id": session.honcho_session_id,
        }])

    def test_create_conclusion_accepts_explicit_user_peer_id(self):
        mgr, session = self._make_cached_manager()
        assistant_peer = MagicMock()
        scope = MagicMock()
        assistant_peer.conclusions_of.return_value = scope
        mgr._get_or_create_peer = MagicMock(return_value=assistant_peer)

        ok = mgr.create_conclusion(session.key, "Robert prefers vinyl", peer=session.user_peer_id)

        assert ok is True
        assistant_peer.conclusions_of.assert_called_once_with(session.user_peer_id)
        scope.create.assert_called_once_with([{
            "content": "Robert prefers vinyl",
            "session_id": session.honcho_session_id,
        }])


class TestConcludeToolDispatch:
    def test_honcho_conclude_defaults_to_user_peer(self):
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._manager.create_conclusion.return_value = True

        result = provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "User prefers dark mode"},
        )

        assert "Conclusion saved for user" in result
        provider._manager.create_conclusion.assert_called_once_with(
            "telegram:123",
            "User prefers dark mode",
            peer="user",
        )

    def test_honcho_conclude_can_target_ai_peer(self):
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._manager.create_conclusion.return_value = True

        result = provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "Assistant likes terse replies", "peer": "ai"},
        )

        assert "Conclusion saved for ai" in result
        provider._manager.create_conclusion.assert_called_once_with(
            "telegram:123",
            "Assistant likes terse replies",
            peer="ai",
        )

    def test_honcho_profile_can_target_explicit_peer_id(self):
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._manager.get_peer_card.return_value = ["Role: Assistant"]

        result = provider.handle_tool_call(
            "honcho_profile",
            {"peer": "hermes"},
        )

        assert "Role: Assistant" in result
        provider._manager.get_peer_card.assert_called_once_with("telegram:123", peer="hermes")

    def test_honcho_search_can_target_explicit_peer_id(self):
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._manager.search_context.return_value = "Assistant self context"

        result = provider.handle_tool_call(
            "honcho_search",
            {"query": "assistant", "peer": "hermes"},
        )

        assert "Assistant self context" in result
        provider._manager.search_context.assert_called_once_with(
            "telegram:123",
            "assistant",
            max_tokens=800,
            peer="hermes",
        )

    def test_honcho_reasoning_can_target_explicit_peer_id(self):
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        provider._manager.dialectic_query.return_value = "Assistant answer"

        result = provider.handle_tool_call(
            "honcho_reasoning",
            {"query": "who are you", "peer": "hermes"},
        )

        assert "Assistant answer" in result
        provider._manager.dialectic_query.assert_called_once_with(
            "telegram:123",
            "who are you",
            reasoning_level=None,
            peer="hermes",
        )

    def test_honcho_conclude_missing_both_params_returns_error(self):
        """Calling honcho_conclude with neither conclusion nor delete_id returns a tool error."""
        import json
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()

        result = provider.handle_tool_call("honcho_conclude", {})

        parsed = json.loads(result)
        assert "error" in parsed or "Missing required" in parsed.get("result", "")
        provider._manager.create_conclusion.assert_not_called()
        provider._manager.delete_conclusion.assert_not_called()


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


class TestPerSessionMigrateGuard:
    """Verify migrate_memory_files is skipped under per-session strategy.

    per-session creates a fresh Honcho session every Hermes run. Uploading
    MEMORY.md/USER.md/SOUL.md to each short-lived session floods the backend
    with duplicate content. The guard was added to prevent orphan sessions
    containing only <prior_memory_file> wrappers.
    """

    def _make_provider_with_strategy(self, strategy, init_on_session_start=True):
        """Create a HonchoMemoryProvider and track migrate_memory_files calls."""
        from plugins.memory.honcho.client import HonchoClientConfig
        from unittest.mock import patch, MagicMock

        cfg = HonchoClientConfig(
            api_key="test-key",
            enabled=True,
            recall_mode="tools",
            init_on_session_start=init_on_session_start,
            session_strategy=strategy,
        )

        provider = HonchoMemoryProvider()

        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []  # empty = new session → triggers migration path
        mock_manager.get_or_create.return_value = mock_session

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-001")

        return provider, mock_manager

    def test_migrate_skipped_for_per_session(self):
        """per-session strategy must NOT call migrate_memory_files."""
        _, mock_manager = self._make_provider_with_strategy("per-session")
        mock_manager.migrate_memory_files.assert_not_called()

    def test_migrate_runs_for_per_directory(self):
        """per-directory strategy with empty session SHOULD call migrate_memory_files."""
        _, mock_manager = self._make_provider_with_strategy("per-directory")
        mock_manager.migrate_memory_files.assert_called_once()


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
# Context token budget enforcement
# ---------------------------------------------------------------------------


class TestTruncateToBudget:
    def test_truncates_oversized_context(self):
        """Text exceeding context_tokens budget is truncated at a word boundary."""
        from plugins.memory.honcho.client import HonchoClientConfig

        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(context_tokens=10)

        long_text = "word " * 200  # ~1000 chars, well over 10*4=40 char budget
        result = provider._truncate_to_budget(long_text)

        assert len(result) <= 50  # budget_chars + ellipsis + word boundary slack
        assert result.endswith(" …")

    def test_no_truncation_within_budget(self):
        """Text within budget passes through unchanged."""
        from plugins.memory.honcho.client import HonchoClientConfig

        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(context_tokens=1000)

        short_text = "Name: Robert, Location: Melbourne"
        assert provider._truncate_to_budget(short_text) == short_text

    def test_no_truncation_when_context_tokens_none(self):
        """When context_tokens is None (explicit opt-out), no truncation."""
        from plugins.memory.honcho.client import HonchoClientConfig

        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(context_tokens=None)

        long_text = "word " * 500
        assert provider._truncate_to_budget(long_text) == long_text

    def test_context_tokens_cap_bounds_prefetch(self):
        """With an explicit token budget, oversized prefetch is bounded."""
        from plugins.memory.honcho.client import HonchoClientConfig

        provider = HonchoMemoryProvider()
        provider._config = HonchoClientConfig(context_tokens=1200)

        # Simulate a massive representation (10k chars)
        huge_text = "x" * 10000
        result = provider._truncate_to_budget(huge_text)

        # 1200 tokens * 4 chars = 4800 chars + " …"
        assert len(result) <= 4805


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


# ---------------------------------------------------------------------------


class TestDialecticCadenceDefaults:
    """Regression tests for dialectic_cadence default value."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        """Create a HonchoMemoryProvider with mocked dependencies."""
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True, recall_mode="hybrid")
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-001")

        return provider

    def test_default_is_3(self):
        """Default dialectic_cadence should be 3 to avoid per-turn LLM calls."""
        provider = self._make_provider()
        assert provider._dialectic_cadence == 3

    def test_config_override(self):
        """dialecticCadence from config overrides the default."""
        provider = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 5}})
        assert provider._dialectic_cadence == 5


class TestBaseContextSummary:
    """Base context injection should include session summary when available."""

    def test_format_includes_summary(self):
        """Session summary should appear first in the formatted context."""
        provider = HonchoMemoryProvider()
        ctx = {
            "summary": "Testing Honcho tools and dialectic depth.",
            "representation": "Eri is a developer.",
            "card": "Name: Eri Barrett",
        }
        formatted = provider._format_first_turn_context(ctx)
        assert "## Session Summary" in formatted
        assert formatted.index("Session Summary") < formatted.index("User Representation")

    def test_format_without_summary(self):
        """No summary key means no summary section."""
        provider = HonchoMemoryProvider()
        ctx = {"representation": "Eri is a developer.", "card": "Name: Eri"}
        formatted = provider._format_first_turn_context(ctx)
        assert "Session Summary" not in formatted
        assert "User Representation" in formatted

    def test_format_empty_summary_skipped(self):
        """Empty summary string should not produce a section."""
        provider = HonchoMemoryProvider()
        ctx = {"summary": "", "representation": "rep", "card": "card"}
        formatted = provider._format_first_turn_context(ctx)
        assert "Session Summary" not in formatted


class TestDialecticDepth:
    """Tests for the dialecticDepth multi-pass system."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True, recall_mode="hybrid")
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-001")

        return provider

    def test_default_depth_is_1(self):
        """Default dialecticDepth should be 1 — single .chat() call."""
        provider = self._make_provider()
        assert provider._dialectic_depth == 1

    def test_depth_from_config(self):
        """dialecticDepth from config sets the depth."""
        provider = self._make_provider(cfg_extra={"dialectic_depth": 2})
        assert provider._dialectic_depth == 2

    def test_depth_clamped_to_3(self):
        """dialecticDepth > 3 gets clamped to 3."""
        provider = self._make_provider(cfg_extra={"dialectic_depth": 7})
        assert provider._dialectic_depth == 3

    def test_depth_clamped_to_1(self):
        """dialecticDepth < 1 gets clamped to 1."""
        provider = self._make_provider(cfg_extra={"dialectic_depth": 0})
        assert provider._dialectic_depth == 1

    def test_depth_levels_from_config(self):
        """dialecticDepthLevels array is read from config."""
        provider = self._make_provider(cfg_extra={
            "dialectic_depth": 2,
            "dialectic_depth_levels": ["minimal", "high"],
        })
        assert provider._dialectic_depth_levels == ["minimal", "high"]

    def test_depth_levels_none_by_default(self):
        """When dialecticDepthLevels is not configured, it's None."""
        provider = self._make_provider()
        assert provider._dialectic_depth_levels is None

    def test_resolve_pass_level_uses_depth_levels(self):
        """Per-pass levels from dialecticDepthLevels override proportional."""
        provider = self._make_provider(cfg_extra={
            "dialectic_depth": 2,
            "dialectic_depth_levels": ["minimal", "high"],
        })
        assert provider._resolve_pass_level(0) == "minimal"
        assert provider._resolve_pass_level(1) == "high"

    def test_resolve_pass_level_proportional_depth_1(self):
        """Depth 1 pass 0 uses the base reasoning level."""
        provider = self._make_provider(cfg_extra={
            "dialectic_depth": 1,
            "dialectic_reasoning_level": "medium",
        })
        assert provider._resolve_pass_level(0) == "medium"

    def test_resolve_pass_level_proportional_depth_2(self):
        """Depth 2: pass 0 is minimal, pass 1 is base level."""
        provider = self._make_provider(cfg_extra={
            "dialectic_depth": 2,
            "dialectic_reasoning_level": "high",
        })
        assert provider._resolve_pass_level(0) == "minimal"
        assert provider._resolve_pass_level(1) == "high"

    def test_cold_start_prompt(self):
        """Cold start (no base context) uses general user query."""
        provider = self._make_provider()
        prompt = provider._build_dialectic_prompt(0, [], is_cold=True)
        assert "preferences" in prompt.lower()
        assert "session" not in prompt.lower()

    def test_warm_session_prompt(self):
        """Warm session (has context) uses session-scoped query."""
        provider = self._make_provider()
        prompt = provider._build_dialectic_prompt(0, [], is_cold=False)
        assert "session" in prompt.lower()
        assert "current conversation" in prompt.lower()

    def test_signal_sufficient_short_response(self):
        """Short responses are not sufficient signal."""
        assert not HonchoMemoryProvider._signal_sufficient("ok")
        assert not HonchoMemoryProvider._signal_sufficient("")
        assert not HonchoMemoryProvider._signal_sufficient(None)

    def test_signal_sufficient_structured_response(self):
        """Structured responses with bullets/headers are sufficient."""
        result = "## Current State\n- Working on Honcho PR\n- Testing dialectic depth\n" + "x" * 50
        assert HonchoMemoryProvider._signal_sufficient(result)

    def test_signal_sufficient_long_unstructured(self):
        """Long responses are sufficient even without structure."""
        assert HonchoMemoryProvider._signal_sufficient("a" * 301)

    def test_run_dialectic_depth_single_pass(self):
        """Depth 1 makes exactly one .chat() call."""
        from unittest.mock import MagicMock
        provider = self._make_provider(cfg_extra={"dialectic_depth": 1})
        provider._manager = MagicMock()
        provider._manager.dialectic_query.return_value = "user prefers zero-fluff"
        provider._session_key = "test"
        provider._base_context_cache = None  # cold start

        result = provider._run_dialectic_depth("hello")
        assert result == "user prefers zero-fluff"
        assert provider._manager.dialectic_query.call_count == 1

    def test_run_dialectic_depth_two_passes(self):
        """Depth 2 makes two .chat() calls when pass 1 signal is weak."""
        from unittest.mock import MagicMock
        provider = self._make_provider(cfg_extra={"dialectic_depth": 2})
        provider._manager = MagicMock()
        provider._manager.dialectic_query.side_effect = [
            "thin response",  # pass 0: weak signal
            "## Synthesis\n- Grounded in evidence\n- Current PR work\n" + "x" * 100,  # pass 1: strong
        ]
        provider._session_key = "test"
        provider._base_context_cache = "existing context"

        result = provider._run_dialectic_depth("test query")
        assert provider._manager.dialectic_query.call_count == 2
        assert "Synthesis" in result

    def test_first_turn_runs_dialectic_synchronously(self):
        """First turn should fire the dialectic synchronously (cold start)."""
        from unittest.mock import MagicMock, patch
        provider = self._make_provider(cfg_extra={"dialectic_depth": 1})
        provider._manager = MagicMock()
        provider._manager.dialectic_query.return_value = "cold start synthesis"
        provider._manager.get_prefetch_context.return_value = None
        provider._manager.pop_context_result.return_value = None
        provider._session_key = "test"
        provider._base_context_cache = ""  # cold start
        provider._last_dialectic_turn = -999  # never fired

        result = provider.prefetch("hello world")
        assert "cold start synthesis" in result
        assert provider._manager.dialectic_query.call_count == 1
        # After first-turn sync, _last_dialectic_turn should be updated
        assert provider._last_dialectic_turn != -999

    def test_first_turn_dialectic_does_not_double_fire(self):
        """After first-turn sync dialectic, queue_prefetch should skip (cadence)."""
        from unittest.mock import MagicMock
        provider = self._make_provider(cfg_extra={"dialectic_depth": 1})
        provider._manager = MagicMock()
        provider._manager.dialectic_query.return_value = "cold start synthesis"
        provider._manager.get_prefetch_context.return_value = None
        provider._manager.pop_context_result.return_value = None
        provider._session_key = "test"
        provider._base_context_cache = ""
        provider._last_dialectic_turn = -999
        provider._turn_count = 0

        # First turn fires sync dialectic
        provider.prefetch("hello")
        assert provider._manager.dialectic_query.call_count == 1

        # Now queue_prefetch on same turn should skip (cadence: 0 - 0 < 3)
        provider._manager.dialectic_query.reset_mock()
        provider.queue_prefetch("hello")
        assert provider._manager.dialectic_query.call_count == 0

    def test_run_dialectic_depth_bails_early_on_strong_signal(self):
        """Depth 2 skips pass 1 when pass 0 returns strong signal."""
        from unittest.mock import MagicMock
        provider = self._make_provider(cfg_extra={"dialectic_depth": 2})
        provider._manager = MagicMock()
        provider._manager.dialectic_query.return_value = (
            "## Full Assessment\n- Strong structured response\n- With evidence\n" + "x" * 200
        )
        provider._session_key = "test"
        provider._base_context_cache = "existing context"

        result = provider._run_dialectic_depth("test query")
        # Only 1 call because pass 0 had sufficient signal
        assert provider._manager.dialectic_query.call_count == 1


# ---------------------------------------------------------------------------
# set_peer_card None guard
# ---------------------------------------------------------------------------


class TestSetPeerCardNoneGuard:
    """set_peer_card must return None (not raise) when peer ID cannot be resolved."""

    def _make_manager(self):
        from plugins.memory.honcho.client import HonchoClientConfig
        from plugins.memory.honcho.session import HonchoSessionManager

        cfg = HonchoClientConfig(api_key="test-key", enabled=True)
        mgr = HonchoSessionManager.__new__(HonchoSessionManager)
        mgr._cache = {}
        mgr._sessions_cache = {}
        mgr._config = cfg
        return mgr

    def test_returns_none_when_peer_resolves_to_none(self):
        """set_peer_card returns None when _resolve_peer_id returns None."""
        from unittest.mock import patch
        mgr = self._make_manager()

        session = HonchoSession(
            key="test",
            honcho_session_id="sid",
            user_peer_id="user-peer",
            assistant_peer_id="ai-peer",
        )
        mgr._cache["test"] = session

        with patch.object(mgr, "_resolve_peer_id", return_value=None):
            result = mgr.set_peer_card("test", ["fact 1", "fact 2"], peer="ghost")

        assert result is None

    def test_returns_none_when_session_missing(self):
        """set_peer_card returns None when session key is not in cache."""
        mgr = self._make_manager()
        result = mgr.set_peer_card("nonexistent", ["fact"], peer="user")
        assert result is None


# ---------------------------------------------------------------------------
# get_session_context cache-miss fallback respects peer param
# ---------------------------------------------------------------------------


class TestGetSessionContextFallback:
    """get_session_context fallback must honour the peer param when honcho_session is absent."""

    def _make_manager_with_session(self, user_peer_id="user-peer", assistant_peer_id="ai-peer"):
        from plugins.memory.honcho.client import HonchoClientConfig
        from plugins.memory.honcho.session import HonchoSessionManager

        cfg = HonchoClientConfig(api_key="test-key", enabled=True)
        mgr = HonchoSessionManager.__new__(HonchoSessionManager)
        mgr._cache = {}
        mgr._sessions_cache = {}
        mgr._config = cfg
        mgr._dialectic_dynamic = True
        mgr._dialectic_reasoning_level = "low"
        mgr._dialectic_max_input_chars = 10000
        mgr._ai_observe_others = True

        session = HonchoSession(
            key="test",
            honcho_session_id="sid-missing-from-sessions-cache",
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
        )
        mgr._cache["test"] = session
        # Deliberately NOT adding to _sessions_cache to trigger fallback path
        return mgr

    def test_fallback_uses_user_peer_for_user(self):
        """On cache miss, peer='user' fetches user peer context."""
        mgr = self._make_manager_with_session()
        fetch_calls = []

        def _fake_fetch(peer_id, search_query=None, *, target=None):
            fetch_calls.append((peer_id, target))
            return {"representation": "user rep", "card": []}

        mgr._fetch_peer_context = _fake_fetch

        mgr.get_session_context("test", peer="user")

        assert len(fetch_calls) == 1
        peer_id, target = fetch_calls[0]
        assert peer_id == "user-peer"
        assert target == "user-peer"

    def test_fallback_uses_ai_peer_for_ai(self):
        """On cache miss, peer='ai' fetches assistant peer context, not user."""
        mgr = self._make_manager_with_session()
        fetch_calls = []

        def _fake_fetch(peer_id, search_query=None, *, target=None):
            fetch_calls.append((peer_id, target))
            return {"representation": "ai rep", "card": []}

        mgr._fetch_peer_context = _fake_fetch

        mgr.get_session_context("test", peer="ai")

        assert len(fetch_calls) == 1
        peer_id, target = fetch_calls[0]
        assert peer_id == "ai-peer", f"expected ai-peer, got {peer_id}"
        assert target == "ai-peer"
