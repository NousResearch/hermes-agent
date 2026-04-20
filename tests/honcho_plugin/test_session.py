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
    def test_conclude_schema_has_no_anyof(self):
        """anyOf/oneOf/allOf breaks Anthropic and Fireworks APIs — schema must be plain object."""
        from plugins.memory.honcho import CONCLUDE_SCHEMA
        params = CONCLUDE_SCHEMA["parameters"]
        assert params["type"] == "object"
        assert "conclusion" in params["properties"]
        assert "delete_id" in params["properties"]
        assert "anyOf" not in params
        assert "oneOf" not in params
        assert "allOf" not in params

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
        assert parsed == {"error": "Exactly one of conclusion or delete_id must be provided."}
        provider._manager.create_conclusion.assert_not_called()
        provider._manager.delete_conclusion.assert_not_called()

    def test_honcho_conclude_rejects_both_params_at_once(self):
        """Sending both conclusion and delete_id should be rejected."""
        import json
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        result = provider.handle_tool_call(
            "honcho_conclude",
            {"conclusion": "User prefers dark mode", "delete_id": "conc-123"},
        )
        parsed = json.loads(result)
        assert parsed == {"error": "Exactly one of conclusion or delete_id must be provided."}
        provider._manager.create_conclusion.assert_not_called()
        provider._manager.delete_conclusion.assert_not_called()

    def test_honcho_conclude_rejects_whitespace_only_conclusion(self):
        """Whitespace-only conclusion should be treated as empty."""
        import json
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        result = provider.handle_tool_call("honcho_conclude", {"conclusion": "   "})
        parsed = json.loads(result)
        assert parsed == {"error": "Exactly one of conclusion or delete_id must be provided."}
        provider._manager.create_conclusion.assert_not_called()

    def test_honcho_conclude_rejects_whitespace_only_delete_id(self):
        """Whitespace-only delete_id should be treated as empty."""
        import json
        provider = HonchoMemoryProvider()
        provider._session_initialized = True
        provider._session_key = "telegram:123"
        provider._manager = MagicMock()
        result = provider.handle_tool_call("honcho_conclude", {"delete_id": "  "})
        parsed = json.loads(result)
        assert parsed == {"error": "Exactly one of conclusion or delete_id must be provided."}
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
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager) as mock_manager_cls, \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-001", **init_kwargs)

        return provider, cfg, mock_manager_cls

    def test_tools_lazy_default(self):
        """tools + initOnSessionStart=false → session NOT initialized after initialize()."""
        provider, _, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=False,
        )
        assert provider._session_initialized is False
        assert provider._manager is None
        assert provider._lazy_init_kwargs is not None

    def test_tools_eager_init(self):
        """tools + initOnSessionStart=true → session IS initialized after initialize()."""
        provider, _, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
        )
        assert provider._session_initialized is True
        assert provider._manager is not None

    def test_tools_eager_prefetch_still_empty(self):
        """tools mode with eager init still returns empty from prefetch() (no auto-injection)."""
        provider, _, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
        )
        assert provider.prefetch("test query") == ""

    def test_tools_lazy_prefetch_empty(self):
        """tools mode with lazy init also returns empty from prefetch()."""
        provider, _, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=False,
        )
        assert provider.prefetch("test query") == ""

    def test_explicit_peer_name_not_overridden_by_user_id(self):
        """Explicit peerName in config must not be replaced by gateway user_id."""
        _, cfg, _ = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
            peer_name="Kathie", user_id="8439114563",
        )
        assert cfg.peer_name == "Kathie"

    def test_user_id_used_when_no_peer_name(self):
        """Gateway user_id is passed separately from config peer_name."""
        _, cfg, mock_manager_cls = self._make_provider_with_config(
            recall_mode="tools", init_on_session_start=True,
            peer_name=None, user_id="8439114563",
        )
        assert cfg.peer_name is None
        assert mock_manager_cls.call_args.kwargs["runtime_user_peer_name"] == "8439114563"


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


def _settle_prewarm(provider):
    """Wait for the session-start prewarm dialectic thread, then return the
    provider to a clean 'nothing fired yet' state so cadence/first-turn/
    trivial-prompt tests can assert from a known baseline."""
    if provider._prefetch_thread:
        provider._prefetch_thread.join(timeout=3.0)
    with provider._prefetch_lock:
        provider._prefetch_result = ""
        provider._prefetch_result_fired_at = -999
    provider._prefetch_thread = None
    provider._prefetch_thread_started_at = 0.0
    provider._last_dialectic_turn = -999
    provider._dialectic_empty_streak = 0
    if getattr(provider, "_manager", None) is not None:
        try:
            provider._manager.dialectic_query.reset_mock()
            provider._manager.prefetch_context.reset_mock()
        except AttributeError:
            pass


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

        _settle_prewarm(provider)
        return provider

    def test_unset_falls_back_to_1(self):
        """Unset dialecticCadence falls back to 1 (every turn) for backwards
        compatibility with existing configs that predate the setting. The
        setup wizard writes 2 explicitly on new configs."""
        provider = self._make_provider()
        assert provider._dialectic_cadence == 1

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

        _settle_prewarm(provider)
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
# Trivial-prompt heuristic + dialectic cadence silent-failure guards
# ---------------------------------------------------------------------------


class TestTrivialPromptHeuristic:
    """Trivial prompts ('ok', 'y', slash commands) must short-circuit injection."""

    @staticmethod
    def _make_provider():
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(api_key="test-key", enabled=True, recall_mode="hybrid")
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-trivial")
        _settle_prewarm(provider)
        return provider

    def test_classifier_catches_common_trivial_forms(self):
        for t in ("ok", "OK", " ok ", "y", "yes", "sure", "thanks", "lgtm", "/help", "", "   "):
            assert HonchoMemoryProvider._is_trivial_prompt(t), f"expected trivial: {t!r}"

    def test_classifier_lets_substantive_prompts_through(self):
        for t in ("hello world", "what's my name", "explain this", "ok so what's next"):
            assert not HonchoMemoryProvider._is_trivial_prompt(t), f"expected non-trivial: {t!r}"

    def test_prefetch_skips_on_trivial_prompt(self):
        provider = self._make_provider()
        provider._session_key = "test"
        provider._base_context_cache = "cached base"
        provider._last_dialectic_turn = 0
        provider._turn_count = 5

        assert provider.prefetch("ok") == ""
        assert provider.prefetch("/help") == ""
        # Dialectic should not have fired
        assert provider._manager.dialectic_query.call_count == 0

    def test_queue_prefetch_skips_on_trivial_prompt(self):
        provider = self._make_provider()
        provider._session_key = "test"
        provider._turn_count = 10
        provider._last_dialectic_turn = -999  # would otherwise fire
        # initialize() pre-warms; clear call counts before the assertion.
        provider._manager.prefetch_context.reset_mock()
        provider._manager.dialectic_query.reset_mock()

        provider.queue_prefetch("y")
        # Trivial prompts short-circuit both context refresh and dialectic fire.
        assert provider._manager.prefetch_context.call_count == 0
        assert provider._manager.dialectic_query.call_count == 0


class TestDialecticCadenceAdvancesOnSuccess:
    """Cadence tracker advances only when the dialectic call returns a
    non-empty result. Empty results (transient API error, sparse representation)
    must retry on the next eligible turn instead of waiting the full cadence."""

    @staticmethod
    def _make_provider():
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig(
            api_key="test-key", enabled=True, recall_mode="hybrid", dialectic_depth=1,
        )
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-session-retry")
        _settle_prewarm(provider)
        return provider

    def test_empty_dialectic_result_does_not_advance_cadence(self):
        import time as _time
        provider = self._make_provider()
        provider._session_key = "test"
        provider._manager.dialectic_query.return_value = ""  # silent failure
        provider._turn_count = 5
        provider._last_dialectic_turn = 0  # would fire (5 - 0 = 5 ≥ 3)

        provider.queue_prefetch("hello")
        # wait for the background thread to settle
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)

        # Dialectic call was attempted
        assert provider._manager.dialectic_query.call_count == 1
        # But cadence tracker did NOT advance — next turn should retry
        assert provider._last_dialectic_turn == 0

    def test_non_empty_dialectic_result_advances_cadence(self):
        provider = self._make_provider()
        provider._session_key = "test"
        provider._manager.dialectic_query.return_value = "real synthesis output"
        provider._turn_count = 5
        provider._last_dialectic_turn = 0

        provider.queue_prefetch("hello")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=2.0)

        assert provider._last_dialectic_turn == 5

    def test_in_flight_thread_is_not_stacked(self):
        import threading as _threading
        import time as _time
        provider = self._make_provider()
        provider._session_key = "test"
        provider._turn_count = 10
        provider._last_dialectic_turn = 0

        # Simulate a prior thread still running (fresh, not stale)
        hold = _threading.Event()

        def _block():
            hold.wait(timeout=5.0)

        fresh = _threading.Thread(target=_block, daemon=True)
        fresh.start()
        provider._prefetch_thread = fresh
        provider._prefetch_thread_started_at = _time.monotonic()  # fresh start

        provider.queue_prefetch("hello")
        # Should have short-circuited — no new dialectic call
        assert provider._manager.dialectic_query.call_count == 0
        hold.set()
        fresh.join(timeout=2.0)


class TestSessionStartDialecticPrewarm:
    """Session-start prewarm fires a depth-aware dialectic whose result is
    consumed by turn 1 — no duplicate .chat() and no dead-cache orphaning."""

    @staticmethod
    def _make_provider(cfg_extra=None, dialectic_result="prewarm synthesis"):
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True, recall_mode="hybrid")
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None
        mock_manager.dialectic_query.return_value = dialectic_result

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-prewarm")
        return provider

    def test_prewarm_populates_prefetch_result(self):
        p = self._make_provider()
        # Wait for prewarm thread to land
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=3.0)
        with p._prefetch_lock:
            assert p._prefetch_result == "prewarm synthesis"
        assert p._last_dialectic_turn == 0

    def test_turn1_consumes_prewarm_without_duplicate_dialectic(self):
        """With prewarm result already in _prefetch_result, turn 1 prefetch
        should NOT fire another dialectic."""
        p = self._make_provider()
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=3.0)
        p._manager.dialectic_query.reset_mock()
        p._session_key = "test-prewarm"
        p._base_context_cache = ""
        p._turn_count = 1

        result = p.prefetch("hello world")
        assert "prewarm synthesis" in result
        # The sync first-turn path must NOT have fired another .chat()
        assert p._manager.dialectic_query.call_count == 0

    def test_turn1_falls_back_to_sync_when_prewarm_missing(self):
        """If the prewarm produced nothing (empty graph, API blip), turn 1
        still fires its own sync dialectic."""
        p = self._make_provider(dialectic_result="")  # prewarm returns empty
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=3.0)
        with p._prefetch_lock:
            assert p._prefetch_result == ""  # prewarm landed nothing
        # Switch dialectic_query to return something on the sync first-turn call
        p._manager.dialectic_query.return_value = "sync recovery"
        p._manager.dialectic_query.reset_mock()
        p._session_key = "test-prewarm"
        p._base_context_cache = ""
        p._turn_count = 1

        result = p.prefetch("hello world")
        assert "sync recovery" in result
        assert p._manager.dialectic_query.call_count == 1


class TestDialecticLiveness:
    """Liveness + observability: stale-thread recovery, stale-result discard,
    empty-streak backoff, and the snapshot method used for diagnostics."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(api_key="test-key", enabled=True, recall_mode="hybrid", timeout=2.0)
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None
        mock_manager.dialectic_query.return_value = ""  # default: silent

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-liveness")
        _settle_prewarm(provider)
        return provider

    def test_stale_thread_is_treated_as_dead(self):
        """A thread older than timeout × multiplier no longer blocks new fires."""
        import threading as _threading
        p = self._make_provider()
        p._session_key = "test"
        p._turn_count = 10
        p._last_dialectic_turn = 0
        p._manager.dialectic_query.return_value = "fresh synthesis"

        # Plant an alive thread with an old timestamp (stale)
        hold = _threading.Event()
        stuck = _threading.Thread(target=lambda: hold.wait(timeout=10.0), daemon=True)
        stuck.start()
        p._prefetch_thread = stuck
        # timeout=2.0, multiplier=2.0, so anything older than 4s is stale
        p._prefetch_thread_started_at = 0.0  # very old (1970 monotonic baseline)

        p.queue_prefetch("hello")
        # New thread should have been spawned since stuck one is stale
        assert p._prefetch_thread is not stuck, "stale thread must be recycled"
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=2.0)
        assert p._manager.dialectic_query.call_count == 1
        hold.set()
        stuck.join(timeout=2.0)

    def test_stale_pending_result_is_discarded_on_read(self):
        """A pending dialectic result from many turns ago is discarded
        instead of injected against a fresh conversational pivot."""
        p = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 2}})
        p._session_key = "test"
        p._base_context_cache = "base ctx"
        with p._prefetch_lock:
            p._prefetch_result = "ancient synthesis"
            p._prefetch_result_fired_at = 1
        # cadence=2, multiplier=2 → stale after 4 turns since fire
        p._turn_count = 10
        p._last_dialectic_turn = 1  # prevents sync first-turn path

        result = p.prefetch("what's new")
        assert "ancient synthesis" not in result, "stale pending must be discarded"
        # Cache slot cleared
        with p._prefetch_lock:
            assert p._prefetch_result == ""
            assert p._prefetch_result_fired_at == -999

    def test_fresh_pending_result_is_kept(self):
        """A pending result within the staleness window is injected normally."""
        p = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 3}})
        p._session_key = "test"
        p._base_context_cache = ""
        with p._prefetch_lock:
            p._prefetch_result = "recent synthesis"
            p._prefetch_result_fired_at = 8
        p._turn_count = 9  # 1 turn since fire, well within cadence × 2 = 6
        p._last_dialectic_turn = 8

        result = p.prefetch("what's new")
        assert "recent synthesis" in result

    def test_empty_streak_widens_effective_cadence(self):
        """After N empty returns, the gate waits cadence + N turns."""
        p = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 1}})
        p._dialectic_empty_streak = 3
        # cadence=1, streak=3 → effective = 4
        assert p._effective_cadence() == 4

    def test_backoff_is_capped(self):
        """Effective cadence is capped at cadence × _BACKOFF_MAX."""
        p = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 2}})
        p._dialectic_empty_streak = 100
        # cadence=2, ceiling = 2 × 8 = 16
        assert p._effective_cadence() == 16

    def test_success_resets_empty_streak(self):
        """A non-empty result zeroes the streak so healthy operation restores
        the base cadence immediately."""
        p = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 1}})
        p._session_key = "test"
        p._dialectic_empty_streak = 5
        p._turn_count = 10
        p._last_dialectic_turn = 0
        p._manager.dialectic_query.return_value = "real output"

        p.queue_prefetch("hello")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=2.0)
        assert p._dialectic_empty_streak == 0
        assert p._last_dialectic_turn == 10

    def test_empty_result_increments_streak(self):
        p = self._make_provider(cfg_extra={"raw": {"dialecticCadence": 1}})
        p._session_key = "test"
        p._turn_count = 5
        p._last_dialectic_turn = 0
        p._manager.dialectic_query.return_value = ""  # empty

        p.queue_prefetch("hello")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=2.0)
        assert p._dialectic_empty_streak == 1
        assert p._last_dialectic_turn == 0  # cadence not advanced

    def test_liveness_snapshot_shape(self):
        p = self._make_provider()
        snap = p.liveness_snapshot()
        for key in (
            "turn_count", "last_dialectic_turn", "pending_result_fired_at",
            "empty_streak", "effective_cadence", "thread_alive", "thread_age_seconds",
        ):
            assert key in snap


class TestDialecticLifecycleSmoke:
    """End-to-end smoke walking a multi-turn session through prewarm,
    turn 1 consume, trivial skip, cadence fire, empty-result retry,
    heuristic bump, and session-end flush."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(
            api_key="test-key", enabled=True, recall_mode="hybrid",
            dialectic_reasoning_level="low", reasoning_heuristic=True,
            reasoning_level_cap="high", dialectic_depth=1,
        )
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.messages = []
        mock_manager.get_or_create.return_value = mock_session
        mock_manager.get_prefetch_context.return_value = None
        mock_manager.pop_context_result.return_value = None

        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            return provider, mock_manager, cfg

    def _await_thread(self, provider):
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=3.0)

    def test_full_multi_turn_session(self):
        """Walks init → turns 1..8 → session end. Asserts at every step that
        the plugin did exactly what it should and nothing more.

        Uses dialecticCadence=3 so we can exercise skip-turns between fires
        and the silent-failure retry path without their gates tripping each
        other. Trivial + slash skips apply independent of cadence.
        """
        from unittest.mock import patch, MagicMock
        provider, mgr, cfg = self._make_provider(
            cfg_extra={"raw": {"dialecticCadence": 3}}
        )

        # Program the dialectic responses in the exact order they'll be requested.
        # An extra or missing call fails the test — strong smoke signal.
        responses = iter([
            "prewarm: user is eri, works on hermes",      # session-start prewarm
            "cadence fire: long query synthesis",         # turn 4 queue_prefetch
            "",                                           # turn 7 fire: silent failure
            "retry success: fresh synthesis",             # turn 8 queue_prefetch retry
        ])
        mgr.dialectic_query.side_effect = lambda *a, **kw: next(responses)

        # ---- init: prewarm fires ----
        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mgr), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="smoke-test")

        self._await_thread(provider)
        with provider._prefetch_lock:
            assert provider._prefetch_result.startswith("prewarm"), \
                "session-start prewarm must land in _prefetch_result"
        assert provider._last_dialectic_turn == 0, "prewarm marks turn 0"
        assert mgr.dialectic_query.call_count == 1

        # ---- turn 1: consume prewarm, no duplicate dialectic ----
        provider.on_turn_start(1, "hey")
        inject1 = provider.prefetch("hey")
        assert "prewarm" in inject1, "turn 1 must surface prewarm"
        provider.sync_turn("hey", "hi there")
        provider.queue_prefetch("hey")  # cadence gate: (1-0)<3 → skip
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 1, \
            "turn 1 must not fire — prewarm covered it and cadence skips"

        # ---- turn 2: trivial 'ok' → skip everything ----
        mgr.prefetch_context.reset_mock()
        provider.on_turn_start(2, "ok")
        assert provider.prefetch("ok") == "", "trivial prompt must short-circuit injection"
        provider.sync_turn("ok", "cool")
        provider.queue_prefetch("ok")
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 1, "trivial must not fire dialectic"
        assert mgr.prefetch_context.call_count == 0, "trivial must not fire context refresh"

        # ---- turn 3: slash '/help' → also skip ----
        provider.on_turn_start(3, "/help")
        assert provider.prefetch("/help") == ""
        provider.queue_prefetch("/help")
        assert mgr.dialectic_query.call_count == 1

        # ---- turn 4: long query → cadence fires + heuristic bumps ----
        long_q = "walk me through " + ("x " * 100)  # ~200 chars → heuristic +1
        provider.on_turn_start(4, long_q)
        provider.prefetch(long_q)
        provider.sync_turn(long_q, "sure")
        provider.queue_prefetch(long_q)  # (4-0)≥3 → fires
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 2, "turn 4 cadence fire"
        _, kwargs = mgr.dialectic_query.call_args
        assert kwargs.get("reasoning_level") in ("medium", "high"), \
            f"long query must bump reasoning level above 'low'; got {kwargs.get('reasoning_level')}"
        assert provider._last_dialectic_turn == 4, "cadence tracker advances on success"

        # ---- turns 5–6: cadence cooldown, no fires ----
        for t in (5, 6):
            provider.on_turn_start(t, "tell me more")
            provider.queue_prefetch("tell me more")
            self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 2, "turns 5–6 blocked by cadence window"

        # ---- turn 7: fires but silent failure (empty dialectic) ----
        provider.on_turn_start(7, "and then what")
        provider.queue_prefetch("and then what")  # (7-4)≥3 → fires
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 3, "turn 7 fires"
        assert provider._last_dialectic_turn == 4, \
            "silent failure must NOT burn the cadence window"

        # ---- turn 8: retries because cadence didn't advance ----
        provider.on_turn_start(8, "try again")
        provider.queue_prefetch("try again")  # (8-4)≥3 → fires again
        self._await_thread(provider)
        assert mgr.dialectic_query.call_count == 4, \
            "turn 8 retries because turn 7's empty result didn't advance cadence"
        assert provider._last_dialectic_turn == 8, "retry success advances"

        # ---- session end: flush messages ----
        provider.on_session_end([])
        mgr.flush_all.assert_called()


class TestReasoningHeuristic:
    """Char-count heuristic that scales the auto-injected reasoning level by
    query length, clamped at reasoning_level_cap."""

    @staticmethod
    def _make_provider(cfg_extra=None):
        from unittest.mock import patch, MagicMock
        from plugins.memory.honcho.client import HonchoClientConfig

        defaults = dict(
            api_key="test-key", enabled=True, recall_mode="hybrid",
            dialectic_reasoning_level="low", reasoning_heuristic=True,
            reasoning_level_cap="high",
        )
        if cfg_extra:
            defaults.update(cfg_extra)
        cfg = HonchoClientConfig(**defaults)
        provider = HonchoMemoryProvider()
        mock_manager = MagicMock()
        mock_manager.get_or_create.return_value = MagicMock(messages=[])
        with patch("plugins.memory.honcho.client.HonchoClientConfig.from_global_config", return_value=cfg), \
             patch("plugins.memory.honcho.client.get_honcho_client", return_value=MagicMock()), \
             patch("plugins.memory.honcho.session.HonchoSessionManager", return_value=mock_manager), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            provider.initialize(session_id="test-heuristic")
        _settle_prewarm(provider)
        return provider

    def test_short_query_stays_at_base(self):
        p = self._make_provider()
        assert p._apply_reasoning_heuristic("low", "hey") == "low"

    def test_medium_query_bumps_one_level(self):
        p = self._make_provider()
        q = "x" * 150
        assert p._apply_reasoning_heuristic("low", q) == "medium"

    def test_long_query_bumps_two_levels(self):
        p = self._make_provider()
        q = "x" * 500
        assert p._apply_reasoning_heuristic("low", q) == "high"

    def test_bump_respects_cap(self):
        p = self._make_provider(cfg_extra={"reasoning_level_cap": "medium"})
        q = "x" * 500  # would hit 'high' without the cap
        assert p._apply_reasoning_heuristic("low", q) == "medium"

    def test_max_never_auto_selected_with_default_cap(self):
        p = self._make_provider(cfg_extra={"dialectic_reasoning_level": "high"})
        q = "x" * 500  # base=high, bump would push to 'max'
        assert p._apply_reasoning_heuristic("high", q) == "high"

    def test_heuristic_disabled_returns_base(self):
        p = self._make_provider(cfg_extra={"reasoning_heuristic": False})
        q = "x" * 500
        assert p._apply_reasoning_heuristic("low", q) == "low"

    def test_resolve_pass_level_applies_heuristic_at_base_mapping(self):
        """Depth=1, pass 0 maps to 'base' → heuristic applies."""
        p = self._make_provider()
        q = "x" * 150
        assert p._resolve_pass_level(0, query=q) == "medium"

    def test_resolve_pass_level_does_not_touch_explicit_per_pass(self):
        """dialecticDepthLevels wins absolutely — no heuristic scaling."""
        p = self._make_provider(cfg_extra={"dialectic_depth_levels": ["minimal"]})
        q = "x" * 500  # heuristic would otherwise bump to 'high'
        assert p._resolve_pass_level(0, query=q) == "minimal"

    def test_resolve_pass_level_does_not_touch_lighter_passes(self):
        """Depth 3 pass 0 is hardcoded 'minimal' — heuristic must not bump it."""
        p = self._make_provider(cfg_extra={"dialectic_depth": 3})
        q = "x" * 500
        assert p._resolve_pass_level(0, query=q) == "minimal"
        # But the 'base' pass (idx 1 for depth 3) does get heuristic
        assert p._resolve_pass_level(1, query=q) == "high"


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
