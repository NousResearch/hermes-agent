"""Tests for Mem0 v3 API — new tool names, paginated responses, update/delete tools."""

import json
import time
import pytest

import plugins.memory.mem0 as mem0_plugin
from plugins.memory.mem0 import Mem0MemoryProvider


class FakeBackend:
    """Fake Mem0Backend for provider-level tests."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or []
        self._all_results = all_results or {"results": [], "count": 0}
        self.captured = []

    def search(self, query, *, filters, top_k=10, rerank=True):
        self.captured.append(("search", query, {"filters": filters, "top_k": top_k, "rerank": rerank}))
        return self._search_results

    def get_all(self, *, filters, page=1, page_size=100):
        self.captured.append(("get_all", {"filters": filters, "page": page, "page_size": page_size}))
        return self._all_results

    def add(self, messages, *, user_id, agent_id, infer=False, metadata=None):
        self.captured.append((
            "add",
            messages,
            {"user_id": user_id, "agent_id": agent_id, "infer": infer, "metadata": metadata},
        ))
        return {"status": "PENDING", "event_id": "evt-test-123"}

    def update(self, memory_id, text):
        self.captured.append(("update", memory_id, text))
        return {"result": "Memory updated.", "memory_id": memory_id}

    def delete(self, memory_id):
        self.captured.append(("delete", memory_id))
        return {"result": "Memory deleted.", "memory_id": memory_id}


class TestMem0V3Tools:
    """Test v3 tool names and response handling."""

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_search_returns_ids(self, monkeypatch):
        backend = FakeBackend(search_results=[{"id": "mem-1", "memory": "foo", "score": 0.9}])
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_search", {"query": "test"}))
        assert result["results"][0]["id"] == "mem-1"

    def test_search_uses_filters(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_search", {"query": "hello", "top_k": 3})
        assert backend.captured[0][2]["filters"] == {"user_id": "u123"}
        assert backend.captured[0][2]["top_k"] == 3

    def test_search_rerank_default_false(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_search", {"query": "test"})
        assert backend.captured[0][2]["rerank"] is False

    def test_search_rerank_override_false(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_search", {"query": "test", "rerank": False})
        assert backend.captured[0][2]["rerank"] is False

    def test_search_rerank_config_default_used_when_arg_absent(self, monkeypatch):
        """The persisted mem0.json ``rerank`` preference is the tool default;
        a per-call arg still wins (see the explicit-False override above)."""
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider._rerank_default = True  # as initialize() sets from config
        provider.handle_tool_call("mem0_search", {"query": "test"})
        assert backend.captured[0][2]["rerank"] is True

    def test_add_uses_content_param(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"}))
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["infer"] is False
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert "event_id" in result

    def test_add_returns_event_id(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {"content": "test"}))
        assert result["event_id"] == "evt-test-123"

    def test_add_missing_content(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_add", {}))
        assert "error" in result

    def test_old_tool_names_return_unknown(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result


class TestMem0UpdateDelete:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_update_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "mem-1", "text": "updated fact"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert backend.captured[0][2] == "updated fact"
        assert result["result"] == "Memory updated."
        assert result["memory_id"] == "mem-1"

    def test_update_missing_memory_id(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_update", {"text": "no id"}))
        assert "error" in result

    def test_update_missing_text(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_update", {"memory_id": "mem-1"}))
        assert "error" in result

    def test_delete_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "mem-1"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert result["result"] == "Memory deleted."

    def test_delete_missing_memory_id(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_delete", {}))
        assert "error" in result


class TestMem0ErrorHandling:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_update_404_no_circuit_breaker(self, monkeypatch):
        backend = FakeBackend()
        backend.update = lambda mid, text: (_ for _ in ()).throw(Exception("404 Not Found"))
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "bad-id", "text": "x"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_delete_404_no_circuit_breaker(self, monkeypatch):
        backend = FakeBackend()
        backend.delete = lambda mid: (_ for _ in ()).throw(Exception("404 not found"))
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "bad-id"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_update_validation_error_no_circuit_breaker(self, monkeypatch):
        """ValidationError (bad UUID format) should not trip circuit breaker."""
        class ValidationError(Exception):
            pass
        backend = FakeBackend()
        backend.update = lambda mid, text: (_ for _ in ()).throw(
            ValidationError('{"error":"memory_id should be a valid UUID"}')
        )
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_update", {"memory_id": "not-a-uuid", "text": "x"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_delete_validation_error_no_circuit_breaker(self, monkeypatch):
        class ValidationError(Exception):
            pass
        backend = FakeBackend()
        backend.delete = lambda mid: (_ for _ in ()).throw(
            ValidationError('{"error":"memory_id should be a valid UUID"}')
        )
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "not-a-uuid"}
        ))
        assert "error" in result
        assert provider._consecutive_failures == 0

    def test_update_5xx_trips_circuit_breaker(self, monkeypatch):
        backend = FakeBackend()
        backend.update = lambda mid, text: (_ for _ in ()).throw(Exception("500 Internal Server Error"))
        provider = self._make_provider(monkeypatch, backend)
        provider.handle_tool_call("mem0_update", {"memory_id": "mem-1", "text": "x"})
        assert provider._consecutive_failures == 1


class TestMem0V3Internal:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_sync_turn_explicit_kwargs(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        provider.sync_turn("user said", "assistant replied", session_id="s1")
        provider._sync_thread.join(timeout=2)
        assert len(backend.captured) == 1
        call = backend.captured[0]
        assert call[2]["user_id"] == "u123"
        assert call[2]["agent_id"] == "hermes"
        assert call[2]["infer"] is True

    def test_old_tool_names_return_unknown(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call("mem0_profile", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_conclude", {}))
        assert "error" in result
        result = json.loads(provider.handle_tool_call("mem0_list", {}))
        assert "error" in result


class TestMem0Prefetch:
    """prefetch() must recall on the CURRENT question, synchronously.

    The old implementation ignored its ``query`` and returned whatever a
    background ``queue_prefetch`` had warmed from the PREVIOUS turn — so the
    first turn injected nothing and later turns injected stale, off-topic
    memories. These lock the corrected behaviour.
    """

    def _make_provider(self, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider

    def test_prefetch_searches_current_query(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "user prefers dark mode"}])
        provider = self._make_provider(backend)
        result = provider.prefetch("what theme do I like?")
        kind, query, opts = backend.captured[0]
        assert kind == "search"
        assert query == "what theme do I like?"
        assert opts["filters"] == {"user_id": "u123"}
        assert opts["top_k"] == 10
        assert opts["rerank"] is False
        assert "## Mem0 Memory" in result
        assert "user prefers dark mode" in result

    def test_prefetch_returns_memories_on_first_call(self):
        # No prior queue_prefetch / warm — the very first call must still recall.
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        provider = self._make_provider(backend)
        result = provider.prefetch("where do I live?")
        assert "lives in Berlin" in result

    def test_on_turn_start_queues_current_query(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        provider = self._make_provider(backend)
        provider.on_turn_start(1, "where do I live?")
        provider._prefetch_thread.join(timeout=1)
        result = provider.prefetch("where do I live?")
        assert "lives in Berlin" in result
        assert len([c for c in backend.captured if c[0] == "search"]) == 1

    def test_slow_prefetch_returns_quickly(self, monkeypatch):
        class SlowBackend(FakeBackend):
            def search(self, query, *, filters, top_k=10, rerank=True):
                time.sleep(0.2)
                return super().search(query, filters=filters, top_k=top_k, rerank=rerank)

        monkeypatch.setattr(mem0_plugin, "_PREFETCH_WAIT_SECS", 0.01)
        provider = self._make_provider(
            SlowBackend(search_results=[{"id": "m1", "memory": "lives in Berlin"}])
        )
        started = time.monotonic()
        assert provider.prefetch("where do I live?") == ""
        assert time.monotonic() - started < 0.1
        provider._prefetch_thread.join(timeout=1)
        assert "lives in Berlin" in provider.prefetch("where do I live?")

    def test_prefetch_empty_results_returns_empty(self):
        backend = FakeBackend(search_results=[])
        provider = self._make_provider(backend)
        assert provider.prefetch("anything") == ""

    def test_prefetch_skips_when_breaker_open(self):
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "x"}])
        provider = self._make_provider(backend)
        provider._consecutive_failures = 5
        provider._breaker_open_until = float("inf")
        assert provider.prefetch("q") == ""
        assert backend.captured == []

    def test_queue_prefetch_fires_no_search(self):
        # prefetch is synchronous now, so the post-turn warm is redundant and
        # must not fire a wasted backend search.
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "x"}])
        provider = self._make_provider(backend)
        provider.queue_prefetch("previous turn text")
        assert backend.captured == []


class TestMem0MinScore:
    """min_score relevance floor on injection (prefetch).

    Default 0.0 keeps everything (pre-existing behavior); a configured floor
    drops low-relevance results so short openers don't inject junk memories
    that read as authoritative context. Results without a score are kept.
    """

    def _make_provider(self, backend, min_score=None):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        if min_score is not None:
            provider._min_score = min_score
        return provider

    def test_floor_filters_low_score_results(self):
        backend = FakeBackend(search_results=[
            {"id": "m1", "memory": "relevant fact", "score": 0.9},
            {"id": "m2", "memory": "junk match", "score": 0.2},
            {"id": "m3", "memory": "borderline", "score": 0.5},
        ])
        provider = self._make_provider(backend, min_score=0.5)
        result = provider.prefetch("what do I like?")
        assert "relevant fact" in result
        assert "borderline" in result  # >= floor is inclusive
        assert "junk match" not in result

    def test_floor_filters_everything_injects_nothing(self):
        backend = FakeBackend(search_results=[
            {"id": "m1", "memory": "junk a", "score": 0.1},
            {"id": "m2", "memory": "junk b", "score": 0.3},
        ])
        provider = self._make_provider(backend, min_score=0.5)
        assert provider.prefetch("ok") == ""

    def test_default_keeps_everything(self, monkeypatch, tmp_path):
        # No min_score configured anywhere → floor is 0.0, nothing filtered.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_MIN_SCORE", raising=False)
        backend = FakeBackend(search_results=[
            {"id": "m1", "memory": "high", "score": 0.9},
            {"id": "m2", "memory": "low", "score": 0.1},
        ])
        provider = self._make_provider(backend)
        assert provider._min_score == 0.0
        result = provider.prefetch("anything")
        assert "high" in result
        assert "low" in result

    def test_missing_score_is_kept(self):
        # Conservative: backends that omit scores keep their current behavior.
        backend = FakeBackend(search_results=[
            {"id": "m1", "memory": "no score here"},
            {"id": "m2", "memory": "scored junk", "score": 0.1},
        ])
        provider = self._make_provider(backend, min_score=0.5)
        result = provider.prefetch("query")
        assert "no score here" in result
        assert "scored junk" not in result

    def test_config_from_mem0_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_MIN_SCORE", raising=False)
        (tmp_path / "mem0.json").write_text('{"min_score": 0.5}')
        provider = Mem0MemoryProvider()
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        provider.initialize("test")
        assert provider._min_score == 0.5

    def test_config_from_env_var(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.setenv("MEM0_MIN_SCORE", "0.7")
        provider = Mem0MemoryProvider()
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        provider.initialize("test")
        assert provider._min_score == 0.7

    @pytest.mark.parametrize("bad", ['"not-a-number"', "-0.5", "1.5"])
    def test_invalid_config_falls_back_with_warning(self, monkeypatch, tmp_path, caplog, bad):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        monkeypatch.delenv("MEM0_MIN_SCORE", raising=False)
        (tmp_path / "mem0.json").write_text('{"min_score": %s}' % bad)
        provider = Mem0MemoryProvider()
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        with caplog.at_level("WARNING", logger="plugins.memory.mem0"):
            provider.initialize("test")
        assert provider._min_score == 0.0
        assert any("min_score" in r.message for r in caplog.records)


class TestMem0V3Config:

    def test_tool_schemas_four_tools(self):
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names == ["mem0_search", "mem0_add", "mem0_update", "mem0_delete"]

    def test_system_prompt_new_tool_names(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "mem0_add" in block
        assert "mem0_update" in block
        assert "mem0_delete" in block
        assert "mem0_list" not in block
        assert "mem0_profile" not in block
        assert "mem0_conclude" not in block

    def test_system_prompt_shows_platform_mode(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "platform"
        block = provider.system_prompt_block()
        assert "platform" in block
        assert "Rerank" in block

    def test_system_prompt_shows_oss_mode(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "oss"
        block = provider.system_prompt_block()
        assert "OSS" in block
        assert "Rerank" not in block

    def test_search_schema_has_rerank(self):
        """rerank property available in SEARCH_SCHEMA for platform mode."""
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        search = next(s for s in schemas if s["name"] == "mem0_search")
        assert "rerank" in search["parameters"]["properties"]
        assert search["parameters"]["properties"]["rerank"]["type"] == "boolean"


class TestMem0ModeSwitch:

    def test_default_mode_is_platform(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"

    def test_missing_mode_key_defaults_platform(self, monkeypatch, tmp_path):
        """Backward compat: old mem0.json without mode key works."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"user_id": "old-user"}')
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        provider.initialize("test")
        assert provider._mode == "platform"
        assert provider._user_id == "old-user"

    def test_is_available_platform_needs_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        provider = Mem0MemoryProvider()
        assert provider.is_available() is False

    def test_is_available_oss_needs_vector(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"mode": "oss", "oss": {"vector_store": {"provider": "qdrant"}}}')
        provider = Mem0MemoryProvider()
        assert provider.is_available() is True

    def test_is_available_oss_no_vector(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_path = tmp_path / "mem0.json"
        config_path.write_text('{"mode": "oss", "oss": {}}')
        provider = Mem0MemoryProvider()
        assert provider.is_available() is False

    def test_tool_schemas_unchanged(self):
        provider = Mem0MemoryProvider()
        schemas = provider.get_tool_schemas()
        names = [s["name"] for s in schemas]
        assert names == ["mem0_search", "mem0_add", "mem0_update", "mem0_delete"]

    def test_system_prompt_includes_mode(self):
        provider = Mem0MemoryProvider()
        provider._user_id = "test"
        provider._mode = "oss"
        block = provider.system_prompt_block()
        assert "mem0_search" in block
        assert "OSS" in block


class TestMem0UserIdResolution:
    """user_id resolution: configured override > gateway-native id > placeholder.

    Same human across CLI / Telegram / Discord / Slack / etc. should map to
    the same memory store when MEM0_USER_ID is set, and only fall back to the
    gateway-native id when it isn't.
    """

    def _provider(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("MEM0_API_KEY", "test-key")
        provider = Mem0MemoryProvider()
        # Skip backend instantiation — we only care about identity resolution.
        provider._create_backend = lambda: None  # type: ignore[method-assign]
        return provider

    def test_env_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MEM0_USER_ID", "ryan@example.com")
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_file_override_beats_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "ryan@example.com"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "ryan@example.com"

    def test_unset_falls_back_to_gateway_native_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"

    def test_unset_and_no_kwargs_falls_back_to_default(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test")
        assert provider._user_id == "hermes-user"

    def test_legacy_placeholder_in_config_does_not_override_kwargs(self, monkeypatch, tmp_path):
        # Setup wizard historically wrote {"user_id": "hermes-user"} as the
        # suggested default. Treat that placeholder as unset so users on
        # gateways still get gateway-native ids — not silent collisions.
        monkeypatch.delenv("MEM0_USER_ID", raising=False)
        (tmp_path / "mem0.json").write_text('{"user_id": "hermes-user"}')
        provider = self._provider(monkeypatch, tmp_path)
        provider.initialize("test", user_id="123456789", platform="telegram")
        assert provider._user_id == "123456789"


class TestMem0WriteMetadata:
    """Writes carry metadata.channel so per-channel filtered views are possible
    without coupling identity to the channel.
    """

    def _make_provider(self, channel: str = "cli"):
        provider = Mem0MemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._channel = channel
        provider._backend = FakeBackend()
        return provider

    def test_add_tool_passes_channel_metadata(self):
        provider = self._make_provider("telegram")
        provider.handle_tool_call("mem0_add", {"content": "user likes dark mode"})
        call = provider._backend.captured[-1]
        assert call[2]["metadata"] == {"channel": "telegram"}

    def test_sync_turn_passes_channel_metadata(self):
        provider = self._make_provider("discord")
        provider.sync_turn("hi", "hello", session_id="s")
        # sync_turn fires a daemon thread; wait for it.
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        adds = [c for c in provider._backend.captured if c[0] == "add"]
        assert adds, "expected an add call from sync_turn"
        assert adds[-1][2]["metadata"] == {"channel": "discord"}


class _SentinelBackend:
    def __init__(self, *args):
        self.args = args


class TestCreateBackendRouting:
    """_create_backend() must pick the backend matching the configured mode/host."""

    def _provider(self, monkeypatch, *, mode="platform", api_key="k", host=""):
        # Neutralize lazy-install so the routing decision is all we exercise.
        monkeypatch.setattr("tools.lazy_deps.ensure", lambda *a, **k: None, raising=False)
        provider = Mem0MemoryProvider()
        provider._mode = mode
        provider._api_key = api_key
        provider._host = host
        provider._config = {"oss": {"vector_store": {"provider": "qdrant"}}}
        return provider

    def test_routes_to_selfhosted_when_host_set(self, monkeypatch):
        captured = {}

        class SH(_SentinelBackend):
            def __init__(self, api_key, host):
                captured["args"] = (api_key, host)

        monkeypatch.setattr("plugins.memory.mem0._backend.SelfHostedBackend", SH)
        provider = self._provider(monkeypatch, host="http://sh:8888", api_key="adminkey")
        backend = provider._create_backend()
        assert isinstance(backend, SH)
        assert captured["args"] == ("adminkey", "http://sh:8888")

    def test_routes_to_platform_when_no_host(self, monkeypatch):
        class PB(_SentinelBackend):
            def __init__(self, api_key):
                pass

        monkeypatch.setattr("plugins.memory.mem0._backend.PlatformBackend", PB)
        provider = self._provider(monkeypatch, host="")
        assert isinstance(provider._create_backend(), PB)

    def test_routes_to_oss_when_mode_oss(self, monkeypatch):
        class OB(_SentinelBackend):
            def __init__(self, cfg):
                pass

        monkeypatch.setattr("plugins.memory.mem0._backend.OSSBackend", OB)
        provider = self._provider(monkeypatch, mode="oss")
        assert isinstance(provider._create_backend(), OB)

    def test_oss_mode_takes_precedence_over_host(self, monkeypatch):
        class OB(_SentinelBackend):
            def __init__(self, cfg):
                pass

        monkeypatch.setattr("plugins.memory.mem0._backend.OSSBackend", OB)
        provider = self._provider(monkeypatch, mode="oss", host="http://sh:8888")
        assert isinstance(provider._create_backend(), OB)

    def test_prompt_label_matches_routing_when_oss_and_host_both_set(self, monkeypatch):
        # system_prompt_block must mirror _create_backend precedence: with both
        # mode=oss and host set, OSS wins the routing, so the prompt must label
        # OSS — not "self-hosted (HTTP API)". Guards the prompt-vs-routing lie.
        provider = self._provider(monkeypatch, mode="oss", host="http://sh:8888")
        provider._user_id = "test"
        block = provider.system_prompt_block()
        assert "OSS" in block
        assert "HTTP API" not in block


class TestSelfHostedConfig:
    """Config plumbing for self-hosted (MEM0_HOST env + is_available)."""

    def test_load_config_reads_mem0_host_env(self, monkeypatch):
        monkeypatch.setenv("MEM0_HOST", "http://localhost:8888")
        assert mem0_plugin._load_config()["host"] == "http://localhost:8888"

    def test_is_available_true_with_host_only(self, monkeypatch):
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        monkeypatch.setenv("MEM0_MODE", "platform")
        monkeypatch.setenv("MEM0_HOST", "http://localhost:8888")
        assert Mem0MemoryProvider().is_available() is True

    def test_is_available_false_without_key_or_host(self, monkeypatch):
        monkeypatch.delenv("MEM0_API_KEY", raising=False)
        monkeypatch.delenv("MEM0_HOST", raising=False)
        monkeypatch.setenv("MEM0_MODE", "platform")
        assert Mem0MemoryProvider().is_available() is False
