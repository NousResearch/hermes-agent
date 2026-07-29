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


    def test_delete_calls_sdk(self, monkeypatch):
        backend = FakeBackend()
        provider = self._make_provider(monkeypatch, backend)
        result = json.loads(provider.handle_tool_call(
            "mem0_delete", {"memory_id": "mem-1"}
        ))
        assert backend.captured[0][1] == "mem-1"
        assert result["result"] == "Memory deleted."


class TestMem0ErrorHandling:

    def _make_provider(self, monkeypatch, backend):
        provider = Mem0MemoryProvider()
        provider.initialize("test-session")
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        provider._backend = backend
        return provider


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


    def test_queue_prefetch_fires_no_search(self):
        # prefetch is synchronous now, so the post-turn warm is redundant and
        # must not fire a wasted backend search.
        backend = FakeBackend(search_results=[{"id": "m1", "memory": "x"}])
        provider = self._make_provider(backend)
        provider.queue_prefetch("previous turn text")
        assert backend.captured == []


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


