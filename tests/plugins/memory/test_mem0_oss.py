"""Tests for Mem0 OSS memory provider — config, filters, tools, circuit breaker."""

import json
import time
import pytest

from plugins.memory.mem0_oss import (
    Mem0OSSMemoryProvider,
    _BREAKER_THRESHOLD,
    _load_config,
)


class FakeMemoryOSS:
    """Fake mem0 Memory client that captures call kwargs."""

    def __init__(self, search_results=None, all_results=None):
        self._search_results = search_results or {"results": []}
        self._all_results = all_results or {"results": []}
        self.captured_search = {}
        self.captured_get_all = {}
        self.captured_add = []

    def search(self, query, **kwargs):
        self.captured_search = {"query": query, **kwargs}
        return self._search_results

    def get_all(self, **kwargs):
        self.captured_get_all = kwargs
        return self._all_results

    def add(self, messages, **kwargs):
        self.captured_add.append({"messages": messages, **kwargs})


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestMem0OSSConfig:
    """Config loads from mem0_oss.json with sensible defaults."""

    def test_load_config_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cfg = _load_config()
        assert cfg["user_id"] == "hermes-user"
        assert cfg["agent_id"] == "hermes"
        assert cfg["llm"]["provider"] == "openai"

    def test_load_config_from_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        config_file = tmp_path / "mem0_oss.json"
        config_file.write_text(json.dumps({
            "llm": {"provider": "ollama", "config": {"model": "llama3.1:8b"}},
            "embedder": {"provider": "ollama", "config": {"model": "nomic-embed-text"}},
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/test-qdrant"}},
            "user_id": "test-user",
            "agent_id": "test-agent",
        }))
        cfg = _load_config()
        assert cfg["user_id"] == "test-user"
        assert cfg["agent_id"] == "test-agent"
        assert cfg["llm"]["provider"] == "ollama"
        assert cfg["embedder"]["provider"] == "ollama"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestMem0OSSDefaults:
    """Default user_id and agent_id are preserved."""

    def test_default_user_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = Mem0OSSMemoryProvider()
        provider.initialize("test")
        assert provider._user_id == "hermes-user"

    def test_default_agent_id(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = Mem0OSSMemoryProvider()
        provider.initialize("test")
        assert provider._agent_id == "hermes"


# ---------------------------------------------------------------------------
# Kwargs (bare user_id=/agent_id= instead of filters={})
# ---------------------------------------------------------------------------


class TestMem0OSSKwargs:
    """OSS uses bare user_id=/agent_id= kwargs, not filters={}."""

    def test_read_kwargs_user_only(self):
        provider = Mem0OSSMemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        assert provider._read_kwargs() == {"user_id": "u123"}

    def test_write_kwargs_user_and_agent(self):
        provider = Mem0OSSMemoryProvider()
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        assert provider._write_kwargs() == {"user_id": "u123", "agent_id": "hermes"}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestMem0OSSAvailability:
    """is_available checks config file + LLM key."""

    def test_available_with_openai_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setattr("plugins.memory.mem0_oss._check_dependencies", lambda cfg: [])
        config_file = tmp_path / "mem0_oss.json"
        config_file.write_text(json.dumps({
            "llm": {"provider": "openai", "config": {"model": "gpt-5.4"}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/q"}},
        }))
        provider = Mem0OSSMemoryProvider()
        assert provider.is_available() is True

    def test_available_with_ollama_no_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("plugins.memory.mem0_oss._check_dependencies", lambda cfg: [])
        config_file = tmp_path / "mem0_oss.json"
        config_file.write_text(json.dumps({
            "llm": {"provider": "ollama", "config": {"model": "llama3.1:8b"}},
            "embedder": {"provider": "ollama", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/q"}},
        }))
        provider = Mem0OSSMemoryProvider()
        assert provider.is_available() is True

    def test_not_available_missing_dependency(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr("plugins.memory.mem0_oss._check_dependencies", lambda cfg: ["ollama"])
        config_file = tmp_path / "mem0_oss.json"
        config_file.write_text(json.dumps({
            "llm": {"provider": "ollama", "config": {"model": "llama3.1:8b"}},
            "embedder": {"provider": "ollama", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {"path": "/tmp/q"}},
        }))
        provider = Mem0OSSMemoryProvider()
        assert provider.is_available() is False

    def test_not_available_no_config_file(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = Mem0OSSMemoryProvider()
        assert provider.is_available() is False

    def test_not_available_missing_api_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr("plugins.memory.mem0_oss._check_dependencies", lambda cfg: [])
        config_file = tmp_path / "mem0_oss.json"
        config_file.write_text(json.dumps({
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
        }))
        provider = Mem0OSSMemoryProvider()
        assert provider.is_available() is False


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------


class TestMem0OSSToolCalls:
    """Tool dispatch uses bare kwargs and correct OSS param names."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0OSSMemoryProvider()
        provider._config = {
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
            "user_id": "hermes-user",
            "agent_id": "hermes",
        }
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_profile_uses_bare_kwargs(self, monkeypatch):
        client = FakeMemoryOSS(all_results={"results": [{"memory": "alpha"}, {"memory": "beta"}]})
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call("mem0_oss_profile", {}))

        assert result["count"] == 2
        assert "alpha" in result["result"]
        assert client.captured_get_all["user_id"] == "u123"
        assert client.captured_get_all["limit"] == 100
        assert "filters" not in client.captured_get_all

    def test_search_uses_bare_kwargs_and_limit(self, monkeypatch):
        client = FakeMemoryOSS(search_results={
            "results": [{"memory": "foo", "score": 0.9}]
        })
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call(
            "mem0_oss_search", {"query": "hello", "top_k": 5, "rerank": True}
        ))

        assert result["count"] == 1
        assert client.captured_search["query"] == "hello"
        assert client.captured_search["user_id"] == "u123"
        assert client.captured_search["limit"] == 5
        assert client.captured_search["rerank"] is True
        assert "filters" not in client.captured_search
        assert "top_k" not in client.captured_search

    def test_search_missing_query(self, monkeypatch):
        client = FakeMemoryOSS()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {}))
        assert "error" in result

    def test_conclude_uses_bare_kwargs_and_infer_false(self, monkeypatch):
        client = FakeMemoryOSS()
        provider = self._make_provider(monkeypatch, client)

        result = json.loads(provider.handle_tool_call(
            "mem0_oss_conclude", {"conclusion": "user likes dark mode"}
        ))

        assert result["result"] == "Fact stored."
        assert len(client.captured_add) == 1
        call = client.captured_add[0]
        assert call["user_id"] == "u123"
        assert call["agent_id"] == "hermes"
        assert call["infer"] is False
        assert "filters" not in call

    def test_conclude_missing_conclusion(self, monkeypatch):
        client = FakeMemoryOSS()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_oss_conclude", {}))
        assert "error" in result

    def test_unknown_tool(self, monkeypatch):
        client = FakeMemoryOSS()
        provider = self._make_provider(monkeypatch, client)
        result = json.loads(provider.handle_tool_call("mem0_oss_unknown", {}))
        assert "error" in result


# ---------------------------------------------------------------------------
# Prefetch and sync
# ---------------------------------------------------------------------------


class TestMem0OSSPrefetchSync:
    """Background prefetch and sync use bare kwargs."""

    def _make_provider(self, monkeypatch, client):
        provider = Mem0OSSMemoryProvider()
        provider._config = {
            "llm": {"provider": "openai", "config": {}},
            "embedder": {"provider": "openai", "config": {}},
            "vector_store": {"provider": "qdrant", "config": {}},
            "user_id": "hermes-user",
            "agent_id": "hermes",
        }
        provider._user_id = "u123"
        provider._agent_id = "hermes"
        monkeypatch.setattr(provider, "_get_client", lambda: client)
        return provider

    def test_prefetch_uses_bare_kwargs(self, monkeypatch):
        client = FakeMemoryOSS(search_results={
            "results": [{"memory": "user prefers dark mode"}]
        })
        provider = self._make_provider(monkeypatch, client)

        provider.queue_prefetch("preferences")
        provider._prefetch_thread.join(timeout=2)
        result = provider.prefetch("preferences")

        assert "dark mode" in result
        assert client.captured_search["user_id"] == "u123"
        assert client.captured_search["rerank"] is True
        assert client.captured_search["limit"] == 5
        assert "filters" not in client.captured_search

    def test_sync_turn_uses_write_kwargs(self, monkeypatch):
        client = FakeMemoryOSS()
        provider = self._make_provider(monkeypatch, client)

        provider.sync_turn("user said this", "assistant replied")
        provider._sync_thread.join(timeout=2)

        assert len(client.captured_add) == 1
        call = client.captured_add[0]
        assert call["user_id"] == "u123"
        assert call["agent_id"] == "hermes"
        assert call["messages"][0]["content"] == "user said this"
        assert call["infer"] is True
        assert "filters" not in call


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestMem0OSSCircuitBreaker:
    """Circuit breaker trips after 5 failures, resets after cooldown."""

    def test_breaker_trips_after_threshold(self):
        provider = Mem0OSSMemoryProvider()
        for _ in range(_BREAKER_THRESHOLD):
            provider._record_failure()
        assert provider._is_breaker_open() is True

    def test_breaker_resets_after_cooldown(self):
        provider = Mem0OSSMemoryProvider()
        for _ in range(_BREAKER_THRESHOLD):
            provider._record_failure()
        provider._breaker_open_until = time.monotonic() - 1
        assert provider._is_breaker_open() is False
        assert provider._consecutive_failures == 0

    def test_breaker_resets_on_success(self):
        provider = Mem0OSSMemoryProvider()
        provider._consecutive_failures = 3
        provider._record_success()
        assert provider._consecutive_failures == 0

    def test_tool_call_returns_error_when_breaker_open(self):
        provider = Mem0OSSMemoryProvider()
        provider._consecutive_failures = _BREAKER_THRESHOLD
        provider._breaker_open_until = time.monotonic() + 999
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {"query": "test"}))
        assert "temporarily unavailable" in result["error"]


# ---------------------------------------------------------------------------
# Response unwrapping
# ---------------------------------------------------------------------------


class TestMem0OSSResponseUnwrapping:
    """_unwrap_results handles dict, list, and edge cases."""

    def test_dict_response(self):
        assert Mem0OSSMemoryProvider._unwrap_results({"results": [1, 2]}) == [1, 2]

    def test_list_response(self):
        assert Mem0OSSMemoryProvider._unwrap_results([3, 4]) == [3, 4]

    def test_empty_dict(self):
        assert Mem0OSSMemoryProvider._unwrap_results({}) == []

    def test_none(self):
        assert Mem0OSSMemoryProvider._unwrap_results(None) == []

    def test_unexpected_type(self):
        assert Mem0OSSMemoryProvider._unwrap_results("string") == []


# ---------------------------------------------------------------------------
# Config schema and save_config
# ---------------------------------------------------------------------------


class TestMem0OSSConfigSchema:
    """Setup wizard config schema and save_config."""

    def test_config_schema_has_required_fields(self):
        provider = Mem0OSSMemoryProvider()
        schema = provider.get_config_schema()
        keys = [f["key"] for f in schema]
        assert "llm_provider" in keys
        assert "llm_model" in keys
        assert "embedder_provider" in keys
        assert "vector_store_provider" in keys
        assert "user_id" in keys

    def test_save_config_writes_nested_json(self, tmp_path):
        provider = Mem0OSSMemoryProvider()
        values = {
            "llm_provider": "ollama",
            "llm_model": "llama3.1:8b",
            "ollama_base_url": "http://localhost:11434",
            "embedder_provider": "ollama",
            "embedder_model": "nomic-embed-text",
            "vector_store_provider": "qdrant",
            "vector_store_path": "/tmp/test-qdrant",
            "user_id": "test-user",
            "agent_id": "test-agent",
        }
        provider.save_config(values, str(tmp_path))

        config_file = tmp_path / "mem0_oss.json"
        assert config_file.exists()
        cfg = json.loads(config_file.read_text())
        assert cfg["llm"]["provider"] == "ollama"
        assert cfg["llm"]["config"]["model"] == "llama3.1:8b"
        assert cfg["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert cfg["embedder"]["provider"] == "ollama"
        assert cfg["embedder"]["config"]["model"] == "nomic-embed-text"
        assert cfg["vector_store"]["provider"] == "qdrant"
        assert cfg["vector_store"]["config"]["path"] == "/tmp/test-qdrant"
        assert cfg["user_id"] == "test-user"
        assert cfg["agent_id"] == "test-agent"

    def test_save_config_ollama_pgvector(self, tmp_path):
        provider = Mem0OSSMemoryProvider()
        values = {
            "llm_provider": "ollama",
            "llm_model": "llama3.1:8b",
            "ollama_base_url": "http://localhost:11434",
            "embedder_provider": "ollama",
            "embedder_model": "nomic-embed-text",
            "vector_store_provider": "pgvector",
            "vector_store_connection_string": "postgresql://user:pass@localhost/db",
            "user_id": "hermes-user",
            "agent_id": "hermes",
        }
        provider.save_config(values, str(tmp_path))

        cfg = json.loads((tmp_path / "mem0_oss.json").read_text())
        assert cfg["llm"]["provider"] == "ollama"
        assert cfg["llm"]["config"]["ollama_base_url"] == "http://localhost:11434"
        assert cfg["vector_store"]["provider"] == "pgvector"
        assert cfg["vector_store"]["config"]["connection_string"] == "postgresql://user:pass@localhost/db"

    def test_save_config_excludes_secrets(self, tmp_path):
        provider = Mem0OSSMemoryProvider()
        values = {
            "llm_provider": "openai",
            "llm_model": "gpt-5.4",
            "llm_api_key": "sk-secret-key",
            "embedder_provider": "openai",
            "embedder_model": "text-embedding-3-small",
            "vector_store_provider": "qdrant",
            "vector_store_path": "/tmp/qdrant",
        }
        provider.save_config(values, str(tmp_path))

        cfg_text = (tmp_path / "mem0_oss.json").read_text()
        assert "sk-secret-key" not in cfg_text
        assert "llm_api_key" not in cfg_text


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


class TestMem0OSSSystemPrompt:
    """System prompt block contains expected content."""

    def test_system_prompt_contains_self_hosted(self):
        provider = Mem0OSSMemoryProvider()
        provider._user_id = "test-user"
        block = provider.system_prompt_block()
        assert "Self-Hosted" in block
        assert "test-user" in block
        assert "mem0_oss_search" in block
        assert "mem0_oss_conclude" in block
        assert "mem0_oss_profile" in block
