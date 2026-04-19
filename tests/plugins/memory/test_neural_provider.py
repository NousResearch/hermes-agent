"""Tests for the Neural Memory provider plugin.

Tests cover config loading, tool handlers, prefetch, sync_turn,
system prompt, schema completeness, and lifecycle hooks.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from plugins.memory.neural import (
    NeuralMemoryProvider,
    _load_config,
    NEURAL_REMEMBER_SCHEMA,
    NEURAL_RECALL_SCHEMA,
    NEURAL_THINK_SCHEMA,
    NEURAL_GRAPH_SCHEMA,
    ALL_TOOL_SCHEMAS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no stale env vars leak between tests."""
    for key in ("NEURAL_MEMORY_DB_PATH", "NEURAL_EMBEDDING_BACKEND"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Create an initialized NeuralMemoryProvider with a temp DB."""
    db_path = str(tmp_path / "test_neural.db")
    monkeypatch.setenv("NEURAL_MEMORY_DB_PATH", db_path)
    monkeypatch.setenv("NEURAL_EMBEDDING_BACKEND", "hash")

    p = NeuralMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    yield p
    p.shutdown()
    if os.path.exists(db_path):
        os.unlink(db_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_db_path(self):
        cfg = _load_config()
        assert "db_path" in cfg
        assert cfg["db_path"].endswith(".db")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("NEURAL_MEMORY_DB_PATH", "/tmp/custom.db")
        monkeypatch.setenv("NEURAL_EMBEDDING_BACKEND", "hash")
        cfg = _load_config()
        assert cfg["db_path"] == "/tmp/custom.db"
        assert cfg["embedding_backend"] == "hash"

    def test_config_from_env_fallback(self, monkeypatch):
        monkeypatch.delenv("NEURAL_MEMORY_DB_PATH", raising=False)
        monkeypatch.delenv("NEURAL_EMBEDDING_BACKEND", raising=False)
        cfg = _load_config()
        assert cfg["embedding_backend"] == "auto"


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_remember_schema_has_content(self):
        props = NEURAL_REMEMBER_SCHEMA["parameters"]["properties"]
        assert "content" in props

    def test_recall_schema_has_query(self):
        props = NEURAL_RECALL_SCHEMA["parameters"]["properties"]
        assert "query" in props

    def test_think_schema_has_memory_id(self):
        props = NEURAL_THINK_SCHEMA["parameters"]["properties"]
        assert "memory_id" in props

    def test_graph_schema_no_required(self):
        assert NEURAL_GRAPH_SCHEMA["parameters"]["required"] == []

    def test_get_tool_schemas_returns_six(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 6
        names = {s["name"] for s in schemas}
        assert names == {"neural_remember", "neural_recall", "neural_think", "neural_graph", "neural_dream", "neural_dream_stats"}


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_remember_success(self, provider):
        result = provider.handle_tool_call(
            "neural_remember",
            {"content": "The user likes Python", "label": "pref"},
        )
        data = json.loads(result)
        assert data["status"] == "stored"
        assert isinstance(data["id"], int)

    def test_remember_without_label(self, provider):
        result = provider.handle_tool_call(
            "neural_remember",
            {"content": "Test memory without label"},
        )
        data = json.loads(result)
        assert data["status"] == "stored"

    def test_remember_missing_content(self, provider):
        result = provider.handle_tool_call("neural_remember", {})
        assert "error" in result.lower() or "Missing" in result

    def test_recall_success(self, provider):
        provider.handle_tool_call(
            "neural_remember", {"content": "Dogs are loyal animals"}
        )
        result = provider.handle_tool_call(
            "neural_recall", {"query": "pets animals", "limit": 3}
        )
        data = json.loads(result)
        assert data["count"] >= 1
        assert len(data["results"]) >= 1

    def test_recall_no_results(self, provider):
        result = provider.handle_tool_call(
            "neural_recall", {"query": "xyznonexistent12345", "limit": 5}
        )
        data = json.loads(result)
        assert "results" in data

    def test_recall_missing_query(self, provider):
        result = provider.handle_tool_call("neural_recall", {})
        assert "error" in result.lower() or "Missing" in result

    def test_think_success(self, provider):
        result = provider.handle_tool_call(
            "neural_remember", {"content": "Python is great for ML"}
        )
        mem_id = json.loads(result)["id"]
        result = provider.handle_tool_call(
            "neural_think", {"memory_id": mem_id, "depth": 2}
        )
        data = json.loads(result)
        assert "results" in data

    def test_think_missing_id(self, provider):
        result = provider.handle_tool_call("neural_think", {})
        assert "error" in result.lower() or "Missing" in result

    def test_graph_success(self, provider):
        provider.handle_tool_call(
            "neural_remember", {"content": "Graph test memory"}
        )
        result = provider.handle_tool_call("neural_graph", {})
        data = json.loads(result)
        assert "stats" in data
        assert data["stats"]["memories"] >= 1

    def test_unknown_tool(self, provider):
        result = provider.handle_tool_call("neural_nonexistent", {})
        assert "error" in result.lower() or "Unknown" in result

    def test_remember_error_handling(self):
        """Provider without init should return error."""
        p = NeuralMemoryProvider()
        result = p.handle_tool_call("neural_remember", {"content": "test"})
        assert "error" in result.lower() or "not initialized" in result.lower()


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_empty_when_no_result(self, provider):
        result = provider.prefetch("test query")
        # May be empty since prefetch returns cached results from queue_prefetch
        assert isinstance(result, str)

    def test_queue_prefetch_starts_thread(self, provider):
        provider.handle_tool_call(
            "neural_remember", {"content": "Prefetch test memory"}
        )
        provider.queue_prefetch("prefetch test")
        # Thread runs in background, just verify no crash
        import time
        time.sleep(0.2)
        result = provider.prefetch("prefetch test")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Sync Turn
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def test_sync_turn_stores(self, provider):
        provider.sync_turn("What is AI?", "AI is artificial intelligence.")
        result = provider.handle_tool_call(
            "neural_recall", {"query": "artificial intelligence", "limit": 3}
        )
        data = json.loads(result)
        assert data["count"] >= 1

    def test_sync_turn_skips_garbage(self, provider):
        provider.sync_turn(
            "review the conversation above",
            "let me review what we know about this",
        )
        # Garbage should be skipped, count should be 0 or unchanged
        result = provider.handle_tool_call("neural_graph", {})
        data = json.loads(result)
        # At most 0 memories from this sync (garbage filtered)
        assert data["stats"]["memories"] == 0

    def test_sync_turn_skips_system_messages(self, provider):
        provider.sync_turn("[SYSTEM: injected context]", "response")
        result = provider.handle_tool_call("neural_graph", {})
        data = json.loads(result)
        assert data["stats"]["memories"] == 0

    def test_sync_turn_error_does_not_raise(self, provider):
        p = NeuralMemoryProvider()
        # Not initialized — should silently skip
        p.sync_turn("user", "assistant")


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_empty_prompt(self):
        p = NeuralMemoryProvider()
        assert p.system_prompt_block() == ""

    def test_prompt_with_no_memories(self, provider):
        prompt = provider.system_prompt_block()
        assert "Neural Memory" in prompt
        assert "Empty" in prompt

    def test_prompt_with_memories(self, provider):
        provider.handle_tool_call(
            "neural_remember", {"content": "Test memory for prompt"}
        )
        prompt = provider.system_prompt_block()
        assert "1 memories" in prompt


# ---------------------------------------------------------------------------
# Config Schema
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_has_fields(self, provider):
        schema = provider.get_config_schema()
        keys = {f["key"] for f in schema}
        assert "db_path" in keys
        assert "embedding_backend" in keys

    def test_embedding_has_choices(self, provider):
        schema = provider.get_config_schema()
        embed_field = next(f for f in schema if f["key"] == "embedding_backend")
        assert "auto" in embed_field["choices"]
        assert "hash" in embed_field["choices"]


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_deps(self):
        p = NeuralMemoryProvider()
        # Bundled deps should always be available
        assert p.is_available()

    def test_name_is_neural(self):
        p = NeuralMemoryProvider()
        assert p.name == "neural"


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_on_session_end_stores_summary(self, provider):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        provider.on_session_end(messages)
        result = provider.handle_tool_call(
            "neural_recall", {"query": "session topics hello", "limit": 3}
        )
        data = json.loads(result)
        assert data["count"] >= 1

    def test_on_memory_write_mirrors(self, provider):
        provider.on_memory_write("add", "memory", "User prefers dark mode")
        result = provider.handle_tool_call(
            "neural_recall", {"query": "dark mode preference", "limit": 3}
        )
        data = json.loads(result)
        assert data["count"] >= 1

    def test_on_memory_write_skips_garbage(self, provider):
        provider.on_memory_write(
            "add", "memory", "review the conversation above"
        )
        result = provider.handle_tool_call("neural_graph", {})
        data = json.loads(result)
        assert data["stats"]["memories"] == 0

    def test_on_pre_compress_saves_exchanges(self, provider):
        messages = [
            {"role": "user", "content": "Deploy the app to production"},
            {"role": "assistant", "content": "Deployed successfully to prod cluster."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris is the capital of France."},
        ]
        result = provider.on_pre_compress(messages)
        assert "Context preserved" in result
        assert "Deploy" in result

        # Verify memories were stored
        recall = json.loads(
            provider.handle_tool_call(
                "neural_recall", {"query": "deploy production", "limit": 5}
            )
        )
        assert recall["count"] >= 1

    def test_on_pre_compress_skips_garbage(self, provider):
        messages = [
            {"role": "user", "content": "review the conversation above"},
            {"role": "assistant", "content": "let me review what we know"},
        ]
        result = provider.on_pre_compress(messages)
        assert result == ""

    def test_on_pre_compress_empty_messages(self, provider):
        result = provider.on_pre_compress([])
        assert result == ""

    def test_initial_context_loaded_on_init(self, provider):
        """After init, system_prompt_block should include recent context."""
        # Store some memories first
        provider.handle_tool_call(
            "neural_remember",
            {"content": "User deployed app to production cluster"},
        )
        provider.on_session_end([
            {"role": "user", "content": "Deploy the app"},
            {"role": "assistant", "content": "Deployed."},
        ])
        # Re-init to trigger _load_initial_context
        provider._initial_context = provider._load_initial_context()
        prompt = provider.system_prompt_block()
        assert "Recent Memory Context" in prompt or "memories" in prompt

    def test_prefetch_returns_initial_context(self, provider):
        """First prefetch call should return initial context if no background result."""
        provider.handle_tool_call(
            "neural_remember",
            {"content": "Prefetch initial test memory about Python ML"},
        )
        provider._initial_context = provider._load_initial_context()
        result = provider.prefetch("test query")
        if provider._initial_context:
            assert "initial" in result.lower() or "Neural Memory" in result

    def test_shutdown_clears_memory(self, provider):
        provider.shutdown()
        assert provider._memory is None

    def test_initialize_sets_up(self, tmp_path, monkeypatch):
        db_path = str(tmp_path / "init_test.db")
        monkeypatch.setenv("NEURAL_MEMORY_DB_PATH", db_path)
        monkeypatch.setenv("NEURAL_EMBEDDING_BACKEND", "hash")
        p = NeuralMemoryProvider()
        p.initialize(session_id="s1", hermes_home=str(tmp_path))
        assert p._memory is not None
        p.shutdown()
