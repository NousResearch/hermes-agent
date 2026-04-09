"""Tests for the Hindsight memory provider plugin.

Tests cover config loading, tool handlers (tags, max_tokens, types),
prefetch (auto_recall, preamble, query truncation), sync_turn (auto_retain,
turn counting, tags), and schema completeness.
"""

import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    RECALL_SCHEMA,
    REFLECT_SCHEMA,
    RETAIN_SCHEMA,
    _load_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no stale env vars leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET", "HINDSIGHT_MODE", "HINDSIGHT_LLM_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_mock_client():
    """Create a mock Hindsight client with async methods."""
    client = MagicMock()
    client.aretain = AsyncMock()
    client.arecall = AsyncMock(
        return_value=SimpleNamespace(
            results=[
                SimpleNamespace(text="Memory 1"),
                SimpleNamespace(text="Memory 2"),
            ]
        )
    )
    client.areflect = AsyncMock(
        return_value=SimpleNamespace(text="Synthesized answer")
    )
    client.aretain_batch = AsyncMock()
    client.aclose = AsyncMock()
    return client


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Create an initialized HindsightMemoryProvider with a mock client."""
    config = {
        "mode": "cloud",
        "apiKey": "test-key",
        "api_url": "http://localhost:9999",
        "bank_id": "test-bank",
        "budget": "mid",
        "memory_mode": "hybrid",
    }
    config_path = tmp_path / "hindsight" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config))

    monkeypatch.setattr(
        "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
    )

    p = HindsightMemoryProvider()
    p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    p._client = _make_mock_client()
    return p


@pytest.fixture()
def provider_with_config(tmp_path, monkeypatch):
    """Create a provider factory that accepts custom config overrides."""
    def _make(**overrides):
        config = {
            "mode": "cloud",
            "apiKey": "test-key",
            "api_url": "http://localhost:9999",
            "bank_id": "test-bank",
            "budget": "mid",
            "memory_mode": "hybrid",
        }
        config.update(overrides)
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))

        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        p = HindsightMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
        p._client = _make_mock_client()
        return p
    return _make


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_retain_schema_has_content(self):
        assert RETAIN_SCHEMA["name"] == "hindsight_retain"
        assert "content" in RETAIN_SCHEMA["parameters"]["properties"]
        assert "content" in RETAIN_SCHEMA["parameters"]["required"]

    def test_recall_schema_has_query(self):
        assert RECALL_SCHEMA["name"] == "hindsight_recall"
        assert "query" in RECALL_SCHEMA["parameters"]["properties"]
        assert "query" in RECALL_SCHEMA["parameters"]["required"]

    def test_reflect_schema_has_query(self):
        assert REFLECT_SCHEMA["name"] == "hindsight_reflect"
        assert "query" in REFLECT_SCHEMA["parameters"]["properties"]

    def test_get_tool_schemas_returns_three(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"hindsight_retain", "hindsight_recall", "hindsight_reflect"}

    def test_context_mode_returns_no_tools(self, provider_with_config):
        p = provider_with_config(memory_mode="context")
        assert p.get_tool_schemas() == []


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_values(self, provider):
        assert provider._auto_retain is True
        assert provider._auto_recall is True
        assert provider._retain_every_n_turns == 1
        assert provider._recall_max_tokens == 4096
        assert provider._recall_max_input_chars == 800
        assert provider._tags is None
        assert provider._recall_tags is None
        assert provider._bank_mission == ""
        assert provider._bank_retain_mission is None
        assert provider._retain_context == "conversation between Hermes Agent and the User"

    def test_custom_config_values(self, provider_with_config):
        p = provider_with_config(
            tags=["tag1", "tag2"],
            recall_tags=["recall-tag"],
            recall_tags_match="all",
            auto_retain=False,
            auto_recall=False,
            retain_every_n_turns=3,
            retain_context="custom-ctx",
            bank_retain_mission="Extract key facts",
            recall_max_tokens=2048,
            recall_types=["world", "experience"],
            recall_prompt_preamble="Custom preamble:",
            recall_max_input_chars=500,
            bank_mission="Test agent mission",
        )
        assert p._tags == ["tag1", "tag2"]
        assert p._recall_tags == ["recall-tag"]
        assert p._recall_tags_match == "all"
        assert p._auto_retain is False
        assert p._auto_recall is False
        assert p._retain_every_n_turns == 3
        assert p._retain_context == "custom-ctx"
        assert p._bank_retain_mission == "Extract key facts"
        assert p._recall_max_tokens == 2048
        assert p._recall_types == ["world", "experience"]
        assert p._recall_prompt_preamble == "Custom preamble:"
        assert p._recall_max_input_chars == 500
        assert p._bank_mission == "Test agent mission"

    def test_config_from_env_fallback(self, tmp_path, monkeypatch):
        """When no config file exists, falls back to env vars."""
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_MODE", "cloud")
        monkeypatch.setenv("HINDSIGHT_API_KEY", "env-key")
        monkeypatch.setenv("HINDSIGHT_BANK_ID", "env-bank")
        monkeypatch.setenv("HINDSIGHT_BUDGET", "high")

        cfg = _load_config()
        assert cfg["apiKey"] == "env-key"
        assert cfg["banks"]["hermes"]["bankId"] == "env-bank"
        assert cfg["banks"]["hermes"]["budget"] == "high"


# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_retain_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "user likes dark mode"}
        ))
        assert result["result"] == "Memory stored successfully."
        provider._client.aretain.assert_called_once()
        call_kwargs = provider._client.aretain.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["content"] == "user likes dark mode"

    def test_retain_with_tags(self, provider_with_config):
        p = provider_with_config(tags=["pref", "ui"])
        p.handle_tool_call("hindsight_retain", {"content": "likes dark mode"})
        call_kwargs = p._client.aretain.call_args.kwargs
        assert call_kwargs["tags"] == ["pref", "ui"]

    def test_retain_without_tags(self, provider):
        provider.handle_tool_call("hindsight_retain", {"content": "hello"})
        call_kwargs = provider._client.aretain.call_args.kwargs
        assert "tags" not in call_kwargs

    def test_retain_missing_content(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {}
        ))
        assert "error" in result

    def test_recall_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "dark mode"}
        ))
        assert "Memory 1" in result["result"]
        assert "Memory 2" in result["result"]

    def test_recall_passes_max_tokens(self, provider_with_config):
        p = provider_with_config(recall_max_tokens=2048)
        p.handle_tool_call("hindsight_recall", {"query": "test"})
        call_kwargs = p._client.arecall.call_args.kwargs
        assert call_kwargs["max_tokens"] == 2048

    def test_recall_passes_tags(self, provider_with_config):
        p = provider_with_config(recall_tags=["tag1"], recall_tags_match="all")
        p.handle_tool_call("hindsight_recall", {"query": "test"})
        call_kwargs = p._client.arecall.call_args.kwargs
        assert call_kwargs["tags"] == ["tag1"]
        assert call_kwargs["tags_match"] == "all"

    def test_recall_passes_types(self, provider_with_config):
        p = provider_with_config(recall_types=["world", "experience"])
        p.handle_tool_call("hindsight_recall", {"query": "test"})
        call_kwargs = p._client.arecall.call_args.kwargs
        assert call_kwargs["types"] == ["world", "experience"]

    def test_recall_no_results(self, provider):
        provider._client.arecall.return_value = SimpleNamespace(results=[])
        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "test"}
        ))
        assert result["result"] == "No relevant memories found."

    def test_recall_missing_query(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {}
        ))
        assert "error" in result

    def test_reflect_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_reflect", {"query": "summarize"}
        ))
        assert result["result"] == "Synthesized answer"

    def test_reflect_missing_query(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_reflect", {}
        ))
        assert "error" in result

    def test_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_unknown", {}
        ))
        assert "error" in result

    def test_retain_error_handling(self, provider):
        provider._client.aretain.side_effect = RuntimeError("connection failed")
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "test"}
        ))
        assert "error" in result
        assert "connection failed" in result["error"]

    def test_recall_error_handling(self, provider):
        provider._client.arecall.side_effect = RuntimeError("timeout")
        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "test"}
        ))
        assert "error" in result


# ---------------------------------------------------------------------------
# Prefetch tests
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_empty_when_no_result(self, provider):
        assert provider.prefetch("test") == ""

    def test_prefetch_default_preamble(self, provider):
        provider._prefetch_result = "- some memory"
        result = provider.prefetch("test")
        assert "Hindsight Memory" in result
        assert "- some memory" in result

    def test_prefetch_custom_preamble(self, provider_with_config):
        p = provider_with_config(recall_prompt_preamble="Custom header:")
        p._prefetch_result = "- memory line"
        result = p.prefetch("test")
        assert result.startswith("Custom header:")
        assert "- memory line" in result

    def test_queue_prefetch_skipped_in_tools_mode(self, provider_with_config):
        p = provider_with_config(memory_mode="tools")
        p.queue_prefetch("test")
        # Should not start a thread
        assert p._prefetch_thread is None

    def test_queue_prefetch_skipped_when_auto_recall_off(self, provider_with_config):
        p = provider_with_config(auto_recall=False)
        p.queue_prefetch("test")
        assert p._prefetch_thread is None

    def test_queue_prefetch_truncates_query(self, provider_with_config):
        p = provider_with_config(recall_max_input_chars=10)
        # Mock _run_sync to capture the query
        original_query = None

        def _capture_recall(**kwargs):
            nonlocal original_query
            original_query = kwargs.get("query", "")
            return SimpleNamespace(results=[])

        p._client.arecall = AsyncMock(side_effect=_capture_recall)

        long_query = "a" * 100
        p.queue_prefetch(long_query)
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)

        # The query passed to arecall should be truncated
        if original_query is not None:
            assert len(original_query) <= 10

    def test_queue_prefetch_passes_recall_params(self, provider_with_config):
        p = provider_with_config(
            recall_tags=["t1"],
            recall_tags_match="all",
            recall_max_tokens=1024,
            recall_types=["world"],
        )
        p.queue_prefetch("test query")
        if p._prefetch_thread:
            p._prefetch_thread.join(timeout=5.0)

        call_kwargs = p._client.arecall.call_args.kwargs
        assert call_kwargs["max_tokens"] == 1024
        assert call_kwargs["tags"] == ["t1"]
        assert call_kwargs["tags_match"] == "all"
        assert call_kwargs["types"] == ["world"]


# ---------------------------------------------------------------------------
# sync_turn tests
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def _get_retain_kwargs(self, provider):
        """Helper to get the kwargs from the aretain_batch call."""
        return provider._client.aretain_batch.call_args.kwargs

    def _get_retain_content(self, provider):
        """Helper to get the raw content string from the first item."""
        kwargs = self._get_retain_kwargs(provider)
        return kwargs["items"][0]["content"]

    def _get_retain_messages(self, provider):
        """Helper to parse the first turn's messages from retained content.

        Content is a JSON array of turns: [[msgs...], [msgs...], ...]
        For single-turn tests, returns the first turn's messages.
        """
        content = self._get_retain_content(provider)
        turns = json.loads(content)
        return turns[0] if len(turns) == 1 else turns

    def test_sync_turn_retains(self, provider):
        provider.sync_turn("hello", "hi there")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        provider._client.aretain_batch.assert_called_once()
        messages = self._get_retain_messages(provider)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello"
        assert "timestamp" in messages[0]
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "hi there"
        assert "timestamp" in messages[1]

    def test_sync_turn_skipped_when_auto_retain_off(self, provider_with_config):
        p = provider_with_config(auto_retain=False)
        p.sync_turn("hello", "hi")
        assert p._sync_thread is None
        p._client.aretain_batch.assert_not_called()

    def test_sync_turn_with_tags(self, provider_with_config):
        p = provider_with_config(tags=["conv", "session1"])
        p.sync_turn("hello", "hi")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["tags"] == ["conv", "session1"]

    def test_sync_turn_uses_aretain_batch(self, provider):
        """sync_turn should use aretain_batch with retain_async."""
        provider.sync_turn("hello", "hi")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        provider._client.aretain_batch.assert_called_once()
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        assert call_kwargs["document_id"] == "test-session"
        assert call_kwargs["retain_async"] is True
        assert len(call_kwargs["items"]) == 1
        assert call_kwargs["items"][0]["context"] == "conversation between Hermes Agent and the User"

    def test_sync_turn_custom_context(self, provider_with_config):
        p = provider_with_config(retain_context="my-agent")
        p.sync_turn("hello", "hi")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["context"] == "my-agent"

    def test_sync_turn_every_n_turns(self, provider_with_config):
        """With retain_every_n_turns=3, only retains on every 3rd turn."""
        p = provider_with_config(retain_every_n_turns=3)

        p.sync_turn("turn1-user", "turn1-asst")
        assert p._sync_thread is None  # not retained yet

        p.sync_turn("turn2-user", "turn2-asst")
        assert p._sync_thread is None  # not retained yet

        p.sync_turn("turn3-user", "turn3-asst")
        assert p._sync_thread is not None  # retained!
        p._sync_thread.join(timeout=5.0)

        p._client.aretain_batch.assert_called_once()
        content = p._client.aretain_batch.call_args.kwargs["items"][0]["content"]
        # Should contain all 3 turns
        assert "turn1-user" in content
        assert "turn2-user" in content
        assert "turn3-user" in content

    def test_sync_turn_accumulates_full_session(self, provider_with_config):
        """Each retain sends the ENTIRE session, not just the latest batch."""
        p = provider_with_config(retain_every_n_turns=2)

        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)

        p._client.aretain_batch.reset_mock()

        p.sync_turn("turn3-user", "turn3-asst")
        p.sync_turn("turn4-user", "turn4-asst")
        if p._sync_thread:
            p._sync_thread.join(timeout=5.0)

        content = p._client.aretain_batch.call_args.kwargs["items"][0]["content"]
        # Should contain ALL turns from the session
        assert "turn1-user" in content
        assert "turn2-user" in content
        assert "turn3-user" in content
        assert "turn4-user" in content

    def test_sync_turn_passes_document_id(self, provider):
        """sync_turn should pass session_id as document_id for dedup."""
        provider.sync_turn("hello", "hi")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        assert call_kwargs["document_id"] == "test-session"

    def test_sync_turn_error_does_not_raise(self, provider):
        """Errors in sync_turn should be swallowed (non-blocking)."""
        provider._client.aretain_batch.side_effect = RuntimeError("network error")
        provider.sync_turn("hello", "hi")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        # Should not raise


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_hybrid_mode_prompt(self, provider):
        block = provider.system_prompt_block()
        assert "Hindsight Memory" in block
        assert "hindsight_recall" in block
        assert "automatically injected" in block

    def test_context_mode_prompt(self, provider_with_config):
        p = provider_with_config(memory_mode="context")
        block = p.system_prompt_block()
        assert "context mode" in block
        assert "hindsight_recall" not in block

    def test_tools_mode_prompt(self, provider_with_config):
        p = provider_with_config(memory_mode="tools")
        block = p.system_prompt_block()
        assert "tools mode" in block
        assert "hindsight_recall" in block


# ---------------------------------------------------------------------------
# Config schema tests
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_has_all_new_fields(self, provider):
        schema = provider.get_config_schema()
        keys = {f["key"] for f in schema}
        expected_keys = {
            "mode", "api_url", "api_key", "llm_provider", "llm_api_key",
            "llm_model", "bank_id", "bank_mission", "bank_retain_mission",
            "recall_budget", "memory_mode", "recall_prefetch_method",
            "tags", "recall_tags", "recall_tags_match",
            "auto_recall", "auto_retain",
            "retain_every_n_turns", "retain_async",
            "retain_context",
            "recall_max_tokens", "recall_max_input_chars",
            "recall_prompt_preamble",
        }
        assert expected_keys.issubset(keys), f"Missing: {expected_keys - keys}"


# ---------------------------------------------------------------------------
# Availability tests
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_API_KEY", "test-key")
        p = HindsightMemoryProvider()
        assert p.is_available()

    def test_not_available_without_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        p = HindsightMemoryProvider()
        assert not p.is_available()

    def test_available_in_local_mode(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_MODE", "local")
        p = HindsightMemoryProvider()
        assert p.is_available()
