"""Tests for the Hindsight memory provider plugin.

Tests cover config loading, tool handlers (tags, max_tokens, types),
prefetch (auto_recall, preamble, query truncation), sync_turn (auto_retain,
turn counting, tags), and schema completeness.
"""

import json
import re
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
    _normalize_retain_tags,
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
        "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_SOURCE",
        "HINDSIGHT_RETAIN_USER_PREFIX", "HINDSIGHT_RETAIN_ASSISTANT_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_mock_client():
    """Create a mock Hindsight client with async methods."""
    async def _aretain(
        bank_id,
        content,
        timestamp=None,
        context=None,
        document_id=None,
        metadata=None,
        entities=None,
        tags=None,
        update_mode=None,
        retain_async=None,
    ):
        return SimpleNamespace(ok=True)

    client = MagicMock()
    client.aretain = AsyncMock(side_effect=_aretain)
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


class _FakeSessionDB:
    def __init__(self, messages=None):
        self._messages = list(messages or [])

    def get_messages_as_conversation(self, session_id):
        return list(self._messages)


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


def test_normalize_retain_tags_accepts_csv_and_dedupes():
    assert _normalize_retain_tags("agent:fakeassistantname, source_system:hermes-agent, agent:fakeassistantname") == [
        "agent:fakeassistantname",
        "source_system:hermes-agent",
    ]


def test_normalize_retain_tags_accepts_json_array_string():
    value = json.dumps(["agent:fakeassistantname", "source_system:hermes-agent"])
    assert _normalize_retain_tags(value) == ["agent:fakeassistantname", "source_system:hermes-agent"]


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_retain_schema_has_content(self):
        assert RETAIN_SCHEMA["name"] == "hindsight_retain"
        assert "content" in RETAIN_SCHEMA["parameters"]["properties"]
        assert "tags" in RETAIN_SCHEMA["parameters"]["properties"]
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
            retain_tags=["tag1", "tag2"],
            retain_source="hermes",
            retain_user_prefix="User (fakeusername)",
            retain_assistant_prefix="Assistant (fakeassistantname)",
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
        assert p._retain_tags == ["tag1", "tag2"]
        assert p._retain_source == "hermes"
        assert p._retain_user_prefix == "User (fakeusername)"
        assert p._retain_assistant_prefix == "Assistant (fakeassistantname)"
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
        p = provider_with_config(retain_tags=["pref", "ui"])
        p.handle_tool_call("hindsight_retain", {"content": "likes dark mode"})
        call_kwargs = p._client.aretain.call_args.kwargs
        assert call_kwargs["tags"] == ["pref", "ui"]

    def test_retain_merges_per_call_tags_with_config_tags(self, provider_with_config):
        p = provider_with_config(retain_tags=["pref", "ui"])
        p.handle_tool_call(
            "hindsight_retain",
            {"content": "likes dark mode", "tags": ["client:x", "ui"]},
        )
        call_kwargs = p._client.aretain.call_args.kwargs
        assert call_kwargs["tags"] == ["pref", "ui", "client:x"]

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
    def test_sync_turn_retains_metadata_rich_turn(self, provider_with_config):
        p = provider_with_config(
            retain_tags=["conv", "session1"],
            retain_source="hermes",
            retain_user_prefix="User (fakeusername)",
            retain_assistant_prefix="Assistant (fakeassistantname)",
        )
        p.initialize(
            session_id="session-1",
            platform="discord",
            user_id="fakeusername-123",
            user_name="fakeusername",
            chat_id="1485316232612941897",
            chat_name="fakeassistantname-forums",
            chat_type="thread",
            thread_id="1491249007475949698",
            agent_identity="fakeassistantname",
        )
        p._client = _make_mock_client()

        p.sync_turn("hello", "hi there")
        p._sync_thread.join(timeout=5.0)

        p._client.aretain_batch.assert_called_once()
        call_kwargs = p._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["document_id"] == "session-1"
        assert call_kwargs["retain_async"] is True
        assert len(call_kwargs["items"]) == 1
        item = call_kwargs["items"][0]
        assert item["context"] == "conversation between Hermes Agent and the User"
        assert item["tags"] == ["conv", "session1"]
        content = json.loads(item["content"])
        assert len(content) == 1
        assert content[0][0]["role"] == "user"
        assert content[0][0]["content"] == "User (fakeusername): hello"
        assert content[0][1]["role"] == "assistant"
        assert content[0][1]["content"] == "Assistant (fakeassistantname): hi there"
        assert item["metadata"]["source"] == "hermes"
        assert item["metadata"]["session_id"] == "session-1"
        assert item["metadata"]["platform"] == "discord"
        assert item["metadata"]["user_id"] == "fakeusername-123"
        assert item["metadata"]["user_name"] == "fakeusername"
        assert item["metadata"]["chat_id"] == "1485316232612941897"
        assert item["metadata"]["chat_name"] == "fakeassistantname-forums"
        assert item["metadata"]["chat_type"] == "thread"
        assert item["metadata"]["thread_id"] == "1491249007475949698"
        assert item["metadata"]["agent_identity"] == "fakeassistantname"
        assert item["metadata"]["turn_index"] == "1"
        assert item["metadata"]["message_count"] == "2"
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?\+00:00", content[0][0]["timestamp"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", item["metadata"]["retained_at"])

    def test_sync_turn_skipped_when_auto_retain_off(self, provider_with_config):
        p = provider_with_config(auto_retain=False)
        p.sync_turn("hello", "hi")
        assert p._sync_thread is None
        p._client.aretain_batch.assert_not_called()

    def test_sync_turn_every_n_turns(self, provider_with_config):
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        p.sync_turn("turn1-user", "turn1-asst")
        assert p._sync_thread is None
        p.sync_turn("turn2-user", "turn2-asst")
        assert p._sync_thread is None
        p.sync_turn("turn3-user", "turn3-asst")
        p._sync_thread.join(timeout=5.0)
        p._client.aretain_batch.assert_called_once()
        call_kwargs = p._client.aretain_batch.call_args.kwargs
        assert call_kwargs["document_id"] == "test-session"
        assert call_kwargs["retain_async"] is False
        item = call_kwargs["items"][0]
        content = json.loads(item["content"])
        assert len(content) == 3
        assert content[-1][0]["role"] == "user"
        assert content[-1][0]["content"] == "User: turn3-user"
        assert content[-1][1]["role"] == "assistant"
        assert content[-1][1]["content"] == "Assistant: turn3-asst"
        assert item["metadata"]["turn_index"] == "3"
        assert item["metadata"]["message_count"] == "6"

    def test_sync_turn_error_does_not_raise(self, provider):
        provider._client.aretain_batch.side_effect = RuntimeError("network error")
        provider.sync_turn("hello", "hi")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)

    def test_sync_turn_preserves_unicode(self, provider_with_config):
        """Non-ASCII text (CJK, ZWJ emoji) must survive JSON round-trip intact."""
        p = provider_with_config()
        p._client = _make_mock_client()
        p.sync_turn("안녕 こんにちは 你好", "👨‍👩‍👧‍👦 family")
        p._sync_thread.join(timeout=5.0)
        p._client.aretain_batch.assert_called_once()
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        # ensure_ascii=False means non-ASCII chars appear as-is in the raw JSON,
        # not as \uXXXX escape sequences.
        raw_json = item["content"]
        assert "안녕" in raw_json
        assert "こんにちは" in raw_json
        assert "你好" in raw_json
        assert "👨‍👩‍👧‍👦" in raw_json


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
            "retain_tags", "retain_source",
            "retain_user_prefix", "retain_assistant_prefix",
            "recall_tags", "recall_tags_match",
            "auto_recall", "auto_retain",
            "retain_every_n_turns", "retain_async", "retain_context",
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

    def test_available_with_snake_case_api_key_in_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "mode": "cloud",
            "api_key": "config-key",
        }))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path,
        )

        p = HindsightMemoryProvider()

        assert p.is_available()
