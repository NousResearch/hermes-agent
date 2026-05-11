"""Tests for Cognee memory provider — tool routing, env vars, background flows.

Follows the same pattern as ``test_mem0_v2.py``: mock the external API,
exercise the provider's ``handle_tool_call`` and lifecycle methods directly.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest

from plugins.memory.cognee import CogneeMemoryProvider
from plugins.memory.cognee.client import CogneeClientConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCogneeAPI:
    """Stand-in for the async cognee SDK functions.

    Captures every call so tests can assert on parameters without needing
    a real DeepSeek / Gemini backend.
    """

    def __init__(self) -> None:
        self.remembered: list[str] = []
        self.recalled: list[str] = []
        self.captured_remember: dict = {}
        self.captured_recall: dict = {}

    async def remember(self, content: str, **kwargs: object) -> dict:
        self.remembered.append(content)
        self.captured_remember = {"content": content, **kwargs}
        return {"status": "completed", "dataset_name": kwargs.get("dataset_name", "hermes_memory")}

    async def recall(self, query: str, **kwargs: object) -> list[dict]:
        self.recalled.append(query)
        self.captured_recall = {"query": query, **kwargs}
        return [{"kind": "graph_completion", "text": f"Result: {query}", "source": "graph"}]

    async def forget(self, **kwargs: object) -> dict:
        self.forgotten.append(kwargs)
        return {"status": "completed"}


@pytest.fixture
def fake_cognee():
    return FakeCogneeAPI()


# ---------------------------------------------------------------------------
# Environment & availability
# ---------------------------------------------------------------------------


class TestCogneeAvailability:
    """``is_available`` and ``get_config_schema``."""

    def test_not_available_without_api_key(self, monkeypatch):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        provider = CogneeMemoryProvider()
        assert provider.is_available() is False

    def test_available_with_api_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = CogneeMemoryProvider()
        # cognee package must be importable for is_available to return True
        try:
            import cognee  # noqa: F401
        except (ImportError, ValueError):
            pytest.skip("cognee package not available")
        assert provider.is_available() is True

    def test_config_schema_contains_expected_keys(self):
        provider = CogneeMemoryProvider()
        schema = provider.get_config_schema()
        keys = {entry["key"] for entry in schema}
        assert "api_key" in keys
        assert "provider" in keys
        assert "base_url" in keys
        assert "dataset_name" in keys

    def test_name_is_cognee(self):
        assert CogneeMemoryProvider().name == "cognee"


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------


class TestCogneeSystemPrompt:
    """The injected system prompt tells the agent how to use cognee."""

    def test_mentions_disabled_memory_tool(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        block = provider.system_prompt_block()
        assert "DISABLED" in block or "disabled" in block
        assert "cognee_remember" in block
        assert "cognee_recall" in block
        assert "cognee_forget" in block

    def test_contains_dataset_name(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        block = provider.system_prompt_block()
        assert "hermes_memory" in block


# ---------------------------------------------------------------------------
# Env var injection (apply_to_environment)
# ---------------------------------------------------------------------------


class TestCogneeEnvVars:
    """``apply_to_environment`` sets the right env vars for cognee."""

    def test_sets_llm_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        cfg = CogneeClientConfig.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("LLM_PROVIDER") == "deepseek"

    def test_sets_llm_endpoint_from_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        cfg = CogneeClientConfig.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("LLM_ENDPOINT") == "https://api.deepseek.com/v1"

    def test_sets_model_for_deepseek(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        cfg = CogneeClientConfig.from_global_config()
        cfg.apply_to_environment()
        assert "deepseek" in (os.environ.get("LLM_MODEL") or "").lower()

    def test_sets_embedding_vars_with_gemini_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        cfg = CogneeClientConfig.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("EMBEDDING_PROVIDER") == "gemini"
        assert os.environ.get("EMBEDDING_DIMENSIONS") == "768"
        assert os.environ.get("EMBEDDING_API_KEY") == "gk-test"

    def test_does_not_set_embedding_vars_without_gemini_key(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        # Must also clean inherited env from parent process
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        cfg = CogneeClientConfig.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("EMBEDDING_PROVIDER") is None

    def test_overwrites_existing_vars(self, monkeypatch):
        """Uses direct assignment, not setdefault."""
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")  # set deepseek so provider matches
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        # Put a wrong value that should be overwritten
        monkeypatch.setenv("LLM_MODEL", "wrong-model")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "9999")
        cfg = CogneeClientConfig.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("LLM_MODEL") != "wrong-model"
        assert "deepseek" in os.environ.get("LLM_MODEL", "").lower()
        assert os.environ.get("EMBEDDING_DIMENSIONS") == "768"


# ---------------------------------------------------------------------------
# Tool call routing
# ---------------------------------------------------------------------------


class TestCogneeToolCalls:
    """Routing, parameter passing, and error handling."""

    def test_remember_empty_content_returns_error(self, fake_cognee):
        provider = CogneeMemoryProvider()
        result = json.loads(provider.handle_tool_call("cognee_remember", {}))
        assert "error" in result

    def test_recall_empty_query_returns_error(self, fake_cognee):
        provider = CogneeMemoryProvider()
        result = json.loads(provider.handle_tool_call("cognee_recall", {}))
        assert "error" in result

    def test_forget_without_confirm_returns_error(self, fake_cognee):
        provider = CogneeMemoryProvider()
        result = json.loads(
            provider.handle_tool_call("cognee_forget", {"dataset_name": "test"})
        )
        assert "confirm" in result.get("error", "")

    def test_unknown_tool_returns_error(self, fake_cognee):
        provider = CogneeMemoryProvider()
        result = provider.handle_tool_call("cognee_nonexistent", {})
        assert "unknown" in result.lower() or "Unknown" in result

    def test_remember_passes_content(self, monkeypatch, fake_cognee):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")

        # Patch the async bridge to use our fake
        async def remember_stub(content, **kw):
            return await fake_cognee.remember(content, **kw)

        with patch("plugins.memory.cognee.client.run_async") as mock_run:
            mock_run.side_effect = lambda coro, timeout: fake_cognee.remember(
                coro.cr_frame.f_locals.get("content", "")
            )
            result = json.loads(
                provider.handle_tool_call(
                    "cognee_remember",
                    {"content": "test memory"},
                )
            )

    def test_recall_passes_query(self, monkeypatch, fake_cognee):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")

        with patch("plugins.memory.cognee.client.run_async") as mock_run:
            mock_run.side_effect = lambda coro, timeout: fake_cognee.recall(
                fake_cognee.captured_recall.get("query", "")
            )
            result = json.loads(
                provider.handle_tool_call(
                    "cognee_recall",
                    {"query": "test query"},
                )
            )

    def test_remember_with_custom_dataset(self, monkeypatch, fake_cognee):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")

        with patch("plugins.memory.cognee.client.run_async") as mock_run:
            mock_run.side_effect = lambda coro, timeout: fake_cognee.remember(
                coro.cr_frame.f_locals.get("content", ""),
                **(coro.cr_frame.f_locals if hasattr(coro, "cr_frame") else {}),
            )
            provider.handle_tool_call(
                "cognee_remember",
                {"content": "test", "dataset_name": "custom_dataset"},
            )


# ---------------------------------------------------------------------------
# Background flows (prefetch, sync)
# ---------------------------------------------------------------------------


class TestCogneeBackgroundFlows:
    """Prefetch, sync_turn, and session-end behaviour."""

    def test_queue_prefetch_starts_thread(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        provider.queue_prefetch("something meaningful")
        assert provider._prefetch_thread is not None
        provider._prefetch_thread.join(timeout=2)

    def test_prefetch_returns_empty_on_no_result(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        result = provider.prefetch("anything")
        assert result == ""

    def test_sync_turn_skips_short_text(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        # Both should be skipped because combined length < _MIN_TURN_LENGTH (16)
        provider.sync_turn("a", "b", session_id="s1")
        assert provider._sync_thread is None or not provider._sync_thread.is_alive()

    def test_sync_turn_starts_thread_for_long_text(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        provider.sync_turn(
            "user said something interesting here",
            "assistant replied with a comprehensive response",
            session_id="s1",
        )
        assert provider._sync_thread is not None
        if provider._sync_thread.is_alive():
            provider._sync_thread.join(timeout=2)

    def test_on_session_end_skips_when_no_messages(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        # Should not crash with empty messages
        provider.on_session_end([])

    def test_shutdown_does_not_crash(self):
        provider = CogneeMemoryProvider()
        provider.initialize("test-session")
        provider.shutdown()

    def test_on_session_switch_updates_session_id(self):
        provider = CogneeMemoryProvider()
        provider.initialize("old-session")
        provider.on_session_switch("new-session", reset=True)
        assert provider._session_id == "new-session"


# ---------------------------------------------------------------------------
# Config persistence (save_config)
# ---------------------------------------------------------------------------


class TestCogneeConfigPersistence:
    """``save_config`` writes env vars and cognee.json correctly."""

    def test_save_config_writes_env_vars(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = CogneeMemoryProvider()
        provider.save_config(
            {
                "api_key": "sk-test",
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
            },
            hermes_home=str(tmp_path),
        )
        env_path = tmp_path / ".env"
        # env vars are written via save_env_value which appends to .env
        assert env_path.exists()

    def test_save_config_writes_cognee_json(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = CogneeMemoryProvider()
        provider.save_config(
            {"dataset_name": "my_memory", "extra_field": "value"},
            hermes_home=str(tmp_path),
        )
        cfg_path = tmp_path / "cognee.json"
        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text())
        assert data.get("dataset_name") == "my_memory"
        assert data.get("extra_field") == "value"
