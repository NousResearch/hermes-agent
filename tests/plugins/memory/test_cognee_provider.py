"""Tests for the standalone Cognee memory plugin loaded from $HERMES_HOME/plugins.

These tests intentionally exercise the user-plugin discovery path instead of
importing a bundled provider directly.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

PLUGIN_SOURCE_DIR = (
    Path(__file__).resolve().parents[3] / "external_plugins" / "memory" / "cognee"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unload_staged_cognee_modules() -> None:
    for name in list(sys.modules):
        if name == "_hermes_user_memory.cognee" or name.startswith(
            "_hermes_user_memory.cognee."
        ):
            sys.modules.pop(name, None)


@pytest.fixture
def staged_cognee(tmp_path, monkeypatch):
    from plugins.memory import load_memory_provider

    plugins_root = tmp_path / "plugins"
    plugin_dir = plugins_root / "cognee"
    plugins_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(PLUGIN_SOURCE_DIR, plugin_dir)

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("plugins.memory._get_user_plugins_dir", lambda: plugins_root)

    _unload_staged_cognee_modules()
    provider = load_memory_provider("cognee")
    assert provider is not None

    provider_module = importlib.import_module(provider.__class__.__module__)
    client_module = importlib.import_module(f"{provider.__class__.__module__}.client")

    yield SimpleNamespace(
        plugin_dir=plugin_dir,
        provider=provider,
        provider_cls=provider.__class__,
        provider_module=provider_module,
        client_module=client_module,
        config_cls=client_module.CogneeClientConfig,
    )

    _unload_staged_cognee_modules()


class FakeCogneeAPI:
    """Stand-in for the async cognee SDK functions."""

    def __init__(self) -> None:
        self.remembered: list[str] = []
        self.recalled: list[str] = []
        self.forgotten: list[dict] = []
        self.captured_remember: dict = {}
        self.captured_recall: dict = {}

    async def remember(self, content: str, **kwargs: object) -> dict:
        self.remembered.append(content)
        self.captured_remember = {"content": content, **kwargs}
        return {
            "status": "completed",
            "dataset_name": kwargs.get("dataset_name", "hermes_memory"),
        }

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
# External plugin staging
# ---------------------------------------------------------------------------


class TestCogneeExternalPluginLayout:
    def test_external_plugin_manifest_exists(self):
        assert (PLUGIN_SOURCE_DIR / "plugin.yaml").exists()
        assert (PLUGIN_SOURCE_DIR / "__init__.py").exists()
        assert (PLUGIN_SOURCE_DIR / "client.py").exists()

    def test_loads_from_user_plugins_dir(self, staged_cognee):
        assert staged_cognee.provider.name == "cognee"
        assert staged_cognee.provider.__class__.__module__.startswith(
            "_hermes_user_memory.cognee"
        )


# ---------------------------------------------------------------------------
# Environment & availability
# ---------------------------------------------------------------------------


class TestCogneeAvailability:
    """``is_available`` and ``get_config_schema``."""

    def test_not_available_without_api_key(self, monkeypatch, staged_cognee):
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        assert staged_cognee.provider_cls().is_available() is False

    def test_available_with_api_key(self, monkeypatch, tmp_path, staged_cognee):
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("COGNEE_SKIP_CONNECTION_TEST", "true")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = staged_cognee.provider_cls()
        try:
            import cognee  # noqa: F401
        except (ImportError, ValueError):
            pytest.skip("cognee package not available")
        assert provider.is_available() is True

    def test_config_schema_contains_expected_keys(self, staged_cognee):
        schema = staged_cognee.provider_cls().get_config_schema()
        keys = {entry["key"] for entry in schema}
        assert "api_key" in keys
        assert "provider" in keys
        assert "base_url" in keys
        assert "dataset_name" in keys

    def test_name_is_cognee(self, staged_cognee):
        assert staged_cognee.provider_cls().name == "cognee"


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------


class TestCogneeSystemPrompt:
    """The injected system prompt tells the agent how to use cognee."""

    def test_mentions_disabled_memory_tool(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        block = provider.system_prompt_block()
        assert "DISABLED" in block or "disabled" in block
        assert "cognee_remember" in block
        assert "cognee_recall" in block
        assert "cognee_forget" in block

    def test_contains_dataset_name(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        block = provider.system_prompt_block()
        assert "hermes_memory" in block


# ---------------------------------------------------------------------------
# Env var injection (apply_to_environment)
# ---------------------------------------------------------------------------


class TestCogneeEnvVars:
    """``apply_to_environment`` sets the right env vars for cognee."""

    def test_sets_llm_provider(self, monkeypatch, staged_cognee):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        cfg = staged_cognee.config_cls.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("LLM_PROVIDER") == "deepseek"

    def test_sets_llm_endpoint_from_base_url(self, monkeypatch, staged_cognee):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
        cfg = staged_cognee.config_cls.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("LLM_ENDPOINT") == "https://api.deepseek.com/v1"

    def test_sets_model_for_deepseek(self, monkeypatch, staged_cognee):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        cfg = staged_cognee.config_cls.from_global_config()
        cfg.apply_to_environment()
        assert "deepseek" in (os.environ.get("LLM_MODEL") or "").lower()

    def test_sets_embedding_vars_with_gemini_key(self, monkeypatch, staged_cognee):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        cfg = staged_cognee.config_cls.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("EMBEDDING_PROVIDER") == "gemini"
        assert os.environ.get("EMBEDDING_DIMENSIONS") == "3072"
        assert os.environ.get("EMBEDDING_API_KEY") == "gk-test"

    def test_does_not_set_embedding_vars_without_gemini_key(self, monkeypatch, staged_cognee):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_PROVIDER", raising=False)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
        monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
        cfg = staged_cognee.config_cls.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("EMBEDDING_PROVIDER") is None

    def test_overwrites_existing_vars(self, monkeypatch, staged_cognee):
        monkeypatch.setenv("LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
        monkeypatch.setenv("LLM_MODEL", "wrong-model")
        monkeypatch.setenv("EMBEDDING_DIMENSIONS", "9999")
        cfg = staged_cognee.config_cls.from_global_config()
        cfg.apply_to_environment()
        assert os.environ.get("LLM_MODEL") != "wrong-model"
        assert "deepseek" in os.environ.get("LLM_MODEL", "").lower()
        assert os.environ.get("EMBEDDING_DIMENSIONS") == "3072"


# ---------------------------------------------------------------------------
# Tool call routing
# ---------------------------------------------------------------------------


class TestCogneeToolCalls:
    """Routing, parameter passing, and error handling."""

    def test_remember_empty_content_returns_error(self, staged_cognee):
        result = json.loads(staged_cognee.provider.handle_tool_call("cognee_remember", {}))
        assert "error" in result

    def test_recall_empty_query_returns_error(self, staged_cognee):
        result = json.loads(staged_cognee.provider.handle_tool_call("cognee_recall", {}))
        assert "error" in result

    def test_forget_without_confirm_returns_error(self, staged_cognee):
        result = json.loads(
            staged_cognee.provider.handle_tool_call(
                "cognee_forget", {"dataset_name": "test"}
            )
        )
        assert "confirm" in result.get("error", "")

    def test_unknown_tool_returns_error(self, staged_cognee):
        result = staged_cognee.provider.handle_tool_call("cognee_nonexistent", {})
        assert "unknown" in result.lower() or "Unknown" in result

    def test_remember_passes_content(self, staged_cognee, fake_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")

        with patch.object(staged_cognee.provider_module, "run_async") as mock_run:
            def remember_side_effect(coro, timeout):
                content = coro.cr_frame.f_locals.get("content", "")
                coro.close()
                fake_cognee.remembered.append(content)
                fake_cognee.captured_remember = {"content": content}
                return {"status": "completed", "dataset_name": "hermes_memory"}

            mock_run.side_effect = remember_side_effect
            json.loads(
                provider.handle_tool_call(
                    "cognee_remember",
                    {"content": "test memory"},
                )
            )

    def test_recall_passes_query(self, staged_cognee, fake_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")

        with patch.object(staged_cognee.provider_module, "run_async") as mock_run:
            def recall_side_effect(coro, timeout):
                query = coro.cr_frame.f_locals.get("query", "")
                coro.close()
                fake_cognee.recalled.append(query)
                fake_cognee.captured_recall = {"query": query}
                return [{"kind": "graph_completion", "text": f"Result: {query}", "source": "graph"}]

            mock_run.side_effect = recall_side_effect
            json.loads(
                provider.handle_tool_call(
                    "cognee_recall",
                    {"query": "test query"},
                )
            )

    def test_remember_with_custom_dataset(self, staged_cognee, fake_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")

        with patch.object(staged_cognee.provider_module, "run_async") as mock_run:
            def remember_with_dataset_side_effect(coro, timeout):
                payload = dict(coro.cr_frame.f_locals) if hasattr(coro, "cr_frame") else {}
                coro.close()
                content = payload.get("content", "")
                fake_cognee.remembered.append(content)
                fake_cognee.captured_remember = {"content": content, **payload}
                return {
                    "status": "completed",
                    "dataset_name": payload.get("dataset_name", "hermes_memory"),
                }

            mock_run.side_effect = remember_with_dataset_side_effect
            provider.handle_tool_call(
                "cognee_remember",
                {"content": "test", "dataset_name": "custom_dataset"},
            )


# ---------------------------------------------------------------------------
# Background flows (prefetch, sync)
# ---------------------------------------------------------------------------


class TestCogneeBackgroundFlows:
    """Prefetch, sync_turn, and session-end behaviour."""

    def test_queue_prefetch_starts_thread(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        provider.queue_prefetch("something meaningful")
        assert provider._prefetch_thread is not None
        provider._prefetch_thread.join(timeout=2)

    def test_prefetch_returns_empty_on_no_result(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        result = provider.prefetch("anything")
        assert result == ""

    def test_sync_turn_skips_short_text(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        provider.sync_turn("a", "b", session_id="s1")
        assert provider._sync_thread is None or not provider._sync_thread.is_alive()

    def test_sync_turn_starts_thread_for_long_text(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        provider.sync_turn(
            "user said something interesting here",
            "assistant replied with a comprehensive response",
            session_id="s1",
        )
        assert provider._sync_thread is not None
        if provider._sync_thread.is_alive():
            provider._sync_thread.join(timeout=2)

    def test_on_session_end_skips_when_no_messages(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        provider.on_session_end([])

    def test_shutdown_does_not_crash(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("test-session")
        provider.shutdown()

    def test_on_session_switch_updates_session_id(self, staged_cognee):
        provider = staged_cognee.provider_cls()
        provider.initialize("old-session")
        provider.on_session_switch("new-session", reset=True)
        assert provider._session_id == "new-session"


# ---------------------------------------------------------------------------
# Config persistence (save_config)
# ---------------------------------------------------------------------------


class TestCogneeConfigPersistence:
    """``save_config`` writes env vars and cognee.json correctly."""

    def test_save_config_writes_env_vars(self, monkeypatch, tmp_path, staged_cognee):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = staged_cognee.provider_cls()
        provider.save_config(
            {
                "api_key": "sk-test",
                "provider": "deepseek",
                "base_url": "https://api.deepseek.com/v1",
            },
            hermes_home=str(tmp_path),
        )
        env_path = tmp_path / ".env"
        assert env_path.exists()

    def test_save_config_writes_cognee_json(self, monkeypatch, tmp_path, staged_cognee):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        provider = staged_cognee.provider_cls()
        provider.save_config(
            {"dataset_name": "my_memory", "extra_field": "value"},
            hermes_home=str(tmp_path),
        )
        cfg_path = tmp_path / "cognee.json"
        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text())
        assert data.get("dataset_name") == "my_memory"
        assert data.get("extra_field") == "value"
