"""Tests for the Hindsight memory provider plugin.

Tests cover config loading, tool handlers (tags, max_tokens, types),
prefetch (auto_recall, preamble, query truncation), sync_turn (auto_retain,
turn counting, tags), and schema completeness.
"""

import json
import os
import re
import stat
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_cli.memory_setup import _CANCELLED
from plugins.memory.hindsight import (
    HindsightMemoryProvider,
    RECALL_SCHEMA,
    REFLECT_SCHEMA,
    RETAIN_SCHEMA,
    _load_config,
    _build_embedded_profile_env,
    _normalize_observation_scopes,
    _normalize_retain_tags,
    _resolve_bank_id_template,
    _run_sync,
    _sanitize_bank_segment,
    _sanitize_hindsight_input,
)
from plugins.memory.hindsight.session_summary import SessionSummaryWrite


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_run_sync_cold_start_returns_without_timeout():
    async def sample():
        return "ok"

    assert _run_sync(sample(), timeout=5) == "ok"


def test_run_sync_falls_back_when_shared_loop_unavailable(monkeypatch):
    async def sample():
        return "ok"

    def fail_loop():
        raise RuntimeError("loop blocked")

    monkeypatch.setattr("plugins.memory.hindsight._get_loop", fail_loop)
    assert _run_sync(sample(), timeout=5) == "ok"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure no stale env vars leak between tests."""
    for key in (
        "HINDSIGHT_API_KEY", "HINDSIGHT_API_URL", "HINDSIGHT_BANK_ID",
        "HINDSIGHT_BUDGET", "HINDSIGHT_MODE", "HINDSIGHT_TIMEOUT",
        "HINDSIGHT_IDLE_TIMEOUT", "HINDSIGHT_LLM_API_KEY",
        "HINDSIGHT_RETAIN_TAGS", "HINDSIGHT_RETAIN_OBSERVATION_SCOPES",
        "HINDSIGHT_RETAIN_SOURCE",
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


def _provider_for_mode(tmp_path, monkeypatch, mode: str):
    """Create an initialized provider without pre-seeding its client."""
    config = {
        "mode": mode,
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

    provider = HindsightMemoryProvider()
    provider.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
    return provider


def _assert_cloud_client_lazy_installed_before_import(tmp_path, monkeypatch, mode: str):
    """Cloud/local-external clients must ensure lazy deps before importing."""
    import builtins

    provider = _provider_for_mode(tmp_path, monkeypatch, mode)
    ensure_calls = []

    def fake_ensure(feature, prompt=True):
        ensure_calls.append((feature, prompt))

    class FakeHindsight:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "hindsight_client":
            if ensure_calls != [("memory.hindsight", False)]:
                raise ModuleNotFoundError("No module named 'hindsight_client'")
            return SimpleNamespace(Hindsight=FakeHindsight)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("tools.lazy_deps.ensure", fake_ensure)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    client = provider._get_client()

    assert ensure_calls == [("memory.hindsight", False)]
    assert isinstance(client, FakeHindsight)
    assert client.kwargs == {
        "base_url": "http://localhost:9999",
        "timeout": 120.0,
        "api_key": "test-key",
    }


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


def test_normalize_observation_scopes_empty_is_none():
    assert _normalize_observation_scopes("") is None
    assert _normalize_observation_scopes(None) is None
    assert _normalize_observation_scopes("   ") is None


def test_normalize_observation_scopes_keywords_pass_through():
    assert _normalize_observation_scopes("per_tag") == "per_tag"
    assert _normalize_observation_scopes("combined") == "combined"
    assert _normalize_observation_scopes(" all_combinations ") == "all_combinations"


def test_normalize_observation_scopes_unknown_keyword_is_none():
    assert _normalize_observation_scopes("nonsense") is None


def test_normalize_observation_scopes_json_list_of_lists():
    value = json.dumps([["user:alice"], ["team:eng"], ["user:alice", "team:eng"]])
    assert _normalize_observation_scopes(value) == [
        ["user:alice"],
        ["team:eng"],
        ["user:alice", "team:eng"],
    ]


def test_normalize_observation_scopes_flat_list_is_single_scope():
    assert _normalize_observation_scopes(["user:alice", "team:eng"]) == [
        ["user:alice", "team:eng"]
    ]


def test_normalize_observation_scopes_list_of_lists():
    assert _normalize_observation_scopes([["user:alice"], ["team:eng"]]) == [
        ["user:alice"],
        ["team:eng"],
    ]


def test_sanitize_hindsight_input_strips_runtime_identifiers():
    dirty = """
Conversation info (untrusted metadata):
```json
{"chat_id":"user:ou_cb923a19782fe748cd9fff99454eee31","message_id":"om_x100b6d364e7e60a0c49f20c620fbbc6"}
```
[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]
open_id: ou_cb923a19782fe748cd9fff99454eee31
ou_cb923a19782fe748cd9fff99454eee31: 查询 retain 语义
"""
    clean = _sanitize_hindsight_input(dirty)

    assert clean == "查询 retain 语义"
    assert "message_id" not in clean
    assert "om_x100" not in clean
    assert "ou_cb" not in clean


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
    def test_cloud_client_lazy_installs_dependency_before_import(self, tmp_path, monkeypatch):
        _assert_cloud_client_lazy_installed_before_import(tmp_path, monkeypatch, "cloud")

    def test_local_external_client_lazy_installs_dependency_before_import(self, tmp_path, monkeypatch):
        _assert_cloud_client_lazy_installed_before_import(
            tmp_path, monkeypatch, "local_external"
        )

    def test_default_values(self, provider):
        assert provider._auto_retain is True
        assert provider._auto_recall is True
        assert provider._retain_every_n_turns == 1
        assert provider._recall_max_tokens == 4096
        assert provider._recall_max_input_chars == 800
        assert provider._session_summary_enabled is False
        assert provider._session_summary_enrich_recall_query is False
        assert provider._session_summary_update_every_n_turns is None
        assert provider._session_summary_min_update_every_n_turns == 2
        assert provider._session_summary_timeout_seconds == 20
        assert provider._session_summary_budget.max_input_chars == 16000
        assert provider._session_summary_budget.max_output_chars == 2000
        assert provider._session_summary_budget.max_recall_query_chars == 800
        assert provider._session_summary_budget.recall_query_budget_ratio == 0.25
        assert provider._session_summary_budget.max_prompt_inject_chars == 1200
        assert provider._session_summary_budget.max_retain_context_chars == 1200
        assert provider._session_summary_budget.min_latest_query_reserve_chars == 400
        assert provider._tags is None
        assert provider._observation_scopes is None
        assert provider._recall_tags is None
        # Default recall narrowed to observation-only; world/experience are
        # aggregate facts that often crowd out concrete-event signal during
        # auto-recall. Users opt back in via the recall_types config key.
        assert provider._recall_types == ["observation"]
        assert provider._bank_mission == ""
        assert provider._bank_retain_mission is None
        assert provider._retain_context == "conversation between Hermes Agent and the User"

    def test_recall_types_default_is_observation_only(self, provider):
        """Auto-recall must filter to observation by default."""
        assert provider._recall_types == ["observation"]

    def test_recall_types_explicit_list_overrides_default(self, provider_with_config):
        p = provider_with_config(recall_types=["world", "experience", "observation"])
        assert p._recall_types == ["world", "experience", "observation"]

    def test_recall_types_csv_string_accepted(self, provider_with_config):
        """For parity with recall_tags, comma-separated strings work too."""
        p = provider_with_config(recall_types="observation, world")
        assert p._recall_types == ["observation", "world"]

    def test_recall_types_empty_list_falls_back_to_default(self, provider_with_config):
        """An empty list shouldn't disable the filter (would be wider than default)."""
        p = provider_with_config(recall_types=[])
        assert p._recall_types == ["observation"]

    def test_observation_scopes_keyword_config(self, provider_with_config):
        p = provider_with_config(observation_scopes="per_tag")
        assert p._observation_scopes == "per_tag"

    def test_observation_scopes_custom_list_config(self, provider_with_config):
        p = provider_with_config(
            observation_scopes=[["user:alice"], ["team:eng"]]
        )
        assert p._observation_scopes == [["user:alice"], ["team:eng"]]

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
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
            session_summary_generator_provider="openai_compatible",
            session_summary_generator_model="summary-model",
            session_summary_generator_base_url="http://localhost:11434/v1",
            session_summary_generator_api_key_env="SUMMARY_API_KEY",
            session_summary_reuse_hindsight_llm_config=False,
            session_summary_update_every_n_turns=7,
            session_summary_min_update_every_n_turns=3,
            session_summary_timeout_seconds=9,
            session_summary_max_input_chars=1234,
            session_summary_max_output_chars=432,
            session_summary_max_recall_query_chars=321,
            session_summary_recall_query_budget_ratio=0.2,
            session_summary_min_latest_query_reserve_chars=111,
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
        assert p._session_summary_enabled is True
        assert p._session_summary_enrich_recall_query is True
        assert p._session_summary_generator_provider == "openai_compatible"
        assert p._session_summary_generator_model == "summary-model"
        assert p._session_summary_generator_base_url == "http://localhost:11434/v1"
        assert p._session_summary_generator_api_key_env == "SUMMARY_API_KEY"
        assert p._session_summary_reuse_hindsight_llm_config is False
        assert p._session_summary_update_every_n_turns == 7
        assert p._session_summary_min_update_every_n_turns == 3
        assert p._session_summary_timeout_seconds == 9
        assert p._session_summary_budget.max_input_chars == 1234
        assert p._session_summary_budget.max_output_chars == 432
        assert p._session_summary_budget.max_recall_query_chars == 321
        assert p._session_summary_budget.recall_query_budget_ratio == 0.2
        assert p._session_summary_budget.max_prompt_inject_chars == 1200
        assert p._session_summary_budget.max_retain_context_chars == 1200
        assert p._session_summary_budget.min_latest_query_reserve_chars == 111

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

    def test_embedded_profile_env_includes_idle_timeout_from_config(self):
        env = _build_embedded_profile_env({
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
            "idle_timeout": 0,
        })

        assert env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "0"

    def test_embedded_profile_env_includes_idle_timeout_from_env(self, monkeypatch):
        monkeypatch.setenv("HINDSIGHT_IDLE_TIMEOUT", "42")

        env = _build_embedded_profile_env({
            "llm_provider": "openai",
            "llm_model": "gpt-4o-mini",
        })

        assert env["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "42"

    def test_get_client_passes_idle_timeout_to_hindsight_embedded(self, monkeypatch):
        captured = {}

        class FakeHindsightEmbedded:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setitem(sys.modules, "hindsight", SimpleNamespace(HindsightEmbedded=FakeHindsightEmbedded))
        monkeypatch.setattr("plugins.memory.hindsight._check_local_runtime", lambda: (True, ""))

        p = HindsightMemoryProvider()
        p._mode = "local_embedded"
        p._config = {
            "profile": "hermes",
            "llm_provider": "openai_compatible",
            "llm_api_key": "test-key",
            "llm_model": "test-model",
            "idle_timeout": 0,
        }
        p._llm_base_url = "http://localhost:8060/v1"

        p._get_client()

        assert captured["idle_timeout"] == 0
        assert captured["llm_provider"] == "openai"


class TestPostSetup:
    def test_setup_cancel_at_mode_picker_writes_nothing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: hermes_home)

        save_config = MagicMock()
        which = MagicMock(return_value="/usr/bin/uv")
        run = MagicMock()
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: _CANCELLED)
        monkeypatch.setattr("shutil.which", which)
        monkeypatch.setattr("subprocess.run", run)
        monkeypatch.setattr("builtins.input", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("getpass.getpass", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("hermes_cli.config.save_config", save_config)

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {"provider": "builtin"}})

        save_config.assert_not_called()
        which.assert_not_called()
        run.assert_not_called()
        assert not (hermes_home / ".env").exists()
        assert not (hermes_home / "hindsight" / "config.json").exists()
        assert not (user_home / ".hindsight" / "profiles" / "hermes.env").exists()

    def test_local_embedded_setup_cancel_at_llm_picker_writes_nothing(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: hermes_home)

        selections = iter([1, _CANCELLED])  # local_embedded, then cancel LLM picker
        save_config = MagicMock()
        which = MagicMock(return_value="/usr/bin/uv")
        run = MagicMock()
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: next(selections))
        monkeypatch.setattr("shutil.which", which)
        monkeypatch.setattr("subprocess.run", run)
        monkeypatch.setattr("builtins.input", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("getpass.getpass", MagicMock(side_effect=AssertionError("prompt should not run")))
        monkeypatch.setattr("hermes_cli.config.save_config", save_config)

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {"provider": "builtin"}})

        save_config.assert_not_called()
        which.assert_not_called()
        run.assert_not_called()
        assert not (hermes_home / ".env").exists()
        assert not (hermes_home / "hindsight" / "config.json").exists()
        assert not (user_home / ".hindsight" / "profiles" / "hermes.env").exists()

    def test_local_embedded_setup_materializes_profile_env(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))

        selections = iter([1, 0])  # local_embedded, openai
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: next(selections))
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-local-test")
        saved_configs = []
        monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: saved_configs.append(cfg.copy()))

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {}})

        assert saved_configs[-1]["memory"]["provider"] == "hindsight"
        env_text = (hermes_home / ".env").read_text()
        assert "HINDSIGHT_LLM_API_KEY=sk-local-test\n" in env_text
        assert "HINDSIGHT_TIMEOUT=120\n" in env_text
        assert "HINDSIGHT_IDLE_TIMEOUT=300\n" in env_text

        profile_env = user_home / ".hindsight" / "profiles" / "hermes.env"
        assert profile_env.exists()
        assert profile_env.read_text() == (
            "HINDSIGHT_API_LLM_PROVIDER=openai\n"
            "HINDSIGHT_API_LLM_API_KEY=sk-local-test\n"
            "HINDSIGHT_API_LLM_MODEL=gpt-4o-mini\n"
            "HINDSIGHT_API_LOG_LEVEL=info\n"
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT=300\n"
        )

    def test_local_embedded_setup_respects_existing_profile_name(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))

        selections = iter([1, 0])  # local_embedded, openai
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: next(selections))
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-local-test")
        monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

        provider = HindsightMemoryProvider()
        provider.save_config({"profile": "coder"}, str(hermes_home))
        provider.post_setup(str(hermes_home), {"memory": {}})

        coder_env = user_home / ".hindsight" / "profiles" / "coder.env"
        hermes_env = user_home / ".hindsight" / "profiles" / "hermes.env"
        assert coder_env.exists()
        assert not hermes_env.exists()

    def test_local_embedded_setup_preserves_existing_key_when_input_left_blank(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))

        selections = iter([1, 0])  # local_embedded, openai
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: next(selections))
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "")
        monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

        env_path = hermes_home / ".env"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text("HINDSIGHT_LLM_API_KEY=existing-key\n")

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {}})

        profile_env = user_home / ".hindsight" / "profiles" / "hermes.env"
        assert profile_env.exists()
        assert "HINDSIGHT_API_LLM_API_KEY=existing-key\n" in profile_env.read_text()


    def test_local_embedded_setup_blank_inputs_preserve_existing_config(self, tmp_path, monkeypatch):
        """Pressing Enter through setup should keep existing Hindsight values."""
        hermes_home = tmp_path / "hermes-home"
        user_home = tmp_path / "user-home"
        user_home.mkdir()
        monkeypatch.setenv("HOME", str(user_home))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: hermes_home)

        existing_config = {
            "mode": "local_embedded",
            "llm_provider": "openai_compatible",
            "llm_base_url": "http://192.168.1.161:8060/v1",
            "llm_api_key": "9913",
            "llm_model": "gemma-4-26B-A4B-it-heretic-oQ4",
            "bank_id": "hermes",
            "recall_budget": "mid",
            "idle_timeout": 0,
            "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT": "0",
            "HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE": "1",
            "timeout": 120,
        }
        provider = HindsightMemoryProvider()
        provider.save_config(existing_config, str(hermes_home))

        # Simulate pressing Enter at the mode and LLM-provider pickers, which
        # should select their current values, and pressing Enter at text prompts.
        monkeypatch.setattr("hermes_cli.memory_setup._curses_select", lambda *args, **kwargs: kwargs.get("default", 0))
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("getpass.getpass", lambda prompt="": "")
        monkeypatch.setattr("hermes_cli.config.save_config", lambda cfg: None)

        provider = HindsightMemoryProvider()
        provider.post_setup(str(hermes_home), {"memory": {}})

        saved = json.loads((hermes_home / "hindsight" / "config.json").read_text())
        assert saved["mode"] == "local_embedded"
        assert saved["llm_provider"] == "openai_compatible"
        assert saved["llm_base_url"] == "http://192.168.1.161:8060/v1"
        assert saved["llm_api_key"] == "9913"
        assert saved["llm_model"] == "gemma-4-26B-A4B-it-heretic-oQ4"
        assert saved["idle_timeout"] == 0
        assert saved["HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"] == "0"
        assert saved["HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE"] == "1"
        assert saved["timeout"] == 120



# ---------------------------------------------------------------------------
# Tool handler tests
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_retain_success(self, provider):
        result = json.loads(provider.handle_tool_call(
            "hindsight_retain", {"content": "user likes dark mode"}
        ))
        assert result["result"] == "Memory stored successfully."
        provider._client.aretain_batch.assert_called_once()
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        item = call_kwargs["items"][0]
        assert item["content"] == "user likes dark mode"
        # bank_id/retain_async are call-level args, never item keys.
        assert "bank_id" not in item
        assert "retain_async" not in item

    def test_retain_with_tags(self, provider_with_config):
        p = provider_with_config(retain_tags=["pref", "ui"])
        p.handle_tool_call("hindsight_retain", {"content": "likes dark mode"})
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["tags"] == ["pref", "ui"]

    def test_retain_merges_per_call_tags_with_config_tags(self, provider_with_config):
        p = provider_with_config(retain_tags=["pref", "ui"])
        p.handle_tool_call(
            "hindsight_retain",
            {"content": "likes dark mode", "tags": ["client:x", "ui"]},
        )
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["tags"] == ["pref", "ui", "client:x"]

    def test_retain_without_tags(self, provider):
        provider.handle_tool_call("hindsight_retain", {"content": "hello"})
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        assert "tags" not in item

    def test_retain_passes_observation_scopes(self, provider_with_config):
        p = provider_with_config(observation_scopes="per_tag")
        p.handle_tool_call("hindsight_retain", {"content": "likes dark mode"})
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["observation_scopes"] == "per_tag"

    def test_retain_omits_observation_scopes_by_default(self, provider):
        provider.handle_tool_call("hindsight_retain", {"content": "hello"})
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        assert "observation_scopes" not in item

    def test_retain_tool_strips_runtime_identifiers(self, provider):
        provider.handle_tool_call(
            "hindsight_retain",
            {
                "content": (
                    "[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]\n"
                    "sender: ou_cb923a19782fe748cd9fff99454eee31\n"
                    "用户喜欢紧凑界面"
                )
            },
        )
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["content"] == "用户喜欢紧凑界面"

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

    def test_recall_tool_strips_runtime_identifiers(self, provider):
        provider.handle_tool_call(
            "hindsight_recall",
            {
                "query": (
                    "Conversation info (untrusted metadata):\n"
                    "```json\n"
                    "{\"message_id\":\"om_x100b6d364e7e60a0c49f20c620fbbc6\"}\n"
                    "```\n"
                    "ou_cb923a19782fe748cd9fff99454eee31: 查询 retain 语义"
                )
            },
        )
        call_kwargs = provider._client.arecall.call_args.kwargs
        assert call_kwargs["query"] == "查询 retain 语义"

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

    def test_reflect_tool_strips_runtime_identifiers(self, provider):
        provider.handle_tool_call(
            "hindsight_reflect",
            {"query": "[open_id: ou_cb923a19782fe748cd9fff99454eee31]\n总结项目状态"},
        )
        call_kwargs = provider._client.areflect.call_args.kwargs
        assert call_kwargs["query"] == "总结项目状态"

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
        provider._client.aretain_batch.side_effect = RuntimeError("connection failed")
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

    def test_local_embedded_recall_reconnects_after_idle_shutdown(self, provider, monkeypatch):
        first_client = _make_mock_client()
        first_client.arecall.side_effect = RuntimeError("Cannot connect to host 127.0.0.1:8888")
        second_client = _make_mock_client()
        second_client.arecall.return_value = SimpleNamespace(
            results=[SimpleNamespace(text="Recovered memory")]
        )
        clients = iter([first_client, second_client])

        provider._mode = "local_embedded"
        provider._client = first_client
        monkeypatch.setattr(provider, "_get_client", lambda: next(clients))

        result = json.loads(provider.handle_tool_call(
            "hindsight_recall", {"query": "test"}
        ))

        assert result["result"] == "1. Recovered memory"
        assert provider._client is second_client
        first_client.arecall.assert_called_once()
        second_client.arecall.assert_called_once()


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

    def test_queue_prefetch_strips_runtime_identifiers_from_query(self, provider):
        provider.queue_prefetch(
            "[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]\n"
            "ou_cb923a19782fe748cd9fff99454eee31: 查询 retain 语义"
        )
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)

        call_kwargs = provider._client.arecall.call_args.kwargs
        assert call_kwargs["query"] == "查询 retain 语义"

    def test_queue_prefetch_skips_empty_query_after_sanitization(self, provider):
        provider.queue_prefetch("[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]")

        assert provider._prefetch_thread is None
        provider._client.arecall.assert_not_called()


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
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        call_kwargs = p._client.aretain_batch.call_args.kwargs
        assert call_kwargs["bank_id"] == "test-bank"
        assert call_kwargs["document_id"].startswith("session-1-")
        assert call_kwargs["retain_async"] is True
        assert len(call_kwargs["items"]) == 1
        item = call_kwargs["items"][0]
        assert item["context"] == "conversation between Hermes Agent and the User"
        assert item["tags"] == ["conv", "session1", "session:session-1"]
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

    def test_sync_turn_with_tags(self, provider_with_config):
        p = provider_with_config(retain_tags=["conv", "session1"])
        p.sync_turn("hello", "hi")
        p._retain_queue.join()
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert "conv" in item["tags"]
        assert "session1" in item["tags"]
        assert "session:test-session" in item["tags"]

    def test_sync_turn_uses_aretain_batch(self, provider):
        """sync_turn should use aretain_batch with retain_async."""
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()
        provider._client.aretain_batch.assert_called_once()
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        assert call_kwargs["document_id"].startswith("test-session-")
        assert call_kwargs["retain_async"] is True
        assert len(call_kwargs["items"]) == 1
        assert call_kwargs["items"][0]["context"] == "conversation between Hermes Agent and the User"

    def test_sync_turn_custom_context(self, provider_with_config):
        p = provider_with_config(retain_context="my-agent")
        p.sync_turn("hello", "hi")
        p._retain_queue.join()
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["context"] == "my-agent"

    def test_sync_turn_every_n_turns(self, provider_with_config):
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        p.sync_turn("turn1-user", "turn1-asst")
        assert p._sync_thread is None
        p.sync_turn("turn2-user", "turn2-asst")
        assert p._sync_thread is None
        p.sync_turn("turn3-user", "turn3-asst")
        p._retain_queue.join()
        p._client.aretain_batch.assert_called_once()
        call_kwargs = p._client.aretain_batch.call_args.kwargs
        assert call_kwargs["document_id"].startswith("test-session-")
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

    def test_sync_turn_retains_only_pending_batch(self, provider_with_config, monkeypatch):
        """Each retain sends only turns since the previous retain."""
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *a, **kw: True,
        )
        p = provider_with_config(retain_every_n_turns=2)

        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        p._retain_queue.join()

        first_content = p._client.aretain_batch.call_args.kwargs["items"][0]["content"]
        assert "turn1-user" in first_content
        assert "turn2-user" in first_content
        assert p._session_turns == []

        p._client.aretain_batch.reset_mock()

        p.sync_turn("turn3-user", "turn3-asst")
        p.sync_turn("turn4-user", "turn4-asst")
        p._retain_queue.join()

        content = p._client.aretain_batch.call_args.kwargs["items"][0]["content"]
        assert "turn1-user" not in content
        assert "turn2-user" not in content
        assert "turn3-user" in content
        assert "turn4-user" in content
        assert p._client.aretain_batch.call_args.kwargs["items"][0]["metadata"]["message_count"] == "4"

    def test_legacy_sync_turn_retains_full_document_batches(self, provider_with_config, monkeypatch):
        """Legacy replace APIs must receive the retained baseline plus pending turns."""
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *a, **kw: False,
        )
        p = provider_with_config(retain_every_n_turns=2)

        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        p._retain_queue.join()

        p._client.aretain_batch.reset_mock()

        p.sync_turn("turn3-user", "turn3-asst")
        p.sync_turn("turn4-user", "turn4-asst")
        p._retain_queue.join()

        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        content = item["content"]
        assert "turn1-user" in content
        assert "turn2-user" in content
        assert "turn3-user" in content
        assert "turn4-user" in content
        assert "update_mode" not in item
        assert item["metadata"]["message_count"] == "8"
        assert p._session_turns == []

    def test_sync_turn_strips_runtime_envelopes_from_retain_content(self, provider):
        dirty_user = """
Conversation info (untrusted metadata):
```json
{"chat_id":"user:ou_cb923a19782fe748cd9fff99454eee31","message_id":"om_x100b6d364e7e60a0c49f20c620fbbc6"}
```

Sender (untrusted metadata):
```json
{"id":"ou_cb923a19782fe748cd9fff99454eee31"}
```

[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]
ou_cb923a19782fe748cd9fff99454eee31: 真实用户问题
[context]
sender: ou_cb923a19782fe748cd9fff99454eee31
channel: feishu
[/context]
<hindsight_memories>old memory</hindsight_memories>
<summary>old summary</summary>
"""
        provider.sync_turn(dirty_user, "正常回复")
        provider._retain_queue.join()

        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        content = item["content"]
        assert "真实用户问题" in content
        assert "正常回复" in content
        assert "message_id" not in content
        assert "ou_cb923a19782fe748cd9fff99454eee31" not in content
        assert "om_x100b6d364e7e60a0c49f20c620fbbc6" not in content
        assert "ou_cb923a19782fe748cd9fff99454eee31:" not in content
        assert "[context]" not in content
        assert "sender:" not in content
        assert "hindsight_memories" not in content
        assert "old summary" not in content

    def test_sync_turn_appends_only_delta_when_append_supported(self, provider_with_config, monkeypatch):
        """On append-capable APIs each retain ships only the new turns, not the whole session."""
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        from plugins.memory.hindsight import _append_capability_cache, _append_capability_lock
        # Clear before AND after: the capability cache is module-global and keyed
        # per api_url, so a stale entry would leak into other tests.
        with _append_capability_lock:
            _append_capability_cache.clear()
        try:
            p = provider_with_config(retain_every_n_turns=2)

            p.sync_turn("turn1-user", "turn1-asst")
            p.sync_turn("turn2-user", "turn2-asst")
            p._retain_queue.join()

            first = p._client.aretain_batch.call_args.kwargs
            first_item = first["items"][0]
            assert first["document_id"] == "test-session"
            assert first_item["update_mode"] == "append"
            assert "turn1-user" in first_item["content"]
            assert "turn2-user" in first_item["content"]

            p._client.aretain_batch.reset_mock()

            p.sync_turn("turn3-user", "turn3-asst")
            p.sync_turn("turn4-user", "turn4-asst")
            p._retain_queue.join()

            second = p._client.aretain_batch.call_args.kwargs
            second_item = second["items"][0]
            assert second["document_id"] == "test-session"
            assert second_item["update_mode"] == "append"
            # Only the delta — the already-retained turns must NOT be resent.
            assert "turn1-user" not in second_item["content"]
            assert "turn2-user" not in second_item["content"]
            assert "turn3-user" in second_item["content"]
            assert "turn4-user" in second_item["content"]
            # message_count reflects only the delta (2 turns -> 4 messages).
            assert second_item["metadata"]["message_count"] == "4"
        finally:
            with _append_capability_lock:
                _append_capability_cache.clear()

    def test_sync_turn_passes_document_id(self, provider):
        """sync_turn should pass document_id (session_id + per-startup ts)."""
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()
        call_kwargs = provider._client.aretain_batch.call_args.kwargs
        # Format: {session_id}-{YYYYMMDD_HHMMSS_microseconds}
        assert call_kwargs["document_id"].startswith("test-session-")
        assert call_kwargs["document_id"] == provider._document_id

    def test_resume_creates_new_document(self, tmp_path, monkeypatch):
        """Resuming a session (re-initializing) gets a new document_id
        so previously stored content is not overwritten."""
        config = {"mode": "cloud", "apiKey": "k", "api_url": "http://x", "bank_id": "b"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p1 = HindsightMemoryProvider()
        p1.initialize(session_id="resumed-session", hermes_home=str(tmp_path), platform="cli")

        # Sleep just enough that the microsecond timestamp differs
        import time
        time.sleep(0.001)

        p2 = HindsightMemoryProvider()
        p2.initialize(session_id="resumed-session", hermes_home=str(tmp_path), platform="cli")

        # Same session, but each process gets its own document_id
        assert p1._document_id != p2._document_id
        assert p1._document_id.startswith("resumed-session-")
        assert p2._document_id.startswith("resumed-session-")

    def test_sync_turn_session_tag(self, provider):
        """Each retain should be tagged with session:<id> for filtering."""
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()
        item = provider._client.aretain_batch.call_args.kwargs["items"][0]
        assert "session:test-session" in item["tags"]

    def test_sync_turn_parent_session_tag(self, tmp_path, monkeypatch):
        """When initialized with parent_session_id, parent tag is added."""
        config = {"mode": "cloud", "apiKey": "k", "api_url": "http://x", "bank_id": "b"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="child-session",
            hermes_home=str(tmp_path),
            platform="cli",
            parent_session_id="parent-session",
        )
        p._client = _make_mock_client()
        p.sync_turn("hello", "hi")
        p._retain_queue.join()

        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert "session:child-session" in item["tags"]
        assert "parent:parent-session" in item["tags"]

    def test_sync_turn_error_does_not_raise(self, provider):
        provider._client.aretain_batch.side_effect = RuntimeError("network error")
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

    def test_failed_sync_turn_retain_restores_pending_batch(self, provider_with_config, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *a, **kw: True,
        )
        p = provider_with_config(retain_every_n_turns=1)
        p._client.aretain_batch.side_effect = RuntimeError("network error")

        p.sync_turn("turn1-user", "turn1-asst")
        p._retain_queue.join()

        assert "turn1-user" in "".join(p._session_turns)
        assert p._retain_inflight is False

        p._client.aretain_batch.side_effect = None
        p._client.aretain_batch.reset_mock()
        p.sync_turn("turn2-user", "turn2-asst")
        p._retain_queue.join()

        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        assert item["update_mode"] == "append"
        assert "turn1-user" in item["content"]
        assert "turn2-user" in item["content"]
        assert p._session_turns == []

    def test_sync_turn_preserves_unicode(self, provider_with_config):
        """Non-ASCII text (CJK, ZWJ emoji) must survive JSON round-trip intact."""
        p = provider_with_config()
        p._client = _make_mock_client()
        p.sync_turn("안녕 こんにちは 你好", "👨‍👩‍👧‍👦 family")
        p._retain_queue.join()
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
# Shutdown / writer tests
# ---------------------------------------------------------------------------


class TestShutdownRace:
    def test_sync_turn_uses_single_writer_thread(self, provider):
        """All retains run through one long-lived writer thread."""
        provider.sync_turn("a", "b")
        provider._retain_queue.join()
        first_writer = provider._writer_thread
        assert first_writer is not None
        assert first_writer.is_alive()

        provider.sync_turn("c", "d")
        provider._retain_queue.join()
        # Same thread reused — no ad-hoc thread per call.
        assert provider._writer_thread is first_writer
        assert provider._client.aretain_batch.call_count == 2

    def test_sync_turn_after_shutdown_is_dropped(self, provider):
        """Once shutdown has fired, new sync_turn() calls are no-ops.

        This is the core of the fix: the plugin must not enqueue a retain
        during interpreter teardown — that's what causes the
        'cannot schedule new futures' RuntimeError + unclosed aiohttp
        sessions on CLI exit.
        """
        client = provider._client
        provider.shutdown()
        before_calls = client.aretain_batch.call_count
        provider.sync_turn("late", "turn")
        # No new enqueue — the retain queue stays empty.
        assert provider._retain_queue.empty()
        # And no new client call (would be impossible anyway since shutdown
        # nulled self._client; we assert via the captured handle).
        assert client.aretain_batch.call_count == before_calls

    def test_queue_prefetch_after_shutdown_is_dropped(self, provider):
        provider.shutdown()
        provider.queue_prefetch("late query")
        assert provider._prefetch_thread is None

    def test_shutdown_drains_pending_retains(self, provider):
        """Shutdown must wait for queued retains to complete, not abandon them.

        Otherwise the LAST in-flight turn — typically the most important —
        is silently lost.
        """
        client = provider._client
        provider.sync_turn("a", "b")
        provider.sync_turn("c", "d")
        provider.shutdown()
        # Both retains drained before shutdown returned.
        assert client.aretain_batch.call_count == 2
        assert provider._retain_queue.empty()

    def test_shutdown_flushes_partial_retain_buffer(self, provider_with_config):
        p = provider_with_config(retain_every_n_turns=3)
        client = p._client

        p.sync_turn("turn1-user", "turn1-asst")
        client.aretain_batch.assert_not_called()

        p.shutdown()

        client.aretain_batch.assert_called_once()
        item = client.aretain_batch.call_args.kwargs["items"][0]
        assert "turn1-user" in item["content"]
        assert p._retain_queue.empty()

    def test_shutdown_is_idempotent(self, provider):
        provider.sync_turn("a", "b")
        provider.shutdown()
        # Second shutdown shouldn't blow up or re-close the client.
        provider.shutdown()
        assert provider._shutting_down.is_set()


# ---------------------------------------------------------------------------
# on_session_switch — flush + prefetch reset behavior
# ---------------------------------------------------------------------------


class TestSessionSwitchBufferFlush:
    def test_buffered_turns_flushed_before_clear(self, provider_with_config):
        """retain_every_n_turns > 1 must not silently drop partial buffers
        on session switch. Whatever's in _session_turns at switch time
        should land in the OLD document under the OLD session id."""
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        old_doc = p._document_id

        # Two turns buffered, no retain yet (boundary is at turn 3). The
        # writer hasn't been started either — sync_turn's early return
        # skips _ensure_writer when no retain is due.
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        assert p._sync_thread is None
        p._client.aretain_batch.assert_not_called()

        # Switch — flush should fire under OLD document_id via the writer queue.
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        kw = p._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        item = kw["items"][0]
        # Both buffered turns must be present in the flushed payload.
        content = json.loads(item["content"])
        flat = json.dumps(content)
        assert "turn1-user" in flat
        assert "turn2-user" in flat
        # Old session id must appear in lineage tags / metadata.
        assert "session:test-session" in item["tags"]
        assert item["metadata"]["session_id"] == "test-session"

        # And the new session must start with a clean slate.
        assert p._session_id == "new-sid"
        assert p._session_turns == []
        assert p._turn_counter == 0
        assert p._document_id != old_doc
        assert p._document_id.startswith("new-sid-")

    def test_no_flush_when_buffer_empty(self, provider):
        """Switch with no buffered turns must not fire a spurious retain."""
        provider.on_session_switch("new-sid")
        # Nothing enqueued — join is immediate.
        provider._retain_queue.join()
        provider._client.aretain_batch.assert_not_called()
        assert provider._session_id == "new-sid"

    def test_legacy_switch_flush_preserves_retained_baseline(self, provider_with_config, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *a, **kw: False,
        )
        p = provider_with_config(retain_every_n_turns=2, retain_async=False)
        old_doc = p._document_id

        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        p._retain_queue.join()
        p._client.aretain_batch.reset_mock()

        p.sync_turn("turn3-user", "turn3-asst")
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        kw = p._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        item = kw["items"][0]
        assert "update_mode" not in item
        assert "turn1-user" in item["content"]
        assert "turn2-user" in item["content"]
        assert "turn3-user" in item["content"]
        assert item["metadata"]["message_count"] == "6"
        assert p._retained_turns == []

    def test_prefetch_result_cleared_on_switch(self, provider):
        """Stale recall text from the old session must not leak into the
        next session's first prefetch read."""
        provider._prefetch_result = "old-session recall: User likes Rust"
        provider.on_session_switch("new-sid")
        assert provider._prefetch_result == ""
        # And subsequent prefetch() should now report empty, not the leftover.
        assert provider.prefetch("anything") == ""

    def test_in_flight_prefetch_thread_drained_on_switch(self, provider, monkeypatch):
        """on_session_switch must wait for an in-flight prefetch from the
        old session to settle before clearing _prefetch_result, otherwise
        the thread can race and re-populate the field after the clear."""
        import threading

        gate = threading.Event()
        finished = threading.Event()

        def _slow_prefetch():
            gate.wait(timeout=5.0)
            with provider._prefetch_lock:
                provider._prefetch_result = "old-session recall"
            finished.set()

        provider._prefetch_thread = threading.Thread(target=_slow_prefetch, daemon=True)
        provider._prefetch_thread.start()

        # Release the prefetch worker so it writes _prefetch_result, then
        # call on_session_switch — it must join the thread before clearing.
        gate.set()
        provider.on_session_switch("new-sid")

        assert finished.is_set(), "switch returned before prefetch thread settled"
        assert provider._prefetch_result == ""

    def test_flush_serializes_behind_pending_retains_via_writer_queue(
        self, provider_with_config
    ):
        """The flush closure must ride the same _retain_queue sync_turn
        uses, so it lands FIFO behind any still-queued old-session
        retains rather than racing them on a separate thread.

        Regression guard: an earlier draft spawned a raw threading.Thread
        for flush, overwriting _sync_thread and racing the writer against
        the same document_id.
        """
        import threading as _threading

        p = provider_with_config(retain_every_n_turns=2, retain_async=False)

        # Block the first writer job until we've enqueued the flush
        # behind it. This proves ordering — the flush MUST wait.
        gate = _threading.Event()
        call_order: list[str] = []

        def _aretain_batch_tracking(**kw):
            idx = kw["items"][0]["metadata"].get("turn_index", "")
            call_order.append(str(idx))
            if idx == "2":
                # First retain blocks until we've enqueued the flush.
                gate.wait(timeout=5.0)

        p._client.aretain_batch = AsyncMock(side_effect=_aretain_batch_tracking)

        # Turn 1+2 → boundary hit → retain enqueued (will block).
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")

        # One more buffered turn so flush has something to land.
        p.sync_turn("turn3-user", "turn3-asst")

        # Switch while the first retain is still blocked on `gate`.
        p.on_session_switch("new-sid", parent_session_id="test-session")

        # Release the first retain. Flush must have been enqueued
        # BEHIND it, and run second.
        gate.set()
        p._retain_queue.join()

        # The flush carries all buffered turns; sync_turn's retain #2
        # carried the batch at boundary time. Two distinct calls.
        assert p._client.aretain_batch.call_count == 2
        # First call landed while buffer was [t1, t2]; flush landed
        # after we added t3. So the second call must be strictly after.
        assert call_order[0] == "2"
        # Flush retain has turn_index matching the buffered count at
        # switch time (3 turns accumulated, _turn_index was set to 3
        # by the last sync_turn).
        assert call_order[1] == "3"


# ---------------------------------------------------------------------------
# update_mode='append' capability probe + retain dispatch
# ---------------------------------------------------------------------------


class TestUpdateModeAppendCapability:
    def _clear_capability_cache(self):
        from plugins.memory.hindsight import _append_capability_cache, _append_capability_lock
        with _append_capability_lock:
            _append_capability_cache.clear()

    def test_legacy_api_falls_back_to_per_process_doc_id(self, provider, monkeypatch):
        """API returns no /version (or pre-0.5.0) — sync_turn must use the
        per-process unique doc_id and NOT pass update_mode."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: None,
        )
        old_doc = provider._document_id
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kw = provider._client.aretain_batch.call_args.kwargs
        assert kw["document_id"] == old_doc
        assert kw["document_id"].startswith("test-session-")
        item = kw["items"][0]
        assert "update_mode" not in item

    def test_modern_api_uses_stable_doc_id_with_append(self, provider, monkeypatch):
        """API on >=0.5.0 — retain uses stable session_id and sets update_mode='append'."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        provider.sync_turn("hello", "hi")
        provider._retain_queue.join()

        kw = provider._client.aretain_batch.call_args.kwargs
        # Stable: just the session id, no per-process timestamp suffix.
        assert kw["document_id"] == "test-session"
        item = kw["items"][0]
        assert item["update_mode"] == "append"

    def test_capability_cached_per_url(self, provider, monkeypatch):
        """The /version probe must run at most once per (process, api_url)."""
        self._clear_capability_cache()
        calls = {"n": 0}

        def _spy(*a, **kw):
            calls["n"] += 1
            return "0.5.6"

        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version", _spy
        )
        provider.sync_turn("a", "b")
        provider._retain_queue.join()
        provider.sync_turn("c", "d")
        provider._retain_queue.join()
        assert calls["n"] == 1

    def test_legacy_warning_emitted_once(self, provider, monkeypatch, caplog):
        """One-time WARN nudges users to upgrade Hindsight."""
        import logging
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.4.22",
        )
        with caplog.at_level(logging.WARNING, logger="plugins.memory.hindsight"):
            provider.sync_turn("a", "b")
            provider._retain_queue.join()
            provider.sync_turn("c", "d")
            provider._retain_queue.join()
        warns = [r for r in caplog.records
                 if r.levelno == logging.WARNING
                 and "older than 0.5.0" in r.getMessage()]
        # Cache hit on the second call → no second warn.
        assert len(warns) == 1

    def test_session_switch_flush_picks_capability_against_old_session(
        self, provider_with_config, monkeypatch
    ):
        """When the API supports append, the flush on /reset must land
        in the OLD session's stable document, not a per-process id."""
        self._clear_capability_cache()
        monkeypatch.setattr(
            "plugins.memory.hindsight._fetch_hindsight_api_version",
            lambda *a, **kw: "0.5.6",
        )
        p = provider_with_config(retain_every_n_turns=3, retain_async=False)
        p.sync_turn("turn1-user", "turn1-asst")
        p.sync_turn("turn2-user", "turn2-asst")
        p.on_session_switch("new-sid", parent_session_id="test-session", reset=True)
        p._retain_queue.join()

        kw = p._client.aretain_batch.call_args.kwargs
        # Flush goes to the OLD session's stable doc, not new-sid's.
        assert kw["document_id"] == "test-session"
        assert kw["items"][0]["update_mode"] == "append"


# ---------------------------------------------------------------------------
# Rolling session summary lifecycle integration
# ---------------------------------------------------------------------------


class TestSessionSummaryIntegration:
    def _seed_summary(self, provider, *, project="active-project"):
        store = provider._get_session_summary_store()
        store.upsert(
            SessionSummaryWrite(
                summary_key=provider._session_summary_key(),
                identity_scope=provider._session_summary_identity_scope(),
                summary_json={
                    "schema_version": 2,
                    "summary_text": f"Continue {project} rollout.",
                },
                summary_text=f"Continue {project} rollout.",
                turn=2,
                turn_hash="turn-hash",
                last_input_hash=f"input-{project}",
            )
        )

    def test_queue_prefetch_enriches_query_with_summary_after_latest_query(
        self, provider_with_config
    ):
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
            session_summary_max_recall_query_chars=90,
            recall_max_input_chars=120,
        )
        self._seed_summary(p, project="active-project")

        p.queue_prefetch("[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]\nWhat is next?")
        p._prefetch_thread.join(timeout=3.0)

        query = p._client.arecall.call_args.kwargs["query"]
        assert query.startswith("What is next?")
        assert "message_id" not in query
        assert "om_x100" not in query
        assert "Rolling session summary:" in query
        assert "active-project" in query
        assert len(query) <= p._recall_max_input_chars

    def test_hindsight_recall_tool_enriches_query_with_summary_in_tools_mode(
        self, provider_with_config
    ):
        p = provider_with_config(
            memory_mode="tools",
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
            session_summary_max_recall_query_chars=90,
            recall_max_input_chars=120,
        )
        self._seed_summary(p, project="tool-recall-project")

        p.handle_tool_call(
            "hindsight_recall",
            {"query": "[message_id: om_x100b6d364e7e60a0c49f20c620fbbc6]\nWhat is next?"},
        )

        query = p._client.arecall.call_args.kwargs["query"]
        assert query.startswith("What is next?")
        assert "message_id" not in query
        assert "om_x100" not in query
        assert "Rolling session summary:" in query
        assert "tool-recall-project" in query
        assert len(query) <= p._recall_max_input_chars

    def test_prefetch_does_not_inject_summary_as_prompt_block(
        self, provider_with_config
    ):
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
        )
        self._seed_summary(p, project="prompt-project")
        p._prefetch_result = "- recalled memory"

        block = p.prefetch("next")

        assert "# Hindsight Memory" in block
        assert "- recalled memory" in block
        assert "<hindsight_session_summary>" not in block
        assert "prompt-project" not in block

    def test_prefetch_does_not_return_summary_without_recall_results(
        self, provider_with_config
    ):
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
        )
        self._seed_summary(p, project="summary-only-project")

        block = p.prefetch("next")

        assert block == ""

    def test_sync_turn_does_not_enrich_retain_context_and_sanitizes_summary_messages(
        self, provider_with_config, monkeypatch
    ):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *a, **kw: False,
        )
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
            session_summary_update_every_n_turns=2,
            retain_async=False,
        )
        self._seed_summary(p, project="retain-project")
        raw_tool_log = "RAW_TOOL_OUTPUT_SHOULD_NOT_BE_SUMMARIZED"
        messages = [
            {"role": "user", "content": "Continue project retain-project."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "content": raw_tool_log, "tool_call_id": "call-1"},
            {"role": "assistant", "content": "Done for retain-project."},
        ]

        p.sync_turn("continue", "ok", messages=messages)
        p._retain_queue.join()
        p.sync_turn("next", "done", messages=messages)
        p._retain_queue.join()

        first_item = p._client.aretain_batch.call_args_list[0].kwargs["items"][0]
        assert first_item["context"] == p._retain_context
        assert "Rolling session summary" not in first_item["context"]
        assert raw_tool_log not in first_item["content"]

        record = p._get_session_summary_store().get(p._session_summary_key())
        assert record is not None
        assert "retain-project" in record.summary_text
        assert raw_tool_log not in record.summary_text

    def test_summary_update_cadence_runs_before_retain_cadence(
        self, provider_with_config
    ):
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
            session_summary_update_every_n_turns=2,
            retain_every_n_turns=5,
            retain_async=False,
        )

        p.sync_turn("Continue project cadence-project.", "ok")
        p.sync_turn("Next cadence-project step.", "done")

        p._client.aretain_batch.assert_not_called()
        assert p._sync_thread is None
        record = p._get_session_summary_store().get(p._session_summary_key())
        assert record is not None
        assert record.turn == 2
        assert "cadence-project" in record.summary_text

        p.sync_turn("Third cadence-project turn.", "noted")
        p.sync_turn("Fourth cadence-project turn.", "noted")
        p.sync_turn("Fifth cadence-project turn.", "noted")
        p._retain_queue.join()

        p._client.aretain_batch.assert_called_once()
        item = p._client.aretain_batch.call_args.kwargs["items"][0]
        content = json.loads(item["content"])
        flat_content = json.dumps(content)
        assert len(content) == 5
        assert "Rolling session summary" not in flat_content
        assert "cadence-project" in flat_content
        assert item["metadata"]["turn_index"] == "5"
        assert item["metadata"]["message_count"] == "10"

    def test_on_pre_compress_updates_without_returning_summary_block(
        self, provider_with_config
    ):
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
        )
        raw_tool_log = "RAW_TOOL_OUTPUT_SHOULD_NOT_BE_SUMMARIZED"

        block = p.on_pre_compress(
            [
                {"role": "user", "content": "Continue project compress-project."},
                {"role": "tool", "content": raw_tool_log},
                {"role": "assistant", "content": "Decision: use compress-project plan."},
            ]
        )

        assert block == ""
        record = p._get_session_summary_store().get(p._session_summary_key())
        assert record is not None
        assert "compress-project" in record.summary_text
        assert raw_tool_log not in record.summary_text

    def test_session_switch_flushes_old_summary_state(self, provider_with_config, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight._check_api_supports_update_mode_append",
            lambda *a, **kw: False,
        )
        p = provider_with_config(
            session_summary_enabled=True,
            session_summary_enrich_recall_query=True,
            retain_every_n_turns=3,
            retain_async=False,
        )
        old_key = p._session_summary_key()
        p.sync_turn("Continue project switch-project.", "ok")
        p.sync_turn("Next switch-project step.", "done")

        p.on_session_switch("new-session", parent_session_id="test-session")
        p._retain_queue.join()

        old_record = p._get_session_summary_store().get(old_key)
        assert old_record is not None
        assert "switch-project" in old_record.summary_text
        assert p._session_summary_messages == []
        assert p._session_summary_key() != old_key


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
            "llm_model", "bank_id", "bank_id_template", "bank_mission", "bank_retain_mission",
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
# bank_id_template tests
# ---------------------------------------------------------------------------


class TestBankIdTemplate:
    def test_sanitize_bank_segment_passthrough(self):
        assert _sanitize_bank_segment("hermes") == "hermes"
        assert _sanitize_bank_segment("my-agent_1") == "my-agent_1"

    def test_sanitize_bank_segment_strips_unsafe(self):
        assert _sanitize_bank_segment("josh@example.com") == "josh-example-com"
        assert _sanitize_bank_segment("chat:#general") == "chat-general"
        assert _sanitize_bank_segment("  spaces  ") == "spaces"

    def test_sanitize_bank_segment_empty(self):
        assert _sanitize_bank_segment("") == ""
        assert _sanitize_bank_segment(None) == ""

    def test_resolve_empty_template_uses_fallback(self):
        result = _resolve_bank_id_template(
            "", fallback="hermes", profile="coder"
        )
        assert result == "hermes"

    def test_resolve_with_profile(self):
        result = _resolve_bank_id_template(
            "hermes-{profile}", fallback="hermes",
            profile="coder", workspace="", platform="", user="", session="",
        )
        assert result == "hermes-coder"

    def test_resolve_with_multiple_placeholders(self):
        result = _resolve_bank_id_template(
            "{workspace}-{profile}-{platform}",
            fallback="hermes",
            profile="coder", workspace="myorg", platform="cli",
            user="", session="",
        )
        assert result == "myorg-coder-cli"

    def test_resolve_collapses_empty_placeholders(self):
        # When user is empty, "hermes-{user}" becomes "hermes-" -> trimmed to "hermes"
        result = _resolve_bank_id_template(
            "hermes-{user}", fallback="default",
            profile="", workspace="", platform="", user="", session="",
        )
        assert result == "hermes"

    def test_resolve_collapses_double_dashes(self):
        # Two empty placeholders with a dash between them should collapse
        result = _resolve_bank_id_template(
            "{workspace}-{profile}-{user}", fallback="fallback",
            profile="coder", workspace="", platform="", user="", session="",
        )
        assert result == "coder"

    def test_resolve_empty_rendered_falls_back(self):
        result = _resolve_bank_id_template(
            "{user}-{profile}", fallback="fallback",
            profile="", workspace="", platform="", user="", session="",
        )
        assert result == "fallback"

    def test_resolve_sanitizes_placeholder_values(self):
        result = _resolve_bank_id_template(
            "user-{user}", fallback="hermes",
            profile="", workspace="", platform="",
            user="josh@example.com", session="",
        )
        assert result == "user-josh-example-com"

    def test_resolve_invalid_template_returns_fallback(self):
        # Unknown placeholder should fall back without raising
        result = _resolve_bank_id_template(
            "hermes-{unknown}", fallback="hermes",
            profile="", workspace="", platform="", user="", session="",
        )
        assert result == "hermes"

    def test_provider_uses_bank_id_template_from_config(self, tmp_path, monkeypatch):
        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "fallback-bank",
            "bank_id_template": "hermes-{profile}",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="s1",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="coder",
            agent_workspace="hermes",
        )
        assert p._bank_id == "hermes-coder"
        assert p._bank_id_template == "hermes-{profile}"

    def test_provider_without_template_uses_static_bank_id(self, tmp_path, monkeypatch):
        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "my-static-bank",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p = HindsightMemoryProvider()
        p.initialize(
            session_id="s1",
            hermes_home=str(tmp_path),
            platform="cli",
            agent_identity="coder",
        )
        assert p._bank_id == "my-static-bank"

    def test_provider_template_with_missing_profile_falls_back(self, tmp_path, monkeypatch):
        config = {
            "mode": "cloud",
            "apiKey": "k",
            "api_url": "http://x",
            "bank_id": "hermes-fallback",
            "bank_id_template": "hermes-{profile}",
        }
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr("plugins.memory.hindsight.get_hermes_home", lambda: tmp_path)

        p = HindsightMemoryProvider()
        # No agent_identity passed — template renders to "hermes-" which collapses to "hermes"
        p.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
        assert p._bank_id == "hermes"


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
        monkeypatch.setattr(
            "plugins.memory.hindsight.importlib.import_module",
            lambda name: object(),
        )
        p = HindsightMemoryProvider()
        assert p.is_available()

    def test_available_with_snake_case_api_key_in_config(self, tmp_path, monkeypatch):
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({
            "mode": "cloud",
            "api_key": "***",
        }))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path,
        )

        p = HindsightMemoryProvider()

        assert p.is_available()

    def test_local_mode_unavailable_when_runtime_import_fails(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home",
            lambda: tmp_path / "nonexistent",
        )
        monkeypatch.setenv("HINDSIGHT_MODE", "local")

        def _raise(_name):
            raise RuntimeError(
                "NumPy was built with baseline optimizations: (x86_64-v2)"
            )

        monkeypatch.setattr(
            "plugins.memory.hindsight.importlib.import_module",
            _raise,
        )
        p = HindsightMemoryProvider()
        assert not p.is_available()

    def test_initialize_disables_local_mode_when_runtime_import_fails(self, tmp_path, monkeypatch):
        config = {"mode": "local_embedded"}
        config_path = tmp_path / "hindsight" / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "plugins.memory.hindsight.get_hermes_home", lambda: tmp_path
        )

        def _raise(_name):
            raise RuntimeError("x86_64-v2 unsupported")

        monkeypatch.setattr(
            "plugins.memory.hindsight.importlib.import_module",
            _raise,
        )

        p = HindsightMemoryProvider()
        p.initialize(session_id="test-session", hermes_home=str(tmp_path), platform="cli")
        assert p._mode == "disabled"


class TestSharedEventLoopLifecycle:
    """Regression tests for #11923 — Hindsight leaking aiohttp ClientSession /
    TCPConnector objects in long-running gateway processes.

    Root cause: the module-global ``_loop`` / ``_loop_thread`` pair is shared
    across every HindsightMemoryProvider instance in the process (the plugin
    loader builds one provider per AIAgent, and the gateway builds one AIAgent
    per concurrent chat session). When a session ended, ``shutdown()`` stopped
    the shared loop, which orphaned every *other* live provider's aiohttp
    ClientSession on a dead loop. Those sessions were never closed and surfaced
    as ``Unclosed client session`` / ``Unclosed connector`` errors.
    """

    def test_shutdown_does_not_stop_shared_event_loop(self, provider_with_config):
        from plugins.memory import hindsight as hindsight_mod

        async def _noop():
            return 1

        # Prime the shared loop by scheduling a trivial coroutine — mirrors
        # the first time any real async call (arecall/aretain/areflect) runs.
        assert hindsight_mod._run_sync(_noop()) == 1

        loop_before = hindsight_mod._loop
        thread_before = hindsight_mod._loop_thread
        assert loop_before is not None and loop_before.is_running()
        assert thread_before is not None and thread_before.is_alive()

        # Build two independent providers (two concurrent chat sessions).
        provider_a = provider_with_config()
        provider_b = provider_with_config()

        # End session A.
        provider_a.shutdown()

        # Module-global loop/thread must still be the same live objects —
        # provider B (and any other sibling provider) is still relying on them.
        assert hindsight_mod._loop is loop_before, (
            "shutdown() swapped out the shared event loop — sibling providers "
            "would have their aiohttp ClientSession orphaned (#11923)"
        )
        assert hindsight_mod._loop.is_running(), (
            "shutdown() stopped the shared event loop — sibling providers' "
            "aiohttp sessions would leak (#11923)"
        )
        assert hindsight_mod._loop_thread is thread_before
        assert hindsight_mod._loop_thread.is_alive()

        # Provider B can still dispatch async work on the shared loop.
        async def _still_working():
            return 42

        assert hindsight_mod._run_sync(_still_working()) == 42

        provider_b.shutdown()

    def test_client_aclose_called_on_cloud_mode_shutdown(self, provider):
        """Per-provider session cleanup still runs even though the shared
        loop is preserved. Each provider's own aiohttp session is closed
        via ``self._client.aclose()``; only the (empty) shared loop survives.
        """
        assert provider._client is not None
        mock_client = provider._client

        provider.shutdown()

        mock_client.aclose.assert_called_once()
        assert provider._client is None


class TestShutdown:
    def test_local_embedded_shutdown_closes_inner_async_client_on_shared_loop(self, provider):
        inner_client = _make_mock_client()
        embedded = MagicMock()
        embedded._client = inner_client
        embedded.close = MagicMock()

        provider._mode = "local_embedded"
        provider._client = embedded

        provider.shutdown()

        inner_client.aclose.assert_awaited_once()
        embedded.close.assert_called_once()
        assert embedded._client is None
        assert provider._client is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits not enforced on Windows")
def test_save_config_sets_owner_only_permissions(tmp_path):
    """hindsight/config.json must be written with 0o600 so API key is not world-readable."""
    provider = HindsightMemoryProvider()
    provider.save_config({"api_key": "hd-test-key"}, str(tmp_path))
    config_file = tmp_path / "hindsight" / "config.json"
    assert config_file.exists()
    mode = stat.S_IMODE(config_file.stat().st_mode)
    assert mode == 0o600, f"Expected 0o600 (owner-only), got {oct(mode)}"
