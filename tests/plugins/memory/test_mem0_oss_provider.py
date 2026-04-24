"""Tests for the Mem0 OSS self-hosted memory provider plugin.

Covers config loading, _build_mem0_config, _extract_results, tool handlers
(search, add, unknown), circuit breaker, prefetch, sync_turn, system prompt,
schema completeness, availability checks, and shutdown.

All tests are fully offline — mem0.Memory is always mocked.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from plugins.memory.mem0_oss import (
    Mem0OSSMemoryProvider,
    SEARCH_SCHEMA,
    ADD_SCHEMA,
    _extract_results,
    _load_config,
    _build_mem0_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Wipe all MEM0_OSS_* env vars between tests."""
    for key in list(os.environ):
        if key.startswith("MEM0_OSS_"):
            monkeypatch.delenv(key, raising=False)


def _make_mock_memory(search_results=None, add_raises=None):
    """Build a mock mem0.Memory with controllable behaviour."""
    mem = MagicMock()

    default_search = [{"memory": "User likes dark mode"}, {"memory": "Project: hermes-agent"}]
    mem.search.return_value = search_results if search_results is not None else default_search

    if add_raises:
        mem.add.side_effect = add_raises
    else:
        mem.add.return_value = [{"id": "abc123", "memory": "stored"}]

    return mem


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    """Initialized provider with a mocked Memory instance.

    _get_memory() is patched to return a fresh MagicMock on every call so that
    tests don't trigger real Qdrant initialisation.  The mock is also exposed as
    ``provider._mock_mem`` for assertion convenience.
    """
    monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
    with patch("plugins.memory.mem0_oss._load_config") as mock_cfg:
        mock_cfg.return_value = {
            "vector_store_path": str(tmp_path / "qdrant"),
            "history_db_path": str(tmp_path / "history.db"),
            "collection": "hermes",
            "user_id": "test-user",
            "llm_provider": "aws_bedrock",
            "llm_model": "us.anthropic.claude-haiku-4-5-20250714-v1:0",
            "embedder_provider": "aws_bedrock",
            "embedder_model": "amazon.titan-embed-text-v2:0",
            "embedder_dims": 1024,
            "top_k": 10,
            "openai_api_key": "",
            "openai_base_url": "",
        }
        p = Mem0OSSMemoryProvider()
        p.initialize("test-session")

    mock_mem = _make_mock_memory()
    # Patch _get_memory so every call returns the same mock without hitting Qdrant.
    p._get_memory = lambda: mock_mem  # type: ignore[assignment]
    # Expose the mock for test assertions.
    p._mock_mem = mock_mem  # type: ignore[attr-defined]
    return p


# ---------------------------------------------------------------------------
# _extract_results
# ---------------------------------------------------------------------------


class TestExtractResults:
    def test_list_of_dicts_memory_key(self):
        results = [{"memory": "A"}, {"memory": "B"}]
        assert _extract_results(results) == ["A", "B"]

    def test_list_of_dicts_text_key(self):
        results = [{"text": "C"}, {"text": "D"}]
        assert _extract_results(results) == ["C", "D"]

    def test_v2_dict_wrapper(self):
        results = {"results": [{"memory": "E"}, {"memory": "F"}]}
        assert _extract_results(results) == ["E", "F"]

    def test_empty_list(self):
        assert _extract_results([]) == []

    def test_empty_v2_dict(self):
        assert _extract_results({"results": []}) == []

    def test_unknown_type_returns_empty(self):
        assert _extract_results(None) == []
        assert _extract_results("string") == []

    def test_filters_empty_strings(self):
        results = [{"memory": "A"}, {"memory": ""}]
        assert _extract_results(results) == ["A"]


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        # Ensure all provider env vars are absent so auto-detect returns "auto"
        # and _load_config falls back to the aws_bedrock default.
        for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                  "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "MEM0_OSS_LLM_PROVIDER"):
            monkeypatch.delenv(k, raising=False)
        cfg = _load_config()
        assert cfg["llm_provider"] == "aws_bedrock"
        assert cfg["embedder_provider"] == "aws_bedrock"
        assert cfg["collection"] == "hermes"
        assert cfg["user_id"] == "hermes-user"
        assert cfg["top_k"] == 10
        assert cfg["embedder_dims"] == 1024

    def test_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.setenv("MEM0_OSS_LLM_PROVIDER", "openai")
        monkeypatch.setenv("MEM0_OSS_LLM_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("MEM0_OSS_USER_ID", "stan")
        monkeypatch.setenv("MEM0_OSS_TOP_K", "5")
        monkeypatch.setenv("MEM0_OSS_EMBEDDER_DIMS", "1536")
        # Clear provider env vars so auto-detect doesn't interfere with provider assertion
        for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                  "AWS_ACCESS_KEY_ID", "AWS_PROFILE"):
            monkeypatch.delenv(k, raising=False)
        cfg = _load_config()
        assert cfg["llm_provider"] == "openai"
        assert cfg["llm_model"] == "gpt-4o-mini"
        assert cfg["user_id"] == "stan"
        assert cfg["top_k"] == 5
        assert cfg["embedder_dims"] == 1536

    def test_json_file_overrides_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.setenv("MEM0_OSS_COLLECTION", "env-collection")
        cfg_file = tmp_path / "mem0_oss.json"
        cfg_file.write_text(json.dumps({"collection": "file-collection"}))
        cfg = _load_config()
        assert cfg["collection"] == "file-collection"

    def test_json_file_skips_null_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.setenv("MEM0_OSS_USER_ID", "env-user")
        cfg_file = tmp_path / "mem0_oss.json"
        cfg_file.write_text(json.dumps({"user_id": None}))
        cfg = _load_config()
        assert cfg["user_id"] == "env-user"

    def test_malformed_json_falls_back_to_env(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.setenv("MEM0_OSS_USER_ID", "env-user")
        cfg_file = tmp_path / "mem0_oss.json"
        cfg_file.write_text("{not valid json")
        cfg = _load_config()
        assert cfg["user_id"] == "env-user"

    def test_paths_scoped_to_hermes_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        cfg = _load_config()
        assert cfg["vector_store_path"].startswith(str(tmp_path))
        assert cfg["history_db_path"].startswith(str(tmp_path))

    def test_aux_config_provider_used(self, tmp_path, monkeypatch):
        """auxiliary.mem0_oss.provider is respected."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config",
                            lambda: {"provider": "openai", "model": "gpt-4o", "api_key": ""})
        cfg = _load_config()
        assert cfg["llm_provider"] == "openai"
        assert cfg["llm_model"] == "gpt-4o"

    def test_env_provider_overrides_aux_config(self, tmp_path, monkeypatch):
        """MEM0_OSS_LLM_PROVIDER env var beats auxiliary.mem0_oss.provider."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config",
                            lambda: {"provider": "anthropic", "model": ""})
        monkeypatch.setenv("MEM0_OSS_LLM_PROVIDER", "ollama")
        cfg = _load_config()
        assert cfg["llm_provider"] == "ollama"

    def test_aux_api_key_used_when_no_env_key(self, tmp_path, monkeypatch):
        """auxiliary.mem0_oss.api_key propagates into cfg['api_key']."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config",
                            lambda: {"provider": "openai", "api_key": "aux-key-123"})
        monkeypatch.delenv("MEM0_OSS_API_KEY", raising=False)
        cfg = _load_config()
        assert cfg["api_key"] == "aux-key-123"

    def test_mem0_oss_api_key_beats_aux_key(self, tmp_path, monkeypatch):
        """MEM0_OSS_API_KEY env var takes precedence over aux config api_key."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config",
                            lambda: {"provider": "openai", "api_key": "aux-key"})
        monkeypatch.setenv("MEM0_OSS_API_KEY", "env-key")
        cfg = _load_config()
        assert cfg["api_key"] == "env-key"

    def test_hermes_provider_aliases_normalised(self, tmp_path, monkeypatch):
        """Hermes aliases 'bedrock' and 'openrouter' map to mem0 provider keys."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        for hermes_alias, expected_mem0 in [("bedrock", "aws_bedrock"),
                                             ("openrouter", "openai"),
                                             ("aws", "aws_bedrock")]:
            monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config",
                                lambda p=hermes_alias: {"provider": p})
            cfg = _load_config()
            assert cfg["llm_provider"] == expected_mem0, f"{hermes_alias} → {expected_mem0}"

    def test_aux_base_url_used(self, tmp_path, monkeypatch):
        """auxiliary.mem0_oss.base_url populates cfg['base_url']."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config",
                            lambda: {"provider": "openai", "base_url": "http://myhost/v1"})
        monkeypatch.delenv("MEM0_OSS_OPENAI_BASE_URL", raising=False)
        cfg = _load_config()
        assert cfg["base_url"] == "http://myhost/v1"

    def test_auxiliary_default_inherited_when_no_mem0_oss_key(self, tmp_path, monkeypatch):
        """auxiliary.default is used when auxiliary.mem0_oss is not set."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        fake_config = {
            "auxiliary": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "default-key",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=fake_config):
            from plugins.memory.mem0_oss import _get_aux_config
            result = _get_aux_config()
        assert result.get("provider") == "openai"
        assert result.get("model") == "gpt-4o-mini"
        assert result.get("api_key") == "default-key"

    def test_auxiliary_mem0_oss_key_overrides_default(self, tmp_path, monkeypatch):
        """auxiliary.mem0_oss-specific keys win over auxiliary.default."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        fake_config = {
            "auxiliary": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                },
                "mem0_oss": {
                    "provider": "anthropic",
                    "model": "claude-haiku-4-5",
                },
            }
        }
        with patch("hermes_cli.config.load_config", return_value=fake_config):
            from plugins.memory.mem0_oss import _get_aux_config
            result = _get_aux_config()
        # mem0_oss-specific values win
        assert result.get("provider") == "anthropic"
        assert result.get("model") == "claude-haiku-4-5"

    def test_auxiliary_default_model_passed_to_load_config(self, tmp_path, monkeypatch):
        """When auxiliary.default sets a model it ends up in llm_model."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        fake_config = {
            "auxiliary": {
                "default": {
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                    "api_key": "sk-default",
                }
            }
        }
        with patch("hermes_cli.config.load_config", return_value=fake_config):
            cfg = _load_config()
        assert cfg["llm_model"] == "gpt-4o-mini"
        assert cfg["llm_provider"] == "openai"


# ---------------------------------------------------------------------------
# _resolve_auto_credentials
# ---------------------------------------------------------------------------


from plugins.memory.mem0_oss import _resolve_auto_credentials


class TestResolveAutoCredentials:
    def test_noop_when_explicit_provider_set(self, monkeypatch):
        """Returns inputs unchanged when aux_provider is already set."""
        result = _resolve_auto_credentials("openai", "gpt-4o", "", "mykey")
        assert result == ("openai", "gpt-4o", "", "mykey")

    def test_noop_when_provider_is_nonempty_non_auto(self, monkeypatch):
        result = _resolve_auto_credentials("anthropic", "", "", "")
        assert result[0] == "anthropic"

    def test_auto_picks_openrouter_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        # Directly test the real function with no aux provider
        provider, model, base_url, api_key = _resolve_auto_credentials("", "", "", "")
        assert provider == "openrouter"
        assert api_key == "sk-or-test"
        assert "openrouter.ai" in base_url

    def test_auto_picks_anthropic_when_no_openrouter(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        provider, model, base_url, api_key = _resolve_auto_credentials("", "", "", "")
        assert provider == "anthropic"
        assert api_key == "sk-ant-test"

    def test_auto_picks_openai_when_no_openrouter_or_anthropic(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        provider, model, base_url, api_key = _resolve_auto_credentials("", "", "", "")
        assert provider == "openai"
        assert api_key == "sk-openai-test"

    def test_auto_picks_bedrock_when_aws_key_set(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        provider, model, base_url, api_key = _resolve_auto_credentials("", "", "", "")
        assert provider == "aws_bedrock"

    def test_auto_returns_auto_when_nothing_configured(self, monkeypatch):
        for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                  "AWS_ACCESS_KEY_ID", "AWS_PROFILE"):
            monkeypatch.delenv(k, raising=False)
        provider, model, base_url, api_key = _resolve_auto_credentials("", "", "", "")
        assert provider == "auto"

    def test_auto_is_also_treated_as_unset(self, monkeypatch):
        """aux_provider='auto' triggers auto-detect, same as empty string."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-auto")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        provider, model, base_url, api_key = _resolve_auto_credentials("auto", "", "", "")
        assert provider == "openrouter"
        assert api_key == "sk-or-auto"

    def test_explicit_aux_api_key_not_overwritten(self, monkeypatch):
        """Existing aux_api_key is never replaced by auto-detected key."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "should-not-use")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        provider, model, base_url, api_key = _resolve_auto_credentials("", "", "", "my-existing-key")
        # Provider resolved to openrouter, but key should be the existing one
        assert api_key == "my-existing-key"


class TestLoadConfigAutoFallthrough:
    """Integration: _load_config picks up provider key via auto-detect."""

    def test_openrouter_key_propagates_when_no_aux_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-load-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        cfg = _load_config()
        assert cfg["api_key"] == "sk-or-load-test"
        assert cfg["llm_provider"] == "openai"  # openrouter normalises to openai

    def test_anthropic_key_propagates_when_no_aux_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-load-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        cfg = _load_config()
        assert cfg["api_key"] == "sk-ant-load-test"
        assert cfg["llm_provider"] == "anthropic"

    def test_explicit_mem0_oss_api_key_beats_auto(self, tmp_path, monkeypatch):
        """MEM0_OSS_API_KEY always wins over auto-detected key."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setattr("plugins.memory.mem0_oss._get_aux_config", lambda: {})
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("MEM0_OSS_API_KEY", "explicit-wins")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        cfg = _load_config()
        assert cfg["api_key"] == "explicit-wins"


# ---------------------------------------------------------------------------
# _build_mem0_config
# ---------------------------------------------------------------------------


class TestBuildMem0Config:
    def _base_cfg(self, **overrides):
        base = {
            "vector_store_path": "/tmp/qdrant",
            "history_db_path": "/tmp/history.db",
            "collection": "hermes",
            "user_id": "hermes-user",
            "llm_provider": "aws_bedrock",
            "llm_model": "some-model",
            "embedder_provider": "aws_bedrock",
            "embedder_model": "some-embed",
            "embedder_dims": 1024,
            "top_k": 10,
            "api_key": "",
            "base_url": "",
            "openai_api_key": "",
            "openai_base_url": "",
        }
        base.update(overrides)
        return base

    def test_vector_store_is_qdrant_local(self):
        out = _build_mem0_config(self._base_cfg())
        assert out["vector_store"]["provider"] == "qdrant"
        assert out["vector_store"]["config"]["path"] == "/tmp/qdrant"
        assert out["vector_store"]["config"]["on_disk"] is True

    def test_collection_name_passed(self):
        out = _build_mem0_config(self._base_cfg(collection="my-col"))
        assert out["vector_store"]["config"]["collection_name"] == "my-col"

    def test_embedding_dims_wired_to_qdrant(self):
        out = _build_mem0_config(self._base_cfg(embedder_dims=768))
        assert out["vector_store"]["config"]["embedding_model_dims"] == 768

    def test_openai_api_key_injected_when_provider_is_openai(self):
        out = _build_mem0_config(self._base_cfg(
            llm_provider="openai",
            openai_api_key="sk-xxx",
        ))
        assert out["llm"]["config"]["api_key"] == "sk-xxx"

    def test_openai_api_key_not_injected_for_bedrock(self):
        out = _build_mem0_config(self._base_cfg(openai_api_key="sk-xxx"))
        assert "api_key" not in out["llm"]["config"]

    def test_openai_base_url_injected_when_set(self):
        out = _build_mem0_config(self._base_cfg(
            llm_provider="openai",
            openai_api_key="k",
            openai_base_url="http://localhost:1234/v1",
        ))
        assert out["llm"]["config"]["openai_base_url"] == "http://localhost:1234/v1"

    def test_history_db_path_passed(self):
        out = _build_mem0_config(self._base_cfg(history_db_path="/tmp/h.db"))
        assert out["history_db_path"] == "/tmp/h.db"

    def test_version_is_v1_1(self):
        out = _build_mem0_config(self._base_cfg())
        assert out["version"] == "v1.1"


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_available_with_bedrock_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
        with patch("plugins.memory.mem0_oss._load_config", return_value={
            "llm_provider": "aws_bedrock", "openai_api_key": "",
        }):
            p = Mem0OSSMemoryProvider()
            assert p.is_available()

    def test_not_available_without_mem0ai(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKID")
        import sys
        mem0_backup = sys.modules.get("mem0")
        sys.modules["mem0"] = None  # type: ignore[assignment]
        try:
            p = Mem0OSSMemoryProvider()
            assert not p.is_available()
        finally:
            if mem0_backup is None:
                sys.modules.pop("mem0", None)
            else:
                sys.modules["mem0"] = mem0_backup

    def test_not_available_without_aws_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        with patch("plugins.memory.mem0_oss._load_config", return_value={
            "llm_provider": "aws_bedrock", "openai_api_key": "",
        }):
            with patch("agent.bedrock_adapter.has_aws_credentials", return_value=False):
                p = Mem0OSSMemoryProvider()
                assert not p.is_available()

    def test_available_via_has_aws_credentials_fallback(self, monkeypatch, tmp_path):
        """is_available returns True when has_aws_credentials() is True even without env vars."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        monkeypatch.delenv("AWS_PROFILE", raising=False)
        with patch("plugins.memory.mem0_oss._load_config", return_value={
            "llm_provider": "aws_bedrock", "openai_api_key": "",
        }):
            with patch("agent.bedrock_adapter.has_aws_credentials", return_value=True):
                p = Mem0OSSMemoryProvider()
                assert p.is_available()

    def test_available_with_openai_key_in_config(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        with patch("plugins.memory.mem0_oss._load_config", return_value={
            "llm_provider": "openai", "openai_api_key": "sk-test",
        }):
            p = Mem0OSSMemoryProvider()
            assert p.is_available()

    def test_not_available_openai_without_key(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch("plugins.memory.mem0_oss._load_config", return_value={
            "llm_provider": "openai", "openai_api_key": "",
        }):
            p = Mem0OSSMemoryProvider()
            assert not p.is_available()

    def test_ollama_always_available(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        with patch("plugins.memory.mem0_oss._load_config", return_value={
            "llm_provider": "ollama", "openai_api_key": "",
        }):
            p = Mem0OSSMemoryProvider()
            assert p.is_available()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------


class TestSchemas:
    def test_search_schema_has_query(self):
        assert SEARCH_SCHEMA["name"] == "mem0_oss_search"
        assert "query" in SEARCH_SCHEMA["parameters"]["properties"]
        assert "query" in SEARCH_SCHEMA["parameters"]["required"]

    def test_add_schema_has_content(self):
        assert ADD_SCHEMA["name"] == "mem0_oss_add"
        assert "content" in ADD_SCHEMA["parameters"]["properties"]
        assert "content" in ADD_SCHEMA["parameters"]["required"]

    def test_get_tool_schemas_returns_both(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 1
        names = {s["name"] for s in schemas}
        assert names == {"mem0_oss_search"}


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


class TestToolHandlers:
    def test_search_success(self, provider):
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {"query": "dark mode"}))
        assert "dark mode" in result["result"] or "hermes-agent" in result["result"]
        provider._mock_mem.search.assert_called_once()
        kwargs = provider._mock_mem.search.call_args.kwargs
        assert kwargs["query"] == "dark mode"
        assert kwargs["top_k"] == 10
        assert kwargs["filters"] == {"user_id": "test-user"}

    def test_search_respects_top_k_arg(self, provider):
        provider.handle_tool_call("mem0_oss_search", {"query": "q", "top_k": 3})
        kwargs = provider._mock_mem.search.call_args.kwargs
        assert kwargs["top_k"] == 3

    def test_search_caps_top_k_at_50(self, provider):
        provider.handle_tool_call("mem0_oss_search", {"query": "q", "top_k": 999})
        kwargs = provider._mock_mem.search.call_args.kwargs
        assert kwargs["top_k"] == 50

    def test_search_no_results(self, provider):
        provider._mock_mem.search.return_value = []
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {"query": "q"}))
        assert result["result"] == "No relevant memories found."

    def test_search_missing_query(self, provider):
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {}))
        assert "error" in result

    def test_search_error_handling(self, provider):
        provider._mock_mem.search.side_effect = RuntimeError("connection refused")
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {"query": "q"}))
        assert "error" in result
        assert "connection refused" in result["error"]

    def test_search_v2_dict_response(self, provider):
        provider._mock_mem.search.return_value = {
            "results": [{"memory": "A"}, {"memory": "B"}]
        }
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {"query": "q"}))
        assert "A" in result["result"]
        assert "B" in result["result"]

    def test_add_success(self, provider):
        result = json.loads(provider.handle_tool_call("mem0_oss_add", {"content": "User is Stan"}))
        assert result["result"] == "Memory stored successfully."
        provider._mock_mem.add.assert_called_once()
        kwargs = provider._mock_mem.add.call_args.kwargs
        assert kwargs["user_id"] == "test-user"
        assert kwargs["infer"] is True
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "User is Stan"

    def test_add_missing_content(self, provider):
        result = json.loads(provider.handle_tool_call("mem0_oss_add", {}))
        assert "error" in result

    def test_add_error_handling(self, provider):
        provider._mock_mem.add.side_effect = RuntimeError("LLM timeout")
        result = json.loads(provider.handle_tool_call("mem0_oss_add", {"content": "test"}))
        assert "error" in result
        assert "LLM timeout" in result["error"]

    def test_unknown_tool(self, provider):
        result = json.loads(provider.handle_tool_call("mem0_oss_unknown", {}))
        assert "error" in result

    def test_search_lock_contention_returns_graceful_message(self, provider):
        """Qdrant lock error produces a non-fatal result, not an error dict."""
        provider._mock_mem.search.side_effect = RuntimeError(
            "Storage folder already accessed by another instance of Qdrant client"
        )
        result = json.loads(provider.handle_tool_call("mem0_oss_search", {"query": "q"}))
        # Should be a graceful result, not a tool_error
        assert "result" in result
        assert "locked" in result["result"].lower() or "unavailable" in result["result"].lower()

    def test_add_lock_contention_returns_graceful_message(self, provider):
        """Qdrant lock error on add produces a non-fatal result."""
        provider._mock_mem.add.side_effect = RuntimeError(
            "Storage folder already accessed by another instance of Qdrant client"
        )
        result = json.loads(provider.handle_tool_call("mem0_oss_add", {"content": "test"}))
        assert "result" in result
        assert "locked" in result["result"].lower() or "unavailable" in result["result"].lower()


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    def test_breaker_trips_after_threshold(self, provider):
        from plugins.memory.mem0_oss import _BREAKER_THRESHOLD
        provider._mock_mem.search.side_effect = RuntimeError("fail")
        for _ in range(_BREAKER_THRESHOLD):
            provider.handle_tool_call("mem0_oss_search", {"query": "q"})
        assert provider._is_tripped()

    def test_breaker_resets_after_cooldown(self, provider):
        from plugins.memory.mem0_oss import _BREAKER_THRESHOLD
        provider._mock_mem.search.side_effect = RuntimeError("fail")
        for _ in range(_BREAKER_THRESHOLD):
            provider.handle_tool_call("mem0_oss_search", {"query": "q"})

        provider._last_fail_ts = time.monotonic() - 999  # simulate expired cooldown
        assert not provider._is_tripped()

    def test_breaker_clears_on_success(self, provider):
        from plugins.memory.mem0_oss import _BREAKER_THRESHOLD
        provider._mock_mem.search.side_effect = RuntimeError("fail")
        for _ in range(_BREAKER_THRESHOLD - 1):
            provider.handle_tool_call("mem0_oss_search", {"query": "q"})

        provider._mock_mem.search.side_effect = None
        provider._mock_mem.search.return_value = [{"memory": "ok"}]
        provider.handle_tool_call("mem0_oss_search", {"query": "q"})
        assert provider._fail_count == 0

    def test_tripped_breaker_skips_prefetch(self, provider):
        from plugins.memory.mem0_oss import _BREAKER_THRESHOLD
        with provider._lock:
            provider._fail_count = _BREAKER_THRESHOLD
            provider._last_fail_ts = time.monotonic()
        provider.queue_prefetch("anything")
        assert provider._prefetch_thread is None


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------


class TestPrefetch:
    def test_prefetch_returns_formatted_result(self, provider):
        provider.queue_prefetch("dark mode preferences")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        result = provider.prefetch("dark mode preferences")
        assert "Mem0 OSS Memory" in result
        assert "dark mode" in result or "hermes-agent" in result

    def test_prefetch_empty_on_no_results(self, provider):
        provider._mock_mem.search.return_value = []
        provider.queue_prefetch("nothing here")
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        assert provider.prefetch("nothing here") == ""

    def test_prefetch_truncates_long_query(self, provider):
        long_query = "x" * 1000
        provider.queue_prefetch(long_query)
        if provider._prefetch_thread:
            provider._prefetch_thread.join(timeout=5.0)
        call_args = provider._mock_mem.search.call_args
        assert len(call_args.kwargs["query"]) <= 500

    def test_prefetch_joins_thread_on_return(self, provider):
        provider.queue_prefetch("test")
        assert provider._prefetch_thread is not None
        result = provider.prefetch("test")  # should join internally
        assert provider._prefetch_thread is None


# ---------------------------------------------------------------------------
# sync_turn
# ---------------------------------------------------------------------------


class TestSyncTurn:
    def test_sync_turn_calls_add(self, provider):
        provider.sync_turn("hello", "hi there")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        provider._mock_mem.add.assert_called_once()
        kwargs = provider._mock_mem.add.call_args.kwargs
        assert kwargs["user_id"] == "test-user"
        assert kwargs["infer"] is True
        msgs = kwargs["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "hi there"

    def test_sync_turn_skipped_for_subagent(self, provider):
        provider._agent_context = "subagent"
        provider.sync_turn("hello", "hi")
        assert provider._sync_thread is None

    def test_sync_turn_error_does_not_raise(self, provider):
        provider._mock_mem.add.side_effect = RuntimeError("network down")
        provider.sync_turn("hello", "hi")
        if provider._sync_thread:
            provider._sync_thread.join(timeout=5.0)
        # No exception propagated

    def test_sync_turn_skipped_when_breaker_tripped(self, provider):
        from plugins.memory.mem0_oss import _BREAKER_THRESHOLD
        with provider._lock:
            provider._fail_count = _BREAKER_THRESHOLD
            provider._last_fail_ts = time.monotonic()
        provider.sync_turn("hello", "hi")
        assert provider._sync_thread is None


# ---------------------------------------------------------------------------
# System prompt block
# ---------------------------------------------------------------------------


class TestSystemPromptBlock:
    def test_mentions_tool_names(self, provider):
        block = provider.system_prompt_block()
        assert "mem0_oss_search" in block

    def test_mentions_long_term_memory(self, provider):
        block = provider.system_prompt_block()
        assert "memory" in block.lower()


# ---------------------------------------------------------------------------
# Provider name
# ---------------------------------------------------------------------------


class TestProviderName:
    def test_name(self, monkeypatch, tmp_path):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        assert Mem0OSSMemoryProvider().name == "mem0_oss"


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class TestConfigSchema:
    def test_schema_has_expected_keys(self, provider):
        schema = provider.get_config_schema()
        keys = {f["key"] for f in schema}
        expected = {
            "llm_provider", "llm_model", "embedder_provider", "embedder_model",
            "embedder_dims", "collection", "user_id", "top_k",
            # New dedicated key (preferred) + legacy alias still present
            "api_key", "openai_api_key",
            # base_url (replaces openai_base_url in name)
            "base_url",
        }
        assert expected.issubset(keys)

    def test_api_key_is_secret(self, provider):
        schema = provider.get_config_schema()
        key_entry = next(f for f in schema if f["key"] == "api_key")
        assert key_entry.get("secret") is True

    def test_openai_api_key_is_secret(self, provider):
        schema = provider.get_config_schema()
        key_entry = next(f for f in schema if f["key"] == "openai_api_key")
        assert key_entry.get("secret") is True


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_creates_storage_directories(self, tmp_path, monkeypatch):
        with patch("plugins.memory.mem0_oss._load_config") as mock_cfg:
            mock_cfg.return_value = {
                "vector_store_path": str(tmp_path / "oss" / "qdrant"),
                "history_db_path": str(tmp_path / "oss" / "history.db"),
                "collection": "hermes",
                "user_id": "u",
                "llm_provider": "aws_bedrock",
                "llm_model": "m",
                "embedder_provider": "aws_bedrock",
                "embedder_model": "e",
                "embedder_dims": 1024,
                "top_k": 10,
                "openai_api_key": "",
                "openai_base_url": "",
            }
            p = Mem0OSSMemoryProvider()
            p.initialize("sess")
        assert (tmp_path / "oss" / "qdrant").is_dir()
        assert (tmp_path / "oss").is_dir()  # parent of history.db

    def test_primary_context_allows_sync(self, provider):
        assert provider._agent_context == "primary"

    def test_default_top_k(self, provider):
        assert provider._top_k == 10


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestOnMemoryWrite:
    """on_memory_write mirrors builtin memory tool calls into mem0."""

    def test_add_action_triggers_mem0_add(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        p.initialize("test-session")
        p._user_id = "hermes-user"

        mock_mem = MagicMock()
        p._get_memory = lambda: mock_mem

        p.on_memory_write("add", "user", "Stan likes light mode")
        # Give the background thread a moment
        import time; time.sleep(0.2)

        mock_mem.add.assert_called_once()
        kwargs = mock_mem.add.call_args.kwargs
        assert kwargs["user_id"] == "hermes-user"
        assert kwargs["infer"] is False
        assert kwargs["metadata"] == {"source": "hermes_memory_tool", "target": "user"}
        msgs = kwargs["messages"]
        assert isinstance(msgs, list)
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Stan likes light mode"

    def test_non_add_action_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        p.initialize("test-session")
        mock_mem = MagicMock()
        p._get_memory = lambda: mock_mem

        p.on_memory_write("replace", "user", "something")
        p.on_memory_write("remove", "memory", "something")
        import time; time.sleep(0.1)

        mock_mem.add.assert_not_called()

    def test_empty_content_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        p.initialize("test-session")
        mock_mem = MagicMock()
        p._get_memory = lambda: mock_mem

        p.on_memory_write("add", "user", "")
        p.on_memory_write("add", "user", "   ")
        import time; time.sleep(0.1)

        mock_mem.add.assert_not_called()


class TestShutdown:
    def test_shutdown_joins_sync_thread(self, provider):
        finished = threading.Event()

        def _slow_sync():
            time.sleep(0.1)
            finished.set()

        t = threading.Thread(target=_slow_sync, daemon=True)
        t.start()
        provider._sync_thread = t
        provider.shutdown()
        assert finished.is_set()

    def test_shutdown_no_error_when_no_threads(self, provider):
        provider._sync_thread = None
        provider._prefetch_thread = None
        provider.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


class TestSaveConfig:
    def test_creates_json_file(self, tmp_path, monkeypatch):
        """save_config writes values to mem0_oss.json."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        p.save_config({"llm_provider": "openai", "user_id": "stan"}, tmp_path)
        cfg_path = tmp_path / "mem0_oss.json"
        assert cfg_path.exists()
        saved = json.loads(cfg_path.read_text())
        assert saved["llm_provider"] == "openai"
        assert saved["user_id"] == "stan"

    def test_merges_with_existing_file(self, tmp_path, monkeypatch):
        """save_config merges into existing file, preserving untouched keys."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        cfg_path = tmp_path / "mem0_oss.json"
        cfg_path.write_text(json.dumps({"collection": "old", "top_k": 5}))
        p.save_config({"collection": "new"}, tmp_path)
        saved = json.loads(cfg_path.read_text())
        assert saved["collection"] == "new"
        assert saved["top_k"] == 5  # untouched

    def test_overwrites_existing_key(self, tmp_path, monkeypatch):
        """save_config overwrites an existing key."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        cfg_path = tmp_path / "mem0_oss.json"
        cfg_path.write_text(json.dumps({"user_id": "old-user"}))
        p.save_config({"user_id": "new-user"}, tmp_path)
        saved = json.loads(cfg_path.read_text())
        assert saved["user_id"] == "new-user"

    def test_graceful_if_existing_file_malformed(self, tmp_path, monkeypatch):
        """save_config ignores malformed JSON and writes fresh."""
        monkeypatch.setattr("plugins.memory.mem0_oss.get_hermes_home", lambda: tmp_path)
        p = Mem0OSSMemoryProvider()
        cfg_path = tmp_path / "mem0_oss.json"
        cfg_path.write_text("{not valid json")
        p.save_config({"llm_provider": "openai"}, tmp_path)
        saved = json.loads(cfg_path.read_text())
        assert saved["llm_provider"] == "openai"
