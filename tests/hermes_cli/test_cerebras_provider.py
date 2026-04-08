"""Tests for Cerebras provider integration."""

import os
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider
from agent.model_metadata import _URL_TO_PROVIDER, _PROVIDER_PREFIXES
from agent.models_dev import PROVIDER_TO_MODELS_DEV, list_agentic_models


# ── Provider Registry ──

class TestCerebrasProviderRegistry:
    def test_cerebras_in_registry(self):
        assert "cerebras" in PROVIDER_REGISTRY

    def test_cerebras_config(self):
        pconfig = PROVIDER_REGISTRY["cerebras"]
        assert pconfig.id == "cerebras"
        assert pconfig.name == "Cerebras"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://api.cerebras.ai/v1"

    def test_cerebras_env_vars(self):
        pconfig = PROVIDER_REGISTRY["cerebras"]
        assert pconfig.api_key_env_vars == ("CEREBRAS_API_KEY",)
        assert pconfig.base_url_env_var == "CEREBRAS_BASE_URL"

    def test_cerebras_base_url(self):
        assert "api.cerebras.ai" in PROVIDER_REGISTRY["cerebras"].inference_base_url


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestCerebrasAliases:
    def test_explicit_cerebras(self):
        assert resolve_provider("cerebras") == "cerebras"

    def test_alias_cerebras_cloud(self):
        assert resolve_provider("cerebras-cloud") == "cerebras"

    def test_alias_cs(self):
        assert resolve_provider("cs") == "cerebras"

    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("cerebras-cloud") == "cerebras"
        assert _PROVIDER_ALIASES.get("cs") == "cerebras"

    def test_normalize_provider(self):
        assert normalize_provider("cerebras") == "cerebras"
        assert normalize_provider("cs") == "cerebras"


# ── Auto-detection ──

class TestCerebrasAutoDetection:
    def test_auto_detects_cerebras_api_key(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-cerebras-key")
        assert resolve_provider("auto") == "cerebras"


# ── Credential Resolution ──

class TestCerebrasCredentials:
    def test_resolve_with_cerebras_api_key(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-secret")
        creds = resolve_api_key_provider_credentials("cerebras")
        assert creds["provider"] == "cerebras"
        assert creds["api_key"] == "cerebras-secret"
        assert creds["base_url"] == "https://api.cerebras.ai/v1"

    def test_resolve_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "key")
        monkeypatch.setenv("CEREBRAS_BASE_URL", "https://custom.cerebras/v1")
        creds = resolve_api_key_provider_credentials("cerebras")
        assert creds["base_url"] == "https://custom.cerebras/v1"

    def test_runtime_cerebras(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "cerebras-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="cerebras")
        assert result["provider"] == "cerebras"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "cerebras-key"
        assert result["base_url"] == "https://api.cerebras.ai/v1"


# ── Model Catalog (dynamic — no static list) ──

class TestCerebrasModelCatalog:
    def test_no_static_model_list(self):
        """Cerebras models are discovered dynamically via models.dev + live API."""
        assert "cerebras" not in _PROVIDER_MODELS

    def test_provider_label(self):
        assert "cerebras" in _PROVIDER_LABELS
        assert _PROVIDER_LABELS["cerebras"] == "Cerebras"


# ── Model Normalization ──

class TestCerebrasModelNormalization:
    def test_passthrough_bare_name(self):
        """Cerebras is a passthrough provider — model names used as-is."""
        assert normalize_model_for_provider("gpt-oss-120b", "cerebras") == "gpt-oss-120b"

    def test_passthrough_llama(self):
        assert normalize_model_for_provider("llama3.1-8b", "cerebras") == "llama3.1-8b"


# ── URL-to-Provider Mapping ──

class TestCerebrasUrlMapping:
    def test_url_to_provider(self):
        assert _URL_TO_PROVIDER.get("api.cerebras.ai") == "cerebras"

    def test_provider_prefix_canonical(self):
        assert "cerebras" in _PROVIDER_PREFIXES

    def test_provider_prefix_alias(self):
        assert "cs" in _PROVIDER_PREFIXES


# ── models.dev Integration ──

class TestCerebrasModelsDev:
    def test_cerebras_mapped(self):
        assert PROVIDER_TO_MODELS_DEV.get("cerebras") == "cerebras"

    def test_list_agentic_models_with_mock_data(self):
        """list_agentic_models filters correctly from mock models.dev data."""
        mock_data = {
            "cerebras": {
                "models": {
                    "gpt-oss-120b": {"tool_call": True},
                    "llama3.1-8b": {"tool_call": True},
                    "some-embedding": {"tool_call": False},
                }
            }
        }
        with patch("agent.models_dev.fetch_models_dev", return_value=mock_data):
            result = list_agentic_models("cerebras")
        assert "gpt-oss-120b" in result
        assert "llama3.1-8b" in result
        assert "some-embedding" not in result


# ── Agent Init (no SyntaxError) ──

class TestCerebrasAgentInit:
    def test_agent_imports_without_error(self):
        """Verify run_agent.py has no SyntaxError."""
        import importlib
        import run_agent
        importlib.reload(run_agent)

    def test_cerebras_agent_uses_chat_completions(self, monkeypatch):
        """Cerebras falls through to chat_completions — no special elif needed."""
        monkeypatch.setenv("CEREBRAS_API_KEY", "test-key")
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from run_agent import AIAgent
            agent = AIAgent(
                model="gpt-oss-120b",
                provider="cerebras",
                api_key="test-key",
                base_url="https://api.cerebras.ai/v1",
            )
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "cerebras"


# ── providers.py New System ──

class TestCerebrasProvidersNew:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "cerebras" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["cerebras"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "CEREBRAS_BASE_URL"

    def test_alias_resolves(self):
        from hermes_cli.providers import normalize_provider as np
        assert np("cerebras") == "cerebras"
        assert np("cs") == "cerebras"
        assert np("cerebras-cloud") == "cerebras"

    def test_label_override(self):
        from hermes_cli.providers import _LABEL_OVERRIDES
        assert _LABEL_OVERRIDES.get("cerebras") == "Cerebras"

    def test_get_label(self):
        from hermes_cli.providers import get_label
        assert get_label("cerebras") == "Cerebras"

    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("cerebras")
        assert pdef is not None
        assert pdef.id == "cerebras"
        assert pdef.transport == "openai_chat"


# ── Auxiliary Model ──

class TestCerebrasAuxiliary:
    def test_aux_model_defined(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "cerebras" in _API_KEY_PROVIDER_AUX_MODELS
        assert _API_KEY_PROVIDER_AUX_MODELS["cerebras"] == "llama3.1-8b"
