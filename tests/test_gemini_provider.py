"""Tests for Google AI Studio (Gemini) provider integration."""

import os
import pytest
from unittest.mock import patch, MagicMock

from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider, resolve_api_key_provider_credentials
from hermes_cli.models import _PROVIDER_MODELS, _PROVIDER_LABELS, _PROVIDER_ALIASES, normalize_provider
from hermes_cli.model_normalize import normalize_model_for_provider, detect_vendor
from agent.model_metadata import get_model_context_length


# ── Provider Registry ──

class TestGeminiProviderRegistry:
    def test_gemini_in_registry(self):
        assert "gemini" in PROVIDER_REGISTRY

    def test_gemini_config(self):
        pconfig = PROVIDER_REGISTRY["gemini"]
        assert pconfig.id == "gemini"
        assert pconfig.name == "Google AI Studio"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_gemini_env_vars(self):
        pconfig = PROVIDER_REGISTRY["gemini"]
        assert pconfig.api_key_env_vars == ("GOOGLE_API_KEY", "GEMINI_API_KEY")
        assert pconfig.base_url_env_var == "GEMINI_BASE_URL"

    def test_gemini_base_url(self):
        assert "generativelanguage.googleapis.com" in PROVIDER_REGISTRY["gemini"].inference_base_url


# ── Provider Aliases ──

PROVIDER_ENV_VARS = (
    "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_BASE_URL",
    "GLM_API_KEY", "ZAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "DEEPSEEK_API_KEY",
)

@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for var in PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestGeminiAliases:
    def test_explicit_gemini(self):
        assert resolve_provider("gemini") == "gemini"

    def test_alias_google(self):
        assert resolve_provider("google") == "gemini"

    def test_alias_google_gemini(self):
        assert resolve_provider("google-gemini") == "gemini"

    def test_alias_google_ai_studio(self):
        assert resolve_provider("google-ai-studio") == "gemini"

    def test_models_py_aliases(self):
        assert _PROVIDER_ALIASES.get("google") == "gemini"
        assert _PROVIDER_ALIASES.get("google-gemini") == "gemini"
        assert _PROVIDER_ALIASES.get("google-ai-studio") == "gemini"

    def test_normalize_provider(self):
        assert normalize_provider("google") == "gemini"
        assert normalize_provider("gemini") == "gemini"
        assert normalize_provider("google-ai-studio") == "gemini"


# ── Auto-detection ──

class TestGeminiAutoDetection:
    def test_auto_detects_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        assert resolve_provider("auto") == "gemini"

    def test_auto_detects_gemini_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
        assert resolve_provider("auto") == "gemini"

    def test_google_api_key_priority_over_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "primary-key")
        monkeypatch.setenv("GEMINI_API_KEY", "alias-key")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["api_key"] == "primary-key"
        assert creds["source"] == "GOOGLE_API_KEY"


# ── Credential Resolution ──

class TestGeminiCredentials:
    def test_resolve_with_google_api_key(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-secret")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["provider"] == "gemini"
        assert creds["api_key"] == "google-secret"
        assert creds["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"

    def test_resolve_with_gemini_api_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["api_key"] == "gemini-secret"

    def test_resolve_with_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "key")
        monkeypatch.setenv("GEMINI_BASE_URL", "https://custom.endpoint/v1")
        creds = resolve_api_key_provider_credentials("gemini")
        assert creds["base_url"] == "https://custom.endpoint/v1"

    def test_runtime_gemini(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider
        result = resolve_runtime_provider(requested="gemini")
        assert result["provider"] == "gemini"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "google-key"
        assert result["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai"


# ── Model Catalog ──

class TestGeminiModelCatalog:
    def test_provider_models_exist(self):
        assert "gemini" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["gemini"]
        assert "gemini-2.5-pro" in models
        assert "gemini-2.5-flash" in models
        assert "gemma-4-31b-it" in models

    def test_provider_label(self):
        assert "gemini" in _PROVIDER_LABELS
        assert _PROVIDER_LABELS["gemini"] == "Google AI Studio"


# ── Model Normalization ──

class TestGeminiModelNormalization:
    def test_passthrough_bare_name(self):
        assert normalize_model_for_provider("gemini-2.5-flash", "gemini") == "gemini-2.5-flash"

    def test_strip_vendor_prefix(self):
        assert normalize_model_for_provider("google/gemini-2.5-flash", "gemini") == "google/gemini-2.5-flash"

    def test_gemma_vendor_detection(self):
        assert detect_vendor("gemma-4-31b-it") == "google"

    def test_gemini_vendor_detection(self):
        assert detect_vendor("gemini-2.5-flash") == "google"

    def test_aggregator_prepends_vendor(self):
        result = normalize_model_for_provider("gemini-2.5-flash", "openrouter")
        assert result == "google/gemini-2.5-flash"

    def test_gemma_aggregator_prepends_vendor(self):
        result = normalize_model_for_provider("gemma-4-31b-it", "openrouter")
        assert result == "google/gemma-4-31b-it"


# ── Context Length ──

class TestGeminiContextLength:
    def test_gemma_4_31b_context(self):
        ctx = get_model_context_length("gemma-4-31b-it", provider="gemini")
        assert ctx == 262144

    def test_gemma_4_e4b_context(self):
        ctx = get_model_context_length("gemma-4-e4b-it", provider="gemini")
        assert ctx == 131072


# ── Agent Init (no SyntaxError) ──

class TestGeminiAgentInit:
    def test_agent_imports_without_error(self):
        """Verify run_agent.py has no SyntaxError (the critical bug)."""
        import importlib
        import run_agent
        importlib.reload(run_agent)

    def test_gemini_agent_uses_chat_completions(self, monkeypatch):
        """Gemini falls through to chat_completions — no special elif needed."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        with patch("run_agent.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from run_agent import AIAgent
            agent = AIAgent(
                model="gemini-2.5-flash",
                provider="gemini",
                api_key="test-key",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            )
            assert agent.api_mode == "chat_completions"
            assert agent.provider == "gemini"
