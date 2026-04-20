"""Tests for ZenMux provider support — OpenAI-compatible aggregator."""

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
    "XAI_API_KEY", "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY", "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
    "ARCEEAI_API_KEY", "OLLAMA_API_KEY",
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestZenmuxProviderRegistry:
    def test_registered(self):
        assert "zenmux" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["zenmux"].name == "ZenMux"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["zenmux"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["zenmux"].inference_base_url == "https://zenmux.ai/api/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["zenmux"].api_key_env_vars == ("ZENMUX_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["zenmux"].base_url_env_var == "ZENMUX_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================


class TestZenmuxAliases:
    def test_alias_zenmux_ai(self, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ZENMUX_API_KEY", "zm-test-12345")
        assert resolve_provider("zenmux.ai") == "zenmux"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("zenmux.ai") == "zenmux"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("zenmux.ai") == "zenmux"


# =============================================================================
# Credentials
# =============================================================================


class TestZenmuxCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("ZENMUX_API_KEY", "zm-test")
        status = get_api_key_provider_status("zenmux")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("ZENMUX_API_KEY", raising=False)
        status = get_api_key_provider_status("zenmux")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_zenmux_configured(self, monkeypatch):
        monkeypatch.delenv("ZENMUX_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("zenmux")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("ZENMUX_API_KEY", "zm-direct-key")
        monkeypatch.delenv("ZENMUX_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("zenmux")
        assert creds["api_key"] == "zm-direct-key"
        assert creds["base_url"] == "https://zenmux.ai/api/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("ZENMUX_API_KEY", "zm-x")
        monkeypatch.setenv("ZENMUX_BASE_URL", "https://custom.zenmux.example/v1")
        creds = resolve_api_key_provider_credentials("zenmux")
        assert creds["base_url"] == "https://custom.zenmux.example/v1"


# =============================================================================
# Model catalog
# =============================================================================


class TestZenmuxModelCatalog:
    def test_static_model_list(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "zenmux" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["zenmux"]
        assert len(models) >= 1

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "zenmux" in slugs


# =============================================================================
# providers.py overlay + aliases
# =============================================================================


class TestZenmuxProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "zenmux" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["zenmux"]
        assert overlay.transport == "openai_chat"
        assert overlay.is_aggregator is True
        assert overlay.base_url_env_var == "ZENMUX_BASE_URL"

    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("zenmux")
        assert pdef is not None
        assert pdef.id == "zenmux"
        assert pdef.transport == "openai_chat"
        assert pdef.is_aggregator is True

    def test_determine_api_mode(self):
        from hermes_cli.providers import determine_api_mode
        assert determine_api_mode("zenmux") == "chat_completions"
