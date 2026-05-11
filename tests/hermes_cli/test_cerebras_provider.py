"""Tests for Cerebras provider support — direct API provider."""

import types

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
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "ARCEEAI_API_KEY", "GMI_API_KEY",
    "NVIDIA_API_KEY", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestCerebrasProviderRegistry:
    def test_registered(self):
        assert "cerebras" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["cerebras"].name == "Cerebras"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["cerebras"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["cerebras"].inference_base_url == "https://api.cerebras.ai/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["cerebras"].api_key_env_vars == ("CEREBRAS_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["cerebras"].base_url_env_var == "CEREBRAS_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================


class TestCerebrasAliases:
    @pytest.mark.parametrize("alias", ["cerebras", "cerebras-cloud"])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("CEREBRAS_API_KEY", "cs-test-12345")
        assert resolve_provider(alias) == "cerebras"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("cerebras-cloud") == "cerebras"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("cerebras-cloud") == "cerebras"


# =============================================================================
# Credentials
# =============================================================================


class TestCerebrasCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "cs-test")
        status = get_api_key_provider_status("cerebras")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        status = get_api_key_provider_status("cerebras")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_cerebras_configured(self, monkeypatch):
        """OpenRouter users should NOT see cerebras as configured."""
        monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("cerebras")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "cs-direct-key")
        monkeypatch.delenv("CEREBRAS_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("cerebras")
        assert creds["api_key"] == "cs-direct-key"
        assert creds["base_url"] == "https://api.cerebras.ai/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("CEREBRAS_API_KEY", "cs-x")
        monkeypatch.setenv("CEREBRAS_BASE_URL", "https://custom.cerebras.example/v1")
        creds = resolve_api_key_provider_credentials("cerebras")
        assert creds["base_url"] == "https://custom.cerebras.example/v1"


# =============================================================================
# Model catalog
# =============================================================================


class TestCerebrasModelCatalog:
    def test_static_model_list(self):
        """Cerebras has a static _PROVIDER_MODELS catalog entry seeded
        from the models.dev catalog. Specific model names change with
        releases and don't belong in tests.
        """
        from hermes_cli.models import _PROVIDER_MODELS
        assert "cerebras" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["cerebras"]) >= 1

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "cerebras" in slugs

    def test_models_dev_preferred(self):
        from hermes_cli.models import _MODELS_DEV_PREFERRED
        assert "cerebras" in _MODELS_DEV_PREFERRED


# =============================================================================
# Model normalization
# =============================================================================


class TestCerebrasNormalization:
    def test_in_matching_prefix_strip_set(self):
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS
        assert "cerebras" in _MATCHING_PREFIX_STRIP_PROVIDERS

    def test_strips_prefix(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("cerebras/gpt-oss-120b", "cerebras") == "gpt-oss-120b"

    def test_bare_name_unchanged(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("llama3.1-8b", "cerebras") == "llama3.1-8b"


# =============================================================================
# URL mapping
# =============================================================================


class TestCerebrasURLMapping:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        # Auto-extended at import time from the cerebras ProviderProfile.
        assert _URL_TO_PROVIDER.get("api.cerebras.ai") == "cerebras"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "cerebras" in _PROVIDER_PREFIXES

    def test_trajectory_compressor_detects_cerebras(self):
        import trajectory_compressor as tc
        comp = tc.TrajectoryCompressor.__new__(tc.TrajectoryCompressor)
        comp.config = types.SimpleNamespace(base_url="https://api.cerebras.ai/v1")
        assert comp._detect_provider() == "cerebras"


# =============================================================================
# providers.py overlay + aliases
# =============================================================================


class TestCerebrasProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "cerebras" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["cerebras"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "CEREBRAS_BASE_URL"
        assert not overlay.is_aggregator

    def test_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS["cerebras"] == "Cerebras"


# =============================================================================
# Provider profile (modern path)
# =============================================================================


class TestCerebrasProfile:
    def test_profile_registered(self):
        from providers import get_provider_profile
        profile = get_provider_profile("cerebras")
        assert profile is not None
        assert profile.name == "cerebras"
        assert profile.base_url == "https://api.cerebras.ai/v1"
        assert "CEREBRAS_API_KEY" in profile.env_vars

    def test_profile_alias(self):
        from providers import get_provider_profile
        # Alias declared on the profile.
        assert get_provider_profile("cerebras-cloud") is get_provider_profile("cerebras")

    def test_default_aux_model(self):
        from providers import get_provider_profile
        profile = get_provider_profile("cerebras")
        assert profile.default_aux_model == "llama3.1-8b"


# =============================================================================
# Auxiliary client — main-model-first design
# =============================================================================


class TestCerebrasAuxiliary:
    def test_main_model_first_design(self):
        """Cerebras uses main-model-first — no entry in the legacy
        _API_KEY_PROVIDER_AUX_MODELS_FALLBACK dict. The cheap aux model
        comes from ProviderProfile.default_aux_model instead.
        """
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS_FALLBACK
        assert "cerebras" not in _API_KEY_PROVIDER_AUX_MODELS_FALLBACK
