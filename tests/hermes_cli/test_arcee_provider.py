"""Tests for Arcee AI provider support — standard direct API provider."""

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
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestArceeProviderRegistry:
    def test_registered(self):
        assert "arcee" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["arcee"].name == "Arcee AI"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["arcee"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["arcee"].inference_base_url == "https://api.arcee.ai/api/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["arcee"].api_key_env_vars == ("ARCEEAI_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["arcee"].base_url_env_var == "ARCEE_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================


class TestArceeAliases:
    @pytest.mark.parametrize("alias", ["arcee", "arcee-ai", "arceeai"])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("ARCEEAI_API_KEY", "arc-test-12345")
        assert resolve_provider(alias) == "arcee"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("arcee-ai") == "arcee"
        assert normalize_provider("arceeai") == "arcee"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("arcee-ai") == "arcee"
        assert normalize_provider("arceeai") == "arcee"


# =============================================================================
# Credentials
# =============================================================================


class TestArceeCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("ARCEEAI_API_KEY", "arc-test")
        status = get_api_key_provider_status("arcee")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("ARCEEAI_API_KEY", raising=False)
        status = get_api_key_provider_status("arcee")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_arcee_configured(self, monkeypatch):
        """OpenRouter users should NOT see arcee as configured."""
        monkeypatch.delenv("ARCEEAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("arcee")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("ARCEEAI_API_KEY", "arc-direct-key")
        monkeypatch.delenv("ARCEE_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("arcee")
        assert creds["api_key"] == "arc-direct-key"
        assert creds["base_url"] == "https://api.arcee.ai/api/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("ARCEEAI_API_KEY", "arc-x")
        monkeypatch.setenv("ARCEE_BASE_URL", "https://custom.arcee.example/v1")
        creds = resolve_api_key_provider_credentials("arcee")
        assert creds["base_url"] == "https://custom.arcee.example/v1"


# =============================================================================
# Model catalog
# =============================================================================


class TestArceeModelCatalog:
    def test_static_model_list(self):
        """Arcee has a static _PROVIDER_MODELS catalog entry. Specific model
        names change with releases and don't belong in tests.
        """
        from hermes_cli.models import _PROVIDER_MODELS
        assert "arcee" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["arcee"]) >= 1

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "arcee" in slugs


# =============================================================================
# Model normalization
# =============================================================================


class TestArceeNormalization:
    def test_in_matching_prefix_strip_set(self):
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS
        assert "arcee" in _MATCHING_PREFIX_STRIP_PROVIDERS

    def test_strips_prefix(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("arcee/trinity-mini", "arcee") == "trinity-mini"

    def test_bare_name_unchanged(self):
        from hermes_cli.model_normalize import normalize_model_for_provider
        assert normalize_model_for_provider("trinity-mini", "arcee") == "trinity-mini"


# =============================================================================
# URL mapping
# =============================================================================


class TestArceeURLMapping:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.arcee.ai") == "arcee"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "arcee" in _PROVIDER_PREFIXES
        assert "arcee-ai" in _PROVIDER_PREFIXES
        assert "arceeai" in _PROVIDER_PREFIXES

    def test_trajectory_compressor_detects_arcee(self):
        import trajectory_compressor as tc
        comp = tc.TrajectoryCompressor.__new__(tc.TrajectoryCompressor)
        comp.config = types.SimpleNamespace(base_url="https://api.arcee.ai/api/v1")
        assert comp._detect_provider() == "arcee"


# =============================================================================
# providers.py overlay + aliases
# =============================================================================


class TestArceeProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "arcee" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["arcee"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "ARCEE_BASE_URL"
        assert not overlay.is_aggregator

    def test_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS["arcee"] == "Arcee AI"


# =============================================================================
# Auxiliary client — main-model-first design
# =============================================================================


class TestArceeAuxiliary:
    def test_main_model_first_design(self):
        """Arcee uses main-model-first — no entry in _API_KEY_PROVIDER_AUX_MODELS."""
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "arcee" not in _API_KEY_PROVIDER_AUX_MODELS
