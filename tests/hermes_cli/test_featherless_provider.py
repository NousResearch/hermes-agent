"""Tests for Featherless provider support — OpenAI-compatible direct API.

Featherless (https://featherless.ai/) serves thousands of open-source models
addressed by their Hugging Face repo IDs (e.g. ``zai-org/GLM-5.2``) through an
OpenAI-compatible endpoint, so it is wired like the other HF-style API-key
providers (GMI) and uses a main-model-first auxiliary design (like Arcee).
"""

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
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "ARCEEAI_API_KEY", "GMI_API_KEY",
    "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestFeatherlessProviderRegistry:
    def test_registered(self):
        assert "featherless" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["featherless"].name == "Featherless"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["featherless"].auth_type == "api_key"

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["featherless"].inference_base_url == "https://api.featherless.ai/v1"

    def test_api_key_env_vars(self):
        assert PROVIDER_REGISTRY["featherless"].api_key_env_vars == ("FEATHERLESS_API_KEY",)

    def test_base_url_env_var(self):
        assert PROVIDER_REGISTRY["featherless"].base_url_env_var == "FEATHERLESS_BASE_URL"


# =============================================================================
# Aliases
# =============================================================================


class TestFeatherlessAliases:
    @pytest.mark.parametrize("alias", ["featherless", "featherless-ai", "featherlessai"])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS + ("OPENROUTER_API_KEY",):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-test-12345")
        assert resolve_provider(alias) == "featherless"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("featherless-ai") == "featherless"
        assert normalize_provider("featherlessai") == "featherless"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("featherless-ai") == "featherless"
        assert normalize_provider("featherlessai") == "featherless"


# =============================================================================
# Credentials
# =============================================================================


class TestFeatherlessCredentials:
    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-test")
        status = get_api_key_provider_status("featherless")
        assert status["configured"]

    def test_status_not_configured(self, monkeypatch):
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        status = get_api_key_provider_status("featherless")
        assert not status["configured"]

    def test_openrouter_key_does_not_make_featherless_configured(self, monkeypatch):
        """OpenRouter users should NOT see featherless as configured."""
        monkeypatch.delenv("FEATHERLESS_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("featherless")
        assert not status["configured"]

    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-direct-key")
        monkeypatch.delenv("FEATHERLESS_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("featherless")
        assert creds["api_key"] == "fl-direct-key"
        assert creds["base_url"] == "https://api.featherless.ai/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("FEATHERLESS_API_KEY", "fl-x")
        monkeypatch.setenv("FEATHERLESS_BASE_URL", "https://custom.featherless.example/v1")
        creds = resolve_api_key_provider_credentials("featherless")
        assert creds["base_url"] == "https://custom.featherless.example/v1"


# =============================================================================
# Config registry (OPTIONAL_ENV_VARS)
# =============================================================================


class TestFeatherlessConfigRegistry:
    def test_optional_env_vars_include_featherless(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "FEATHERLESS_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["FEATHERLESS_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FEATHERLESS_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["FEATHERLESS_API_KEY"]["url"] == "https://featherless.ai/"

        assert "FEATHERLESS_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["FEATHERLESS_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["FEATHERLESS_BASE_URL"]["password"] is False


# =============================================================================
# Model catalog
# =============================================================================


class TestFeatherlessModelCatalog:
    def test_static_model_list(self):
        """Featherless has a static _PROVIDER_MODELS catalog entry. Specific
        model names track upstream releases and don't belong in tests.
        """
        from hermes_cli.models import _PROVIDER_MODELS
        assert "featherless" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["featherless"]) >= 1

    def test_default_model_is_glm(self):
        """The first catalog entry is the default offered on selection."""
        from hermes_cli.models import _PROVIDER_MODELS
        assert _PROVIDER_MODELS["featherless"][0] == "zai-org/GLM-5.2"

    def test_canonical_provider_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "featherless" in slugs


# =============================================================================
# URL mapping + provider prefixes
# =============================================================================


class TestFeatherlessURLMapping:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.featherless.ai") == "featherless"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "featherless" in _PROVIDER_PREFIXES
        assert "featherless-ai" in _PROVIDER_PREFIXES
        assert "featherlessai" in _PROVIDER_PREFIXES

    def test_trajectory_compressor_detects_featherless(self):
        import trajectory_compressor as tc
        comp = tc.TrajectoryCompressor.__new__(tc.TrajectoryCompressor)
        comp.config = types.SimpleNamespace(base_url="https://api.featherless.ai/v1")
        assert comp._detect_provider() == "featherless"


# =============================================================================
# providers.py overlay + label
# =============================================================================


class TestFeatherlessProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "featherless" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["featherless"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "FEATHERLESS_BASE_URL"
        assert not overlay.is_aggregator

    def test_label(self):
        from hermes_cli.models import _PROVIDER_LABELS
        assert _PROVIDER_LABELS["featherless"] == "Featherless"


# =============================================================================
# Doctor
# =============================================================================


class TestFeatherlessDoctor:
    def test_provider_env_hints_include_featherless(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS
        assert "FEATHERLESS_API_KEY" in _PROVIDER_ENV_HINTS


# =============================================================================
# Auxiliary client — main-model-first design
# =============================================================================


class TestFeatherlessAuxiliary:
    def test_main_model_first_design(self):
        """Featherless uses main-model-first — no entry in _API_KEY_PROVIDER_AUX_MODELS."""
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        assert "featherless" not in _API_KEY_PROVIDER_AUX_MODELS


# =============================================================================
# Context length — provider-served value (from /v1/models) wins over native
# =============================================================================


class TestFeatherlessContextLength:
    def test_endpoint_served_context_wins_over_native(self):
        """Featherless serves models at a capped context (e.g. zai-org/GLM-5.2 at
        256K, not its native 1M). The featherless branch in get_model_context_length
        must prefer the /v1/models context_length over the hardcoded native fallback.
        """
        from unittest.mock import patch
        from agent.model_metadata import get_model_context_length

        with patch(
            "agent.model_metadata.get_cached_context_length", return_value=None,
        ), patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value={"zai-org/GLM-5.2": {"context_length": 262144}},
        ), patch(
            "agent.models_dev.lookup_models_dev_context", return_value=None,
        ), patch(
            "agent.model_metadata.fetch_model_metadata", return_value={},
        ):
            result = get_model_context_length(
                "zai-org/GLM-5.2",
                base_url="https://api.featherless.ai/v1",
                api_key="fl-test-key",
                provider="featherless",
            )

        # 262144 (served) — NOT the native 1,048,576 hardcoded for glm-5.2.
        assert result == 262144
