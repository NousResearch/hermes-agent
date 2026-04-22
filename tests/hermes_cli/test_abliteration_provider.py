"""Tests for abliteration.ai provider support."""

import types
from unittest.mock import patch

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)


_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "DEEPSEEK_API_KEY",
    "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY",
    "XIAOMI_API_KEY",
    "ARCEEAI_API_KEY",
    "NVIDIA_API_KEY",
    "HF_TOKEN",
)


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    for key in _OTHER_PROVIDER_KEYS + ("ABLITERATION_API_KEY", "ABLITERATION_BASE_URL"):
        monkeypatch.delenv(key, raising=False)


class TestAbliterationProviderRegistry:
    def test_registered(self):
        assert "abliteration" in PROVIDER_REGISTRY

    def test_config(self):
        pconfig = PROVIDER_REGISTRY["abliteration"]
        assert pconfig.id == "abliteration"
        assert pconfig.name == "abliteration.ai"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://api.abliteration.ai/v1"
        assert pconfig.api_key_env_vars == ("ABLITERATION_API_KEY",)
        assert pconfig.base_url_env_var == "ABLITERATION_BASE_URL"


class TestAbliterationAliases:
    @pytest.mark.parametrize(
        "alias",
        ["abliteration", "abliteration-ai", "abliteration.ai", "abliterationai"],
    )
    def test_auth_aliases(self, alias):
        assert resolve_provider(alias) == "abliteration"

    def test_models_aliases(self):
        from hermes_cli.models import normalize_provider

        assert normalize_provider("abliteration-ai") == "abliteration"
        assert normalize_provider("abliteration.ai") == "abliteration"

    def test_providers_aliases(self):
        from hermes_cli.providers import normalize_provider

        assert normalize_provider("abliteration-ai") == "abliteration"
        assert normalize_provider("abliteration.ai") == "abliteration"

    def test_auto_detects_primary_env_var(self, monkeypatch):
        monkeypatch.setenv("ABLITERATION_API_KEY", "sk-ablit-primary")
        assert resolve_provider("auto") == "abliteration"


class TestAbliterationCredentials:
    def test_status_configured_with_primary_env_var(self, monkeypatch):
        monkeypatch.setenv("ABLITERATION_API_KEY", "sk-ablit-primary")
        status = get_api_key_provider_status("abliteration")
        assert status["configured"]

    def test_resolve_credentials_with_primary_env_var(self, monkeypatch):
        monkeypatch.setenv("ABLITERATION_API_KEY", "sk-ablit-primary")
        creds = resolve_api_key_provider_credentials("abliteration")
        assert creds["provider"] == "abliteration"
        assert creds["api_key"] == "sk-ablit-primary"
        assert creds["base_url"] == "https://api.abliteration.ai/v1"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("ABLITERATION_API_KEY", "sk-ablit-primary")
        monkeypatch.setenv("ABLITERATION_BASE_URL", "https://proxy.example.com/v1")
        creds = resolve_api_key_provider_credentials("abliteration")
        assert creds["base_url"] == "https://proxy.example.com/v1"


class TestAbliterationRuntime:
    def test_runtime_provider_resolution(self, monkeypatch):
        from hermes_cli.runtime_provider import resolve_runtime_provider

        monkeypatch.setenv("ABLITERATION_API_KEY", "sk-ablit-primary")
        result = resolve_runtime_provider(requested="abliteration")

        assert result["provider"] == "abliteration"
        assert result["api_mode"] == "chat_completions"
        assert result["api_key"] == "sk-ablit-primary"
        assert result["base_url"] == "https://api.abliteration.ai/v1"
        assert result["requested_provider"] == "abliteration"


class TestAbliterationModelCatalog:
    def test_static_fallback_model_list_exists(self):
        from hermes_cli.models import _PROVIDER_MODELS

        assert "abliteration" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["abliteration"]) >= 1

    def test_canonical_provider_entry_exists(self):
        from hermes_cli.models import CANONICAL_PROVIDERS

        slugs = [entry.slug for entry in CANONICAL_PROVIDERS]
        assert "abliteration" in slugs

    def test_provider_label(self):
        from hermes_cli.models import _PROVIDER_LABELS

        assert _PROVIDER_LABELS["abliteration"] == "abliteration.ai"

    def test_provider_model_ids_prefers_live_catalog(self):
        from hermes_cli.models import provider_model_ids

        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={
                "provider": "abliteration",
                "api_key": "sk-ablit-primary",
                "base_url": "https://api.abliteration.ai/v1",
                "source": "env",
            },
        ), patch(
            "hermes_cli.models.fetch_api_models",
            return_value=["abliterated-model-live", "vision-model-live"],
        ):
            assert provider_model_ids("abliteration") == [
                "abliterated-model-live",
                "vision-model-live",
            ]


class TestAbliterationNormalization:
    def test_in_matching_prefix_strip_set(self):
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS

        assert "abliteration" in _MATCHING_PREFIX_STRIP_PROVIDERS

    def test_strips_canonical_prefix(self):
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert (
            normalize_model_for_provider(
                "abliteration/abliterated-model", "abliteration"
            )
            == "abliterated-model"
        )

    def test_strips_alias_prefix(self):
        from hermes_cli.model_normalize import normalize_model_for_provider

        assert (
            normalize_model_for_provider(
                "abliteration.ai/abliterated-model", "abliteration"
            )
            == "abliterated-model"
        )


class TestAbliterationMetadata:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.abliteration.ai") == "abliteration"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "abliteration" in _PROVIDER_PREFIXES
        assert "abliteration-ai" in _PROVIDER_PREFIXES
        assert "abliteration.ai" in _PROVIDER_PREFIXES

    def test_context_length_for_documented_model(self):
        from agent.model_metadata import get_model_context_length

        assert get_model_context_length("abliterated-model") == 150000

    def test_trajectory_compressor_detects_provider_from_base_url(self):
        import trajectory_compressor as tc

        comp = tc.TrajectoryCompressor.__new__(tc.TrajectoryCompressor)
        comp.config = types.SimpleNamespace(base_url="https://api.abliteration.ai/v1")
        assert comp._detect_provider() == "abliteration"


class TestAbliterationProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS, determine_api_mode

        overlay = HERMES_OVERLAYS["abliteration"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "ABLITERATION_BASE_URL"
        assert not overlay.is_aggregator
        assert determine_api_mode("abliteration") == "chat_completions"


class TestAbliterationAuxiliary:
    def test_aux_model_is_in_provider_catalog(self):
        from agent.auxiliary_client import _API_KEY_PROVIDER_AUX_MODELS
        from hermes_cli.models import _PROVIDER_MODELS

        aux_model = _API_KEY_PROVIDER_AUX_MODELS["abliteration"]
        assert aux_model in _PROVIDER_MODELS["abliteration"]
