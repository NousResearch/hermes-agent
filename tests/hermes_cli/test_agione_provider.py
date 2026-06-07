"""Focused tests for AGIone first-class provider wiring."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from agent.auxiliary_client import resolve_provider_client
from hermes_cli.auth import resolve_provider
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _PROVIDER_LABELS,
    normalize_provider,
    provider_model_ids,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GLM_API_KEY",
        "KIMI_API_KEY",
        "GMI_API_KEY",
        "AGIONE_API_KEY",
        "AGIONE_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)


class TestAgioneAliases:
    @pytest.mark.parametrize("alias", ["agione", "agi-one", "agione-pro"])
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("AGIONE_API_KEY", "agione-test-key")
        assert resolve_provider(alias) == "agione"

    def test_models_normalize_provider(self):
        assert normalize_provider("agi-one") == "agione"
        assert normalize_provider("agione-pro") == "agione"

    def test_providers_normalize_provider(self):
        from hermes_cli.providers import normalize_provider as normalize_provider_in_providers

        assert normalize_provider_in_providers("agi-one") == "agione"
        assert normalize_provider_in_providers("agione-pro") == "agione"


class TestAgioneConfigRegistry:
    def test_optional_env_vars_include_agione(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        assert "AGIONE_API_KEY" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["AGIONE_API_KEY"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["AGIONE_API_KEY"]["password"] is True
        assert OPTIONAL_ENV_VARS["AGIONE_API_KEY"]["url"] == "https://agione.pro/"

        assert "AGIONE_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["AGIONE_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["AGIONE_BASE_URL"]["password"] is False


class TestAgioneModelCatalog:
    def test_canonical_provider_entry(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "agione" in slugs
        assert _PROVIDER_LABELS["agione"] == "AGIone"

    def test_provider_model_ids_prefers_live_models_endpoint(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "agione-live-key",
                "base_url": "https://agione.pro/hyperone/xapi/api/v1",
                "source": "AGIONE_API_KEY",
            },
        )

        from providers import get_provider_profile

        profile = get_provider_profile("agione")
        assert profile is not None

        called = {}

        def fake_fetch_models(*, api_key=None, timeout=8.0):
            called["api_key"] = api_key
            called["models_url"] = profile.models_url
            return ["deepseek/deepseek-v4-pro/d3462", "openai/GPT-5.5/c6fbe"]

        monkeypatch.setattr(profile, "fetch_models", fake_fetch_models)

        assert provider_model_ids("agione") == [
            "deepseek/deepseek-v4-pro/d3462",
            "openai/GPT-5.5/c6fbe",
        ]
        assert called == {
            "api_key": "agione-live-key",
            "models_url": "https://agione.pro/hyperone/xapi/api/models",
        }

    def test_provider_model_ids_falls_back_to_profile_models(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "",
                "base_url": "https://agione.pro/hyperone/xapi/api/v1",
                "source": "AGIONE_API_KEY",
            },
        )

        assert provider_model_ids("agione") == [
            "deepseek/deepseek-v4-pro/d3462",
            "openai/GPT-5.5/c6fbe",
            "anthropic/Claude-opus-4.7/a4d5d",
        ]


class TestAgioneProvidersModule:
    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS, get_provider

        assert "agione" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["agione"]
        assert overlay.transport == "openai_chat"
        assert overlay.extra_env_vars == ("AGIONE_API_KEY",)
        assert overlay.base_url_override == "https://agione.pro/hyperone/xapi/api/v1"
        assert overlay.base_url_env_var == "AGIONE_BASE_URL"
        assert overlay.is_aggregator

        pdef = get_provider("agione")
        assert pdef is not None
        assert pdef.name == "AGIone"
        assert pdef.base_url == "https://agione.pro/hyperone/xapi/api/v1"

    def test_profile_models_endpoint(self):
        from providers import get_provider_profile

        profile = get_provider_profile("agione")
        assert profile is not None
        assert profile.models_url == "https://agione.pro/hyperone/xapi/api/models"
        assert profile.default_aux_model == "deepseek/deepseek-v4-pro/d3462"


class TestAgioneDoctor:
    def test_provider_env_hints_include_agione(self):
        from hermes_cli.doctor import _PROVIDER_ENV_HINTS

        assert "AGIONE_API_KEY" in _PROVIDER_ENV_HINTS


class TestAgioneModelMetadata:
    def test_url_to_provider_auto_registered_from_profile(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("agione.pro") == "agione"

    def test_provider_prefixes(self):
        from agent.model_metadata import _PROVIDER_PREFIXES

        assert "agione" in _PROVIDER_PREFIXES
        assert "agi-one" in _PROVIDER_PREFIXES
        assert "agione-pro" in _PROVIDER_PREFIXES

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://agione.pro/hyperone/xapi/api/v1") == "agione"

class TestAgioneAuxiliary:
    def test_resolve_provider_client_uses_agione_aux_default(self, monkeypatch):
        monkeypatch.setenv("AGIONE_API_KEY", "agione-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = object()
            client, model = resolve_provider_client("agione")

        assert client is not None
        assert model == "deepseek/deepseek-v4-pro/d3462"
        assert mock_openai.call_args.kwargs["api_key"] == "agione-test-key"
        assert mock_openai.call_args.kwargs["base_url"] == "https://agione.pro/hyperone/xapi/api/v1"
