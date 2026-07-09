"""Focused tests for Meta Model API (Muse Spark) first-class provider wiring."""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.models import (
    CANONICAL_PROVIDERS,
    _PROVIDER_LABELS,
    provider_model_ids,
)
from agent.model_metadata import get_model_context_length


_META_ENV = (
    "MODEL_API_KEY",
    "META_API_KEY",
    "META_MODEL_API_KEY",
    "META_BASE_URL",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    for key in _META_ENV:
        monkeypatch.delenv(key, raising=False)


class TestMetaAiAliases:
    @pytest.mark.parametrize(
        "alias",
        ["meta-ai", "meta", "muse", "llama-api", "model-api"],
    )
    def test_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("MODEL_API_KEY", "meta-test-key")
        assert resolve_provider(alias) == "meta-ai"

    def test_provider_registry_entry(self):
        assert "meta-ai" in PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY["meta-ai"]
        assert pconfig.name == "Meta Model API"
        assert pconfig.auth_type == "api_key"
        assert pconfig.inference_base_url == "https://api.meta.ai/v1"
        assert pconfig.api_key_env_vars == (
            "MODEL_API_KEY",
            "META_API_KEY",
            "META_MODEL_API_KEY",
        )
        assert pconfig.base_url_env_var == "META_BASE_URL"


class TestMetaAiConfigRegistry:
    def test_optional_env_vars_include_meta(self):
        from hermes_cli.config import OPTIONAL_ENV_VARS

        for key in ("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY"):
            assert key in OPTIONAL_ENV_VARS
            assert OPTIONAL_ENV_VARS[key]["category"] == "provider"
            assert OPTIONAL_ENV_VARS[key]["password"] is True
            assert OPTIONAL_ENV_VARS[key]["url"] == "https://dev.meta.ai/"

        assert "META_BASE_URL" in OPTIONAL_ENV_VARS
        assert OPTIONAL_ENV_VARS["META_BASE_URL"]["category"] == "provider"
        assert OPTIONAL_ENV_VARS["META_BASE_URL"]["password"] is False


class TestMetaAiModelCatalog:
    def test_canonical_provider_entry(self):
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "meta-ai" in slugs
        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "meta-ai")
        assert entry.label == "Meta Model API"
        assert "Muse Spark" in entry.tui_desc

    def test_provider_label(self):
        assert _PROVIDER_LABELS["meta-ai"] == "Meta Model API"

    def test_provider_model_ids_prefers_live_api(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "meta-live-key",
                "base_url": "https://api.meta.ai/v1",
                "source": "MODEL_API_KEY",
            },
        )
        # Profile-based path uses ProviderProfile.fetch_models, not fetch_api_models.
        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            lambda self, *, api_key=None, base_url=None, timeout=8.0: [
                "muse-spark-1.1",
                "muse-spark-1.0",
            ],
        )

        assert provider_model_ids("meta-ai") == [
            "muse-spark-1.1",
            "muse-spark-1.0",
        ]

    def test_provider_model_ids_falls_back_to_profile_models(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "meta-live-key",
                "base_url": "https://api.meta.ai/v1",
                "source": "MODEL_API_KEY",
            },
        )
        monkeypatch.setattr(
            "hermes_cli.models.fetch_api_models",
            lambda api_key, base_url: None,
        )
        # Generic profile path uses ProviderProfile.fetch_models (urllib), not
        # fetch_api_models — must stub it or CI can hit the real endpoint.
        monkeypatch.setattr(
            "providers.base.ProviderProfile.fetch_models",
            lambda self, *, api_key=None, base_url=None, timeout=8.0: None,
        )

        models = provider_model_ids("meta-ai")
        assert models == ["muse-spark-1.1"]


class TestMetaAiCredentials:
    def test_resolve_with_model_api_key(self, monkeypatch):
        monkeypatch.setenv("MODEL_API_KEY", "model-key")
        creds = resolve_api_key_provider_credentials("meta-ai")
        assert creds["provider"] == "meta-ai"
        assert creds["api_key"] == "model-key"
        assert creds["base_url"] == "https://api.meta.ai/v1"

    def test_env_var_priority(self, monkeypatch):
        """MODEL_API_KEY wins over META_API_KEY / META_MODEL_API_KEY."""
        monkeypatch.setenv("MODEL_API_KEY", "first")
        monkeypatch.setenv("META_API_KEY", "second")
        monkeypatch.setenv("META_MODEL_API_KEY", "third")
        creds = resolve_api_key_provider_credentials("meta-ai")
        assert creds["api_key"] == "first"

    def test_fallback_meta_api_key(self, monkeypatch):
        monkeypatch.setenv("META_API_KEY", "meta-key")
        creds = resolve_api_key_provider_credentials("meta-ai")
        assert creds["api_key"] == "meta-key"

    def test_custom_base_url_override(self, monkeypatch):
        monkeypatch.setenv("MODEL_API_KEY", "key")
        monkeypatch.setenv("META_BASE_URL", "https://custom.meta.example/v1")
        creds = resolve_api_key_provider_credentials("meta-ai")
        assert creds["base_url"] == "https://custom.meta.example/v1"

    def test_runtime_provider_base_url(self, monkeypatch):
        monkeypatch.setenv("MODEL_API_KEY", "meta-key")
        from hermes_cli.runtime_provider import resolve_runtime_provider

        result = resolve_runtime_provider(requested="meta-ai")
        assert result["provider"] == "meta-ai"
        assert result["api_mode"] == "chat_completions"
        assert result["base_url"] == "https://api.meta.ai/v1"


class TestMetaAiModelMetadata:
    def test_url_to_provider(self):
        from agent.model_metadata import _URL_TO_PROVIDER

        assert _URL_TO_PROVIDER.get("api.meta.ai") == "meta-ai"

    def test_infer_from_url(self):
        from agent.model_metadata import _infer_provider_from_url

        assert _infer_provider_from_url("https://api.meta.ai/v1") == "meta-ai"

    def test_muse_spark_context_length(self):
        with patch(
            "agent.model_metadata.get_cached_context_length",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_endpoint_model_metadata",
            return_value={},
        ), patch(
            "agent.models_dev.lookup_models_dev_context",
            return_value=None,
        ), patch(
            "agent.model_metadata.fetch_model_metadata",
            return_value={},
        ):
            result = get_model_context_length(
                "muse-spark-1.1",
                base_url="https://api.meta.ai/v1",
                api_key="meta-test-key",
                provider="meta-ai",
            )

        assert result == 1_048_576
