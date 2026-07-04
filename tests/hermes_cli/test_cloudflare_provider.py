"""Focused tests for Cloudflare Workers AI provider wiring."""

from __future__ import annotations

import sys
import types

import pytest

if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    setattr(fake_dotenv, "load_dotenv", lambda *args, **kwargs: None)
    sys.modules["dotenv"] = fake_dotenv

from agent.cloudflare_workers_ai import (
    cloudflare_ai_models_search_url,
    cloudflare_model_names,
    model_capabilities_from_cloudflare_catalog_entry,
)
from agent.models_dev import ModelCapabilities, get_model_capabilities
from hermes_cli.auth import resolve_api_key_provider_credentials, resolve_provider
from hermes_cli.models import normalize_provider, provider_model_ids
from providers import get_provider_profile


@pytest.fixture(autouse=True)
def _clear_cloudflare_env(monkeypatch):
    for key in (
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_BASE_URL",
        "CLOUDFLARE_ACCOUNT_ID",
    ):
        monkeypatch.delenv(key, raising=False)


class TestCloudflareAliases:
    @pytest.mark.parametrize("alias", ["cloudflare", "cloudflare-workers-ai", "workers-ai", "workersai"])
    def test_auth_alias_resolves(self, alias, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-test-key")
        monkeypatch.setenv(
            "CLOUDFLARE_BASE_URL",
            "https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1",
        )
        assert resolve_provider(alias) == "cloudflare"

    def test_models_normalize_provider(self):
        assert normalize_provider("workers-ai") == "cloudflare"
        assert normalize_provider("workersai") == "cloudflare"

    def test_profile_aliases_resolve(self):
        profile = get_provider_profile("custom:cloudflare")
        assert profile is not None
        assert profile.name == "cloudflare"

        derived = get_provider_profile("custom:cloudflare-workers-ai")
        assert derived is not None
        assert derived.name == "cloudflare"


class TestCloudflareCatalogHelpers:
    def test_search_url_translation(self):
        assert cloudflare_ai_models_search_url(
            "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"
        ) == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/models/search"

    def test_catalog_parsing_and_capabilities(self):
        payload = {
            "result": [
                {
                    "name": "@cf/zai-org/glm-5.2",
                    "task": {"name": "Text Generation"},
                    "properties": [
                        {"property_id": "context_window", "value": "262144"},
                        {"property_id": "function_calling", "value": "true"},
                        {"property_id": "reasoning", "value": "true"},
                        {"property_id": "vision", "value": "true"},
                    ],
                },
                {
                    "name": "@cf/baai/bge-m3",
                    "task": {"name": "Text Embeddings"},
                    "properties": [],
                },
            ]
        }
        assert cloudflare_model_names(payload) == ["@cf/zai-org/glm-5.2"]
        assert cloudflare_model_names(payload, text_generation_only=False) == [
            "@cf/zai-org/glm-5.2",
            "@cf/baai/bge-m3",
        ]
        caps = model_capabilities_from_cloudflare_catalog_entry(payload["result"][0])
        assert caps == ModelCapabilities(
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=True,
            context_window=262144,
            max_output_tokens=8192,
            model_family="zai-org/glm-5.2",
        )


class TestCloudflareCapabilitiesFallback:
    def test_models_dev_falls_back_to_profile_live_caps(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text(
            "custom_providers:\n"
            "  - name: Cloudflare Workers AI\n"
            "    base_url: https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1\n"
            "    api_key: test-token\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "agent.cloudflare_workers_ai.fetch_cloudflare_model_catalog",
            lambda api_key, base_url, timeout=8.0: [
                {
                    "name": "@cf/openai/gpt-oss-120b",
                    "task": {"name": "Text Generation"},
                    "properties": [
                        {"property_id": "context_window", "value": "128000"},
                        {"property_id": "function_calling", "value": "true"},
                        {"property_id": "reasoning", "value": "true"},
                    ],
                },
                {
                    "name": "@cf/baai/bge-m3",
                    "task": {"name": "Text Embeddings"},
                    "properties": [
                        {"property_id": "context_window", "value": "8192"},
                    ],
                },
            ],
        )

        caps = get_model_capabilities("custom:cloudflare-workers-ai", "@cf/openai/gpt-oss-120b")
        assert caps is not None
        assert caps.supports_reasoning is True
        assert caps.supports_tools is True
        assert caps.context_window == 128000

        embedding_caps = get_model_capabilities("custom:cloudflare-workers-ai", "@cf/baai/bge-m3")
        assert embedding_caps is not None
        assert embedding_caps.supports_reasoning is False
        assert embedding_caps.supports_tools is False
        assert embedding_caps.context_window == 8192


class TestCloudflareModelCatalog:
    def test_provider_model_ids_prefers_live_catalog(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": "cf-live-key",
                "base_url": "https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1",
                "source": "CLOUDFLARE_API_KEY",
            },
        )
        monkeypatch.setattr(
            "agent.cloudflare_workers_ai.fetch_cloudflare_model_catalog",
            lambda api_key, base_url, timeout=8.0: [
                {"name": "@cf/zai-org/glm-5.2", "task": {"name": "Text Generation"}},
                {"name": "@cf/openai/gpt-oss-120b", "task": {"name": "Text Generation"}},
                {"name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}},
            ],
        )

        assert provider_model_ids("cloudflare") == [
            "@cf/zai-org/glm-5.2",
            "@cf/openai/gpt-oss-120b",
        ]

    def test_custom_provider_model_ids_use_cloudflare_profile_resolution(self, monkeypatch, tmp_path):
        home = tmp_path / ".hermes"
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        (home / "config.yaml").write_text(
            "custom_providers:\n"
            "  - name: Cloudflare Workers AI\n"
            "    base_url: https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1\n"
            "    api_key: test-token\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "agent.cloudflare_workers_ai.fetch_cloudflare_model_catalog",
            lambda api_key, base_url, timeout=8.0: [
                {"name": "@cf/zai-org/glm-5.2", "task": {"name": "Text Generation"}},
                {"name": "@cf/openai/gpt-oss-120b", "task": {"name": "Text Generation"}},
                {"name": "@cf/baai/bge-m3", "task": {"name": "Text Embeddings"}},
            ],
        )

        assert provider_model_ids("custom:cloudflare-workers-ai") == [
            "@cf/zai-org/glm-5.2",
            "@cf/openai/gpt-oss-120b",
        ]


class TestCloudflareCredentials:
    def test_missing_account_id_does_not_return_placeholder_base_url(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_API_KEY", "cf-test-key")
        creds = resolve_api_key_provider_credentials("cloudflare")
        assert creds["api_key"] == "cf-test-key"
        assert creds["base_url"] == ""
