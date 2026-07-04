"""Tests for Cloudflare Workers AI provider integration.

Covers:
- Provider registry and aliases (from #16398 + #10386)
- Base URL resolution with CLOUDFLARE_ACCOUNT_ID templating (from #10386)
- Auto-detect hijack guard (from #16398)
- Credential pool safeguards (from #10386)
- Dynamic catalog discovery, caching, and filtering (from our PR)
- Capability metadata extraction (from our PR)
- Name normalization: meta-llama → meta (from #53105)
- Custom-provider slug resolution (from our PR)
"""

import json
import os
import sys
import types

import pytest

# Ensure dotenv doesn't interfere
if "dotenv" not in sys.modules:
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = fake_dotenv

from agent.cloudflare_workers_ai import (
    cloudflare_ai_models_search_url,
    cloudflare_model_names,
    fetch_cloudflare_model_catalog,
    is_cloudflare_workers_ai_base_url,
    model_capabilities_from_cloudflare_catalog_entry,
    parse_cloudflare_model_search_response,
)
from hermes_cli.auth import (
    CLOUDFLARE_WORKERS_AI_BASE_URL_TEMPLATE,
    PROVIDER_REGISTRY,
    AuthError,
    _cloudflare_base_url_is_resolved,
    _cloudflare_has_endpoint_config,
    _resolve_cloudflare_base_url,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
    resolve_provider,
)
from hermes_cli.cloudflare import (
    is_cloudflare_provider_name,
    resolve_cloudflare_runtime_credentials,
)


# =============================================================================
# Env var cleanup helpers
# =============================================================================

_CF_KEYS = (
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_API_KEY",
    "CLOUDFLARE_ACCOUNT_ID",
    "CLOUDFLARE_BASE_URL",
    "CLOUDFLARE_WORKERS_AI_BASE_URL",
    "CLOUDFLARE_GATEWAY_ID",
)

_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "DASHSCOPE_API_KEY", "XAI_API_KEY", "KIMI_API_KEY",
    "MINIMAX_API_KEY", "AI_GATEWAY_API_KEY", "KILOCODE_API_KEY",
    "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "ARCEEAI_API_KEY", "NVIDIA_API_KEY",
    "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
)


def _clear_cf_env(monkeypatch):
    for k in _CF_KEYS:
        monkeypatch.delenv(k, raising=False)


def _clear_other_provider_env(monkeypatch):
    for k in _OTHER_PROVIDER_KEYS:
        monkeypatch.delenv(k, raising=False)


def _clear_all_provider_keys(monkeypatch):
    _clear_cf_env(monkeypatch)
    _clear_other_provider_env(monkeypatch)


_CF_DEFAULT = CLOUDFLARE_WORKERS_AI_BASE_URL_TEMPLATE


# =============================================================================
# Provider Registry
# =============================================================================


class TestCloudflareProviderRegistry:
    """Verify Cloudflare is registered correctly."""

    def test_registered(self):
        assert "cloudflare" in PROVIDER_REGISTRY

    def test_name(self):
        assert PROVIDER_REGISTRY["cloudflare"].name == "Cloudflare Workers AI"

    def test_auth_type(self):
        assert PROVIDER_REGISTRY["cloudflare"].auth_type == "api_key"

    def test_inference_base_url_is_template(self):
        url = PROVIDER_REGISTRY["cloudflare"].inference_base_url
        assert "{account_id}" in url or "ACCOUNT_ID" in url

    def test_api_key_env_var(self):
        """CLOUDFLARE_API_TOKEN is the primary env var."""
        env_vars = PROVIDER_REGISTRY["cloudflare"].api_key_env_vars
        assert "CLOUDFLARE_API_TOKEN" in env_vars


# =============================================================================
# Base URL Resolution
# =============================================================================


class TestCloudflareBaseUrlResolution:
    """Verify _resolve_cloudflare_base_url substitutes CLOUDFLARE_ACCOUNT_ID."""

    def test_substitutes_account_id(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "abc123")
        url = _resolve_cloudflare_base_url()
        assert url == "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "abc123")
        monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://custom.example/v1")
        url = _resolve_cloudflare_base_url("https://custom.example/v1")
        assert url == "https://custom.example/v1"

    def test_missing_account_id_returns_empty(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
        url = _resolve_cloudflare_base_url()
        assert url == ""

    def test_gateway_url_when_gateway_id_set(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "abc123")
        monkeypatch.setenv("CLOUDFLARE_GATEWAY_ID", "my-gw")
        url = _resolve_cloudflare_base_url()
        assert "gateway.ai.cloudflare.com" in url
        assert "abc123" in url
        assert "my-gw" in url

    def test_base_url_is_resolved_true(self):
        assert _cloudflare_base_url_is_resolved("https://api.cloudflare.com/client/v4/accounts/abc/ai/v1")

    def test_base_url_is_resolved_false_for_placeholder(self):
        assert not _cloudflare_base_url_is_resolved("https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1")
        assert not _cloudflare_base_url_is_resolved("https://api.cloudflare.com/client/v4/accounts/ACCOUNT_ID/ai/v1")

    def test_base_url_is_resolved_false_for_empty(self):
        assert not _cloudflare_base_url_is_resolved("")

    def test_has_endpoint_config_with_account_id(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "abc123")
        assert _cloudflare_has_endpoint_config()

    def test_has_endpoint_config_with_base_url(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_BASE_URL", "https://custom.example/v1")
        assert _cloudflare_has_endpoint_config()

    def test_has_endpoint_config_false(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        assert not _cloudflare_has_endpoint_config()


# =============================================================================
# Aliases
# =============================================================================


class TestCloudflareAliases:
    """Accepted aliases should resolve to 'cloudflare'."""

    @pytest.mark.parametrize("alias", [
        "cloudflare",
        "cloudflare-workers-ai",
        "workers-ai",
        "workersai",
        "cf",
        "cf-ai",
    ])
    def test_alias_resolves(self, alias, monkeypatch):
        _clear_all_provider_keys(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        assert resolve_provider(alias) == "cloudflare"


# =============================================================================
# Auto-detect hijack guard
# =============================================================================


class TestCloudflareAutoDetectGuard:
    """A bare CLOUDFLARE_API_TOKEN must not auto-route to cloudflare
    unless CLOUDFLARE_ACCOUNT_ID is also set."""

    def test_token_alone_does_not_auto_route(self, monkeypatch):
        _clear_all_provider_keys(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        with pytest.raises(AuthError):
            resolve_provider("auto")

    def test_token_plus_account_id_auto_routes(self, monkeypatch):
        _clear_all_provider_keys(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        assert resolve_provider("auto") == "cloudflare"


# =============================================================================
# Provider Status & Credentials
# =============================================================================


class TestCloudflareProviderStatus:
    """get_api_key_provider_status should report configured correctly."""

    def test_configured_with_token_and_account(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        status = get_api_key_provider_status("cloudflare")
        assert status["configured"] is True
        assert "test-account" in status["base_url"]

    def test_not_configured_without_account_id(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        status = get_api_key_provider_status("cloudflare")
        assert status["configured"] is False

    def test_not_configured_without_token(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        status = get_api_key_provider_status("cloudflare")
        assert status["configured"] is False


class TestCloudflareCredentialsResolution:
    """resolve_api_key_provider_credentials should resolve or raise."""

    def test_resolves_with_token_and_account(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
        creds = resolve_api_key_provider_credentials("cloudflare")
        assert creds["api_key"] == "cf-test-token-12345678"
        assert "test-account" in creds["base_url"]

    def test_raises_without_account_id(self, monkeypatch):
        _clear_cf_env(monkeypatch)
        monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "cf-test-token-12345678")
        with pytest.raises(AuthError, match="CLOUDFLARE_ACCOUNT_ID"):
            resolve_api_key_provider_credentials("cloudflare")


# =============================================================================
# Catalog Helpers (our PR)
# =============================================================================


class TestCloudflareCatalogHelpers:
    """URL detection and catalog endpoint mapping."""

    def test_is_cloudflare_base_url(self):
        assert is_cloudflare_workers_ai_base_url(
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"
        )

    def test_is_not_cloudflare_base_url(self):
        assert not is_cloudflare_workers_ai_base_url("https://api.openai.com/v1")
        assert not is_cloudflare_workers_ai_base_url("")

    def test_search_url_mapping(self):
        search = cloudflare_ai_models_search_url(
            "https://api.cloudflare.com/client/v4/accounts/abc/ai/v1"
        )
        assert search == "https://api.cloudflare.com/client/v4/accounts/abc/ai/models/search"

    def test_search_url_none_for_non_cloudflare(self):
        assert cloudflare_ai_models_search_url("https://api.openai.com/v1") is None


class TestCloudflareCatalogParsing:
    """Parse Cloudflare catalog responses correctly."""

    _SAMPLE_PAYLOAD = {
        "result": [
            {
                "name": "@cf/meta/llama-4-scout-17b",
                "task": {"name": "Text Generation"},
                "properties": [
                    {"property_id": "context_window", "value": 128000},
                    {"property_id": "function_calling", "value": True},
                    {"property_id": "reasoning", "value": False},
                    {"property_id": "vision", "value": False},
                ],
            },
            {
                "name": "@cf/meta-llama/llama-3.1-8b-instruct",
                "task": {"name": "Text Generation"},
                "properties": [],
            },
            {
                "name": "@cf/baai/bge-large-en-v1.5",
                "task": {"name": "Text Embeddings"},
                "properties": [],
            },
            {
                "name": "@cf/openai/whisper-large-v3",
                "task": {"name": "Automatic Speech Recognition"},
                "properties": [],
            },
        ]
    }

    def test_parse_response(self):
        entries = parse_cloudflare_model_search_response(self._SAMPLE_PAYLOAD)
        assert len(entries) == 4

    def test_text_generation_only(self):
        models = cloudflare_model_names(self._SAMPLE_PAYLOAD, text_generation_only=True)
        assert len(models) == 2
        assert "@cf/meta/llama-4-scout-17b" in models
        # Embeddings and ASR filtered out
        assert not any("bge" in m for m in models)
        assert not any("whisper" in m for m in models)

    def test_all_tasks(self):
        models = cloudflare_model_names(self._SAMPLE_PAYLOAD, text_generation_only=False)
        assert len(models) == 4

    def test_name_normalization_meta_llama(self):
        """meta-llama in catalog should be normalized to meta for inference."""
        models = cloudflare_model_names(self._SAMPLE_PAYLOAD, text_generation_only=True)
        assert "@cf/meta/llama-3.1-8b-instruct" in models
        assert not any("meta-llama" in m for m in models)

    def test_empty_payload(self):
        assert cloudflare_model_names({"result": []}) == []
        assert cloudflare_model_names({}) == []
        assert cloudflare_model_names(None) == []


class TestCloudflareCapabilityExtraction:
    """Convert catalog entries to ModelCapabilities."""

    def test_capability_extraction(self):
        entry = {
            "name": "@cf/meta/llama-4-scout-17b",
            "task": {"name": "Text Generation"},
            "properties": [
                {"property_id": "context_window", "value": 128000},
                {"property_id": "max_output_tokens", "value": 4096},
                {"property_id": "function_calling", "value": True},
                {"property_id": "reasoning", "value": False},
                {"property_id": "vision", "value": False},
            ],
        }
        caps = model_capabilities_from_cloudflare_catalog_entry(entry)
        assert caps.context_window == 128000
        assert caps.max_output_tokens == 4096
        assert caps.supports_tools is True
        assert caps.supports_reasoning is False
        assert caps.supports_vision is False

    def test_capability_defaults(self):
        """Missing properties should fall back to sensible defaults."""
        entry = {
            "name": "@cf/test/model",
            "task": {"name": "Text Generation"},
            "properties": [],
        }
        caps = model_capabilities_from_cloudflare_catalog_entry(entry)
        assert caps.context_window == 200000
        assert caps.max_output_tokens == 8192
        assert caps.supports_tools is False

    def test_capability_vision(self):
        entry = {
            "name": "@cf/test/vision-model",
            "task": {"name": "Text Generation"},
            "properties": [
                {"property_id": "vision", "value": True},
                {"property_id": "function_calling", "value": True},
            ],
        }
        caps = model_capabilities_from_cloudflare_catalog_entry(entry)
        assert caps.supports_vision is True
        assert caps.supports_tools is True


# =============================================================================
# Custom-provider slug resolution (our PR)
# =============================================================================


class TestCloudflareSlugResolution:
    """is_cloudflare_provider_name should recognize all variants."""

    @pytest.mark.parametrize("name", [
        "cloudflare",
        "cloudflare-workers-ai",
        "workers-ai",
        "workersai",
        "cf",
        "cf-ai",
        "custom:cloudflare",
        "custom:cloudflare-workers-ai",
        "custom:workers-ai",
        "custom:workersai",
    ])
    def test_recognizes_provider_name(self, name):
        assert is_cloudflare_provider_name(name)

    @pytest.mark.parametrize("name", [
        "openai",
        "anthropic",
        "custom:openai",
        "",
        "random-provider",
    ])
    def test_not_cloudflare(self, name):
        assert not is_cloudflare_provider_name(name)


# =============================================================================
# Model metadata integration
# =============================================================================


class TestCloudflareModelMetadata:
    """Cloudflare models appear in curated list and context length lookups."""

    def test_curated_models_exist(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "cloudflare" in _PROVIDER_MODELS
        models = _PROVIDER_MODELS["cloudflare"]
        assert len(models) > 0
        assert all(m.startswith("@cf/") or m.startswith("@hf/") for m in models)

    def test_url_to_provider_mapping(self):
        from agent.model_metadata import _URL_TO_PROVIDER
        assert _URL_TO_PROVIDER.get("api.cloudflare.com") == "cloudflare"
        assert _URL_TO_PROVIDER.get("gateway.ai.cloudflare.com") == "cloudflare"

    def test_provider_prefixes_include_cloudflare(self):
        from agent.model_metadata import _PROVIDER_PREFIXES
        assert "cloudflare" in _PROVIDER_PREFIXES
        assert "workers-ai" in _PROVIDER_PREFIXES


# =============================================================================
# Provider overlay integration
# =============================================================================


class TestCloudflareProviderOverlay:
    """Cloudflare appears in overlays and label overrides."""

    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "cloudflare" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["cloudflare"]
        assert overlay.transport == "openai_chat"

    def test_label_override(self):
        from hermes_cli.providers import _LABEL_OVERRIDES
        assert _LABEL_OVERRIDES.get("cloudflare") == "Cloudflare Workers AI"

    def test_canonical_providers_entry(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "cloudflare" in slugs
