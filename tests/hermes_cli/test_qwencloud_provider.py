"""Tests for QwenCloud Token Plan provider support.

QwenCloud (home.qwencloud.com) offers a Token Plan subscription ($6/mo,
2500 credits/week) with an OpenAI-compatible API at a distinct endpoint
(token-plan.ap-southeast-1.maas.aliyuncs.com) that differs from the
existing `alibaba` (DashScope) and `alibaba-coding-plan` providers.
"""

import pytest

from hermes_cli.auth import (
    PROVIDER_REGISTRY,
    resolve_provider,
    get_api_key_provider_status,
    resolve_api_key_provider_credentials,
)


# Other provider env vars to clear during auto-detection tests
_OTHER_PROVIDER_KEYS = (
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "DASHSCOPE_API_KEY",
    "XAI_API_KEY", "KIMI_API_KEY", "KIMI_CN_API_KEY",
    "MINIMAX_API_KEY", "MINIMAX_CN_API_KEY", "AI_GATEWAY_API_KEY",
    "KILOCODE_API_KEY", "HF_TOKEN", "GLM_API_KEY", "ZAI_API_KEY",
    "XIAOMI_API_KEY", "TOKENHUB_API_KEY", "OPENROUTER_API_KEY",
    "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN", "ARCEEAI_API_KEY",
)


QWENCLOUD_BASE_URL = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


# =============================================================================
# Provider Registry
# =============================================================================


class TestQwencloudProviderRegistry:
    """Verify qwencloud is registered correctly in the PROVIDER_REGISTRY."""

    def test_registered(self):
        assert "qwencloud" in PROVIDER_REGISTRY

    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["qwencloud"].inference_base_url == QWENCLOUD_BASE_URL

    def test_not_custom(self, monkeypatch):
        """qwencloud must be a first-class provider, not the custom workaround."""
        for key in _OTHER_PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("QWENCLOUD_API_KEY", "sk-sp-test")
        assert resolve_provider("qwencloud") == "qwencloud"


# =============================================================================
# Aliases
# =============================================================================


class TestQwencloudAliases:
    """All aliases should resolve to 'qwencloud'."""

    @pytest.mark.parametrize("alias", [
        "qwencloud", "qwencloud-token-plan", "qwencloud_token_plan", "qwen-cloud",
    ])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("QWENCLOUD_API_KEY", "sk-sp-test")
        assert resolve_provider(alias) == "qwencloud"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("qwencloud-token-plan") == "qwencloud"
        assert normalize_provider("qwencloud_token_plan") == "qwencloud"
        assert normalize_provider("qwen-cloud") == "qwencloud"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("qwencloud-token-plan") == "qwencloud"
        assert normalize_provider("qwencloud_token_plan") == "qwencloud"
        assert normalize_provider("qwen-cloud") == "qwencloud"


# =============================================================================
# Credentials
# =============================================================================


class TestQwencloudCredentials:
    """Test credential resolution for the qwencloud provider."""

    def test_resolve_credentials_with_default_base_url(self, monkeypatch):
        monkeypatch.setenv("QWENCLOUD_API_KEY", "sk-sp-test")
        monkeypatch.delenv("QWENCLOUD_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("qwencloud")
        assert creds["api_key"] == "sk-sp-test"
        assert creds["base_url"] == QWENCLOUD_BASE_URL

    def test_resolve_credentials_with_env_base_url_override(self, monkeypatch):
        monkeypatch.setenv("QWENCLOUD_API_KEY", "sk-sp-test")
        monkeypatch.setenv("QWENCLOUD_BASE_URL", "https://custom.example.com/v1")
        creds = resolve_api_key_provider_credentials("qwencloud")
        assert creds["base_url"] == "https://custom.example.com/v1"

    def test_status_configured(self, monkeypatch):
        monkeypatch.setenv("QWENCLOUD_API_KEY", "sk-sp-test")
        status = get_api_key_provider_status("qwencloud")
        assert status["configured"]
        assert status["base_url"] == QWENCLOUD_BASE_URL

    def test_openrouter_key_does_not_make_qwencloud_configured(self, monkeypatch):
        """OpenRouter users should NOT see qwencloud as configured."""
        monkeypatch.delenv("QWENCLOUD_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("qwencloud")
        assert not status["configured"]


# =============================================================================
# Model catalog
# =============================================================================


class TestQwencloudModelCatalog:
    """QwenCloud Token Plan static model list."""

    def test_static_model_list_exists(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "qwencloud" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["qwencloud"]) >= 1

    def test_default_model_is_in_catalog(self):
        """The non-interactive default must come from the static catalog so a
        missing model never escalates to an unbilled flagship."""
        from hermes_cli.models import get_default_model_for_provider, _PROVIDER_MODELS
        default = get_default_model_for_provider("qwencloud")
        assert default
        assert default in _PROVIDER_MODELS["qwencloud"]


# =============================================================================
# CANONICAL_PROVIDERS (hermes model picker)
# =============================================================================


class TestQwencloudCanonicalProvider:
    """QwenCloud Token Plan appears in the interactive model picker."""

    def test_in_canonical_providers(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        slugs = [p.slug for p in CANONICAL_PROVIDERS]
        assert "qwencloud" in slugs

    def test_folded_into_qwen_group(self):
        from hermes_cli.models import provider_group_for_slug
        assert provider_group_for_slug("qwencloud") == "qwen"


# =============================================================================
# _KNOWN_PROVIDER_NAMES (models.py)
# =============================================================================


class TestQwencloudKnownProviderNames:
    """Verify qwencloud and its aliases are recognized as valid provider
    names for the ``provider:model`` syntax."""

    @pytest.mark.parametrize("alias", [
        "qwencloud", "qwencloud-token-plan", "qwencloud_token_plan", "qwen-cloud",
    ])
    def test_alias_known(self, alias):
        from hermes_cli.models import _KNOWN_PROVIDER_NAMES
        assert alias in _KNOWN_PROVIDER_NAMES


# =============================================================================
# providers.py (unified provider module)
# =============================================================================


class TestQwencloudProvidersModule:
    """Test QwenCloud in the unified providers module."""

    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "qwencloud" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["qwencloud"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "QWENCLOUD_BASE_URL"
        assert overlay.base_url_override == QWENCLOUD_BASE_URL
        assert not overlay.is_aggregator

    def test_api_mode_is_chat_completions(self):
        from hermes_cli.providers import HERMES_OVERLAYS, TRANSPORT_TO_API_MODE
        overlay = HERMES_OVERLAYS["qwencloud"]
        api_mode = TRANSPORT_TO_API_MODE[overlay.transport]
        assert api_mode == "chat_completions"

    def test_get_provider(self):
        from hermes_cli.providers import get_provider
        pdef = get_provider("qwencloud", allow_network=False)
        assert pdef is not None
        assert pdef.id == "qwencloud"
        assert pdef.transport == "openai_chat"
        assert pdef.base_url == QWENCLOUD_BASE_URL
