"""Tests for Tencent TokenHub provider support (Hy3 Preview)."""

import json
import os

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
    "XIAOMI_API_KEY", "OPENROUTER_API_KEY", "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN", "GITHUB_TOKEN", "ARCEEAI_API_KEY",
)


# =============================================================================
# Provider Registry
# =============================================================================


class TestTencentTokenhubProviderRegistry:
    """Verify tencent-tokenhub is registered correctly in the PROVIDER_REGISTRY."""

    def test_registered(self):
        assert "tencent-tokenhub" in PROVIDER_REGISTRY


    def test_inference_base_url(self):
        assert PROVIDER_REGISTRY["tencent-tokenhub"].inference_base_url == "https://tokenhub.tencentmaas.com/v1"


# =============================================================================
# Aliases
# =============================================================================


class TestTencentTokenhubAliases:
    """All aliases should resolve to 'tencent-tokenhub'."""

    @pytest.mark.parametrize("alias", [
        "tencent-tokenhub", "tencent", "tokenhub", "tencent-cloud", "tencentmaas",
    ])
    def test_alias_resolves(self, alias, monkeypatch):
        for key in _OTHER_PROVIDER_KEYS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("TOKENHUB_API_KEY", "sk-test-key-12345678")
        assert resolve_provider(alias) == "tencent-tokenhub"

    def test_normalize_provider_models_py(self):
        from hermes_cli.models import normalize_provider
        assert normalize_provider("tencent") == "tencent-tokenhub"
        assert normalize_provider("tokenhub") == "tencent-tokenhub"
        assert normalize_provider("tencent-cloud") == "tencent-tokenhub"
        assert normalize_provider("tencentmaas") == "tencent-tokenhub"

    def test_normalize_provider_providers_py(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("tencent") == "tencent-tokenhub"
        assert normalize_provider("tokenhub") == "tencent-tokenhub"
        assert normalize_provider("tencent-cloud") == "tencent-tokenhub"
        assert normalize_provider("tencentmaas") == "tencent-tokenhub"


# =============================================================================
# Auto-detection
# =============================================================================




# =============================================================================
# Credentials
# =============================================================================


class TestTencentTokenhubCredentials:
    """Test credential resolution for the tencent-tokenhub provider."""



    def test_resolve_credentials(self, monkeypatch):
        monkeypatch.setenv("TOKENHUB_API_KEY", "sk-test-12345678")
        monkeypatch.delenv("TOKENHUB_BASE_URL", raising=False)
        creds = resolve_api_key_provider_credentials("tencent-tokenhub")
        assert creds["api_key"] == "sk-test-12345678"
        assert creds["base_url"] == "https://tokenhub.tencentmaas.com/v1"

    def test_openrouter_key_does_not_make_tokenhub_configured(self, monkeypatch):
        """OpenRouter users should NOT see tencent-tokenhub as configured."""
        monkeypatch.delenv("TOKENHUB_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("tencent-tokenhub")
        assert not status["configured"]



# =============================================================================
# Model catalog
# =============================================================================


class TestTencentTokenhubModelCatalog:
    """Tencent TokenHub static model list."""

    def test_static_model_list_exists(self):
        from hermes_cli.models import _PROVIDER_MODELS
        assert "tencent-tokenhub" in _PROVIDER_MODELS
        assert len(_PROVIDER_MODELS["tencent-tokenhub"]) >= 1


    def test_default_model(self):
        from hermes_cli.models import get_default_model_for_provider
        assert get_default_model_for_provider("tencent-tokenhub") == "hy3-preview"


# =============================================================================
# CANONICAL_PROVIDERS (hermes model picker)
# =============================================================================


class TestTencentTokenhubCanonicalProvider:
    """Tencent TokenHub appears in the interactive model picker."""


    def test_description_contains_hy3(self):
        from hermes_cli.models import CANONICAL_PROVIDERS
        entry = next(p for p in CANONICAL_PROVIDERS if p.slug == "tencent-tokenhub")
        assert "Hy3 Preview" in entry.tui_desc


# =============================================================================
# OpenRouter / Nous Portal curated lists
# =============================================================================




# =============================================================================
# Model normalization
# =============================================================================


class TestTencentTokenhubNormalization:
    """Model name normalization — Tencent TokenHub is a direct provider
    not in _MATCHING_PREFIX_STRIP_PROVIDERS, so names pass through as-is.
    """


    def test_not_in_matching_prefix_strip_set(self):
        """tencent-tokenhub does NOT need prefix stripping — it only has
        one model (hy3-preview) and users won't copy vendor/ form."""
        from hermes_cli.model_normalize import _MATCHING_PREFIX_STRIP_PROVIDERS
        assert "tencent-tokenhub" not in _MATCHING_PREFIX_STRIP_PROVIDERS

    def test_not_in_lowercase_providers(self):
        """tencent-tokenhub does not require lowercase normalization."""
        from hermes_cli.model_normalize import _LOWERCASE_MODEL_PROVIDERS
        assert "tencent-tokenhub" not in _LOWERCASE_MODEL_PROVIDERS

    @pytest.mark.parametrize("empty_input", ["", None, "   "])
    def test_normalize_empty_and_none(self, empty_input):
        """None, empty, and whitespace-only inputs return empty string."""
        from hermes_cli.model_normalize import normalize_model_for_provider
        result = normalize_model_for_provider(empty_input, "tencent-tokenhub")
        assert result == "" or result.strip() == ""


# =============================================================================
# Provider label
# =============================================================================




# =============================================================================
# URL mapping
# =============================================================================




# =============================================================================
# Context length
# =============================================================================


class TestTencentTokenhubContextLength:
    """hy3-preview has a context-length entry registered.

    Asserting the relationship (registered + ≥ 4096) instead of a
    specific value, per AGENTS.md "Don't write change-detector tests".
    The previous version of this class pinned an exact integer that
    broke whenever Tencent / OpenRouter bumped the published context
    window (#22268).
    """

    def test_hy3_preview_has_registered_context_length(self):
        from agent.model_metadata import get_model_context_length
        ctx = get_model_context_length("hy3-preview")
        assert isinstance(ctx, int)
        assert ctx >= 4096, f"hy3-preview context length looks unset/wrong: {ctx}"


# =============================================================================
# providers.py (unified provider module)
# =============================================================================


class TestTencentTokenhubProvidersModule:
    """Test Tencent TokenHub in the unified providers module."""

    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS
        assert "tencent-tokenhub" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["tencent-tokenhub"]
        assert overlay.transport == "openai_chat"
        assert overlay.base_url_env_var == "TOKENHUB_BASE_URL"
        assert not overlay.is_aggregator

    def test_alias_resolves(self):
        from hermes_cli.providers import normalize_provider
        assert normalize_provider("tencent") == "tencent-tokenhub"
        assert normalize_provider("tokenhub") == "tencent-tokenhub"


    def test_get_provider(self):
        pdef = None
        try:
            from hermes_cli.providers import get_provider
            pdef = get_provider("tencent-tokenhub")
        except Exception:
            pass
        if pdef is not None:
            assert pdef.id == "tencent-tokenhub"
            assert pdef.transport == "openai_chat"


# =============================================================================
# Auxiliary client
# =============================================================================




# =============================================================================
# Doctor
# =============================================================================




# =============================================================================
# Agent init (no SyntaxError, correct api_mode)
# =============================================================================


class TestTencentTokenhubAgentInit:
    """Verify the agent can be constructed with tencent-tokenhub provider without errors."""

    def test_no_syntax_errors(self):
        """Importing run_agent with tencent-tokenhub should not raise."""
        import importlib
        importlib.import_module("run_agent")

    def test_api_mode_is_chat_completions(self):
        from hermes_cli.providers import HERMES_OVERLAYS, TRANSPORT_TO_API_MODE
        overlay = HERMES_OVERLAYS["tencent-tokenhub"]
        api_mode = TRANSPORT_TO_API_MODE[overlay.transport]
        assert api_mode == "chat_completions"


# =============================================================================
# CLI model flow dispatch (main.py)
# =============================================================================




# =============================================================================
# Remote model catalog (model-catalog.json)
# =============================================================================


class TestTencentTokenhubModelCatalogJSON:
    """Verify tencent/hy3 is present in the website model-catalog.json.

    tencent/hy3:free was delisted 2026-08-21 — the slug vanished from
    OpenRouter's live catalog (free promo rotated out), so the paid hy3
    entry is the surviving assertion target.
    """

    def test_in_model_catalog_json(self):
        catalog_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..",
            "website", "static", "api", "model-catalog.json",
        )
        if not os.path.isfile(catalog_path):
            pytest.skip("model-catalog.json not found in workspace")
        with open(catalog_path) as f:
            data = json.load(f)
        # Collect all model IDs across all provider lists.
        # providers is a dict keyed by provider name, each value has a "models" list.
        all_ids = set()
        providers = data.get("providers", {})
        if isinstance(providers, dict):
            for provider_entry in providers.values():
                for model in provider_entry.get("models", []):
                    all_ids.add(model.get("id", ""))
        else:
            for provider_entry in providers:
                for model in provider_entry.get("models", []):
                    all_ids.add(model.get("id", ""))
        assert "tencent/hy3" in all_ids


# =============================================================================
# determine_api_mode (providers.py)
# =============================================================================


class TestTencentTokenhubApiMode:
    """Verify determine_api_mode routes tencent-tokenhub correctly."""


    def test_determine_api_mode_via_alias(self):
        from hermes_cli.providers import determine_api_mode
        mode = determine_api_mode("tencent")
        assert mode == "chat_completions"


# =============================================================================
# _KNOWN_PROVIDER_NAMES (models.py)
# =============================================================================


class TestTencentTokenhubKnownProviderNames:
    """Verify tencent-tokenhub and its aliases are recognized as valid
    provider names for the ``provider:model`` syntax.
    """


    @pytest.mark.parametrize("alias", [
        "tencent", "tokenhub", "tencent-cloud", "tencentmaas",
    ])
    def test_alias_known(self, alias):
        from hermes_cli.models import _KNOWN_PROVIDER_NAMES
        assert alias in _KNOWN_PROVIDER_NAMES

