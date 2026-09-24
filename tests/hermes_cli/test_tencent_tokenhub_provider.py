"""Tests for Tencent TokenHub provider support (Hy4 preview)."""


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
        assert creds["base_url"] == PROVIDER_REGISTRY["tencent-tokenhub"].inference_base_url

    def test_openrouter_key_does_not_make_tokenhub_configured(self, monkeypatch):
        """OpenRouter users should NOT see tencent-tokenhub as configured."""
        monkeypatch.delenv("TOKENHUB_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        status = get_api_key_provider_status("tencent-tokenhub")
        assert not status["configured"]



# =============================================================================
# Model catalog
# =============================================================================




# =============================================================================
# CANONICAL_PROVIDERS (hermes model picker)
# =============================================================================




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




# =============================================================================
# Auxiliary client
# =============================================================================




# =============================================================================
# Doctor
# =============================================================================




# =============================================================================
# Agent init (no SyntaxError, correct api_mode)
# =============================================================================




# =============================================================================
# CLI model flow dispatch (main.py)
# =============================================================================




# =============================================================================
# Remote model catalog (model-catalog.json)
# =============================================================================




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



