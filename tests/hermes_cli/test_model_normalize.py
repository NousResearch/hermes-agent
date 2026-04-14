"""Tests for hermes_cli.model_normalize — provider-aware model name normalization.

Covers issue #5211: opencode-go model names with dots (e.g. minimax-m2.7)
must NOT be mangled to hyphens (minimax-m2-7).
"""
import pytest

from hermes_cli.model_normalize import (
    normalize_model_for_provider,
    _DOT_TO_HYPHEN_PROVIDERS,
    _AGGREGATOR_PROVIDERS,
    detect_vendor,
)


# ── Regression: issue #5211 ────────────────────────────────────────────

class TestIssue5211OpenCodeGoDotPreservation:
    """OpenCode Go model names with dots must pass through unchanged."""

    @pytest.mark.parametrize("model,expected", [
        ("minimax-m2.7", "minimax-m2.7"),
        ("minimax-m2.5", "minimax-m2.5"),
        ("glm-4.5", "glm-4.5"),
        ("kimi-k2.5", "kimi-k2.5"),
        ("some-model-1.0.3", "some-model-1.0.3"),
    ])
    def test_opencode_go_preserves_dots(self, model, expected):
        result = normalize_model_for_provider(model, "opencode-go")
        assert result == expected, f"Expected {expected!r}, got {result!r}"

    def test_opencode_go_not_in_dot_to_hyphen_set(self):
        """opencode-go must NOT be in the dot-to-hyphen provider set."""
        assert "opencode-go" not in _DOT_TO_HYPHEN_PROVIDERS


# ── Anthropic dot-to-hyphen conversion (regression) ────────────────────

class TestAnthropicDotToHyphen:
    """Anthropic API still needs dots→hyphens."""

    @pytest.mark.parametrize("model,expected", [
        ("claude-sonnet-4.6", "claude-sonnet-4-6"),
        ("claude-opus-4.5", "claude-opus-4-5"),
    ])
    def test_anthropic_converts_dots(self, model, expected):
        result = normalize_model_for_provider(model, "anthropic")
        assert result == expected

    def test_anthropic_strips_vendor_prefix(self):
        result = normalize_model_for_provider("anthropic/claude-sonnet-4.6", "anthropic")
        assert result == "claude-sonnet-4-6"


# ── OpenCode Zen regression ────────────────────────────────────────────

class TestOpenCodeZenModelNormalization:
    """OpenCode Zen preserves dots for most models, but Claude stays hyphenated."""

    @pytest.mark.parametrize("model,expected", [
        ("claude-sonnet-4.6", "claude-sonnet-4-6"),
        ("opencode-zen/claude-opus-4.5", "claude-opus-4-5"),
        ("glm-4.5", "glm-4.5"),
        ("glm-5.1", "glm-5.1"),
        ("gpt-5.4", "gpt-5.4"),
        ("minimax-m2.5-free", "minimax-m2.5-free"),
        ("kimi-k2.5", "kimi-k2.5"),
    ])
    def test_zen_normalizes_models(self, model, expected):
        result = normalize_model_for_provider(model, "opencode-zen")
        assert result == expected

    def test_zen_strips_vendor_prefix(self):
        result = normalize_model_for_provider("opencode-zen/claude-sonnet-4.6", "opencode-zen")
        assert result == "claude-sonnet-4-6"

    def test_zen_strips_vendor_prefix_for_non_claude(self):
        result = normalize_model_for_provider("opencode-zen/glm-5.1", "opencode-zen")
        assert result == "glm-5.1"


# ── Copilot dot preservation (regression) ──────────────────────────────

class TestCopilotDotPreservation:
    """Copilot preserves dots in model names."""

    @pytest.mark.parametrize("model,expected", [
        ("claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("gpt-5.4", "gpt-5.4"),
    ])
    def test_copilot_preserves_dots(self, model, expected):
        result = normalize_model_for_provider(model, "copilot")
        assert result == expected


# ── Aggregator providers (regression) ──────────────────────────────────

class TestAggregatorProviders:
    """Aggregators need vendor/model slugs."""

    def test_openrouter_prepends_vendor(self):
        result = normalize_model_for_provider("claude-sonnet-4.6", "openrouter")
        assert result == "anthropic/claude-sonnet-4.6"

    def test_nous_prepends_vendor(self):
        result = normalize_model_for_provider("gpt-5.4", "nous")
        assert result == "openai/gpt-5.4"

    def test_vendor_already_present(self):
        result = normalize_model_for_provider("anthropic/claude-sonnet-4.6", "openrouter")
        assert result == "anthropic/claude-sonnet-4.6"


class TestIssue6211NativeProviderPrefixNormalization:
    @pytest.mark.parametrize("model,target_provider,expected", [
        ("zai/glm-5.1", "zai", "glm-5.1"),
        ("google/gemini-2.5-pro", "gemini", "google/gemini-2.5-pro"),
        ("moonshot/kimi-k2.5", "kimi-coding", "kimi-k2.5"),
        ("anthropic/claude-sonnet-4.6", "openrouter", "anthropic/claude-sonnet-4.6"),
        ("Qwen/Qwen3.5-397B-A17B", "huggingface", "Qwen/Qwen3.5-397B-A17B"),
        ("modal/zai-org/GLM-5-FP8", "custom", "modal/zai-org/GLM-5-FP8"),
    ])
    def test_native_provider_prefixes_are_only_stripped_on_matching_provider(
        self, model, target_provider, expected
    ):
        assert normalize_model_for_provider(model, target_provider) == expected


# ── detect_vendor ──────────────────────────────────────────────────────

class TestDetectVendor:
    @pytest.mark.parametrize("model,expected", [
        ("claude-sonnet-4.6", "anthropic"),
        ("gpt-5.4-mini", "openai"),
        ("minimax-m2.7", "minimax"),
        ("glm-4.5", "z-ai"),
        ("kimi-k2.5", "moonshotai"),
    ])
    def test_detects_known_vendors(self, model, expected):
        assert detect_vendor(model) == expected
