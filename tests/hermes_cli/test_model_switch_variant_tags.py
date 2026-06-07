"""Tests for OpenRouter variant tag preservation in model switching.

Regression test for GitHub PR #6088 / Discord report: OpenRouter model IDs
with variant suffixes like ``:free``, ``:extended``, ``:fast`` were being
mangled by the colon-to-slash conversion in model_switch.py Step c.

The fix: Step c now skips colon→slash conversion when the model name already
contains a forward slash (i.e. is already in ``vendor/model`` format), since
the colon is a variant tag, not a vendor separator.
"""
import pytest
from unittest.mock import patch

from hermes_cli.model_switch import switch_model


# Shared mock context — skip network calls, credential resolution, catalog lookups
_MOCK_VALIDATION = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run_switch(raw_input: str, current_provider: str = "openrouter") -> str:
    """Run switch_model with mocked dependencies, return the resolved model name."""
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "test", "base_url": "", "api_mode": "chat_completions"}), \
         patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None):
        result = switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="anthropic/claude-sonnet-4.6",
        )
        assert result.success, f"switch_model failed: {result.error_message}"
        return result.new_model


class TestVariantTagPreservation:
    """OpenRouter variant tags (:free, :extended, :fast) must survive model switching."""

    @pytest.mark.parametrize("model,expected", [
        ("nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-super-120b-a12b:free"),
        ("anthropic/claude-sonnet-4.6:extended", "anthropic/claude-sonnet-4.6:extended"),
        ("meta-llama/llama-4-maverick:fast", "meta-llama/llama-4-maverick:fast"),
    ])
    def test_slash_format_preserves_variant_tag(self, model, expected):
        """Models already in vendor/model:tag format must not have their tag mangled."""
        assert _run_switch(model) == expected

    def test_legacy_colon_format_converts_to_slash(self):
        """Legacy vendor:model (no slash) should still be converted to vendor/model."""
        result = _run_switch("nvidia:nemotron-3-super-120b-a12b")
        assert result == "nvidia/nemotron-3-super-120b-a12b"

    def test_legacy_colon_format_with_tag_converts_first_colon_only(self):
        """vendor:model:free (no slash) → vendor/model:free — first colon becomes slash."""
        result = _run_switch("nvidia:nemotron-3-super-120b-a12b:free")
        assert result == "nvidia/nemotron-3-super-120b-a12b:free"

    def test_bare_model_name_unaffected(self):
        """Bare model names without colons or slashes should work normally."""
        result = _run_switch("claude-sonnet-4.6")
        assert result == "anthropic/claude-sonnet-4.6"

    def test_already_correct_slug_no_tag(self):
        """Standard vendor/model slugs without tags pass through unchanged."""
        result = _run_switch("anthropic/claude-sonnet-4.6")
        assert result == "anthropic/claude-sonnet-4.6"


class TestColonProviderModelSyntax:
    """Colon-based provider:model syntax should work from non-aggregator providers.

    Regression test for #40852: users typing ``/model xai-oauth:grok-4.3``
    on a non-aggregator provider (e.g. deepseek) had the colon ignored,
    causing the full string to be treated as a model name and validated
    against the current provider's catalog.
    """

    def test_colon_syntax_switches_provider(self):
        """provider:model on a non-aggregator should route to PATH A."""
        from unittest.mock import MagicMock

        mock_pdef = MagicMock()
        mock_pdef.id = "xai-oauth"
        mock_pdef.name = "xAI OAuth"
        mock_pdef.base_url = "https://api.x.ai/v1"

        with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
             patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
             patch("hermes_cli.model_switch.resolve_provider_full", return_value=mock_pdef), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"api_key": "test", "base_url": "https://api.x.ai/v1", "api_mode": "chat_completions"}), \
             patch("hermes_cli.models.validate_requested_model",
                   return_value={"accepted": True, "persist": True, "recognized": True, "message": None}), \
             patch("hermes_cli.model_switch.get_model_info", return_value=None), \
             patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None):
            result = switch_model(
                raw_input="xai-oauth:grok-4.3",
                current_provider="deepseek",
                current_model="deepseek-v4-pro",
            )
            assert result.success, f"switch_model failed: {result.error_message}"
            assert result.target_provider == "xai-oauth"
            assert result.new_model == "grok-4.3"
            assert result.provider_changed is True

    def test_colon_syntax_unknown_provider_falls_through(self):
        """provider:model with unknown provider should not be parsed as provider switch."""
        with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
             patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
             patch("hermes_cli.model_switch.resolve_provider_full", return_value=None), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"api_key": "test", "base_url": "", "api_mode": "chat_completions"}), \
             patch("hermes_cli.models.validate_requested_model",
                   return_value={"accepted": True, "persist": True, "recognized": True, "message": None}), \
             patch("hermes_cli.model_switch.get_model_info", return_value=None), \
             patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None):
            result = switch_model(
                raw_input="unknown-provider:some-model",
                current_provider="deepseek",
                current_model="deepseek-v4-pro",
            )
            assert result.success, f"switch_model failed: {result.error_message}"
            # Should NOT switch provider — the colon should be left in the model name
            assert result.target_provider == "deepseek"

    def test_colon_syntax_not_applied_on_aggregators(self):
        """Colons on aggregators should follow existing step c behavior (vendor:model -> vendor/model)."""
        with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
             patch("hermes_cli.model_switch.list_provider_models",
                   return_value=["nvidia/nemotron-3-super-120b-a12b"]), \
             patch("hermes_cli.runtime_provider.resolve_runtime_provider",
                   return_value={"api_key": "test", "base_url": "", "api_mode": "chat_completions"}), \
             patch("hermes_cli.models.validate_requested_model",
                   return_value={"accepted": True, "persist": True, "recognized": True, "message": None}), \
             patch("hermes_cli.model_switch.get_model_info", return_value=None), \
             patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
             patch("hermes_cli.models.detect_provider_for_model", return_value=None):
            result = switch_model(
                raw_input="nvidia:nemotron-3-super-120b-a12b",
                current_provider="openrouter",
                current_model="anthropic/claude-sonnet-4.6",
            )
            assert result.success, f"switch_model failed: {result.error_message}"
            # On aggregators, colon is converted to slash (existing behavior)
            assert result.new_model == "nvidia/nemotron-3-super-120b-a12b"
