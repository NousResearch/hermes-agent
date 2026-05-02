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
