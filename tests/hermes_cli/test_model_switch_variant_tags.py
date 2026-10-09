"""Tests for OpenRouter variant tag preservation in model switching.

Regression coverage for colon-bearing model IDs and qualified model refs.

Variant suffixes on slash-qualified model IDs remain part of the model ID.
A leading ``provider:model`` form is parsed as model identity only when the
left side is a known provider; unknown prefixes remain native model IDs.
"""
import pytest
from unittest.mock import patch

from hermes_cli.model_switch import switch_model


# Shared mock context — skip network calls, credential resolution, catalog lookups
_MOCK_VALIDATION = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _run_switch_result(raw_input: str, current_provider: str = "openrouter"):
    """Run switch_model with network/catalog dependencies mocked."""
    with patch("hermes_cli.model_switch.list_provider_models", return_value=[]), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "test", "base_url": "", "api_mode": "chat_completions"}), \
         patch("hermes_cli.models_validate.validate_requested_model", return_value=_MOCK_VALIDATION), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.query_model_metadata", return_value=None):
        result = switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="anthropic/claude-sonnet-4.6",
        )
        assert result.success, f"switch_model failed: {result.error_message}"
        return result


def _run_switch(raw_input: str, current_provider: str = "openrouter") -> str:
    return _run_switch_result(raw_input, current_provider).new_model


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

    def test_known_provider_colon_is_a_qualified_model_ref(self):
        result = _run_switch_result("nvidia:nemotron-3-super-120b-a12b")
        assert result.target_provider == "nvidia"
        assert result.new_model == "nvidia/nemotron-3-super-120b-a12b"




class TestColonFormOffAggregators:
    """Known provider prefixes are identity; unknown prefixes remain model-native."""

    def test_known_provider_colon_selects_that_provider(self):
        result = _run_switch_result("Alibaba:qwen3.6-plus", current_provider="anthropic")
        assert result.target_provider == "alibaba"
        assert result.new_model == "qwen3.6-plus"

    def test_non_provider_left_side_keeps_colon(self):
        assert _run_switch("qwen3.5:4b", current_provider="alibaba") == "qwen3.5:4b"
