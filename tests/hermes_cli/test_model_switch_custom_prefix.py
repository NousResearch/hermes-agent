"""Regression tests for /model auto-detect not hijacking custom:* providers.

When current_provider starts with "custom:" (e.g. "custom:custom",
"custom:neuralwatt"), detect_provider_for_model() must NOT be called —
these are user-defined endpoints that should never be auto-redirected to
OpenRouter or another catalog provider.

Regression for #16259: is_custom guard only checked for bare "custom" and
"local", so "custom:custom" was treated as non-custom, allowing auto-detect
to silently switch to openrouter when it recognized the model slug.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}

_CUSTOM_PROVIDERS = [
    {
        "name": "My Nous Portal",
        "slug": "custom:custom",
        "base_url": "https://portal.nousresearch.com/v1",
        "api_key": "np-test-key",
        "model": "nvidia/nemotron-3-super-120b-a12b",
    }
]


def _run_switch(raw_input, current_provider, current_base_url="https://portal.nousresearch.com/v1",
                current_model="nvidia/nemotron-3-super-120b-a12b",
                detect_return=None, custom_providers=None):
    if custom_providers is None:
        custom_providers = _CUSTOM_PROVIDERS
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch("hermes_cli.model_switch.list_provider_models", return_value=[]),
        patch("hermes_cli.model_switch.is_aggregator", return_value=False),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "np-test-key",
                "base_url": current_base_url,
                "api_mode": "chat_completions",
                "provider": current_provider,
            },
        ),
        patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch(
            "hermes_cli.models.detect_provider_for_model",
            return_value=detect_return,
        ) as mock_detect,
    ):
        result = switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model=current_model,
            current_base_url=current_base_url,
            current_api_key="np-test-key",
            custom_providers=custom_providers,
        )
        return result, mock_detect


class TestCustomPrefixIsCustomGuard:

    def test_custom_colon_custom_does_not_call_detect(self):
        """custom:custom provider must not invoke detect_provider_for_model."""
        result, mock_detect = _run_switch(
            raw_input="nvidia/nemotron-super-2",
            current_provider="custom:custom",
        )
        mock_detect.assert_not_called()

    def test_custom_colon_named_does_not_call_detect(self):
        """custom:neuralwatt (any named custom: prefix) must not invoke detect."""
        result, mock_detect = _run_switch(
            raw_input="hermes-3-70b",
            current_provider="custom:neuralwatt",
            current_base_url="https://api.neuralwatt.ai/v1",
            custom_providers=[
                {
                    "name": "NeuralWatt",
                    "slug": "custom:neuralwatt",
                    "base_url": "https://api.neuralwatt.ai/v1",
                    "api_key": "nw-key",
                    "model": "hermes-3-70b",
                }
            ],
        )
        mock_detect.assert_not_called()

    def test_bare_custom_still_does_not_call_detect(self):
        """Bare 'custom' provider remains protected (pre-existing behavior)."""
        result, mock_detect = _run_switch(
            raw_input="some-model",
            current_provider="custom",
            current_base_url="https://custom.example.com/v1",
        )
        mock_detect.assert_not_called()

    def test_non_custom_provider_does_call_detect(self):
        """Non-custom providers like openrouter still trigger detect (unchanged behavior)."""
        _result, mock_detect = _run_switch(
            raw_input="nvidia/nemotron-super-2",
            current_provider="openrouter",
            current_base_url="",
            detect_return=None,  # detect returns None → provider unchanged
        )
        mock_detect.assert_called_once()

    def test_detect_cannot_redirect_custom_colon_to_openrouter(self):
        """Even if detect_provider_for_model would return openrouter, it must
        not be called for custom:* providers — provider must stay unchanged."""
        result, mock_detect = _run_switch(
            raw_input="nvidia/nemotron-3-super-120b-a12b",
            current_provider="custom:custom",
            detect_return=("openrouter", "nvidia/nemotron-3-super-120b-a12b"),
        )
        mock_detect.assert_not_called()
        assert result.target_provider == "custom:custom"
