"""Regression tests for provider:model syntax in /model command (#40852).

When the current provider is NOT an aggregator (e.g. DeepSeek), typing
/model xai-oauth:grok-4.3 should parse ``xai-oauth`` as a provider slug
and ``grok-4.3`` as the model name — instead of treating the entire string
as a model name on the current provider.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model
from hermes_cli.providers import ProviderDef

_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}

_XAI_PDEF = ProviderDef(
    id="xai-oauth",
    name="xAI",
    transport="codex_responses",
    api_key_env_vars=("XAI_API_KEY",),
    base_url="https://api.x.ai/v1",
    base_url_env_var="",
    is_aggregator=False,
    auth_type="oauth_external",
    doc="",
    source="models.dev",
)


def _resolve_provider_side_effect(slug, user_provs, custom_provs):
    if slug == "xai-oauth":
        return _XAI_PDEF
    return None


class TestProviderColonModelSyntax:
    """provider:model should be parsed when current provider is not an aggregator."""

    def test_switch_to_different_provider_via_colon_syntax(self):
        """xai-oauth:grok-4.3 on deepseek → provider=xai-oauth, model=grok-4.3."""
        with (
            patch(
                "hermes_cli.model_switch.resolve_provider_full",
                side_effect=_resolve_provider_side_effect,
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "test-key",
                    "base_url": "https://api.x.ai/v1",
                    "api_mode": "",
                },
            ),
            patch(
                "hermes_cli.models.validate_requested_model",
                return_value=_MOCK_VALIDATION,
            ),
        ):
            result = switch_model(
                raw_input="xai-oauth:grok-4.3",
                current_provider="deepseek",
                current_model="deepseek-v4-pro",
                current_base_url="https://api.deepseek.com",
                current_api_key="deepseek-key",
            )

        assert result.success is True
        assert result.new_model == "grok-4.3"
        assert result.target_provider == "xai-oauth"

    def test_aggregator_colon_syntax_preserved(self):
        """openai:gpt-4 on openrouter → model=openai/gpt-4 (aggregator slug)."""
        with (
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "or-key",
                    "base_url": "https://openrouter.ai/api/v1",
                    "api_mode": "",
                },
            ),
            patch(
                "hermes_cli.models.validate_requested_model",
                return_value=_MOCK_VALIDATION,
            ),
        ):
            result = switch_model(
                raw_input="openai:gpt-4",
                current_provider="openrouter",
                current_model="some-model",
                current_base_url="https://openrouter.ai/api/v1",
                current_api_key="or-key",
            )

        assert result.success is True
        assert result.new_model == "openai/gpt-4"
        assert result.target_provider == "openrouter"

    def test_unknown_left_part_not_misparsed(self):
        """unknown:variant on deepseek → falls through to normal resolution."""
        with (
            patch(
                "hermes_cli.model_switch.resolve_provider_full",
                return_value=None,
            ),
            patch(
                "hermes_cli.runtime_provider.resolve_runtime_provider",
                return_value={
                    "api_key": "test-key",
                    "base_url": "https://api.deepseek.com",
                    "api_mode": "",
                },
            ),
            patch(
                "hermes_cli.models.validate_requested_model",
                return_value=_MOCK_VALIDATION,
            ),
        ):
            result = switch_model(
                raw_input="unknown-model:variant",
                current_provider="deepseek",
                current_model="deepseek-v4-pro",
                current_base_url="https://api.deepseek.com",
                current_api_key="deepseek-key",
            )

        # Should NOT switch provider — unknown-model is not a provider
        assert result.target_provider != "xai-oauth"
