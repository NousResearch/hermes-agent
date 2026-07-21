"""Regression tests for inline provider syntax in the shared /model pipeline.

The gateway/CLI /model command uses hermes_cli.model_switch.switch_model(), not
hermes_cli.models.parse_model_input(). Provider-qualified inputs must therefore
be recognized in this pipeline directly.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {"accepted": True, "persist": True, "recognized": True, "message": None}


def _switch(raw_input: str, *, current_provider: str = "openrouter", catalog=None):
    """Run switch_model with network/catalog/metadata calls mocked out."""
    catalog = [] if catalog is None else catalog
    with patch("hermes_cli.model_switch.resolve_alias", return_value=None), \
         patch("hermes_cli.model_switch.list_provider_models", return_value=catalog), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value={"api_key": "test", "base_url": "https://example.invalid/v1", "api_mode": "chat_completions"}), \
         patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION), \
         patch("hermes_cli.model_switch.get_model_info", return_value=None), \
         patch("hermes_cli.model_switch.get_model_capabilities", return_value=None), \
         patch("hermes_cli.models.detect_provider_for_model", return_value=None):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="anthropic/claude-sonnet-4.5",
            current_base_url="https://openrouter.ai/api/v1",
            current_api_key="test",
        )


def test_colon_provider_model_switches_provider_before_catalog_logic():
    result = _switch("openai-codex:gpt-5.5", current_provider="openrouter")

    assert result.success, result.error_message
    assert result.provider_changed is True
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-5.5"


def test_colon_provider_model_does_not_get_normalized_on_old_provider():
    result = _switch("openai-codex:gpt-5.5", current_provider="claude-api-proxy")

    assert result.success, result.error_message
    assert result.provider_changed is True
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-5.5"


def test_slash_provider_model_switches_from_non_aggregator_provider():
    result = _switch("nous/hermes-4", current_provider="claude-api-proxy")

    assert result.success, result.error_message
    assert result.provider_changed is True
    assert result.target_provider == "nous"
    assert result.new_model == "hermes-4"


def test_slash_openrouter_catalog_slug_stays_on_current_aggregator():
    result = _switch(
        "anthropic/claude-sonnet-4.5",
        current_provider="openrouter",
        catalog=["anthropic/claude-sonnet-4.5"],
    )

    assert result.success, result.error_message
    assert result.provider_changed is False
    assert result.target_provider == "openrouter"
    assert result.new_model == "anthropic/claude-sonnet-4.5"


def test_slash_alias_namespace_does_not_switch_provider_when_slug_exists():
    # `openai` is an alias to OpenRouter in provider resolution. Do not treat
    # every vendor namespace as an inline provider switch; OpenRouter slugs such
    # as openai/gpt-5.5 must keep working.
    result = _switch(
        "openai/gpt-5.5",
        current_provider="openrouter",
        catalog=["openai/gpt-5.5"],
    )

    assert result.success, result.error_message
    assert result.provider_changed is False
    assert result.target_provider == "openrouter"
    assert result.new_model == "openai/gpt-5.5"
