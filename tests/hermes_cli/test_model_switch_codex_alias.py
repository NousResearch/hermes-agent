"""Regression tests for the /model codex alias.

The short alias used to point at a nonexistent openai/codex catalog entry,
which made `/model codex` fail even though Codex models are available.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.detect_provider_for_model", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
def test_codex_alias_resolves_on_openai_codex(
    _validate,
    _detect,
    _capabilities,
    _info,
):
    """/model codex should resolve to a real Codex model on openai-codex."""
    with patch(
        "hermes_cli.model_switch.list_provider_models",
        return_value=["gpt-5.4-mini", "gpt-5.4", "gpt-5.3-codex", "gpt-5.2-codex"],
    ):
        result = switch_model(
            raw_input="codex",
            current_provider="openai-codex",
            current_model="gpt-5.4",
            current_base_url="https://chatgpt.com/backend-api/codex",
            current_api_key="***",
        )

    assert result.success is True
    assert result.target_provider == "openai-codex"
    assert result.new_model == "gpt-5.3-codex"


@patch("hermes_cli.model_switch.get_model_info", return_value=None)
@patch("hermes_cli.model_switch.get_model_capabilities", return_value=None)
@patch("hermes_cli.models.detect_provider_for_model", return_value=None)
@patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION)
def test_codex_alias_resolves_on_openrouter(
    _validate,
    _detect,
    _capabilities,
    _info,
):
    """The alias should also work when switching through openrouter."""
    with patch(
        "hermes_cli.model_switch.list_provider_models",
        return_value=["openai/gpt-5.4", "openai/gpt-5.4-mini", "openai/gpt-5.3-codex"],
    ):
        result = switch_model(
            raw_input="codex",
            current_provider="openrouter",
            current_model="openai/gpt-5.4",
            current_base_url="https://openrouter.ai/api/v1",
            current_api_key="***",
        )

    assert result.success is True
    assert result.target_provider == "openrouter"
    assert result.new_model == "openai/gpt-5.3-codex"
