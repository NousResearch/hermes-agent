"""Regression tests for Copilot auto model picker in /model switch flow.

MVP behavior:
- ``/model auto --provider copilot`` should resolve to a concrete Copilot model.
- ``/model copilot:auto`` should imply provider switch to Copilot.
- Selection should use fallback order when top preference is unavailable.
"""

from unittest.mock import patch

from hermes_cli.model_switch import switch_model


_MOCK_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


def _switch_auto(*, raw_input: str, current_provider: str = "openrouter", explicit_provider: str = ""):
    with (
        patch("hermes_cli.model_switch.resolve_alias", return_value=None),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "api_key": "ghu_test_token",
                "base_url": "https://api.githubcopilot.com",
                "api_mode": "chat_completions",
            },
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.validate_requested_model", return_value=_MOCK_VALIDATION),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        return switch_model(
            raw_input=raw_input,
            current_provider=current_provider,
            current_model="gpt-4o",
            current_base_url="",
            current_api_key="",
            explicit_provider=explicit_provider,
        )


def test_copilot_auto_with_explicit_provider_selects_concrete_model():
    with patch(
        "hermes_cli.model_switch.list_provider_models",
        return_value=["gpt-5.4", "gpt-5.4-mini", "claude-sonnet-4.6"],
    ):
        result = _switch_auto(raw_input="auto", explicit_provider="copilot")

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.target_provider == "github-copilot"
    assert result.new_model == "gpt-5.4"
    assert "Auto picker selected" in (result.warning_message or "")


def test_copilot_auto_uses_fallback_when_top_model_missing():
    with patch(
        "hermes_cli.model_switch.list_provider_models",
        return_value=["gpt-5.4-mini", "claude-sonnet-4.6"],
    ):
        result = _switch_auto(raw_input="auto", explicit_provider="copilot")

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.new_model == "claude-sonnet-4.6"


def test_copilot_auto_short_syntax_implies_provider_switch():
    with patch(
        "hermes_cli.model_switch.list_provider_models",
        return_value=["gpt-5.4", "claude-sonnet-4.6"],
    ):
        result = _switch_auto(raw_input="copilot:auto", current_provider="openrouter")

    assert result.success, f"switch_model failed: {result.error_message}"
    assert result.target_provider == "github-copilot"
    assert result.new_model == "gpt-5.4"
