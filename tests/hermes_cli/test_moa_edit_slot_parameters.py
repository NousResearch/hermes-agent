"""Regression tests for editing existing MoA slot parameters in place."""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.moa_cmd import _pick_slot


CURRENT = {
    "provider": "openrouter",
    "model": "anthropic/claude-opus-4.8",
    "enabled": False,
    "max_tokens": 600,
    "reasoning_effort": "low",
}


def test_existing_slot_parameters_can_be_edited_without_model_picker():
    with patch(
        "hermes_cli.moa_cmd._prompt_choice",
        side_effect=[0, 2, 6],  # edit, set max_tokens, high effort
    ), patch("builtins.input", return_value="1200") as prompt, patch(
        "hermes_cli.moa_cmd._model_options",
        side_effect=AssertionError("provider/model picker must not run"),
    ):
        slot = _pick_slot(CURRENT)

    assert slot == {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4.8",
        "enabled": False,
        "max_tokens": 1200,
        "reasoning_effort": "high",
    }
    prompt.assert_called_once()


def test_existing_slot_parameter_editor_can_keep_current_values():
    with patch("hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0, 0]), patch(
        "builtins.input", side_effect=AssertionError("value prompt must not run")
    ):
        assert _pick_slot(CURRENT) == CURRENT


def test_existing_slot_parameter_editor_can_clear_overrides():
    with patch("hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 1, 1]), patch(
        "builtins.input", side_effect=AssertionError("value prompt must not run")
    ):
        slot = _pick_slot(CURRENT)

    assert slot == {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4.8",
        "enabled": False,
    }


def test_existing_slot_can_still_change_provider_and_model():
    providers = [{"slug": "openai-codex", "name": "OpenAI Codex", "models": ["gpt-5.6-sol"]}]
    with patch("hermes_cli.moa_cmd._model_options", return_value=providers), patch(
        "hermes_cli.moa_cmd._prompt_choice",
        side_effect=[1, 0, 0],  # change model, provider, model
    ):
        slot = _pick_slot(CURRENT)

    assert slot == {"provider": "openai-codex", "model": "gpt-5.6-sol"}


def test_custom_max_tokens_reprompts_until_positive_integer(capsys):
    with patch("hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 2, 0]), patch(
        "builtins.input", side_effect=["invalid", "0", "2048"]
    ):
        slot = _pick_slot(CURRENT)

    assert slot["max_tokens"] == 2048
    assert capsys.readouterr().out.count("positive integer") == 2
