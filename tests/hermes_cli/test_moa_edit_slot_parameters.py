"""Regression tests for editing existing MoA slot parameters in place."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.moa_cmd import _pick_slot, _prompt_positive_int, cmd_moa


CURRENT = {
    "provider": "openrouter",
    "model": "anthropic/claude-opus-4.8",
    "enabled": False,
    "max_tokens": 600,
    "reasoning_effort": "low",
}

PROVIDER = {
    "slug": "openrouter",
    "name": "OpenRouter",
    "models": ["anthropic/claude-opus-4.8", "openai/gpt-4.1"],
    "capabilities": {
        "anthropic/claude-opus-4.8": {"reasoning": True},
        "openai/gpt-4.1": {"reasoning": False},
    },
}


def test_existing_slot_parameters_can_be_edited_without_model_picker():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 2, 0]
    ) as choice, patch("builtins.input", return_value="1200") as prompt, patch(
        "hermes_cli.main._prompt_reasoning_effort_selection", return_value="high"
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
    assert [call.args[0] for call in choice.call_args_list] == [
        "Edit existing slot",
        "Max tokens",
        "Reasoning effort",
    ]


def test_existing_slot_parameter_editor_can_keep_current_values():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0, 0]
    ), patch("builtins.input", side_effect=AssertionError("value prompt must not run")), patch(
        "hermes_cli.main._prompt_reasoning_effort_selection", return_value=None
    ):
        assert _pick_slot(CURRENT) == CURRENT


def test_existing_slot_parameter_editor_can_clear_max_tokens_override():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 1, 0]
    ), patch("builtins.input", side_effect=AssertionError("value prompt must not run")), patch(
        "hermes_cli.main._prompt_reasoning_effort_selection", return_value="none"
    ):
        slot = _pick_slot(CURRENT)

    assert slot == {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4.8",
        "enabled": False,
        "reasoning_effort": "none",
    }


def test_existing_slot_can_still_change_provider_and_model():
    providers = [
        {
            "slug": "openai-codex",
            "name": "OpenAI Codex",
            "models": ["gpt-5.6-sol"],
            "capabilities": {"gpt-5.6-sol": {"reasoning": True}},
        }
    ]
    with patch("hermes_cli.moa_cmd._model_options", return_value=providers), patch(
        "hermes_cli.moa_cmd._prompt_choice",
        side_effect=[1, 0, 0],  # change model, provider, model
    ), patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value="medium"):
        slot = _pick_slot(CURRENT)

    assert slot == {
        "provider": "openai-codex",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
    }


def test_replacing_with_same_model_preserves_effort_when_selection_is_skipped():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[1, 0, 0]
    ), patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value=None):
        slot = _pick_slot(CURRENT)

    assert slot == {
        "provider": "openrouter",
        "model": "anthropic/claude-opus-4.8",
        "reasoning_effort": "low",
    }


def test_replacing_with_changed_model_does_not_carry_effort_when_selection_is_skipped():
    provider = {
        "slug": "openai-codex",
        "name": "OpenAI Codex",
        "models": ["gpt-5.6-sol"],
        "capabilities": {"gpt-5.6-sol": {"reasoning": True}},
    }
    with patch("hermes_cli.moa_cmd._model_options", return_value=[provider]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[1, 0, 0]
    ), patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value=None):
        slot = _pick_slot(CURRENT)

    assert slot == {"provider": "openai-codex", "model": "gpt-5.6-sol"}


def test_custom_max_tokens_reprompts_until_positive_integer(capsys):
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 2, 0]
    ), patch("builtins.input", side_effect=["invalid", "0", "2048"]), patch(
        "hermes_cli.main._prompt_reasoning_effort_selection", return_value=None
    ):
        slot = _pick_slot(CURRENT)

    assert slot["max_tokens"] == 2048
    assert capsys.readouterr().out.count("positive integer") == 2


def test_custom_max_tokens_keyboard_interrupt_aborts_edit():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 2]
    ), patch("builtins.input", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            _pick_slot(CURRENT)


def test_custom_max_tokens_eof_keeps_current_value():
    with patch("builtins.input", side_effect=EOFError):
        assert _prompt_positive_int("Max tokens", 600) == 600


def test_non_reasoning_slot_does_not_offer_reasoning_control(capsys):
    current = {**CURRENT, "model": "openai/gpt-4.1"}
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0]
    ), patch("hermes_cli.main._prompt_reasoning_effort_selection") as prompt:
        slot = _pick_slot(current)

    assert slot == {key: value for key, value in current.items() if key != "reasoning_effort"}
    prompt.assert_not_called()
    assert (
        "Note: openrouter:openai/gpt-4.1 does not support reasoning; "
        "dropping the existing reasoning_effort override."
    ) in capsys.readouterr().out


def test_unknown_reasoning_capability_preserves_existing_override():
    provider = {**PROVIDER, "capabilities": {}}
    with patch("hermes_cli.moa_cmd._model_options", return_value=[provider]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0]
    ), patch("hermes_cli.main._prompt_reasoning_effort_selection") as prompt:
        slot = _pick_slot(CURRENT)

    assert slot == CURRENT
    prompt.assert_not_called()


def test_existing_slot_parameter_editor_can_unset_reasoning_effort():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0, 1]
    ), patch("hermes_cli.main._prompt_reasoning_effort_selection") as prompt:
        slot = _pick_slot(CURRENT)

    assert slot == {key: value for key, value in CURRENT.items() if key != "reasoning_effort"}
    prompt.assert_not_called()


def test_aggregator_editor_does_not_offer_or_persist_max_tokens():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0]
    ) as choice, patch(
        "hermes_cli.main._prompt_reasoning_effort_selection", return_value="high"
    ):
        slot = _pick_slot(CURRENT, role="aggregator")

    assert "max_tokens" not in slot
    assert [call.args[0] for call in choice.call_args_list] == [
        "Edit existing slot",
        "Reasoning effort",
    ]
    assert choice.call_args_list[0].args[1][0] == "Keep provider/model; edit reasoning_effort"


def test_configure_marks_reference_and_aggregator_slot_roles():
    config = {
        "moa": {
            "default_preset": "default",
            "presets": {
                "default": {
                    "reference_models": [CURRENT],
                    "aggregator": CURRENT,
                }
            },
        }
    }
    with patch("hermes_cli.moa_cmd.load_config", return_value=config), patch(
        "hermes_cli.moa_cmd._pick_slot",
        side_effect=[dict(CURRENT), dict(CURRENT)],
    ) as pick_slot, patch("hermes_cli.moa_cmd._prompt_choice", return_value=1), patch(
        "hermes_cli.moa_cmd.save_config"
    ), patch("hermes_cli.moa_cmd._print_config"):
        cmd_moa(SimpleNamespace(moa_command="configure", name="default"))

    assert pick_slot.call_args_list[0].kwargs == {"role": "reference"}
    assert pick_slot.call_args_list[1].kwargs == {"role": "aggregator"}
