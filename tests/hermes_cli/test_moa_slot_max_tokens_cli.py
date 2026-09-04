"""CLI coverage for per-reference max_tokens overrides."""

from unittest.mock import patch

import pytest

from hermes_cli.moa_cmd import (
    _format_slot,
    _pick_slot,
    _print_config,
    _prompt_max_tokens_override,
)


PROVIDER = {
    "slug": "openrouter",
    "name": "OpenRouter",
    "models": ["deepseek/deepseek-v4-pro"],
}


def test_format_slot_shows_max_tokens_override():
    assert _format_slot(
        {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-pro",
            "max_tokens": 600,
        }
    ) == "openrouter:deepseek/deepseek-v4-pro [max_tokens=600]"


def test_pick_reference_slot_prompts_for_max_tokens_override():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0]
    ), patch("builtins.input", return_value="600"):
        slot = _pick_slot()

    assert slot == {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-pro",
        "max_tokens": 600,
    }


def test_blank_max_tokens_override_inherits_preset_default():
    with patch("builtins.input", return_value=""):
        assert _prompt_max_tokens_override() is None


def test_invalid_max_tokens_override_reprompts(capsys):
    with patch("builtins.input", side_effect=["0", "many", "800"]):
        assert _prompt_max_tokens_override() == 800

    assert capsys.readouterr().out.count("Enter a positive integer") == 2


@pytest.mark.parametrize("interruption", [EOFError(), KeyboardInterrupt()])
def test_max_tokens_override_interruptions_cancel_cleanly(interruption):
    with patch("builtins.input", side_effect=interruption), pytest.raises(
        SystemExit, match="MoA configuration cancelled"
    ):
        _prompt_max_tokens_override()


def test_aggregator_slot_does_not_offer_reference_override():
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0]
    ), patch("builtins.input") as prompt:
        slot = _pick_slot(configure_max_tokens=False)

    assert slot == {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-pro",
    }
    prompt.assert_not_called()


def test_print_config_hides_reference_only_override_on_aggregator(capsys):
    _print_config(
        {
            "moa": {
                "presets": {
                    "custom": {
                        "reference_models": [
                            {
                                "provider": "openrouter",
                                "model": "reference",
                                "max_tokens": 600,
                            }
                        ],
                        "aggregator": {
                            "provider": "openrouter",
                            "model": "aggregator",
                            "reasoning_effort": "high",
                            "max_tokens": 777,
                        },
                    }
                },
                "default_preset": "custom",
            }
        }
    )

    output = capsys.readouterr().out
    assert "openrouter:reference [max_tokens=600]" in output
    assert "Aggregator: openrouter:aggregator [reasoning=high]" in output
    assert "max_tokens=777" not in output


def test_format_slot_shows_reasoning_and_max_tokens_overrides():
    assert _format_slot(
        {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning_effort": "low",
            "max_tokens": 600,
        }
    ) == "openrouter:deepseek/deepseek-v4-pro [reasoning=low, max_tokens=600]"
