"""CLI coverage for per-reference max_tokens overrides."""

from unittest.mock import patch

from hermes_cli.moa_cmd import (
    _format_slot,
    _pick_slot,
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


def test_format_slot_shows_reasoning_and_max_tokens_overrides():
    assert _format_slot(
        {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning_effort": "low",
            "max_tokens": 600,
        }
    ) == "openrouter:deepseek/deepseek-v4-pro [reasoning=low, max_tokens=600]"
