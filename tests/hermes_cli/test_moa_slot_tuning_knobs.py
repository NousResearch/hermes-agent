"""Per-slot MoA tuning knobs in ``hermes moa configure`` (#102582, #102584, #102585).

``moa.presets.<name>.reference_models[].reasoning_effort`` and ``.max_tokens``
have been first-class schema fields (``hermes_cli/moa_config.py::_clean_slot``)
that the interactive flow never set, so the only way to use them was hand-editing
config JSON. These tests pin the three behaviors that closed that gap:

* ``_pick_slot`` prompts for both knobs and persists them on the slot.
* ``_format_slot`` surfaces both in ``hermes moa list``.
* An existing slot can be retuned in place, without re-picking provider/model.
"""

from unittest.mock import patch

import pytest

from hermes_cli import moa_cmd
from hermes_cli.moa_config import normalize_moa_config

PROVIDERS = [
    {
        "slug": "openrouter",
        "name": "OpenRouter",
        "models": ["anthropic/claude-opus-4.8", "deepseek/deepseek-v4-pro"],
        "capabilities": {
            "anthropic/claude-opus-4.8": {"fast": False, "reasoning": True},
            "deepseek/deepseek-v4-pro": {"fast": False, "reasoning": True},
        },
    },
    {
        "slug": "nous",
        "name": "Nous",
        "models": ["hermes-4"],
        # A route that takes no reasoning parameter at all.
        "capabilities": {"hermes-4": {"fast": False, "reasoning": False}},
    },
]


@pytest.fixture()
def providers():
    with patch.object(moa_cmd, "_model_options", return_value=PROVIDERS):
        yield PROVIDERS


class TestFormatSlot:
    def test_bare_slot_has_no_bracket(self):
        assert moa_cmd._format_slot({"provider": "nous", "model": "hermes-4"}) == "nous:hermes-4"

    def test_reasoning_effort_is_shown(self):
        slot = {"provider": "nous", "model": "hermes-4", "reasoning_effort": "high"}
        assert moa_cmd._format_slot(slot) == "nous:hermes-4 [reasoning=high]"

    def test_max_tokens_is_shown(self):
        # #102584: _format_slot used to drop the per-slot cap entirely, so a
        # configured override was invisible in `hermes moa list`.
        slot = {"provider": "nous", "model": "hermes-4", "max_tokens": 2048}
        assert moa_cmd._format_slot(slot) == "nous:hermes-4 [max_tokens=2048]"

    def test_both_knobs_are_shown(self):
        slot = {
            "provider": "nous",
            "model": "hermes-4",
            "reasoning_effort": "low",
            "max_tokens": 512,
        }
        assert moa_cmd._format_slot(slot) == "nous:hermes-4 [reasoning=low, max_tokens=512]"

    def test_non_positive_max_tokens_is_not_shown(self):
        slot = {"provider": "nous", "model": "hermes-4", "max_tokens": 0}
        assert moa_cmd._format_slot(slot) == "nous:hermes-4"


class TestPromptOptionalMaxTokens:
    def test_blank_keeps_current(self):
        with patch("builtins.input", return_value=""):
            assert moa_cmd._prompt_optional_max_tokens(4096) == (False, 4096)

    def test_number_sets_override(self):
        with patch("builtins.input", return_value="1234"):
            assert moa_cmd._prompt_optional_max_tokens(None) == (True, 1234)

    @pytest.mark.parametrize("text", ["none", "default", "clear", "0"])
    def test_sentinels_clear_the_override(self, text):
        with patch("builtins.input", return_value=text):
            assert moa_cmd._prompt_optional_max_tokens(4096) == (True, None)

    def test_garbage_keeps_current(self):
        with patch("builtins.input", return_value="lots"):
            assert moa_cmd._prompt_optional_max_tokens(2048) == (False, 2048)

    def test_negative_falls_back_to_preset_default(self):
        with patch("builtins.input", return_value="-5"):
            assert moa_cmd._prompt_optional_max_tokens(2048) == (True, None)

    def test_interrupt_keeps_current(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            assert moa_cmd._prompt_optional_max_tokens(2048) == (False, 2048)


class TestPromptSlotReasoningEffort:
    def test_skipped_for_a_non_reasoning_model(self, providers):
        # The picker already knows this route takes no reasoning parameter, so
        # the prompt must not even be offered.
        with patch("hermes_cli.main._prompt_reasoning_effort_selection") as prompt:
            changed, value = moa_cmd._prompt_slot_reasoning_effort(PROVIDERS[1], "hermes-4", None)
        prompt.assert_not_called()
        assert (changed, value) == (False, None)

    def test_selection_is_returned(self, providers):
        with patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value="high"):
            assert moa_cmd._prompt_slot_reasoning_effort(
                PROVIDERS[0], "anthropic/claude-opus-4.8", None
            ) == (True, "high")

    def test_skip_keeps_current(self, providers):
        with patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value=None):
            assert moa_cmd._prompt_slot_reasoning_effort(
                PROVIDERS[0], "anthropic/claude-opus-4.8", "low"
            ) == (False, "low")

    def test_disable_refused_on_reasoning_mandatory_route(self):
        provider = {
            "slug": "nous",
            "models": ["mandatory-thinker"],
            "capabilities": {
                "mandatory-thinker": {"reasoning": True, "can_disable_reasoning": False}
            },
        }
        with patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value="none"):
            assert moa_cmd._prompt_slot_reasoning_effort(provider, "mandatory-thinker", "high") == (
                False,
                "high",
            )


class TestPickSlotTuningKnobs:
    def test_new_slot_records_both_knobs(self, providers):
        # provider idx 0, model idx 0, then the two knob prompts.
        with patch.object(moa_cmd, "_prompt_choice", side_effect=[0, 0]), patch(
            "hermes_cli.main._prompt_reasoning_effort_selection", return_value="medium"
        ), patch("builtins.input", return_value="3000"):
            slot = moa_cmd._pick_slot()

        assert slot == {
            "provider": "openrouter",
            "model": "anthropic/claude-opus-4.8",
            "reasoning_effort": "medium",
            "max_tokens": 3000,
        }
        # And the write boundary keeps both fields.
        cleaned = normalize_moa_config(
            {"presets": {"p": {"reference_models": [slot], "aggregator": slot}}}
        )["presets"]["p"]["reference_models"][0]
        assert cleaned["reasoning_effort"] == "medium"
        assert cleaned["max_tokens"] == 3000

    def test_edit_in_place_keeps_provider_and_model(self, providers):
        # #102585: choosing "Keep provider/model" must not walk the provider or
        # model picker again — exactly one _prompt_choice call (the edit menu).
        current = {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning_effort": "low",
            "max_tokens": 512,
        }
        with patch.object(moa_cmd, "_prompt_choice", side_effect=[0]) as choice, patch(
            "hermes_cli.main._prompt_reasoning_effort_selection", return_value="high"
        ), patch("builtins.input", return_value="4096"):
            slot = moa_cmd._pick_slot(current)

        assert choice.call_count == 1
        assert slot == {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning_effort": "high",
            "max_tokens": 4096,
        }

    def test_edit_in_place_preserves_untouched_knobs(self, providers):
        current = {
            "provider": "openrouter",
            "model": "deepseek/deepseek-v4-pro",
            "reasoning_effort": "low",
            "max_tokens": 512,
        }
        with patch.object(moa_cmd, "_prompt_choice", side_effect=[0]), patch(
            "hermes_cli.main._prompt_reasoning_effort_selection", return_value=None
        ), patch("builtins.input", return_value=""):
            slot = moa_cmd._pick_slot(current)

        assert slot["reasoning_effort"] == "low"
        assert slot["max_tokens"] == 512

    def test_changing_model_drops_stale_knobs(self, providers):
        # Knobs are per-model; carrying an effort/cap onto a different model
        # would silently apply a budget the user chose for something else.
        current = {
            "provider": "openrouter",
            "model": "anthropic/claude-opus-4.8",
            "reasoning_effort": "high",
            "max_tokens": 512,
        }
        # edit menu -> "Change provider/model", provider idx 0, model idx 1.
        with patch.object(moa_cmd, "_prompt_choice", side_effect=[1, 0, 1]), patch(
            "hermes_cli.main._prompt_reasoning_effort_selection", return_value=None
        ), patch("builtins.input", return_value=""):
            slot = moa_cmd._pick_slot(current)

        assert slot == {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"}

    def test_no_edit_menu_when_current_model_is_unselectable(self, providers):
        # A slot naming a model the picker no longer offers must fall through to
        # the full flow rather than letting "keep" pin a dead pairing.
        current = {"provider": "openrouter", "model": "retired/model"}
        with patch.object(moa_cmd, "_prompt_choice", side_effect=[0, 0]) as choice, patch(
            "hermes_cli.main._prompt_reasoning_effort_selection", return_value=None
        ), patch("builtins.input", return_value=""):
            slot = moa_cmd._pick_slot(current)

        # provider + model pickers only; no edit menu.
        assert choice.call_count == 2
        assert slot == {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"}

    def test_non_reasoning_model_never_gets_an_effort(self, providers):
        # provider idx 1 (nous), model idx 0 (hermes-4, reasoning: False).
        with patch.object(moa_cmd, "_prompt_choice", side_effect=[1, 0]), patch(
            "builtins.input", return_value=""
        ):
            slot = moa_cmd._pick_slot()

        assert slot == {"provider": "nous", "model": "hermes-4"}
