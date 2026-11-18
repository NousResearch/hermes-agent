"""Regression tests for per-slot reasoning effort in ``hermes moa configure``.

Issue #102582: ``moa.presets.<name>.reference_models[].reasoning_effort`` and
``.aggregator.reasoning_effort`` are first-class config fields that
``hermes moa list`` displays, but ``_pick_slot`` never offered them — the only
way to set one was hand-editing the config JSON. The slot picker must now reuse
the primary-model effort prompt whenever the picked model's capability map says
it supports reasoning.
"""

from __future__ import annotations

from unittest.mock import patch

from hermes_cli.moa_cmd import _pick_slot

PROVIDER = {
    "slug": "z-ai",
    "name": "Z AI",
    "models": ["glm-5.3", "glm-5.3-air"],
    "capabilities": {
        "glm-5.3": {"fast": False, "reasoning": True},
        "glm-5.3-air": {"fast": True, "reasoning": False},
    },
}


def _run_pick_slot(*, model_idx: int = 0, current: dict | None = None):
    with patch("hermes_cli.moa_cmd._model_options", return_value=[PROVIDER]), patch(
        "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, model_idx]
    ) as choice:
        slot = _pick_slot(current)
    return slot, choice


def _run_with_selection(selection, *, model_idx: int = 0, current: dict | None = None):
    with patch("hermes_cli.main._prompt_reasoning_effort_selection", return_value=selection) as prompt:
        slot, choice = _run_pick_slot(model_idx=model_idx, current=current)
    return slot, choice, prompt


class TestPickSlotReasoningEffort:
    def test_capable_model_prompts_and_persists_effort(self):
        slot, _, prompt = _run_with_selection("high")
        assert slot == {
            "provider": "z-ai",
            "model": "glm-5.3",
            "reasoning_effort": "high",
        }
        prompt.assert_called_once()
        args, kwargs = prompt.call_args
        assert args[0] == [
            "minimal", "low", "medium", "high", "xhigh", "max", "ultra",
        ]
        assert kwargs.get("current_effort") == ""

    def test_disable_reasoning_persists_none(self):
        slot, _, _ = _run_with_selection("none")
        assert slot["reasoning_effort"] == "none"

    def test_skip_on_new_slot_omits_key(self):
        slot, _, prompt = _run_with_selection(None)
        assert slot == {"provider": "z-ai", "model": "glm-5.3"}
        prompt.assert_called_once()

    def test_skip_keeps_existing_effort(self):
        slot, _, prompt = _run_with_selection(
            None, current={"provider": "z-ai", "model": "glm-5.3", "reasoning_effort": "low"}
        )
        assert slot["reasoning_effort"] == "low"
        assert prompt.call_args.kwargs.get("current_effort") == "low"

    def test_non_reasoning_model_not_prompted(self):
        slot, _, prompt = _run_with_selection("high", model_idx=1)
        assert slot == {"provider": "z-ai", "model": "glm-5.3-air"}
        prompt.assert_not_called()

    def test_missing_capability_entry_not_prompted(self):
        provider = dict(PROVIDER, models=["glm-5.3-flash"], capabilities={"glm-5.3": {"reasoning": True}})
        with patch("hermes_cli.moa_cmd._model_options", return_value=[provider]), patch(
            "hermes_cli.moa_cmd._prompt_choice", side_effect=[0, 0]
        ), patch(
            "hermes_cli.main._prompt_reasoning_effort_selection", return_value="high"
        ) as prompt:
            slot = _pick_slot(None)
        assert slot == {"provider": "z-ai", "model": "glm-5.3-flash"}
        prompt.assert_not_called()
