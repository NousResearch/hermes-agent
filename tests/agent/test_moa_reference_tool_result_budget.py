"""Per-preset ``reference_tool_result_budget`` for the MoA advisory view.

The advisor sees tool CALLS in full and tool RESULTS head+tail previewed; the preview length was a
module constant. These tests pin the new per-preset override, its clamps, and the fact that the
default behaviour (4000 chars) is unchanged when the key is absent.
"""

from __future__ import annotations

import pytest

from agent.moa_loop import (
    _REFERENCE_TOOL_RESULT_BUDGET,
    _preset_tool_result_budget,
    _reference_messages,
)
from hermes_cli.moa_config import (
    DEFAULT_REFERENCE_TOOL_RESULT_BUDGET,
    normalize_moa_config,
    resolve_moa_preset,
)

LONG_RESULT = "A" * 5_000 + "B" * 5_000  # 10 000 chars


def _tool_turn(result: str = LONG_RESULT):
    return [
        {"role": "user", "content": "run the thing"},
        {"role": "assistant", "content": "calling it",
         "tool_calls": [{"id": "1", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "1", "name": "terminal", "content": result},
    ]


def _advisory_size(budget: int) -> int:
    view = _reference_messages(_tool_turn(), tool_result_budget=budget)
    return sum(len(str(m.get("content") or "")) for m in view)


class TestPresetToolResultBudget:
    def test_defaults_when_absent_or_invalid(self):
        assert _preset_tool_result_budget({}) == _REFERENCE_TOOL_RESULT_BUDGET
        assert _preset_tool_result_budget(None) == _REFERENCE_TOOL_RESULT_BUDGET
        assert _preset_tool_result_budget({"reference_tool_result_budget": ""}) == _REFERENCE_TOOL_RESULT_BUDGET
        assert _preset_tool_result_budget({"reference_tool_result_budget": "nope"}) == _REFERENCE_TOOL_RESULT_BUDGET

    def test_accepts_int_and_float_string(self):
        assert _preset_tool_result_budget({"reference_tool_result_budget": 8000}) == 8000
        assert _preset_tool_result_budget({"reference_tool_result_budget": "8000"}) == 8000
        assert _preset_tool_result_budget({"reference_tool_result_budget": 8000.0}) == 8000

    def test_floors_at_minimum(self):
        assert _preset_tool_result_budget({"reference_tool_result_budget": 1}) == 200


class TestAdvisoryViewUsesBudget:
    def test_wider_budget_keeps_more_of_a_tool_result(self):
        narrow = _advisory_size(200)
        wide = _advisory_size(8000)
        assert wide > narrow
        # 10k-char result: an 8000-char budget keeps ~8000 chars, a 200-char budget keeps ~200.
        assert wide - narrow > 7_000

    def test_default_budget_matches_module_constant(self):
        assert _advisory_size(_REFERENCE_TOOL_RESULT_BUDGET) == _advisory_size(4000)

    def test_short_results_are_never_truncated(self):
        short = "x" * 100
        view = _reference_messages(_tool_turn(short), tool_result_budget=200)
        assert short in str(view[-2].get("content") or view[-1].get("content"))
        assert "chars omitted" not in str(view)


class TestNormalization:
    def test_default_applied_when_absent(self):
        cfg = normalize_moa_config({"presets": {"p": {"reference_models": [
            {"provider": "nous", "model": "m"}], "aggregator": {"provider": "nous", "model": "a"}}}})
        assert cfg["presets"]["p"]["reference_tool_result_budget"] == DEFAULT_REFERENCE_TOOL_RESULT_BUDGET

    def test_flattened_view_carries_the_key(self):
        cfg = normalize_moa_config({"presets": {"default": {
            "reference_models": [{"provider": "nous", "model": "m"}],
            "aggregator": {"provider": "nous", "model": "a"},
            "reference_tool_result_budget": 9000,
        }}})
        assert cfg["reference_tool_result_budget"] == 9000

    @pytest.mark.parametrize("raw,expected", [
        (0, 200), (10, 200), (4000, 4000), (123456, 50_000), (-5, 200),
    ])
    def test_clamped(self, raw, expected):
        cfg = normalize_moa_config({"presets": {"default": {
            "reference_models": [{"provider": "nous", "model": "m"}],
            "aggregator": {"provider": "nous", "model": "a"},
            "reference_tool_result_budget": raw,
        }}})
        assert cfg["reference_tool_result_budget"] == expected

    def test_resolved_preset_exposes_it(self):
        preset = resolve_moa_preset({"presets": {"default": {
            "reference_models": [{"provider": "nous", "model": "m"}],
            "aggregator": {"provider": "nous", "model": "a"},
            "reference_tool_result_budget": 6000,
        }}})
        assert preset["reference_tool_result_budget"] == 6000
