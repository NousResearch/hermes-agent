"""Which requests a keep-alive may replay with its tiny output cap.

A Chat Completions gateway turns a requested effort into a thinking budget for Claude models that
predate adaptive thinking, and the bounded replay then keys a different cache entry (measured on Nous
Portal, claude-haiku-4.5, 30K prompt: replay with effort read 11,457 / wrote 18,886; without it read
30,343 / wrote 0). Every wire shape of the effort counts; adaptive models and effort ``none`` replay."""

import pytest

from agent.prompt_cache_lifecycle import is_replayable

_EFFORT_SHAPES = [
    {"reasoning_effort": "medium"},
    {"extra_body": {"reasoning_effort": "high"}},
    {"extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}},
]


@pytest.mark.parametrize("shape", _EFFORT_SHAPES)
def test_budget_era_claude_with_any_effort_shape_is_not_replayed(shape):
    for model in ("anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-4.5"):
        assert not is_replayable({"model": model, "messages": [], **shape})


@pytest.mark.parametrize("shape", _EFFORT_SHAPES)
def test_adaptive_claude_and_effort_off_stay_replayable(shape):
    assert is_replayable({"model": "anthropic/claude-sonnet-4.6", "messages": [], **shape})
    for off in ({"reasoning_effort": "none"}, {"extra_body": {"reasoning": {"enabled": False, "effort": "none"}}}):
        assert is_replayable({"model": "anthropic/claude-haiku-4.5", "messages": [], **off})
