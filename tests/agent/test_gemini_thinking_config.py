"""Disabling reasoning must actually stop (or floor) Gemini thinking.

``includeThoughts: False`` only hides thought parts; the model still reasons
internally and bills thought tokens against maxOutputTokens, starving small
budgets (title generation's 64 tokens).

- Gemini 2.5 documents ``thinkingBudget: 0`` as the real off switch (#91927).
- Gemini 3 cannot disable thinking at all: Flash-Lite tiers reject
  ``thinkingBudget: 0`` with 400 INVALID_ARGUMENT (#123512), and the rest of
  the generation only tolerate the field undocumented. ``thinkingLevel:
  "minimal"`` is the documented closest-to-zero level and is accepted across
  the generation, including where the budget is rejected.
"""

import pytest

from agent.transports.chat_completions import (
    _build_gemini_thinking_config,
    _snake_case_gemini_thinking_config,
)


@pytest.mark.parametrize(
    "model,mode",
    [
        ("gemini-2.5-flash", "budget0"),        # documented off switch (#91927)
        ("gemini-3.6-flash", "minimal"),        # 3.x: cannot disable (#123512)
        ("gemini-3.1-pro", "minimal"),
        ("gemini-3.5-flash-lite", "minimal"),   # rejects budget 0 with 400
        ("gemini-flash-latest", "minimal"),
        ("gemini-1.5-flash", "hidden-only"),    # pre-2.5: thinkingBudget undocumented
    ],
)
def test_disabled_reasoning_sends_the_closest_supported_off(model, mode):
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        assert config is not None
        assert config.get("includeThoughts") is False
        if mode == "budget0":
            assert config.get("thinkingBudget") == 0
            assert "thinkingLevel" not in config
        elif mode == "minimal":
            assert config.get("thinkingLevel") == "minimal"
            assert "thinkingBudget" not in config
        else:
            assert "thinkingBudget" not in config
            assert "thinkingLevel" not in config


def test_enabled_reasoning_never_zeroes_budget_and_non_gemini_gets_nothing():
    # Enabled reasoning must not be silently strangled by a zero budget.
    for reasoning in ({"enabled": True}, {"effort": "medium"}):
        config = _build_gemini_thinking_config("gemini-2.5-flash", reasoning)
        assert config is not None
        assert config.get("includeThoughts") is True
        assert "thinkingBudget" not in config
    # Non-Gemini models on the same provider 400 on the field entirely (#17426).
    assert _build_gemini_thinking_config("gpt-4o", {"enabled": False}) is None
    assert _build_gemini_thinking_config("gemma-2b", {"enabled": False}) is None


def test_snake_case_translation_carries_thinking_budget():
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False, "thinkingBudget": 0})
    assert translated == {"include_thoughts": False, "thinking_budget": 0}
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False})
    assert translated == {"include_thoughts": False}
