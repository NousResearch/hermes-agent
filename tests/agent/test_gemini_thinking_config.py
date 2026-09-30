"""Disabling reasoning must actually stop Gemini thinking (#91927).

``includeThoughts: False`` only hides thought parts; the model still reasons
internally and bills thought tokens against maxOutputTokens, starving small
budgets (title generation's 64 tokens). ``thinkingBudget: 0`` is the real
off switch on families that document it.
"""

import pytest

from agent.transports.chat_completions import (
    _build_gemini_thinking_config,
    _snake_case_gemini_thinking_config,
)


@pytest.mark.parametrize(
    "model,expect_budget_zero",
    [
        ("gemini-2.5-flash", True),
        ("gemini-3.6-flash", True),
        ("gemini-3.1-pro", True),
        ("gemini-flash-latest", True),
        ("gemini-1.5-flash", False),  # pre-2.5: thinkingBudget undocumented
    ],
)
def test_disabled_reasoning_zeroes_thinking_budget_where_supported(model, expect_budget_zero):
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        assert config is not None
        assert config.get("includeThoughts") is False
        assert (config.get("thinkingBudget") == 0) is expect_budget_zero
        if not expect_budget_zero:
            assert "thinkingBudget" not in config


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
