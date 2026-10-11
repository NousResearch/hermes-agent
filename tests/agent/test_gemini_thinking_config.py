"""Disabling reasoning must actually stop Gemini thinking (#91927).

``includeThoughts: False`` only hides thought parts; the model still reasons
internally and bills thought tokens against maxOutputTokens, starving small
budgets (title generation's 64 tokens). ``thinkingBudget: 0`` is the real off
switch on Gemini 2.5. Gemini 3 deprecates ``thinkingBudget`` (silently remapped
today, rejected on upcoming models), so 3.x must ask for the documented
``thinkingLevel`` floor instead (#135176).
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
        ("gemini-1.5-flash", False),  # pre-2.5: thinkingBudget undocumented
    ],
)
def test_disabled_reasoning_zeroes_thinking_budget_on_gemini_2_5(model, expect_budget_zero):
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        assert config is not None
        assert config.get("includeThoughts") is False
        assert (config.get("thinkingBudget") == 0) is expect_budget_zero
        assert "thinkingLevel" not in config


@pytest.mark.parametrize(
    "model,level",
    [
        ("gemini-3.6-flash", "minimal"),  # flash family documents minimal
        ("gemini-flash-latest", "minimal"),  # alias tracks the current 3.x flash
        ("gemini-3.1-pro", "low"),  # 3 Pro rejects minimal
        ("gemini-3-something", "low"),  # unnamed family: don't guess minimal
    ],
)
def test_disabled_reasoning_uses_thinking_level_on_gemini_3(model, level):
    # Gemini 3 deprecates thinkingBudget; a level must replace it, never both.
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        assert config is not None
        assert config.get("includeThoughts") is False
        assert config.get("thinkingLevel") == level
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
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False, "thinkingLevel": "minimal"})
    assert translated == {"include_thoughts": False, "thinking_level": "minimal"}
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False})
    assert translated == {"include_thoughts": False}
