"""Disabling reasoning must actually stop (or minimise) Gemini thinking (#91927).

``includeThoughts: False`` only hides thought parts; the model still reasons
internally and bills thought tokens against maxOutputTokens, starving small
budgets (title generation's 64 tokens). Gemini 2.5 documents ``thinkingBudget: 0``
as the real off switch; Gemini 3+ deprecated that field in favour of the
``thinkingLevel`` string enum (which has no "none"), so the disable path uses its
lowest universally supported level ("low").
"""

import pytest

from agent.transports.chat_completions import (
    _build_gemini_thinking_config,
    _snake_case_gemini_thinking_config,
)


@pytest.mark.parametrize(
    "model,disabled_field",
    [
        ("gemini-2.5-flash", ("thinkingBudget", 0)),      # 2.5 still documents thinkingBudget
        ("gemini-3.6-flash", ("thinkingLevel", "low")),   # 3+ deprecated thinkingBudget → thinkingLevel
        ("gemini-3.1-pro", ("thinkingLevel", "low")),
        ("gemini-flash-latest", ("thinkingLevel", "low")),
        ("gemini-1.5-flash", None),                        # pre-2.5: thinking control undocumented
    ],
)
def test_disabled_reasoning_bounds_thinking_where_supported(model, disabled_field):
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        assert config is not None
        assert config.get("includeThoughts") is False
        if disabled_field is None:
            assert "thinkingBudget" not in config
            assert "thinkingLevel" not in config
        else:
            field, value = disabled_field
            assert config.get(field) == value
            other = "thinkingLevel" if field == "thinkingBudget" else "thinkingBudget"
            assert other not in config


def test_enabled_reasoning_never_bounds_thinking_and_non_gemini_gets_nothing():
    # Enabled reasoning must not be silently strangled by a zero budget / low level.
    for reasoning in ({"enabled": True}, {"effort": "medium"}):
        config = _build_gemini_thinking_config("gemini-2.5-flash", reasoning)
        assert config is not None
        assert config.get("includeThoughts") is True
        assert "thinkingBudget" not in config
        assert "thinkingLevel" not in config
    # Non-Gemini models on the same provider 400 on the field entirely (#17426).
    assert _build_gemini_thinking_config("gpt-4o", {"enabled": False}) is None
    assert _build_gemini_thinking_config("gemma-2b", {"enabled": False}) is None


def test_snake_case_translation_carries_thinking_controls():
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False, "thinkingBudget": 0})
    assert translated == {"include_thoughts": False, "thinking_budget": 0}
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False, "thinkingLevel": "low"})
    assert translated == {"include_thoughts": False, "thinking_level": "low"}
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False})
    assert translated == {"include_thoughts": False}
