"""Disabling reasoning must actually stop Gemini thinking (#91927).

``includeThoughts: False`` only hides thought parts; the model still reasons
internally and bills thought tokens against maxOutputTokens, starving small
budgets (title generation's 64 tokens). ``thinkingBudget: 0`` is the real
off switch on families that document it — except Gemini 3, which cannot
disable thinking and rejects a zero budget on some models; there the request
degrades to the documented closest-to-zero level (#123512).
"""

import pytest

from agent.transports.chat_completions import (
    _build_gemini_thinking_config,
    _snake_case_gemini_thinking_config,
)


@pytest.mark.parametrize(
    "model,expect_budget_zero,expect_level",
    [
        ("gemini-2.5-flash", True, None),
        # Gemini 3 cannot disable thinking: Flash takes MINIMAL, others LOW. (#123512)
        ("gemini-3.6-flash", False, "minimal"),
        ("gemini-3.5-flash-lite", False, "minimal"),
        ("gemini-3.1-pro", False, "low"),
        ("gemini-flash-latest", True, None),
        ("gemini-1.5-flash", False, None),  # pre-2.5: thinkingBudget undocumented
    ],
)
def test_disabled_reasoning_zeroes_thinking_budget_where_supported(model, expect_budget_zero, expect_level):
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        assert config is not None
        assert config.get("includeThoughts") is False
        assert (config.get("thinkingBudget") == 0) is expect_budget_zero
        if not expect_budget_zero:
            assert "thinkingBudget" not in config
        assert config.get("thinkingLevel") == expect_level


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


def test_gemini3_flash_lite_never_receives_zero_budget_when_disabling():
    # 3.5 Flash-Lite rejects ``thinkingBudget: 0`` with HTTP 400 INVALID_ARGUMENT,
    # which silently degraded every title to the derived fallback; the documented
    # closest-to-zero level must go on the wire instead. (#123512)
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config("gemini-3.5-flash-lite", reasoning)
        assert config == {"includeThoughts": False, "thinkingLevel": "minimal"}


def test_gemini3_minimal_level_still_requests_output_headroom():
    # MINIMAL/LOW still spend thought tokens against maxOutputTokens; the #91927
    # starvation guard must keep raising small caps on the level shape too.
    from agent.gemini_native_adapter import _effective_gemini_max_output_tokens

    for model in ("gemini-3.5-flash-lite", "gemini-3.1-pro"):
        config = _build_gemini_thinking_config(model, {"enabled": False})
        assert _effective_gemini_max_output_tokens(64, config) > 64


def test_snake_case_translation_carries_thinking_budget():
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False, "thinkingBudget": 0})
    assert translated == {"include_thoughts": False, "thinking_budget": 0}
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False})
    assert translated == {"include_thoughts": False}
