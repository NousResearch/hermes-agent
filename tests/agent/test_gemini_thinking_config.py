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


@pytest.mark.parametrize("model", ["gemma-4-26b-a4b-it", "google/gemma-4-31b-it", "gemma4:31b-cloud"])
def test_gemma4_rides_thinking_level_only(model):
    """Gemma 4 on the gemini provider thinks by default and accepts only ``thinkingLevel``
    (``minimal``/``high``); ``thinkingBudget`` is HTTP 400 and no config at all burns a small
    output cap on hidden thought. Off and low efforts map to ``minimal``, high efforts to ``high``."""
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        assert _build_gemini_thinking_config(model, reasoning) == {"thinkingLevel": "minimal", "includeThoughts": False}
    for effort in ("minimal", "low", "medium"):
        assert _build_gemini_thinking_config(model, {"effort": effort}) == {"thinkingLevel": "minimal", "includeThoughts": True}
    for effort in ("high", "xhigh", "max"):
        assert _build_gemini_thinking_config(model, {"effort": effort}) == {"thinkingLevel": "high", "includeThoughts": True}
    for config in (_build_gemini_thinking_config(model, {"effort": e}) for e in ("none", "low", "high")):
        assert "thinkingBudget" not in config


def test_gemma4_minimal_off_switch_keeps_small_output_caps():
    """The off switch spends no thought tokens, so the 64-token title call keeps its cap instead
    of being raised to the 65,535 ceiling; a real thinking level still gets the headroom."""
    from agent.gemini_native_adapter import _effective_gemini_max_output_tokens

    off = _build_gemini_thinking_config("gemma-4-31b-it", {"enabled": False})
    assert _effective_gemini_max_output_tokens(64, off) == 64
    high = _build_gemini_thinking_config("gemma-4-31b-it", {"effort": "high"})
    assert _effective_gemini_max_output_tokens(64, high) == 65535
