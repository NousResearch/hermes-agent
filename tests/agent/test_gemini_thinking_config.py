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
        ("gemini-4-argon", ("thinkingLevel", "low")),     # a new generation is not treated as legacy
        ("google/gemini-4-argon", ("thinkingLevel", "low")),
        ("gemini-flash-lite-latest", ("thinkingLevel", "low")),  # every -latest alias, not just flash
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


@pytest.mark.parametrize(
    "reasoning,level,shown,budget",
    [
        ({"effort": "none"}, "low", False, None),  # Gemini 4 cannot stop thinking: floor level, hidden
        ({"effort": "minimal"}, "low", True, None),  # "minimal" 400s on Gemini 4
        ({"effort": "medium"}, "medium", True, None),
        ({"effort": "high"}, "high", True, None),
        ({"effort": "xhigh"}, "high", True, 262_144),  # above high buys long-decode budget instead
        ({"effort": "max"}, "high", True, 1_048_576),
    ],
)
def test_gemini4_maps_effort_to_thinking_level_and_long_decode_budget(reasoning, level, shown, budget):
    from providers import get_provider_profile

    config = _build_gemini_thinking_config("gemini-4-argon", reasoning)
    assert config == {"includeThoughts": shown, "thinkingLevel": level}  # never thinkingBudget (400 with a level)
    profile = get_provider_profile("gemini")
    native = profile.build_extra_body(model="gemini-4-argon", reasoning_config=reasoning,
                                      base_url="https://generativelanguage.googleapis.com/v1beta")
    assert native.get("max_output_tokens") == budget
    # The OpenAI-compat subpath has no long decode: no budget rides there.
    compat = profile.build_extra_body(model="gemini-4-argon", reasoning_config=reasoning,
                                      base_url="https://generativelanguage.googleapis.com/v1beta/openai")
    assert "max_output_tokens" not in compat


@pytest.mark.parametrize(
    "model,effort,level",
    [
        # Argon documents low/medium/high and 400s on "minimal" — clamp, never forward it.
        ("gemini-4-argon", "minimal", "low"),
        ("gemini-4-argon", "low", "low"),
        ("gemini-4-argon", "medium", "medium"),
        ("gemini-4-argon", "xhigh", "high"),
        ("google/gemini-4-argon", "high", "high"),
        ("gemini-3.8-flash", "medium", "medium"),
        ("gemini-3.1-pro", "medium", "low"),  # 3 Pro stays strict (low/high)
        ("gemini-3.1-pro", "high", "high"),
        ("gemini-pro-latest", "high", "high"),
    ],
)
def test_enabled_reasoning_maps_effort_to_thinking_level_on_gemini_3_and_later(model, effort, level):
    config = _build_gemini_thinking_config(model, {"enabled": True, "effort": effort})
    assert config == {"includeThoughts": True, "thinkingLevel": level}


def test_unversioned_non_alias_gemini_ids_get_no_thinking_level():
    # Only a version >= 3 or a -latest alias earns a level; unknown ids keep the no-guess default.
    assert _build_gemini_thinking_config("gemini-exp-1206", {"effort": "high"}) == {"includeThoughts": True}
    assert _build_gemini_thinking_config("gemini-1.5-pro", {"effort": "high"}) == {"includeThoughts": True}
