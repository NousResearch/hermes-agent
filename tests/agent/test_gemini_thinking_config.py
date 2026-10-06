"""Gemini thinking controls use verified levels; none is the minimum, not off."""

import pytest

from agent.transports.chat_completions import (
    _build_gemini_thinking_config,
    _snake_case_gemini_thinking_config,
)


@pytest.mark.parametrize("model,lowest", [
    ("gemini-3-flash-preview", "minimal"),
    ("gemini-3.1-pro-preview", "low"),
    ("gemini-2.5-flash", None),
    ("gemini-flash-latest", None),
    ("gemini-3.1-pro", None),
])
def test_disabled_reasoning_uses_verified_minimum_or_model_defaults(model, lowest):
    for reasoning in ({"enabled": False}, {"effort": "none"}):
        config = _build_gemini_thinking_config(model, reasoning)
        if lowest is None:
            assert config is None
        else:
            assert config == {"includeThoughts": False, "thinkingLevel": lowest}


def test_enabled_reasoning_uses_levels_only_on_verified_models():
    assert _build_gemini_thinking_config("gemini-3-flash-preview", {"effort": "medium"}) == {
        "includeThoughts": True, "thinkingLevel": "medium"}
    for model in ("gemini-2.5-flash", "gpt-4o", "gemma-2b"):
        assert _build_gemini_thinking_config(model, {"effort": "medium"}) is None


def test_snake_case_translation_carries_thinking_level():
    translated = _snake_case_gemini_thinking_config({"includeThoughts": False, "thinkingLevel": "minimal"})
    assert translated == {"include_thoughts": False, "thinking_level": "minimal"}
