"""Tests for the Copilot catalog max_prompt_tokens under-report correction.

GitHub's Copilot /models catalog advertises ``max_prompt_tokens: 200000`` for
the Claude family, but the API enforces the true ~1M window. These tests pin the
correction helper's behaviour and its deliberately narrow, tier-safe scope.
"""
from __future__ import annotations

import pytest

from hermes_cli.models import (
    _correct_copilot_max_prompt,
    _COPILOT_TRUE_PROMPT_WINDOW,
    _COPILOT_UNDERREPORTED_PROMPT,
)


@pytest.mark.parametrize(
    "model_id",
    [
        "claude-opus-4.6",
        "claude-opus-4.7",
        "claude-opus-4.8",
        "claude-sonnet-4.6",
        "CLAUDE-OPUS-4.8",  # case-insensitive
    ],
)
def test_corrects_known_1m_models_at_underreported_value(model_id: str) -> None:
    """The exact verified-wrong 200000 is lifted to the true 1M window."""
    assert (
        _correct_copilot_max_prompt(model_id, _COPILOT_UNDERREPORTED_PROMPT)
        == _COPILOT_TRUE_PROMPT_WINDOW
    )


@pytest.mark.parametrize(
    "model_id, value",
    [
        # Non-underreported values pass through untouched, even for known models —
        # we only correct the EXACT value we proved false.
        ("claude-opus-4.8", 168000),
        ("claude-opus-4.8", 1000000),
        ("claude-opus-4.8", 264000),
        # Unknown / unmatched models are never inflated, even at 200000.
        ("gpt-5.5", _COPILOT_UNDERREPORTED_PROMPT),
        ("gemini-2.5-pro", _COPILOT_UNDERREPORTED_PROMPT),
        ("claude-haiku-4.5", _COPILOT_UNDERREPORTED_PROMPT),
        ("claude-sonnet-4.5", _COPILOT_UNDERREPORTED_PROMPT),
        # Older opus that was never verified at 1M is left alone.
        ("claude-opus-4.5", _COPILOT_UNDERREPORTED_PROMPT),
    ],
)
def test_leaves_unmatched_or_non_underreported_values_untouched(
    model_id: str, value: int
) -> None:
    """Tier-safety: only the exact (known-model, 200000) pair is corrected."""
    assert _correct_copilot_max_prompt(model_id, value) == value


def test_empty_model_id_is_safe() -> None:
    assert (
        _correct_copilot_max_prompt("", _COPILOT_UNDERREPORTED_PROMPT)
        == _COPILOT_UNDERREPORTED_PROMPT
    )
