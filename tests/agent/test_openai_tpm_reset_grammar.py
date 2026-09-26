"""OpenAI's "Please try again in Ns" 429 grammar feeds the reset/cooldown datum.

Without the grammar the same 429 reads as "no declared reset": the primary cooldown falls back to
the 60 s exponential guess and ``fallback.min_switch_reset_seconds`` can never defer the model
switch it exists to defer (the deferral needs a provider-declared future reset).
"""

import time
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error
from agent.retry_utils import reset_delay_from_message


class MockAPIError(Exception):
    """Minimal OpenAI-SDK-shaped APIStatusError."""

    def __init__(self, message, status_code=None, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


def _openai_tpm_message(suffix: str) -> str:
    return (
        "Rate limit reached for gpt-6-luna in organization org-T8tvh6eTUm4ObxoCkkkHFIaz on tokens "
        f"per min (TPM): Limit 200000, Used 67806, Requested 152604. Please try again in {suffix}. "
        "Visit https://platform.openai.com/account/rate-limits to learn more."
    )


@pytest.mark.parametrize(
    ("suffix", "expected"),
    [
        ("6.123s", 6.123),
        ("42.746s", 42.746),
        ("1.5 seconds", 1.5),
        ("2 minutes", 120.0),
        ("250ms", 0.25),
    ],
)
def test_try_again_in_grammar_parses(suffix, expected):
    assert reset_delay_from_message(_openai_tpm_message(suffix)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "message",
    [
        # No number: a hedged "in a moment" carries no window and must not invent one.
        "Rate limit reached on tokens per min (TPM). Please try again in a moment.",
        "Rate limit reached on tokens per min (TPM). Please try again in a few seconds.",
        # Number without a unit is ambiguous — refuse rather than guess.
        "Rate limit reached. Please try again in 30.",
    ],
)
def test_try_again_in_grammar_ignores_hedges(message):
    assert reset_delay_from_message(message) is None


def test_explicit_retry_after_still_wins_over_try_again_in():
    """The table's precedence contract: an explicit "retry after N s" is the shorter, asked-for wait."""
    message = "Please retry after 3s. Rate limit reached: try again in 9s."
    assert reset_delay_from_message(message) == pytest.approx(3.0)


def test_classifier_carries_reset_at_for_openai_tpm_429():
    """Production entry: the parsed window must land in ``error_context['reset_at']``, which is the
    datum the primary cooldown and ``switch_deferred_by_reset`` read."""
    error = MockAPIError(_openai_tpm_message("6.123s"), status_code=429)
    before = time.time()
    classified = classify_api_error(error, provider="openai", model="gpt-6-luna")
    assert classified.reason == FailoverReason.rate_limit
    assert classified.should_fallback is True
    reset_at = classified.error_context.get("reset_at")
    assert reset_at is not None, "reset_at missing — the cooldown falls back to the 60 s exponential guess"
    assert before + 5.9 <= reset_at <= before + 6.6
