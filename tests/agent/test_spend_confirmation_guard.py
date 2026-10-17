"""A pre-flight spend refusal is about money, not tokens.

Some routers price a call before running it and decline when the estimate is large
against the balance left on the key, offering a confirm flag instead:

    This call is estimated at about $1.87, more than half of your $3.63 balance
    (26,827 prompt tokens + up to 32,000 output tokens). Send "th_confirm_spend":
    true to run it anyway, or reduce the prompt or max_tokens.

That 400 used to classify as ``context_overflow``: its own closing advice contains
"max_tokens", a ``_CONTEXT_OVERFLOW_PATTERNS`` entry, and ``_400_TAIL_RULES`` checks
overflow before billing — while no ``_BILLING_PATTERNS`` entry matches "more than half
of your $3.63 balance". Hermes then compressed the transcript three times, re-sent,
collected the identical 400 each round (the estimate is dominated by the 32,000-token
output budget, which compression never touches) and gave up with "the conversation is
too long for the model (21,101 tokens)" — against a 1,050,000-token context window.

These tests pin the money reading, and pin that genuine overflow still compresses.
"""

from __future__ import annotations

from agent.conversation_loop import (
    _billing_failure_result,
    _billing_terminal_label,
)
from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    is_spend_confirmation_guard,
)


class MockAPIError(Exception):
    def __init__(self, message, status_code=None, body=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


_SPEND_GUARD_BODY = (
    'This call is estimated at about $1.87, more than half of your $3.63 balance '
    '(26,827 prompt tokens + up to 32,000 output tokens). Send "th_confirm_spend": true '
    'to run it anyway, or reduce the prompt or max_tokens.'
)

# The session this arrived on: nowhere near the window, which is the tell.
_SESSION = {"approx_tokens": 28201, "context_length": 1050000, "num_messages": 15}


def _classify(message, *, status_code=400, **overrides):
    kwargs = {**_SESSION, **overrides}
    error = MockAPIError(
        message,
        status_code=status_code,
        body={"error": {"type": "invalid_request_error", "message": message}},
    )
    return classify_api_error(error, provider="custom", model="gpt-6-astra", **kwargs)


# ── Classification ───────────────────────────────────────────────────────────


class TestSpendGuardClassification:
    def test_not_context_overflow(self):
        """The regression itself: no compression may be armed."""
        classified = _classify(_SPEND_GUARD_BODY)
        assert classified.reason is FailoverReason.billing
        assert classified.should_compress is False

    def test_does_not_retry_the_identical_request(self):
        """Same body, same price, same refusal — retrying burns the ladder. A different
        route can still carry the turn, so the fallback chain stays armed."""
        classified = _classify(_SPEND_GUARD_BODY)
        assert classified.retryable is False
        assert classified.should_fallback is True

    def test_marked_spend_guard_and_unverified(self):
        """Funds remain: the pool gets a short cooldown, not a permanent write-off."""
        classified = _classify(_SPEND_GUARD_BODY)
        assert classified.spend_guard is True
        assert classified.billing_unverified is True

    def test_status_less_relay_reads_the_same(self):
        """A relay that strips the status hits _by_message, where the tail rules check
        billing before overflow — but no billing pattern matches this wording."""
        classified = _classify(_SPEND_GUARD_BODY, status_code=None)
        assert classified.reason is FailoverReason.billing
        assert classified.should_compress is False

    def test_confirm_flag_alone_is_enough(self):
        """Another router, same shape, wording we have not seen."""
        assert is_spend_confirmation_guard(
            'refused: set "confirm_spend": true to proceed'
        ) is True

    def test_one_signal_alone_is_not_enough(self):
        """A cost phrase or a balance phrase on its own must not claim the verdict."""
        assert is_spend_confirmation_guard("this will cost more than expected") is False
        assert is_spend_confirmation_guard("your balance is shown in the dashboard") is False


# ── What must not change ─────────────────────────────────────────────────────


class TestNeighbouringVerdicts:
    def test_real_overflow_still_compresses(self):
        classified = _classify(
            "This model's maximum context length is 128000 tokens, however you requested "
            "191875 tokens. Please reduce the length of the messages."
        )
        assert classified.reason is FailoverReason.context_overflow
        assert classified.should_compress is True

    def test_genuine_exhaustion_stays_assertive_billing(self):
        classified = _classify("Your credit balance is too low to access the API.")
        assert classified.reason is FailoverReason.billing
        assert classified.spend_guard is False


# ── What the user is told ────────────────────────────────────────────────────


class TestSpendGuardCopy:
    def test_label_does_not_claim_exhaustion(self):
        label = _billing_terminal_label("boom", True, True)
        assert not label.startswith("Billing or credits exhausted")
        assert "content-filter" not in label  # the #82154 hedge is a different story
        assert "cost" in label

    def test_label_contract_unchanged_without_the_flag(self):
        assert _billing_terminal_label("boom", False) == "Billing or credits exhausted: boom"
        assert "unverified" in _billing_terminal_label("boom", True)

    def test_result_names_the_three_real_fixes(self):
        result = _billing_failure_result(
            classified=_classify(_SPEND_GUARD_BODY),
            summary="HTTP 400: spend confirmation required",
            messages=[],
            api_call_count=1,
            provider="custom",
            base_url="https://router.example/v1",
            model="gpt-6-astra",
        )
        final = result["final_response"]
        assert "max_tokens" in final
        assert "top up" in final.lower()
        assert "confirm" in final.lower()
        # And it must say the thing the old path got exactly backwards.
        assert "Compressing the conversation will not help" in final
        assert result["spend_guard"] is True
        assert result["failure_reason"] == "billing"
        assert result["failure_retryable"] is False
