"""Regression tests for issue #27405.

The preflight compression gate must trigger when *either* the message
count exceeds the protected ranges OR the cheap char-based token
estimate already crosses the configured threshold. Pre-fix, only the
message-count condition was checked, so a session with a small number
of huge messages would silently skip compression and eventually hit a
hard context-overflow error.
"""

from agent.turn_context import _should_run_preflight_estimate


# Protected-range counts mirror the compressor defaults. THRESHOLD_TOKENS is an
# arbitrary test threshold passed explicitly into the helper — it is NOT the
# live runtime threshold (which is max(0.5*window, MINIMUM_CONTEXT_LENGTH) per
# model); the helper takes the threshold as a parameter so the tests are
# self-contained and independent of model metadata.
PROTECT_FIRST_N = 3
PROTECT_LAST_N = 20
THRESHOLD_TOKENS = 64_000


def _msg(content: str) -> dict:
    return {"role": "user", "content": content}


def test_few_messages_huge_content_triggers_gate():
    """The bug from #27405: 8 messages with one massive content blob."""
    # ~280K chars in one message ~= 70K tokens at 4 chars/token.
    big = "x" * 280_000
    messages = [_msg("hi")] * 7 + [_msg(big)]
    assert len(messages) <= PROTECT_FIRST_N + PROTECT_LAST_N + 1  # would fail old gate
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True






def test_content_above_threshold_triggers():
    """A single message comfortably above the threshold trips branch (b)."""
    # ~threshold*4 chars => ~threshold tokens; +1000 tokens of margin so the
    # test doesn't depend on per-message dict-wrapping overhead in the
    # shared estimator's (chars+3)//4 rounding.
    messages = [_msg("x" * ((THRESHOLD_TOKENS + 1000) * 4))]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_content_below_threshold_does_not_trigger():
    """A single message comfortably below the threshold (and few messages)
    must not trigger — the estimator stays under and the count gate is not
    tripped."""
    messages = [_msg("x" * ((THRESHOLD_TOKENS - 1000) * 4))]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False






