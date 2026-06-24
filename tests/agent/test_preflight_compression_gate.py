"""Regression tests for issue #27405.

The preflight compression gate must trigger when *either* the message
count exceeds the protected ranges OR the cheap char-based token
estimate already crosses the configured threshold. Pre-fix, only the
message-count condition was checked, so a session with a small number
of huge messages would silently skip compression and eventually hit a
hard context-overflow error.
"""

from agent.turn_context import _should_run_preflight_estimate


# Compressor defaults mirrored in the tests so we don't depend on
# MINIMUM_CONTEXT_LENGTH / model metadata.
PROTECT_FIRST_N = 3
PROTECT_LAST_N = 20
THRESHOLD_TOKENS = 64_000  # matches the production floor


def _msg(content: str) -> dict:
    return {"role": "user", "content": content}


def test_few_messages_huge_content_triggers_gate():
    """The bug from #27405: 8 messages with one massive content blob."""
    # ~280K chars in one message ≈ 70K tokens at 4 chars/token.
    big = "x" * 280_000
    messages = [_msg("hi")] * 7 + [_msg(big)]
    assert len(messages) <= PROTECT_FIRST_N + PROTECT_LAST_N + 1  # would fail old gate
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_few_messages_small_content_does_not_trigger():
    """Regression guard: tiny sessions should not pay the estimator cost."""
    messages = [_msg("hello world")] * 8
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


def test_many_small_messages_still_triggers_via_count():
    """The historical path: > protect_first + protect_last + 1 messages."""
    messages = [_msg("ok")] * (PROTECT_FIRST_N + PROTECT_LAST_N + 2)  # 25
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_content_exactly_at_threshold_triggers():
    """Boundary: approx_tokens >= threshold should trigger (inclusive)."""
    # 4 chars per token, so threshold * 4 chars produces exactly threshold tokens.
    messages = [_msg("x" * (THRESHOLD_TOKENS * 4))]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True


def test_content_just_below_threshold_does_not_trigger():
    """Boundary: one token below threshold and few messages must not trigger."""
    messages = [_msg("x" * ((THRESHOLD_TOKENS - 1) * 4))]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


def test_message_with_none_content_is_treated_as_empty():
    """Assistant turns mid-tool-call carry content=None — must not crash."""
    messages = [{"role": "assistant", "content": None}] * 5
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is False


def test_message_with_list_content_uses_repr_length():
    """Multimodal content lists are handled by str(); approximate is fine.

    The downstream real estimator is authoritative; the gate just needs to
    not crash and to err on the side of admitting suspicious payloads.
    """
    parts = [{"type": "text", "text": "x" * 300_000}]
    messages = [{"role": "user", "content": parts}]
    assert _should_run_preflight_estimate(
        messages, PROTECT_FIRST_N, PROTECT_LAST_N, THRESHOLD_TOKENS
    ) is True
