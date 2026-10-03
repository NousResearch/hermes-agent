"""Over-limit pre-API dispatch must be pressure-based, not identity.

NO request whose estimated context remains at or over
``context_compressor.context_length`` may reach the provider merely because
compression was in cooldown, failed, skipped, made no progress, returned a
new list, or only partially shrank. Same-object over-limit no-op already
fail-closes; this file pins the remaining holes on the real
``run_conversation`` seam with an inert provider.

Lock-skip still defers (may send). Noisy-estimate deferral is left armed.
No real 180s/300s waits.
"""

from __future__ import annotations

from unittest.mock import patch

from tests.run_agent.test_413_compression import _mock_response, agent  # noqa: F401


CONTEXT_LENGTH = 272_000
THRESHOLD_TOKENS = 130_000
OVER_LIMIT = 302_000
UNDER_LIMIT = 10_000
SHRUNK_UNDER = 80_000
PARTIAL_OVER = 280_000


def _patch_estimates(over_limit_tokens: int, *, turn_context_tokens: int = 10_000):
    return (
        patch(
            "agent.turn_context.estimate_request_tokens_rough",
            return_value=turn_context_tokens,
        ),
        patch(
            "agent.conversation_loop.estimate_request_tokens_rough",
            return_value=over_limit_tokens,
        ),
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=over_limit_tokens,
        ),
    )


def _run_turn(agent, *, history, compress, cooldown=None, estimates=OVER_LIMIT):
    agent.context_compressor.context_length = CONTEXT_LENGTH
    agent.context_compressor.threshold_tokens = THRESHOLD_TOKENS
    agent.client.chat.completions.create.return_value = _mock_response(
        content="inert provider boundary"
    )
    patches = _patch_estimates(estimates)
    with (
        patches[0],
        patches[1],
        patches[2],
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=False,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value=cooldown,
        ),
        patch.object(
            agent.context_compressor,
            "should_compress",
            return_value=True,
        ),
        patch.object(
            agent.context_compressor,
            "should_compress_info",
            return_value=(True, "summary_failure_cooldown"),
        ),
        patch.object(agent, "_compress_context", side_effect=compress) as compress_mock,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=history)
    return result, compress_mock, agent.client.chat.completions.create.call_count


def _history():
    return [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]


def test_below_limit_dispatch_allowed(agent):
    """Cell 1: under the resolved window, a normal turn may reach the provider."""
    agent.context_compressor.context_length = CONTEXT_LENGTH
    agent.context_compressor.threshold_tokens = THRESHOLD_TOKENS
    agent.client.chat.completions.create.return_value = _mock_response(
        content="inert provider boundary"
    )
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=UNDER_LIMIT),
        patch(
            "agent.conversation_loop.estimate_request_tokens_rough",
            return_value=UNDER_LIMIT,
        ),
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=UNDER_LIMIT,
        ),
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=False,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value=None,
        ),
        patch.object(agent, "_compress_context") as compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_history())
    assert compress.call_count == 0
    assert agent.client.chat.completions.create.call_count == 1
    assert result.get("failed") is not True
    assert result.get("completed") is True


def test_same_object_over_limit_noop_blocked(agent):
    """Cell 2: timeout/failure no-op that returns the same list must not send."""

    def no_shrink(messages, *args, **kwargs):
        return messages, agent._cached_system_prompt

    result, compress, dispatched = _run_turn(
        agent, history=_history(), compress=no_shrink
    )
    assert compress.call_count == 1
    assert dispatched == 0
    assert result.get("failed") is True


def test_active_failure_cooldown_over_limit_blocked(agent):
    """Cell 3: failure cooldown skips compaction but must not send over-limit."""

    def no_shrink(messages, *args, **kwargs):
        return messages, agent._cached_system_prompt

    result, compress, dispatched = _run_turn(
        agent,
        history=_history(),
        compress=no_shrink,
        cooldown={"reason": "test-summary-failure"},
    )
    assert compress.call_count == 0
    assert dispatched == 0
    assert result.get("failed") is True


def test_compression_failure_over_limit_blocked(agent):
    """Cell 4: a failed compaction that leaves pressure over the window must not send."""

    def failed_compress(messages, *args, **kwargs):
        return messages, agent._cached_system_prompt

    result, compress, dispatched = _run_turn(
        agent, history=_history(), compress=failed_compress
    )
    assert compress.call_count == 1
    assert dispatched == 0
    assert result.get("failed") is True


def test_new_list_no_meaningful_shrink_blocked(agent):
    """Cell 5: a new list with the same messages is not compression success."""

    def copy_list(messages, *args, **kwargs):
        return list(messages), agent._cached_system_prompt

    result, compress, dispatched = _run_turn(
        agent, history=_history(), compress=copy_list
    )
    assert compress.call_count >= 1
    assert dispatched == 0
    assert result.get("failed") is True


def test_partial_shrink_still_over_safe_limit_blocked(agent):
    """Cell 6: shrinking some tokens but remaining over context_length must not send."""

    def partial(messages, *args, **kwargs):
        return [{"role": "user", "content": "PARTIAL"}], agent._cached_system_prompt

    def remaining(msgs, *args, **kwargs):
        blob = str(msgs)
        if "PARTIAL" in blob:
            return PARTIAL_OVER
        return OVER_LIMIT

    agent.context_compressor.context_length = CONTEXT_LENGTH
    agent.context_compressor.threshold_tokens = THRESHOLD_TOKENS
    agent.client.chat.completions.create.return_value = _mock_response(
        content="inert provider boundary"
    )
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
        patch(
            "agent.conversation_loop.estimate_request_tokens_rough",
            side_effect=remaining,
        ),
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            side_effect=remaining,
        ),
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=False,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value=None,
        ),
        patch.object(agent.context_compressor, "should_compress", return_value=True),
        patch.object(
            agent.context_compressor,
            "should_compress_info",
            return_value=(True, None),
        ),
        patch.object(agent, "_compress_context", side_effect=partial) as compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_history())
    assert compress.call_count >= 1
    assert agent.client.chat.completions.create.call_count == 0
    assert result.get("failed") is True


def test_successful_shrink_below_safe_limit_allowed(agent):
    """Cell 7: a real shrink under context_length may continue to the provider."""

    def shrink(messages, *args, **kwargs):
        return [{"role": "user", "content": "SHRUNK"}], agent._cached_system_prompt

    def remaining(msgs, *args, **kwargs):
        blob = str(msgs)
        if "SHRUNK" in blob:
            return SHRUNK_UNDER
        return OVER_LIMIT

    agent.context_compressor.context_length = CONTEXT_LENGTH
    agent.context_compressor.threshold_tokens = THRESHOLD_TOKENS
    agent.client.chat.completions.create.return_value = _mock_response(
        content="inert provider boundary"
    )
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
        patch(
            "agent.conversation_loop.estimate_request_tokens_rough",
            side_effect=remaining,
        ),
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            side_effect=remaining,
        ),
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=False,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value=None,
        ),
        patch.object(
            agent.context_compressor,
            "should_compress",
            side_effect=lambda tokens: tokens >= THRESHOLD_TOKENS,
        ),
        patch.object(
            agent.context_compressor,
            "should_compress_info",
            side_effect=lambda tokens: (
                tokens >= THRESHOLD_TOKENS,
                None,
            ),
        ),
        patch.object(agent, "_compress_context", side_effect=shrink) as compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_history())
    assert compress.call_count == 1
    assert agent.client.chat.completions.create.call_count == 1
    assert result.get("failed") is not True
    assert result.get("completed") is True


def test_exhausted_attempts_over_limit_blocked(agent):
    """Attempt budget exhaustion must not become an over-limit send."""
    agent.max_compression_attempts = 0

    def no_shrink(messages, *args, **kwargs):
        return messages, agent._cached_system_prompt

    result, compress, dispatched = _run_turn(
        agent, history=_history(), compress=no_shrink
    )
    assert compress.call_count == 0
    assert dispatched == 0
    assert result.get("failed") is True


def test_lock_skip_over_limit_still_defers_to_provider_path(agent):
    """Lock-skip still defers preflight (does not burn the attempt budget),
    but that deferral is not a dispatch-safety exemption. An over-limit
    assembled request must not reach the provider."""

    def lock_skip(messages, *args, **kwargs):
        agent._compression_skipped_due_to_lock = "pid=1:tid=2:agent=aa:nonce=bb"
        return messages, agent._cached_system_prompt

    agent.context_compressor.context_length = CONTEXT_LENGTH
    agent.context_compressor.threshold_tokens = THRESHOLD_TOKENS
    agent.client.chat.completions.create.return_value = _mock_response(
        content="inert provider boundary"
    )
    with (
        patch("agent.turn_context.estimate_request_tokens_rough", return_value=10_000),
        patch(
            "agent.conversation_loop.estimate_request_tokens_rough",
            return_value=OVER_LIMIT,
        ),
        patch(
            "agent.conversation_loop.estimate_messages_tokens_rough",
            return_value=OVER_LIMIT,
        ),
        patch(
            "agent.conversation_compression.estimate_request_tokens_rough",
            return_value=OVER_LIMIT,
        ),
        patch.object(
            agent.context_compressor,
            "should_defer_preflight_to_real_usage",
            return_value=False,
        ),
        patch.object(
            agent.context_compressor,
            "get_active_compression_failure_cooldown",
            return_value=None,
        ),
        patch.object(agent.context_compressor, "should_compress", return_value=True),
        patch.object(
            agent.context_compressor,
            "should_compress_info",
            return_value=(True, None),
        ),
        patch.object(agent, "_compress_context", side_effect=lock_skip) as compress,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("hello", conversation_history=_history())
    assert compress.call_count == 1
    assert agent.client.chat.completions.create.call_count == 0
    assert result.get("failed") is True
    assert not result.get("compression_exhausted")
