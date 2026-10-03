"""Final provider-bound request pressure must not exceed the safe limit.

Compression admission exemptions (lock skip, stale lock, cooldown,
awaiting-real-usage, recent-real anchor, compression-disabled,
review-first, exhausted attempts) are not dispatch-safety exemptions.
Raw transcript fit is not sufficient: the fully assembled provider
payload — including ephemeral system text, retained post-compression
material, and request-middleware growth — is what must fit.

Uses the real estimator. Only the turn-prologue estimate is forced low
so the pre-API loop is reached deterministically. Inert provider.
No 180s/300s waits.
"""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tests.run_agent.test_413_compression import agent, _mock_response  # noqa: F401
from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_messages_tokens_rough

WINDOW = 1000
THRESHOLD = 500


def history(size=4400):
    return [
        {"role": "user", "content": "x" * size},
        {"role": "assistant", "content": "previous answer"},
    ]


def run_local(a, *, messages, compress=None, cooldown=None, middleware=None):
    a.tools = []
    a.context_compressor.context_length = WINDOW
    a.context_compressor.threshold_tokens = THRESHOLD
    a.client.chat.completions.create.return_value = _mock_response(
        content="inert boundary"
    )
    if compress is None:
        compress = lambda rows, *args, **kwargs: (rows, a._cached_system_prompt)
    with ExitStack() as stack:
        stack.enter_context(
            patch("agent.turn_context.estimate_request_tokens_rough", return_value=10)
        )
        stack.enter_context(
            patch.object(
                a.context_compressor,
                "get_active_compression_failure_cooldown",
                return_value=cooldown,
            )
        )
        comp = stack.enter_context(patch.object(a, "_compress_context", side_effect=compress))
        status = stack.enter_context(patch.object(a, "_emit_status"))
        stack.enter_context(patch.object(a, "_persist_session"))
        stack.enter_context(patch.object(a, "_save_trajectory"))
        stack.enter_context(patch.object(a, "_cleanup_task_resources"))
        if middleware:
            stack.enter_context(
                patch(
                    "hermes_cli.middleware.apply_llm_request_middleware",
                    side_effect=middleware,
                )
            )
        result = a.run_conversation("hello", conversation_history=messages)
    create = a.client.chat.completions.create
    payloads = [c.kwargs for c in create.call_args_list]
    wire_pressure = [
        estimate_messages_tokens_rough(p.get("messages", [])) for p in payloads
    ]
    texts = [str(c.args[0]) for c in status.call_args_list if c.args]
    return result, create, comp, texts, wire_pressure


def assert_unsafe_stopped(observation):
    result, create, _comp, _texts, pressure = observation
    assert create.call_count == 0, (
        f"unsafe request DISPATCHED: estimated_messages={pressure}, window={WINDOW}"
    )
    assert result.get("failed") is True
    assert result.get("completed") is False


def test_below_limit_final_assembled_request_dispatches(agent):
    """Cell 1: a below-limit fully assembled request may reach the provider."""
    result, create, comp, _texts, pressure = run_local(
        agent, messages=history(200)
    )
    assert create.call_count == 1
    assert pressure[0] < WINDOW
    assert result.get("completed") is True
    assert comp.call_count == 0


def test_successful_compression_final_assembled_below_limit_dispatches(agent):
    """Cell 2: compaction whose FINAL assembled request is under the window may send."""

    def compact(rows, *args, **kwargs):
        agent.context_compressor.last_prompt_tokens = -1
        agent.context_compressor.awaiting_real_usage_after_compression = True
        return [{"role": "user", "content": "s" * 200}], agent._cached_system_prompt

    result, create, comp, _texts, pressure = run_local(
        agent, messages=history(), compress=compact
    )
    assert comp.call_count == 1
    assert create.call_count == 1
    assert pressure[0] < WINDOW
    assert result.get("completed") is True


@pytest.mark.parametrize(
    "mode",
    [
        "lock",
        "stale_lock_cooldown",
        "awaiting_usage",
        "recent_real_anchor",
        "compression_disabled",
        "review_first_request",
        "exhausted_attempts",
    ],
)
def test_admission_exemption_cannot_dispatch_final_over_limit(agent, mode):
    """Cells 3-8 + exhaustion: preflight exemptions are not dispatch-safety waivers."""
    a = agent
    cooldown = None
    compress = None
    if mode == "lock":
        def lock_skip(rows, *args, **kwargs):
            a._compression_skipped_due_to_lock = "other-holder"
            return rows, a._cached_system_prompt
        compress = lock_skip
    elif mode == "stale_lock_cooldown":
        a._compression_skipped_due_to_lock = "stale-prior-holder"
        cooldown = {"reason": "summary-failure"}
    elif mode == "awaiting_usage":
        a.context_compressor.awaiting_real_usage_after_compression = True
    elif mode == "recent_real_anchor":
        a.context_compressor.last_real_prompt_tokens = 100
        a.context_compressor.last_rough_tokens_when_real_prompt_fit = 2000
    elif mode == "compression_disabled":
        a.compression_enabled = False
    elif mode == "review_first_request":
        a._review_defer_compaction_before_first_response = True
    elif mode == "exhausted_attempts":
        a.max_compression_attempts = 0
    assert_unsafe_stopped(
        run_local(a, messages=history(), compress=compress, cooldown=cooldown)
    )


def test_raw_transcript_safe_assembled_unsafe_cannot_dispatch(agent):
    """Cell 9: post-compaction raw transcript can fit while assembled request does not."""
    a = agent
    a.ephemeral_system_prompt = "e" * 2800

    def compact(rows, *args, **kwargs):
        a.context_compressor.last_prompt_tokens = -1
        a.context_compressor.awaiting_real_usage_after_compression = True
        return [{"role": "user", "content": "s" * 2000}], a._cached_system_prompt

    assert_unsafe_stopped(run_local(a, messages=history(), compress=compact))


def test_request_middleware_growth_final_unsafe_cannot_dispatch(agent):
    """Cell 10: middleware may grow the payload after preflight; final must still fail closed."""

    def inflate(payload, **kwargs):
        copied = dict(payload)
        copied["messages"] = list(payload["messages"]) + [
            {"role": "user", "content": "m" * 4400}
        ]
        return SimpleNamespace(payload=copied, original_payload=payload, trace=[])

    assert_unsafe_stopped(
        run_local(agent, messages=history(200), middleware=inflate)
    )


def test_partial_compaction_still_unsafe_blocked_with_truthful_diagnostic(agent):
    """Cell 11: partial shrink can drop transcript yet remain unsafe; do not claim none dropped."""

    def partial(rows, *args, **kwargs):
        return [{"role": "user", "content": "p" * 4100}], agent._cached_system_prompt

    result, create, comp, texts, _pressure = run_local(
        agent, messages=history(), compress=partial
    )
    assert create.call_count == 0
    assert comp.call_count == 1
    assert result.get("failed") is True
    assert len(result["messages"]) < 3
    assert not any("No messages were dropped" in t for t in texts), (
        "partial rewrite is incorrectly described as unchanged/no messages dropped"
    )


def test_codex_shaped_kwargs_count_instructions_and_input():
    """Codex Responses wire uses instructions+input, not chat messages."""
    from agent.conversation_compression import estimate_provider_bound_request_pressure

    over = {
        "model": "gpt-5.5",
        "instructions": "sys " + ("i" * 2000),
        "input": [{"role": "user", "content": "u" * 2500}],
        "tools": [],
    }
    under = {
        "model": "gpt-5.5",
        "instructions": "short",
        "input": [{"role": "user", "content": "hi"}],
        "tools": [],
    }
    assert estimate_provider_bound_request_pressure(over) >= WINDOW
    assert estimate_provider_bound_request_pressure(under) < WINDOW
    chat = {
        "model": "x",
        "messages": [{"role": "user", "content": "u" * 2500}],
    }
    assert estimate_provider_bound_request_pressure(chat) > 0


def test_output_reservation_reduces_compaction_trigger_not_dispatch_ceiling():
    """Cell 17: established #43547 contract.

    ``max_tokens`` is subtracted from ``context_length`` when computing the
    compaction *trigger* (usable input budget). The dispatch-time INPUT
    refuse ceiling remains ``context_length``. Output-cap recovery is a
    separate post-error path that clamps ``max_tokens`` without inventing a
    second pre-dispatch input ceiling.
    """
    reserved = ContextCompressor._compute_threshold_tokens(200_000, 0.50, 40_000)
    unreserved = ContextCompressor._compute_threshold_tokens(200_000, 0.50, None)
    assert reserved < unreserved
    from agent.conversation_compression import resolved_safe_dispatch_limit

    compressor = SimpleNamespace(context_length=1000, max_tokens=200)
    bound = SimpleNamespace(context_compressor=compressor, max_tokens=200)
    assert resolved_safe_dispatch_limit(bound) == 1000
    assert resolved_safe_dispatch_limit(bound) != 1000 - 200


def test_input_under_context_length_with_output_reservation_is_not_input_overflow(agent):
    """Cell 17 companion: 843 input vs context 1000 with max_tokens 200 may send.

    The independent probe expected input <= context-max_tokens (800). That is
    the compaction-trigger budget, not the established dispatch refuse
    ceiling. This pins the source contract rather than adopting the probe.
    """
    agent.max_tokens = 200
    observation = run_local(
        agent,
        messages=history(3200),
        cooldown={"reason": "summary-failure"},
    )
    result, create, _comp, _texts, pressure = observation
    if create.call_count:
        payload = create.call_args.kwargs
        reservation = payload.get("max_tokens", payload.get("max_completion_tokens"))
        assert reservation == 200
        assert all(p < WINDOW for p in pressure)
        assert any(p >= WINDOW - reservation for p in pressure)
    assert create.call_count == 1
    assert result.get("completed") is True
    assert result.get("failed") is not True
