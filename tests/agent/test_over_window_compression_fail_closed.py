"""An over-window session that compression cannot shrink must not block the host toward the
compression ceiling or send a request the model cannot accept (#116472).

Two invariants: turn-start preflight ends the turn at once with ``/new`` guidance when a pass made
no progress on a request still above the model window (a fitting or unknown-window request keeps
the send-as-is behaviour); and the pre-commit wait for an over-window request is bounded by ONE
inactivity budget (``compression.context_timeout_seconds``) instead of the generic ceiling, so a
summary that keeps streaming while reclaiming nothing cannot hold the session for minutes.
"""

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_compression import run_compress_context_with_progress_timeout
from agent.turn_context import PreflightCompressionTimedOut
from agent.turn_context_compaction import CompactionOutcome, _run_preflight_passes


def _agent(*, context_length):
    compressor = SimpleNamespace(
        protect_first_n=3, protect_last_n=3, threshold_tokens=65_536, context_length=context_length,
        summary_target_ratio=0.5, should_compress=lambda tokens: tokens >= 65_536,
    )
    agent = SimpleNamespace(
        context_compressor=compressor, session_id="s1", model="m", max_compression_attempts=3,
        _emit_status=MagicMock(), _compression_blocked_transient=None,
    )
    agent._compress_context = lambda msgs, system, **kw: (msgs, system)  # a pass that reclaims nothing
    return agent


def _preflight(agent, request_tokens):
    out = CompactionOutcome(
        messages=[{"role": "user", "content": "hi"}], active_system_prompt="sys",
        conversation_history=None, current_turn_user_idx=0,
    )
    with patch("agent.turn_context._preflight_request_tokens", return_value=request_tokens), patch(
        "agent.turn_context_compaction.automatic_compaction_status_message", return_value=""
    ):
        _run_preflight_passes(agent, out, agent.context_compressor, request_tokens, "sys", "t")
    return out


def test_no_progress_preflight_fails_closed_only_when_the_request_exceeds_the_window():
    with pytest.raises(PreflightCompressionTimedOut, match="Start a new session with /new"):
        _preflight(_agent(context_length=131_072), 356_113)

    # Over threshold but inside the window: sent as-is (blocked so the loop does not retry).
    assert _preflight(_agent(context_length=1_000_000), 356_113).blocked is True
    # Window unresolvable (0): never guess — the previous behaviour stands.
    assert _preflight(_agent(context_length=0), 356_113).blocked is True
    # The pass no-op'd on a transient guard (summary-failure cooldown): a defer, not proof.
    cooling = _agent(context_length=131_072)
    cooling._compression_blocked_transient = "cooldown:42"
    assert _preflight(cooling, 356_113).blocked is True


def test_over_window_wait_is_bounded_by_one_inactivity_budget():
    msgs = [{"role": "user", "content": "x"}]

    def trickle(fence):
        end = time.monotonic() + 5
        while time.monotonic() < end and not (fence.is_cancelled or fence.deadline_exceeded):
            fence.touch_progress()  # a token every 50ms: never idle, never committing
            time.sleep(0.05)
        return msgs, ""

    def waited(*, request_exceeds_window):
        started = time.monotonic()
        result = run_compress_context_with_progress_timeout(
            worker=trickle, messages=msgs, system_prompt_fallback="sys", idle_timeout_seconds=0.4,
            total_ceiling_seconds=2.0, request_exceeds_window=request_exceeds_window, stall_fallback=False,
        )
        assert result[0] is msgs
        return time.monotonic() - started

    assert waited(request_exceeds_window=True) < 1.5   # one idle budget (0.4s), not the 2s ceiling
    assert waited(request_exceeds_window=False) >= 1.9  # a fitting request keeps the full ceiling
