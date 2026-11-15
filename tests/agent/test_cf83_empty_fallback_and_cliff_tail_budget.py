"""Unit tests for #CF-83:
1. Empty-response fallback arms `_rate_limited_until` via `FailoverReason.overloaded` and sets
   `_empty_fallback_retry_pending = True` so `prepare_iteration` does not cancel the fallback
   in 32ms before the fallback provider is called.
2. `lean_tail_budget` scales `effective_window` with `min(context_length, threshold_tokens * 2)`
   when `threshold_tokens` is set below 0.50 of `context_length` (e.g., 0.15 = 157,286 tokens on
   a 1,048,576-token window), reclaiming 15k-22.5k tokens of post-compaction runway.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent import turn_empty_response as ter
from agent.context_compressor_tail import lean_tail_budget
from agent.error_classifier import FailoverReason
from agent.fallback_cooldown import _arm_rate_limit_cooldown
from agent.turn_iteration_prep import prepare_iteration


def _make_empty_fallback_agent():
    activated_reasons = []

    a = MagicMock()
    a.model = "google/gemini-4-argon"
    a.provider = "openrouter"
    a._primary_runtime = {"provider": "openrouter", "model": "google/gemini-4-argon"}
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.return_value = False
    a._strip_think_blocks.side_effect = lambda t: t or ""
    a._last_content_with_tools = None
    a._thinking_prefill_retries = 2
    a._empty_content_retries = 1
    a._fallback_chain = [{"provider": "vertex", "model": "google/gemini-4-argon"}]
    a._fallback_activated = False
    a._content_filter_fallback_sticky = False
    a._rebuilt_restarts_since_response = 0
    a._rate_limited_until = 0.0
    a._rate_limit_backoff_count = 0
    a._empty_fallback_retry_pending = False
    a._api_call_count = 5
    a._nous_wire_pending = None
    a.step_callback = None
    a._skill_nudge_interval = 0
    a.valid_tool_names = set()
    a._drain_pending_steer.return_value = None
    a.run_budget_seconds = None
    a.session_id = "s-cf83"
    a._sanitize_tool_call_arguments.return_value = 0
    a._last_flushed_db_idx = 0
    a._session_db = None
    a.iteration_budget = SimpleNamespace(remaining=10, refund=MagicMock())
    a.max_iterations = 20

    def _try_activate_fallback(reason=None, reset_at=None):
        activated_reasons.append(reason)
        _arm_rate_limit_cooldown(a, reason, reset_at)
        a.provider = "vertex"
        a._fallback_activated = True
        return True

    a._try_activate_fallback.side_effect = _try_activate_fallback
    a._activated_reasons = activated_reasons
    return a


def test_empty_response_fallback_arms_cooldown_and_survives_prepare_iteration():
    agent = _make_empty_fallback_agent()
    msg = SimpleNamespace(reasoning=None, reasoning_content=None, reasoning_details=None)
    with patch.object(ter, "_retry_empty", return_value=(None, None, False)), \
            patch("agent.conversation_loop._sync_failover_system_message", return_value="sys"):
        verdict = ter.recover_empty_response(
            agent,
            msg,
            None,
            "stop",
            final_response="",
            messages=[{"role": "user", "content": "hi"}],
            api_messages=[],
            conversation_history=[],
            active_system_prompt="sys",
            api_call_count=5,
            turn_exit_reason=None,
            preflight_compression_blocked=False,
        )
    assert verdict.action == "continue"
    assert verdict.api_call_count == 4
    assert agent._activated_reasons == [FailoverReason.overloaded]
    assert agent._rate_limited_until > time.monotonic() + 30.0
    assert agent._empty_fallback_retry_pending is True

    # Simulate the immediate outer-loop re-entry into prepare_iteration on step >= 2.
    prepare_iteration(
        agent,
        messages=[{"role": "user", "content": "hi"}],
        api_call_count=verdict.api_call_count,
        current_turn_user_idx=0,
        user_message="hi",
    )
    agent._restore_primary_runtime.assert_not_called()
    assert agent._empty_fallback_retry_pending is False


def test_cliff_capped_lean_tail_budget_reclaims_runway_without_affecting_defaults():
    # At 0.15 threshold on 1M window (157,286 tokens), effective_window = 314,572 -> 10,000 floor.
    assert lean_tail_budget(1_048_576, 157_286) == 10_000
    # Default 0.50 threshold on 1M window (524,288 tokens) stays at 25,000 cap.
    assert lean_tail_budget(1_048_576, 524_288) == 25_000
    # Default 0.50 threshold on 128k/131k windows stays at 10,000 floor.
    assert lean_tail_budget(131_072, 65_536) == 10_000
