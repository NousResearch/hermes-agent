"""Opt-in soft usage limits (agent/usage_limits.py) through the real loop and finalizer."""
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent import conversation_loop as loop
from agent.turn_finalizer import finalize_turn
from agent.usage_limits import TurnUsageTracker, UsageLimitsConfig


@pytest.mark.parametrize("key, code, tokens, cost", [
    ("turn_wall_clock_seconds", "usage_limit_wall_clock", 0, 0.0),
    ("turn_total_tokens", "usage_limit_turn_tokens", 10, 0.0),
    ("session_total_tokens", "usage_limit_session_tokens", 10, 0.0),
    ("session_cost_usd", "usage_limit_session_cost", 0, 10.0),
])
def test_each_limit_trips_at_its_ceiling_and_junk_disables(key, code, tokens, cost):
    now = [0.0]
    tracker = TurnUsageTracker(UsageLimitsConfig.from_mapping({key: 10}), 0, clock=lambda: now[0])
    now[0] = 9.0
    assert tracker.check(max(tokens - 1, 0), max(cost - 0.01, 0.0)) is None
    now[0] = 10.0
    breach = tracker.check(tokens, cost)
    assert breach is not None and breach.code == code
    assert f"usage_limits.{key}" in breach.message and "continue" in breach.message
    # A YAML typo (bool/inf/nan/text/negative) disables that field instead of tripping at once.
    for junk in (True, float("inf"), float("nan"), "abc", -5):
        assert not UsageLimitsConfig.from_mapping({key: junk}).enabled


def _loop_agent():
    agent = MagicMock()
    agent._voice_turn_pending = False
    agent._voice_route_state = None
    agent._interrupt_requested = False
    agent._review_input_token_budget = None
    agent._drain_pending_redirect.return_value = None
    agent._budget_grace_call = False
    agent.quiet_mode = True
    agent.session_total_tokens = 0
    agent.max_iterations = 100
    agent.iteration_budget.remaining = 100
    agent.context_compressor.last_prompt_tokens = 0
    agent._tool_guardrail_halt_decision = None
    agent._skill_nudge_interval = 0
    agent.skip_background_review = True
    agent._persist_disabled = True
    agent._drain_pending_steer.return_value = None
    agent._last_streamed_text = ""
    agent._response_was_previewed = False
    agent._reused_response_text = None
    agent.platform = "cli"
    agent.request_overrides = {}
    agent._turn_failed_file_mutations = {}
    agent._turn_completion_explainer_enabled.return_value = False
    return agent


def test_session_over_cost_limit_stops_before_any_model_call(monkeypatch):
    agent = _loop_agent()
    agent.session_estimated_cost_usd = 5.02
    agent.api_mode = "chat_completions"
    agent._current_turn_user_idx = 0
    agent._current_turn_id = "turn-1"
    ctx: dict[str, Any] = {name.lstrip("_"): None for name in loop._CTX_FIELDS}
    ctx.update(user_message="task", original_user_message="task", messages=[{"role": "user", "content": "task"}],
               conversation_history=[], current_turn_user_idx=0, turn_id="turn-1", should_review_memory=False)
    monkeypatch.setattr(loop, "build_turn_context", lambda *a, **kw: SimpleNamespace(**ctx))
    monkeypatch.setattr(loop, "begin_fast_mode_turn", lambda *a: None)
    monkeypatch.setattr(loop, "current_usage_limits", lambda: UsageLimitsConfig(session_cost_usd=5.0))

    result = loop.run_conversation(agent, "task", moa_config={})

    assert result["api_calls"] == 0
    assert result["turn_exit_reason"] == "usage_limit_session_cost"
    assert result["completed"] is False and result["partial"] is True
    assert "usage_limits.session_cost_usd" in result["final_response"]
    agent._handle_max_iterations.assert_not_called()


@pytest.mark.parametrize("previewed", [False, True])
def test_pending_verification_answer_survives_a_limit_stop(previewed):
    agent = _loop_agent()
    result = finalize_turn(
        agent, final_response=None, api_call_count=0, interrupted=False, failed=False, messages=[],
        conversation_history=[], effective_task_id=None, turn_id="t", user_message="task",
        original_user_message="task", _should_review_memory=False, _turn_exit_reason="usage_limit_session_cost",
        _pending_verification_response="Verified candidate text",
        _pending_verification_response_previewed=previewed)
    assert result["final_response"] == "Verified candidate text"
    assert result["response_previewed"] is previewed
    assert result["completed"] is False and result["partial"] is True
    agent._handle_max_iterations.assert_not_called()
