"""Cooperative limits through current config, phase dispatch and finalization."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from agent.usage_limits import UsageLimitsConfig, TurnUsageTracker
from agent.agent_init import _apply_display_config
from agent import conversation_loop as loop
from agent.turn_iteration_prep import begin_iteration
from agent.turn_finalizer import finalize_turn

@pytest.mark.parametrize("bad", [True, False, float("inf"), -float("inf"), float("nan"), "bad", None])
def test_config_invalid_field_does_not_discard_other_limits(bad):
    agent = SimpleNamespace()
    _apply_display_config(agent, {"usage_limits": {"turn_total_tokens": bad, "session_total_tokens": "20"}}, "cli")
    assert agent._usage_limits_config.turn_total_tokens == 0
    assert agent._usage_limits_config.session_total_tokens == 20

@pytest.mark.parametrize("mode", ["turn_total_tokens", "session_total_tokens", "turn_wall_clock_seconds"])
def test_phase_boundary_and_real_finalizer(mode, monkeypatch):
    agent = MagicMock()
    _apply_display_config(agent, {"usage_limits": {mode: 10}}, "cli")
    agent._interrupt_requested = False
    agent._review_input_token_budget = None
    agent._drain_pending_redirect.return_value = None
    agent._budget_grace_call = False
    agent.quiet_mode = True
    agent.session_total_tokens = 0
    agent.max_iterations = 100
    agent.iteration_budget.remaining = 100
    agent.iteration_budget.consume.return_value = True
    now = [0]
    state = SimpleNamespace(messages=[], conversation_history=[], original_user_message="task", api_call_count=0, interrupted=False, _turn_exit_reason="unknown", _usage_tracker=TurnUsageTracker(agent._usage_limits_config, 0, clock=lambda: now[0]))
    assert loop._run_phase(begin_iteration, agent, state).action == "fallthrough"
    # The crossing operation has finished; no attempt to cancel its in-flight work.
    agent.session_total_tokens = 10
    now[0] = 10
    assert loop._run_phase(begin_iteration, agent, state).action == "break"
    assert state.api_call_count == 1
    agent.iteration_budget.consume.assert_called_once()
    agent.context_compressor.last_prompt_tokens = 0
    agent._tool_guardrail_halt_decision = None
    agent._skill_nudge_interval = 0
    agent.skip_background_review = True
    agent._persist_disabled = True
    agent._drain_pending_steer.return_value = None
    agent._last_streamed_text = ""
    agent._response_was_previewed = False
    agent.platform = "cli"
    agent.request_overrides = {}
    agent._turn_failed_file_mutations = {}
    agent._turn_completion_explainer_enabled.return_value = False
    result = finalize_turn(agent, final_response=None, api_call_count=1, interrupted=False, failed=False, messages=[], conversation_history=[], effective_task_id=None, turn_id="t", user_message="task", original_user_message="task", _should_review_memory=False, _turn_exit_reason=state._turn_exit_reason)
    assert result["turn_exit_reason"].startswith("usage_limit_")
    assert result["completed"] is False
    assert result["partial"] is True
    assert "usage limit" in result["final_response"]
    agent._handle_max_iterations.assert_not_called()
    if mode == "session_total_tokens":
        # Exercise the actual orchestrator, state construction and public result envelope.
        ctx = {name.lstrip("_"): None for name in loop._CTX_FIELDS}
        ctx.update(user_message="task", original_user_message="task", messages=[{"role": "user", "content": "task"}], conversation_history=[], current_turn_user_idx=0, turn_id="turn-1", should_review_memory=False)
        monkeypatch.setattr(loop, "build_turn_context", lambda *a, **kw: SimpleNamespace(**ctx))
        monkeypatch.setattr(loop, "begin_fast_mode_turn", lambda *a: None)
        agent.api_mode = "chat_completions"
        agent._current_turn_user_idx = 0
        agent._current_turn_id = "turn-1"
        result = loop.run_conversation(agent, "task", moa_config={})
        assert result["api_calls"] == 0
        assert result["turn_exit_reason"] == "usage_limit_session_tokens"
        assert result["completed"] is False
        assert "usage limit" in result["final_response"]
        agent._handle_max_iterations.assert_not_called()
