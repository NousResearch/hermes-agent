"""Qualification-only counter-contracts; no live provider calls."""
from unittest.mock import MagicMock
import pytest
from agent.turn_finalizer import finalize_turn

@pytest.mark.parametrize('previewed', [False, True])
@pytest.mark.parametrize('reason', ['usage_limit_session_tokens', 'usage_limit_turn_tokens', 'usage_limit_wall_clock'])
def test_pending_verification_survives_usage_exit(previewed, reason):
    agent = MagicMock()
    agent.max_iterations = 100
    agent.iteration_budget.remaining = 100
    agent.context_compressor.last_prompt_tokens = 0
    agent._tool_guardrail_halt_decision = None
    agent._skill_nudge_interval = 0
    agent.skip_background_review = True
    agent._persist_disabled = True
    agent._drain_pending_steer.return_value = None
    agent._last_streamed_text = ''
    agent._response_was_previewed = False
    agent._reused_response_text = None
    agent.platform = 'cli'
    agent.request_overrides = {}
    agent._turn_failed_file_mutations = {}
    agent._turn_completion_explainer_enabled.return_value = False
    result = finalize_turn(agent, final_response=None, api_call_count=0,
        interrupted=False, failed=False, messages=[], conversation_history=[],
        effective_task_id=None, turn_id='qualification', user_message='task',
        original_user_message='task', _should_review_memory=False,
        _turn_exit_reason=reason, _pending_verification_response='Verified candidate text',
        _pending_verification_response_previewed=previewed)
    assert result['final_response'] == 'Verified candidate text'
    assert result['response_previewed'] is previewed
    assert result['response_reused'] is previewed
    assert result['completed'] is False
    assert result['partial'] is True
    assert result['turn_exit_reason'] == reason
    assert result['api_calls'] == 0
    agent._handle_max_iterations.assert_not_called()
