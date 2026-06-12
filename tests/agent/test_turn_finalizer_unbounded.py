"""Regression coverage for unbounded turn finalization."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.iteration_budget import IterationBudget
from agent.turn_finalizer import finalize_turn


def _agent_for_finalizer(max_iterations=0):
    agent = SimpleNamespace()
    agent.max_iterations = max_iterations
    agent.iteration_budget = IterationBudget(max_iterations)
    agent.quiet_mode = True
    agent.model = "test/model"
    agent.provider = "test-provider"
    agent.base_url = "https://example.test"
    agent.session_id = "session-test"
    agent.platform = "cli"
    agent.context_compressor = SimpleNamespace(last_prompt_tokens=0)
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "unknown"
    agent.session_cost_source = "none"
    agent._tool_guardrail_halt_decision = None
    agent._response_was_previewed = False
    agent._interrupt_message = None
    agent._stream_callback = None
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = set()
    agent._emit_status = MagicMock()
    agent._safe_print = MagicMock()
    agent._handle_max_iterations = MagicMock(return_value="summary")
    agent._save_trajectory = MagicMock()
    agent._cleanup_task_resources = MagicMock()
    agent._drop_trailing_empty_response_scaffolding = MagicMock()
    agent._persist_session = MagicMock()
    agent._file_mutation_verifier_enabled = MagicMock(return_value=False)
    agent._turn_completion_explainer_enabled = MagicMock(return_value=False)
    agent._format_turn_completion_explanation = MagicMock(return_value="")
    agent._drain_pending_steer = MagicMock(return_value=None)
    agent.clear_interrupt = MagicMock()
    agent._sync_external_memory_for_turn = MagicMock()
    agent._spawn_background_review = MagicMock()
    return agent


def test_unbounded_turn_finalizer_does_not_treat_high_api_count_as_exhausted():
    agent = _agent_for_finalizer(max_iterations=0)
    messages = [
        {"role": "user", "content": "work"},
        {"role": "assistant", "content": "done"},
    ]

    result = finalize_turn(
        agent,
        final_response="done",
        api_call_count=999,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="task-test",
        turn_id="turn-test",
        user_message="work",
        original_user_message="work",
        _should_review_memory=False,
        _turn_exit_reason="text_response(final)",
    )

    assert result["completed"] is True
    assert result["final_response"] == "done"
    agent._handle_max_iterations.assert_not_called()
