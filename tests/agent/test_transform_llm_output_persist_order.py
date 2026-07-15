from unittest.mock import MagicMock, patch
import pytest

from agent.turn_finalizer import finalize_turn

def test_transform_llm_output_runs_before_persist(monkeypatch):
    call_order = []
    persisted_messages = []

    # Mock invoke_hook
    def mock_invoke_hook(hook_name, **kwargs):
        call_order.append(("invoke_hook", hook_name))
        if hook_name == "transform_llm_output":
            return ["Transformed Content"]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", mock_invoke_hook)

    # Setup agent mock with all accessed attributes
    agent = MagicMock()
    agent.session_id = "test-session-id"
    agent.model = "test-model"
    agent.provider = "test-provider"
    agent.max_iterations = 5
    agent.quiet_mode = True
    agent.session_input_tokens = 0
    agent.session_output_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_reasoning_tokens = 0
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_estimated_cost_usd = 0.0
    agent.session_cost_status = "ok"
    agent.session_cost_source = "cache"
    agent.base_url = "http://localhost:8000"
    agent.request_overrides = {}
    agent.context_compressor = MagicMock()
    agent.context_compressor.last_prompt_tokens = 0
    agent._tool_guardrail_halt_decision = None
    agent._drain_pending_steer.return_value = None
    agent._interrupt_message = None
    agent.clear_interrupt = MagicMock()
    agent._stream_callback = None
    agent._skill_nudge_interval = 0
    agent._iters_since_skill = 0
    agent.valid_tool_names = []
    agent._memory_nudge_interval = 0
    agent._turns_since_memory = 0
    agent._user_turn_count = 0
    agent.iteration_budget = MagicMock()
    agent.iteration_budget.remaining = 99

    def mock_persist_session(messages, history):
        call_order.append(("persist_session",))
        persisted_messages.append([dict(m) for m in messages])

    agent._persist_session = mock_persist_session
    agent._file_mutation_verifier_enabled.return_value = False
    agent._turn_completion_explainer_enabled.return_value = False

    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Original Content"}
    ]

    result = finalize_turn(
        agent=agent,
        final_response="Original Content",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=messages,
        conversation_history=[],
        effective_task_id="test-task-id",
        turn_id="test-turn-id",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="stop",
    )

    # Verify that the hook was called before persist_session
    assert ("invoke_hook", "transform_llm_output") in call_order
    assert ("persist_session",) in call_order
    assert call_order.index(("invoke_hook", "transform_llm_output")) < call_order.index(("persist_session",))

    # Verify that the persisted message and final_response were updated
    assert result["final_response"] == "Transformed Content"
    assert result["response_transformed"] is True

    assert len(persisted_messages) == 1
    assert persisted_messages[0][-1] == {"role": "assistant", "content": "Transformed Content"}
