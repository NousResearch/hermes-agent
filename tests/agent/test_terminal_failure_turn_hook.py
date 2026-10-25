"""Provider retry exhaustion must still finish the turn lifecycle (#137357)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from tests.agent.test_turn_finalizer_final_response_persistence import FakeAgent


def test_retry_exhaustion_fires_one_failed_session_end_hook():
    """The retry-loop's terminal result reaches the normal end-of-turn hook."""
    from agent.conversation_loop import _finalize_terminal_api_result

    agent = FakeAgent()
    agent.platform = "telegram"
    state = SimpleNamespace(
        final_response=None,
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "hello"}],
        conversation_history=[],
        effective_task_id="turn-task",
        turn_id="turn-1",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="api_error",
    )
    terminal_result = {
        "final_response": "The provider stayed unavailable.",
        "messages": state.messages,
        "api_calls": 1,
        "completed": False,
        "failed": True,
        "failure_reason": "server_error",
        "failure_retryable": True,
    }

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]) as invoke_hook:
        result = _finalize_terminal_api_result(agent, state, terminal_result)

    end_calls = [call for call in invoke_hook.call_args_list if call.args[0] == "on_session_end"]
    assert result["failed"] is True
    assert result["failure_reason"] == "server_error"
    assert len(end_calls) == 1
    assert end_calls[0].kwargs == {
        "session_id": "sess-test",
        "task_id": "turn-task",
        "turn_id": "turn-1",
        "completed": False,
        "failed": True,
        "interrupted": False,
        "turn_exit_reason": "api_error",
        "model": "test-model",
        "platform": "telegram",
    }
