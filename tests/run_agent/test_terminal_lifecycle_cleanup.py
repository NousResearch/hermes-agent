"""Regression tests for terminal-resource cleanup at the turn boundary."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _agent(cleanup):
    return SimpleNamespace(
        _conversation_root_id=lambda: "root-session",
        _session_db=None,
        session_id="session-id",
        _cleanup_task_resources=cleanup,
    )


@pytest.mark.parametrize(
    "result",
    [
        {"final_response": "completed", "completed": True},
        {"final_response": "provider failed", "completed": False, "failed": True},
        {"final_response": "interrupted", "completed": False, "interrupted": True},
    ],
    ids=["success", "failure", "cancellation"],
)
def test_all_terminal_outcomes_run_shared_cleanup(result):
    cleanup = MagicMock()
    agent = _agent(cleanup)

    with patch("agent.conversation_loop.run_conversation", return_value=result):
        assert AIAgent.run_conversation(agent, "hello", task_id="task-1") == result

    cleanup.assert_called_once_with("task-1")


def test_cleanup_runs_when_conversation_raises_without_masking_error():
    cleanup = MagicMock()
    agent = _agent(cleanup)

    with patch(
        "agent.conversation_loop.run_conversation",
        side_effect=RuntimeError("provider disconnected"),
    ):
        with pytest.raises(RuntimeError, match="provider disconnected"):
            AIAgent.run_conversation(agent, "hello", task_id="task-2")

    cleanup.assert_called_once_with("task-2")


def test_cleanup_failure_does_not_mask_terminal_result():
    cleanup = MagicMock(side_effect=OSError("teardown unavailable"))
    agent = _agent(cleanup)

    with patch(
        "agent.conversation_loop.run_conversation",
        return_value={"final_response": "still delivered"},
    ):
        result = AIAgent.run_conversation(agent, "hello", task_id="task-3")

    assert result["final_response"] == "still delivered"
    cleanup.assert_called_once_with("task-3")