from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.continuation_engine import (
    apply_bounded_continuation_engine,
    build_runtime_resume_message,
    should_use_continuation_engine,
)
from run_agent import AIAgent


class _FakeChild:
    def __init__(self, follow_up_results):
        self.follow_up_results = list(follow_up_results)
        self.calls = []
        self.session_id = "child-1"

    def run_conversation(self, user_message: str):
        self.calls.append(user_message)
        return self.follow_up_results.pop(0)


def _make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://example.test/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent.runtime_activation_state = {"runtime_mode": "ralph"}
    agent.get_runtime_activation_state = lambda: dict(agent.runtime_activation_state)
    return agent


def test_ralph_loop_retries_failed_work_without_open_todos():
    child = _FakeChild([
        {"completed": True, "final_response": "Recovered on retry.", "api_calls": 1}
    ])

    initial_result = {
        "failed": True,
        "final_response": "First attempt failed.",
        "api_calls": 1,
    }

    continuation = apply_bounded_continuation_engine(child, initial_result, runtime_mode="ralph")

    assert continuation["resume_count"] == 1
    assert continuation["attempt_count"] == 2
    assert continuation["result"]["final_response"] == "Recovered on retry."
    assert continuation["snapshot"]["outcomeStatus"] == "completed"
    assert child.calls
    assert "Ralph-loop continuation engine retry" in child.calls[0]
    assert "Previous outcome: failed" in child.calls[0]


def test_ralph_loop_honors_explicit_stop_signal():
    snapshot = {
        "outcomeStatus": "interrupted",
        "activeTodos": [{"id": "todo-1", "content": "Stop here", "status": "in_progress"}],
        "stopRequested": True,
    }

    assert should_use_continuation_engine("ralph", snapshot) is False


def test_ultrawork_does_not_retry_failed_attempt_without_open_todos():
    snapshot = {
        "outcomeStatus": "failed",
        "activeTodos": [],
    }

    assert should_use_continuation_engine("ultrawork", snapshot) is False
    assert should_use_continuation_engine("ralph", snapshot) is True


def test_mode_specific_resume_messages_are_distinct():
    snapshot = {
        "outcomeStatus": "interrupted",
        "activeTodos": [{"id": "todo-1", "content": "Finish the task", "status": "in_progress"}],
    }

    ultrawork_message = build_runtime_resume_message(snapshot, runtime_mode="ultrawork", attempt=1, max_attempts=2)
    ralph_message = build_runtime_resume_message(snapshot, runtime_mode="ralph", attempt=1, max_attempts=2)

    assert "Ultrawork continuation engine resume" in ultrawork_message
    assert "Do not stop at a status update" in ultrawork_message
    assert "Ralph-loop continuation engine retry" in ralph_message
    assert "Stop cleanly if the loop should end" in ralph_message


def test_aiagent_ralph_requeues_failed_text_response_without_open_todos():
    agent = _make_agent()
    assistant_message = SimpleNamespace(content="First pass failed.", tool_calls=[], failed=True)
    messages = []

    queued = agent._maybe_enqueue_runtime_continuation(
        messages=messages,
        assistant_message=assistant_message,
        finish_reason="stop",
        final_response="First pass failed.",
        runtime_resume_count=0,
        turn_outcome=agent._build_runtime_continuation_turn_outcome(
            assistant_message=assistant_message,
            final_response="First pass failed.",
            interrupted=False,
        ),
    )

    assert queued is True
    assert messages[-1]["role"] == "user"
    assert "Ralph-loop continuation engine retry" in messages[-1]["content"]
    assert "Previous outcome: failed" in messages[-1]["content"]


def test_aiagent_ralph_requeues_interrupted_text_response_without_open_todos():
    agent = _make_agent()
    assistant_message = SimpleNamespace(content="Interrupted before completion.", tool_calls=[], interrupted=True)
    messages = []

    queued = agent._maybe_enqueue_runtime_continuation(
        messages=messages,
        assistant_message=assistant_message,
        finish_reason="stop",
        final_response="Interrupted before completion.",
        runtime_resume_count=0,
        turn_outcome=agent._build_runtime_continuation_turn_outcome(
            assistant_message=assistant_message,
            final_response="Interrupted before completion.",
            interrupted=False,
        ),
    )

    assert queued is True
    assert messages[-1]["role"] == "user"
    assert "Ralph-loop continuation engine retry" in messages[-1]["content"]
    assert "Previous outcome: interrupted" in messages[-1]["content"]
