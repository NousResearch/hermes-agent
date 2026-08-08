"""End-to-end regression for active TodoStore continuation in the real loop."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


def _response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="test/model",
        usage=None,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            session_id="todo-completion-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=2,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = lambda *_a, **_k: None
    instance._save_trajectory = lambda *_a, **_k: None
    instance.valid_tool_names = ["terminal"]
    instance._todo_completion_continuation = True
    return instance


def test_active_todo_stop_is_reprompted_through_real_conversation_loop(agent, monkeypatch):
    """A narration-only stop is not final while Hermes marks work in progress."""
    agent._todo_store.write(
        [{"id": "s4-ci", "content": "commit and verify CI", "status": "in_progress"}]
    )
    calls = 0

    def model_call(_api_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("Need commit.")
        # This models the required todo update after the actual commit/CI work.
        agent._todo_store.write(
            [{"id": "s4-ci", "content": "commit and verify CI", "status": "completed"}]
        )
        return _response("Committed and CI passed.")

    agent._interruptible_api_call = model_call
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("finish the release")

    assert calls == 2
    assert result["final_response"] == "Committed and CI passed."
    assert result["turn_exit_reason"] == "text_response(finish_reason=stop)"
    assert result["completed"] is True
    assert not any(
        message.get("_todo_completion_synthetic")
        for message in result["messages"]
        if isinstance(message, dict)
    )


def test_active_todo_continuation_exhaustion_is_visible_not_silent(agent, monkeypatch):
    """A model that ignores three nudges must not be reported as successful."""
    agent.max_iterations = 4
    agent.iteration_budget.max_total = 4
    agent._todo_store.write(
        [{"id": "s4-ci", "content": "commit and verify CI", "status": "in_progress"}]
    )
    calls = 0

    def model_call(_api_kwargs):
        nonlocal calls
        calls += 1
        return _response("Need commit.")

    agent._interruptible_api_call = model_call
    monkeypatch.setenv("HERMES_VERIFY_ON_STOP", "0")

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = agent.run_conversation("finish the release")

    assert calls == 4
    assert "Task remains incomplete" in result["final_response"]
    assert result["turn_exit_reason"] == "todo_completion_exhausted"
    assert result["completed"] is False
