from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _response(text: str):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=text, tool_calls=[]),
            )
        ],
        usage=None,
    )


def _make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def _set_runtime_mode(agent: AIAgent, runtime_mode: str) -> None:
    def _activate(_message: str):
        state = {"runtime_mode": runtime_mode}
        agent.runtime_activation_state = dict(state)
        agent._runtime_activation_state = dict(state)
        return state

    agent._activate_runtime_for_turn = _activate
    agent.get_runtime_activation_state = lambda: dict(agent.runtime_activation_state)


def test_ultrawork_requeues_when_open_todos_remain():
    agent = _make_agent()
    _set_runtime_mode(agent, "ultrawork")
    agent._todo_store.write([
        {"id": "todo-1", "content": "Finish the remaining implementation", "status": "in_progress"}
    ])

    def _api_side_effect(*_args, **_kwargs):
        if not hasattr(_api_side_effect, "calls"):
            _api_side_effect.calls = 0
        _api_side_effect.calls += 1
        if _api_side_effect.calls == 1:
            return _response("Status update only.")
        agent._todo_store.write([
            {"id": "todo-1", "content": "Finish the remaining implementation", "status": "completed"}
        ])
        return _response("Completed after closing the todo.")

    with patch.object(agent, "_interruptible_api_call", side_effect=_api_side_effect):
        result = agent.run_conversation("Finish the implementation in ultrawork mode.")

    assert result["final_response"] == "Completed after closing the todo."
    continuation_notes = [
        msg["content"]
        for msg in result["messages"]
        if msg.get("role") == "user" and "Ultrawork continuation engine resume" in str(msg.get("content"))
    ]
    assert len(continuation_notes) == 1
    assert "Finish the remaining implementation" in continuation_notes[0]
    assert result["api_calls"] == 2


def test_default_mode_keeps_one_shot_completion_behavior():
    agent = _make_agent()
    _set_runtime_mode(agent, "default")
    agent._todo_store.write([
        {"id": "todo-1", "content": "Finish the remaining implementation", "status": "in_progress"}
    ])

    with patch.object(agent, "_interruptible_api_call", return_value=_response("Status update only.")) as api_call:
        result = agent.run_conversation("Finish the implementation.")

    assert result["final_response"] == "Status update only."
    assert api_call.call_count == 1
    assert not any(
        msg.get("role") == "user" and "continuation engine" in str(msg.get("content"))
        for msg in result["messages"]
    )
