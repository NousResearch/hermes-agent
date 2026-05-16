"""Characterization tests for the minimal TurnRunner runtime seam."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent, TurnRunner


def _tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent() -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=3,
        )

    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _run_one_turn(
    runner: TurnRunner,
    user_message: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    return runner.run_conversation(
        user_message,
        conversation_history=conversation_history,
        task_id="sym-273-turn",
    )


def test_aiagent_satisfies_turn_runner_protocol() -> None:
    assert isinstance(_make_agent(), TurnRunner)


def test_turn_runner_stop_turn_preserves_result_shape_and_history() -> None:
    agent = _make_agent()
    agent.client.chat.completions.create.return_value = _mock_response("Final answer")
    history = [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]

    with (
        patch("hermes_cli.plugins.invoke_hook", return_value=[]),
        patch.object(agent, "_ensure_db_session"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = _run_one_turn(agent, "hello", conversation_history=history)

    assert history == [
        {"role": "user", "content": "prior question"},
        {"role": "assistant", "content": "prior answer"},
    ]
    assert result["final_response"] == "Final answer"
    assert result["api_calls"] == 1
    assert result["completed"] is True
    assert result["interrupted"] is False
    assert result["partial"] is False
    assert result["messages"][:2] == history
    assert result["messages"][-2] == {"role": "user", "content": "hello"}
    assistant_message = result["messages"][-1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "Final answer"
    assert assistant_message["reasoning"] is None
    assert assistant_message["finish_reason"] == "stop"
