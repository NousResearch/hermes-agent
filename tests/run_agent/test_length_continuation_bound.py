"""Regression coverage for the per-turn length-continuation bound."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _response(content: str, finish_reason: str = "length"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason=finish_reason,
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
            session_id="length-continuation-bound-test",
            api_key="test-key",
            base_url="https://example.invalid/v1",
            provider="openai-compat",
            model="test/model",
            max_iterations=12,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )

    instance._cached_system_prompt = "stable test prompt"
    instance._session_db = None
    instance._session_json_enabled = False
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance._cleanup_task_resources = MagicMock()
    instance._save_trajectory = MagicMock()
    return instance


def _run(agent, prompt: str):
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        return agent.run_conversation(prompt)


def test_length_continuation_bound_stops_before_a_fifth_submission(agent):
    agent.client.chat.completions.create.side_effect = [
        _response(f"part-{index}") for index in range(5)
    ]

    result = _run(agent, "generate a long response")

    assert agent.client.chat.completions.create.call_count == 4
    assert result["api_calls"] == 4
    assert result["completed"] is False
    assert result["partial"] is True
    agent._cleanup_task_resources.assert_called_once()


def test_length_continuation_bound_retains_all_partial_output(agent):
    agent.client.chat.completions.create.side_effect = [
        _response(f"segment-{index}|") for index in range(4)
    ]

    result = _run(agent, "retain every segment")

    assert result["final_response"] == "segment-0|segment-1|segment-2|segment-3|"
    assert result["completed"] is False
    assert result["partial"] is True


def test_new_turn_resets_length_continuation_counter(agent):
    agent.client.chat.completions.create.side_effect = [
        *[_response(f"first-{index}|") for index in range(4)],
        _response("second-part-1|"),
        _response("second-part-2", finish_reason="stop"),
    ]

    first = _run(agent, "first turn")
    second = _run(agent, "second turn")

    assert first["completed"] is False
    assert first["partial"] is True
    assert second["completed"] is True
    assert second["partial"] is False
    assert second["final_response"] == "second-part-1|second-part-2"
    assert agent.client.chat.completions.create.call_count == 6
    assert agent._cleanup_task_resources.call_count == 2
