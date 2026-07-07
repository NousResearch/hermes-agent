import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
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


def _make_agent(*tool_names: str):
    hermes_home = Path(tempfile.mkdtemp(prefix="hermes-test-home-"))
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("run_agent._hermes_home", hermes_home),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.tool_delay = 0
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _mock_tool_call(name: str, args: dict, call_id: str = "call_1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def _mock_response(content="", finish_reason="tool_calls", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def test_run_conversation_returns_decision_packet_for_git_commit_without_second_model_call():
    agent = _make_agent("terminal")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            tool_calls=[
                _mock_tool_call("terminal", {"command": "git commit -m test"}, "c1")
            ]
        ),
        AssertionError("run_conversation should halt before a follow-up model call"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("commit it")

    assert result["turn_exit_reason"] == "decision_policy_halt"
    assert result["decision_packet"]["status"] == "NEEDS_CHAD"
    assert "status: NEEDS_CHAD" in result["final_response"]
    assert "git commit" in result["final_response"]
    assert agent.client.chat.completions.create.call_count == 1


def test_non_terminal_external_send_stops_before_dispatch():
    agent = _make_agent("send_message")
    tool_call = _mock_tool_call(
        "send_message",
        {"action": "send", "message": "hello", "target": "telegram:123"},
        "send-1",
    )
    assistant_message = SimpleNamespace(content="", tool_calls=[tool_call])
    messages = []

    with patch("run_agent.handle_function_call", side_effect=AssertionError("should not dispatch")):
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    assert agent._decision_policy_halt_packet is not None
    assert agent._decision_policy_halt_packet.status == "NEEDS_CHAD"
    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload["status"] == "needs_chad"
    assert payload["decision_packet"]["status"] == "NEEDS_CHAD"


def test_cron_context_gets_decision_packet_not_vague_question(monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    agent = _make_agent("terminal")
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            tool_calls=[
                _mock_tool_call("terminal", {"command": "git push origin main"}, "c1")
            ]
        )
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("publish the branch")

    assert result["turn_exit_reason"] == "decision_policy_halt"
    assert result["final_response"].startswith("status: NEEDS_CHAD")
    assert "approve:" in result["final_response"]
    assert "?" not in result["final_response"].splitlines()[0]
