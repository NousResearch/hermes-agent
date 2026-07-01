import json
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
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


def _mock_tool_call(name="terminal", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _assistant_message(*tool_calls):
    return SimpleNamespace(tool_calls=list(tool_calls))


def _make_agent(*tool_names: str) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",  # gitleaks:allow (fixture constant, not a secret)
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


def _tool_result_payload(messages):
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    return json.loads(tool_messages[0]["content"])


def test_resume_summary_only_blocks_forward_tool_call_and_logs(caplog):
    agent = _make_agent("terminal")
    agent._resume_summary_only = True
    agent.tool_progress_callback = MagicMock()
    messages = []
    assistant_message = _assistant_message(
        _mock_tool_call("terminal", '{"command":"touch should-not-run"}', "call-1")
    )

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    mock_hfc.assert_not_called()
    assert agent._resume_summary_only is True
    payload = _tool_result_payload(messages)
    assert "summarize-only" in payload["error"]
    assert payload["resume_autocontinue_block"]["tool_name"] == "terminal"
    assert "RESUME_AUTOCONTINUE_VIOLATION" in caplog.text
    agent.tool_progress_callback.assert_called_once()
    assert agent.tool_progress_callback.call_args.args[0] == "resume.autocontinue_violation"


def test_resume_summary_only_blocks_every_tool_round_in_same_turn():
    agent = _make_agent("terminal")
    agent._resume_summary_only = True

    with patch("run_agent.handle_function_call", return_value="executed") as mock_hfc:
        agent._execute_tool_calls_sequential(
            _assistant_message(_mock_tool_call("terminal", "{}", "call-1")),
            [],
            "task-1",
        )
        agent._execute_tool_calls_sequential(
            _assistant_message(_mock_tool_call("terminal", "{}", "call-2")),
            [],
            "task-1",
        )

    mock_hfc.assert_not_called()
    assert agent._resume_summary_only is True


def test_resume_summary_only_read_tool_block_does_not_emit_violation(caplog):
    agent = _make_agent("read_file")
    agent._resume_summary_only = True
    agent.tool_progress_callback = MagicMock()
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(
            _assistant_message(_mock_tool_call("read_file", '{"path":"README.md"}', "call-1")),
            messages,
            "task-1",
        )

    mock_hfc.assert_not_called()
    payload = _tool_result_payload(messages)
    assert payload["resume_autocontinue_block"]["tool_name"] == "read_file"
    assert "RESUME_AUTOCONTINUE_VIOLATION" not in caplog.text
    agent.tool_progress_callback.assert_not_called()


def test_normal_turn_executes_tools_without_interference():
    agent = _make_agent("terminal")
    agent._resume_summary_only = False
    messages = []
    assistant_message = _assistant_message(_mock_tool_call("terminal", "{}", "call-1"))

    with patch("run_agent.handle_function_call", return_value="executed") as mock_hfc:
        agent._execute_tool_calls_sequential(assistant_message, messages, "task-1")

    mock_hfc.assert_called_once()
    assert messages[0]["content"] == "executed"
