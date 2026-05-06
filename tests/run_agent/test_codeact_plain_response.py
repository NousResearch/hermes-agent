import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from run_agent import AIAgent


def _agent_stub():
    agent = object.__new__(AIAgent)
    agent._codeact_kernel = MagicMock()
    agent.tools = [{"type": "function", "function": {"name": "run_code"}}]
    agent._build_assistant_message = MagicMock(
        return_value={"role": "assistant", "content": "raw envelope"}
    )
    agent._emit_status = MagicMock()
    agent._invoke_run_code = MagicMock(return_value="tool output")
    agent._save_session_log = MagicMock()
    agent._session_messages = []
    agent._stream_needs_break = False
    return agent


def test_codeact_plain_text_envelope_is_executed_as_synthetic_tool_call():
    agent = _agent_stub()
    messages = []
    assistant_message = SimpleNamespace(
        content=json.dumps(
            {
                "thoughts": "search first",
                "code": "result = web_search(query='GLP-1 GIP drugs', limit=5)",
            }
        )
    )

    handled = agent._maybe_execute_codeact_plain_response(
        assistant_message,
        messages,
        "stop",
    )

    assert handled is True
    assert len(messages) == 2
    assistant_turn = messages[0]
    tool_turn = messages[1]
    assert assistant_turn["content"] == ""
    tool_call = assistant_turn["tool_calls"][0]
    assert tool_call["function"]["name"] == "run_code"
    args = json.loads(tool_call["function"]["arguments"])
    assert args["thoughts"] == "search first"
    assert "web_search" in args["code"]
    assert tool_turn["role"] == "tool"
    assert tool_turn["name"] == "run_code"
    assert tool_turn["tool_call_id"] == tool_call["id"]
    assert tool_turn["content"] == "tool output"
    agent._invoke_run_code.assert_called_once_with(args)
    agent._save_session_log.assert_called_once_with(messages)


def test_codeact_plain_text_non_envelope_is_not_executed():
    agent = _agent_stub()
    messages = []
    assistant_message = SimpleNamespace(content="Here is the final answer.")

    handled = agent._maybe_execute_codeact_plain_response(
        assistant_message,
        messages,
        "stop",
    )

    assert handled is False
    assert messages == []
    agent._invoke_run_code.assert_not_called()
