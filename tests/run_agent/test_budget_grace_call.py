import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

import run_agent
from run_agent import AIAgent

def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]

def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )

def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        function_call=None
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model")

@pytest.fixture()
def test_agent():
    with (
        patch(
            "run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")
        ),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        return a

def test_grace_call_strips_tools_and_includes_summary_instruction(test_agent):
    """Verify that during the grace call, tool schemas are stripped and summary instruction is appended."""
    test_agent.max_iterations = 2
    
    tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
    tool_resp = _mock_response(
        content="", finish_reason="tool_calls", tool_calls=[tc]
    )
    summary_resp = _mock_response(
        content="This is the final summary.", finish_reason="stop"
    )
    
    test_agent.client.chat.completions.create.side_effect = [
        tool_resp, tool_resp, summary_resp
    ]
    
    # Capture kwargs passed to completions.create
    create_calls_kwargs = []
    def mock_create(*args, **kwargs):
        create_calls_kwargs.append(kwargs)
        return summary_resp if len(create_calls_kwargs) >= 3 else tool_resp

    test_agent.client.chat.completions.create.side_effect = mock_create

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(test_agent, "_persist_session"),
        patch.object(test_agent, "_save_trajectory"),
        patch.object(test_agent, "_cleanup_task_resources"),
    ):
        result = test_agent.run_conversation("test budget grace call task")

    # Should have run 3 times: 2 normal iterations + 1 grace summary iteration
    assert len(create_calls_kwargs) == 3
    assert result["completed"] is False  # Loop ended due to max iterations limit
    
    # The third API call (grace summary call) must have tools=None (or omitted)
    third_call_kwargs = create_calls_kwargs[2]
    assert third_call_kwargs.get("tools") is None

    # The messages for the third call must include the summary instruction
    messages = third_call_kwargs.get("messages", [])
    assert len(messages) > 0
    last_user_message = [m for m in messages if m.get("role") == "user"][-1]
    assert "maximum number of tool-calling iterations allowed" in last_user_message["content"]


def test_grace_call_empty_response_triggers_fallback_summary(test_agent):
    """Verify that if the grace call returns an empty response, the direct handle_max_iterations fallback runs."""
    test_agent.max_iterations = 2
    
    tc = _mock_tool_call(name="web_search", arguments="{}", call_id="c1")
    tool_resp = _mock_response(
        content="", finish_reason="tool_calls", tool_calls=[tc]
    )
    # The grace call returns empty content/None response
    empty_grace_resp = _mock_response(
        content="", finish_reason="stop"
    )
    fallback_summary_resp = _mock_response(
        content="Fallback summary response content.", finish_reason="stop"
    )
    
    test_agent.client.chat.completions.create.side_effect = [
        tool_resp, tool_resp, empty_grace_resp, fallback_summary_resp
    ]

    with (
        patch("run_agent.handle_function_call", return_value="ok"),
        patch.object(test_agent, "_persist_session"),
        patch.object(test_agent, "_save_trajectory"),
        patch.object(test_agent, "_cleanup_task_resources"),
        patch.object(test_agent, "_handle_max_iterations", return_value="Fallback summary response content.") as mock_fallback,
    ):
        result = test_agent.run_conversation("test fallback task")

    # Assert that the fallback handler was called because the grace call returned an empty response
    assert mock_fallback.call_count == 1
    assert result["final_response"] == "Fallback summary response content."
