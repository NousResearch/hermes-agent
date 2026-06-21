"""Shared mock-builder helpers for the run_agent unit-test suite.

Extracted verbatim from the former monolithic ``test_run_agent.py`` when it was
split into per-theme files (it had outgrown the per-file CI timeout). These are
plain helper functions called directly by tests; the pytest *fixtures*
(``agent`` / ``agent_with_memory_tool``) live in ``conftest.py``.
"""

import uuid
from types import SimpleNamespace


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
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


def _mock_assistant_msg(
    content="Hello",
    tool_calls=None,
    reasoning=None,
    reasoning_content=None,
    reasoning_details=None,
):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletionMessage."""
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        msg.reasoning = reasoning
    if reasoning_content is not None:
        msg.reasoning_content = reasoning_content
    if reasoning_details is not None:
        msg.reasoning_details = reasoning_details
    return msg


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    """Return a SimpleNamespace mimicking a tool call object."""
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(
    content="Hello",
    finish_reason="stop",
    tool_calls=None,
    reasoning=None,
    reasoning_content=None,
    reasoning_details=None,
    usage=None,
):
    """Return a SimpleNamespace mimicking an OpenAI ChatCompletion response."""
    msg = _mock_assistant_msg(
        content=content,
        tool_calls=tool_calls,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        reasoning_details=reasoning_details,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    resp = SimpleNamespace(choices=[choice], model="test/model")
    if usage:
        resp.usage = SimpleNamespace(**usage)
    else:
        resp.usage = None
    return resp


def _make_chunk(content=None, tool_calls=None, finish_reason=None, model="test/model"):
    """Build a SimpleNamespace mimicking an OpenAI streaming chunk."""
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(model=model, choices=[choice])


def _make_tc_delta(index=0, tc_id=None, name=None, arguments=None):
    """Build a SimpleNamespace mimicking a streaming tool_call delta."""
    func = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=func)
