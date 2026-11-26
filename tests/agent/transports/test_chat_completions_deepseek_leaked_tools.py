"""Tests for recovering leaked DeepSeek tool-call tokens in ChatCompletionsTransport.normalize_response.

When OpenAI-compatible gateways (e.g. OpenRouter routing DeepSeek V3/V3.1/V3.2) occasionally emit
DeepSeek's chat-template tool-call syntax (``function<｜tool▁sep｜><name>\\njson\\n{...}``) inside
``message.content`` with ``tool_calls=None`` and ``finish_reason="stop"``, the transport recovers
the structured ``ToolCall`` objects and strips the leaked scaffolding so the turn loop executes
the tool instead of terminating mid-task.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport


def _fake_completion(content: str, *, tool_calls=None, finish_reason: str = "stop"):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls, reasoning=None, reasoning_details=None)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.fixture
def transport():
    import agent.transports.chat_completions
    return get_transport("chat_completions")


def test_recovers_bare_deepseek_tool_sep_call_and_strips_scaffolding(transport):
    raw = (
        "Created server.py.\n"
        "function<\uff5ctool\u2581sep\uff5c>terminal\n"
        "json\n"
        '{"command": "python /app/server.py", "background": true, "notify": true}\n'
    )
    out = transport.normalize_response(_fake_completion(raw))

    assert out.finish_reason == "tool_calls"
    assert out.tool_calls is not None and len(out.tool_calls) == 1
    assert out.tool_calls[0].name == "terminal"
    assert json.loads(out.tool_calls[0].arguments) == {
        "command": "python /app/server.py",
        "background": True,
        "notify": True,
    }
    assert out.content == "Created server.py."


def test_recovers_multiple_wrapped_deepseek_calls_with_fenced_json(transport):
    raw = (
        "Running checks.\n"
        "<\uff5ctool\u2581calls\u2581begin\uff5c>"
        "<\uff5ctool\u2581call\u2581begin\uff5c>function<\uff5ctool\u2581sep\uff5c>read_file\n"
        "```json\n"
        '{"path": "/app/a.py"}\n'
        "```<\uff5ctool\u2581call\u2581end\uff5c>"
        "<\uff5ctool\u2581call\u2581begin\uff5c>function<\uff5ctool\u2581sep\uff5c>terminal\n"
        "```json\n"
        '{"command": "echo {nested}"}\n'
        "```<\uff5ctool\u2581call\u2581end\uff5c>"
        "<\uff5ctool\u2581calls\u2581end\uff5c>"
    )
    out = transport.normalize_response(_fake_completion(raw))

    assert out.finish_reason == "tool_calls"
    assert out.tool_calls is not None and len(out.tool_calls) == 2
    assert [tc.name for tc in out.tool_calls] == ["read_file", "terminal"]
    assert json.loads(out.tool_calls[0].arguments) == {"path": "/app/a.py"}
    assert json.loads(out.tool_calls[1].arguments) == {"command": "echo {nested}"}
    assert out.content == "Running checks."


def test_does_not_duplicate_when_structured_tool_calls_already_present(transport):
    real_tc = SimpleNamespace(
        id="call_real_1",
        function=SimpleNamespace(name="terminal", arguments='{"command": "ls"}'),
    )
    raw = (
        "function<\uff5ctool\u2581sep\uff5c>terminal\n"
        'json\n{"command": "ls"}\n'
    )
    out = transport.normalize_response(
        _fake_completion(raw, tool_calls=[real_tc], finish_reason="tool_calls")
    )

    assert out.tool_calls is not None and len(out.tool_calls) == 1
    assert out.tool_calls[0].id == "call_real_1"


def test_malformed_json_leaves_content_untouched_and_returns_no_tool_calls(transport):
    raw = "function<\uff5ctool\u2581sep\uff5c>terminal\njson\n{not valid json"
    out = transport.normalize_response(_fake_completion(raw))

    assert out.tool_calls is None
    assert out.finish_reason == "stop"
    assert out.content == raw
