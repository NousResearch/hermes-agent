"""Dict tool-result content is serialized for strict OpenAI-compatible endpoints."""

import json

from agent.transports.chat_completions import _stringify_tool_content


def test_dict_tool_content_becomes_json_string():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "content": {"success": True, "output": "ok"}},
    ]
    original = json.loads(json.dumps(messages))
    out = _stringify_tool_content(messages)
    assert out[1]["content"] == '{"success":true,"output":"ok"}'
    assert out[0] == messages[0]
    assert messages == original  # no mutation of stored history


def test_string_tool_content_is_unchanged():
    messages = [{"role": "tool", "content": "already a string"}]
    assert _stringify_tool_content(messages) is messages


def test_non_tool_dict_content_is_unchanged():
    messages = [{"role": "user", "content": {"not": "a tool"}}]
    assert _stringify_tool_content(messages) is messages
