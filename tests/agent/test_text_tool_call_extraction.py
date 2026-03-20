import json
from types import SimpleNamespace

import pytest

from agent.text_tool_call_extraction import extract_text_tool_calls


def _make_tool_call(name, arguments):
    fn = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    return SimpleNamespace(id="tc-1", type="function", function=fn)


def test_no_extraction_when_structured_tool_calls_present():
    existing = [_make_tool_call("terminal", {"command": "ls"})]
    content = "some response text"
    result_content, result_calls = extract_text_tool_calls(content, existing)
    assert result_content == content
    assert result_calls is None


def test_extracts_single_hermes_tool_call():
    content = '<tool_call>{"name":"terminal","arguments":{"command":"ls"}}</tool_call>'
    _, calls = extract_text_tool_calls(content, None)
    assert calls is not None
    assert len(calls) == 1
    assert calls[0].function.name == "terminal"
    assert json.loads(calls[0].function.arguments) == {"command": "ls"}


def test_extracts_multiple_tool_calls():
    content = (
        '<tool_call>{"name":"terminal","arguments":{"command":"ls"}}</tool_call>'
        '<tool_call>{"name":"read_file","arguments":{"path":"/etc/hosts"}}</tool_call>'
    )
    _, calls = extract_text_tool_calls(content, None)
    assert calls is not None
    assert len(calls) == 2


def test_preserves_content_before_tool_call():
    content = 'Let me check.\n<tool_call>{"name":"terminal","arguments":{"command":"ls"}}</tool_call>'
    result_content, calls = extract_text_tool_calls(content, None)
    assert result_content == "Let me check."
    assert calls is not None
    assert len(calls) == 1


def test_no_extraction_for_plain_text():
    content = "Hello, how can I help?"
    result_content, calls = extract_text_tool_calls(content, None)
    assert result_content == content
    assert calls is None


def test_no_extraction_for_none_content():
    result_content, calls = extract_text_tool_calls(None, None)
    assert result_content is None
    assert calls is None


def test_no_extraction_for_empty_content():
    result_content, calls = extract_text_tool_calls("", None)
    assert result_content == ""
    assert calls is None


def test_malformed_json_returns_original():
    content = "<tool_call>not valid json</tool_call>"
    result_content, calls = extract_text_tool_calls(content, None)
    assert result_content == content
    assert calls is None


def test_tool_calls_have_required_attributes():
    content = '<tool_call>{"name":"terminal","arguments":{"command":"pwd"}}</tool_call>'
    _, calls = extract_text_tool_calls(content, None)
    assert calls is not None
    for tc in calls:
        assert isinstance(tc.id, str)
        assert isinstance(tc.function.name, str)
        assert isinstance(tc.function.arguments, str)
        json.loads(tc.function.arguments)  # must be valid JSON
