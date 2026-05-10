"""Tests for JSON-aware inline XML tool-call extraction."""

import json

from agent.inline_tool_calls import (
    extract_inline_tool_calls,
    strip_inline_tool_call_blocks,
)


def test_extracts_tool_use_with_nested_input_object():
    text = (
        '<tool_use>{"id":"t1","name":"terminal",'
        '"input":{"command":"ls","timeout":300}}</tool_use>'
    )

    result = extract_inline_tool_calls(text)

    assert result.content is None
    assert result.parsed_spans == [(0, len(text))]
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.id == "t1"
    assert call.name == "terminal"
    assert json.loads(call.arguments) == {"command": "ls", "timeout": 300}


def test_extracts_tool_use_with_closing_tag_inside_json_string():
    text = (
        '<tool_use>{"id":"t1","name":"write_file",'
        '"input":{"content":"</tool_use>"}}</tool_use>'
    )

    result = extract_inline_tool_calls(text)

    assert result.content is None
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "write_file"
    assert json.loads(result.tool_calls[0].arguments) == {
        "content": "</tool_use>",
    }


def test_strips_tool_call_with_closing_tag_inside_json_string():
    text = (
        'before <tool_call>{"name":"write_file",'
        '"arguments":{"content":"</tool_call>"}}</tool_call> after'
    )

    result = strip_inline_tool_call_blocks(text)

    assert result == "before  after"
    assert "</tool_call>" not in result
    assert '"}}' not in result


def test_extracts_function_call_shape():
    text = (
        '<function_call>{"function":{"name":"read_file",'
        '"arguments":{"path":"README.md"}}}</function_call>'
    )

    result = extract_inline_tool_calls(text)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "read_file"
    assert json.loads(result.tool_calls[0].arguments) == {"path": "README.md"}


def test_extracts_plural_tool_calls_array_and_preserves_surrounding_text():
    text = (
        'before <tool_calls>[{"id":"a","name":"read_file",'
        '"arguments":{"path":"a.md"}},{"id":"b","name":"terminal",'
        '"arguments":{"command":"pwd"}}]</tool_calls> after'
    )

    result = extract_inline_tool_calls(text)

    assert result.content == "before  after"
    assert [call.id for call in result.tool_calls] == ["a", "b"]
    assert [call.name for call in result.tool_calls] == ["read_file", "terminal"]


def test_prose_mention_is_not_extracted():
    text = "Use <tool_use> in docs when describing Anthropic blocks."

    result = extract_inline_tool_calls(text)

    assert result.tool_calls == []
    assert result.parsed_spans == []
    assert result.content == text


def test_prose_mention_before_actual_tool_use_is_skipped():
    text = (
        "Use <tool_use> in docs first.\n"
        '<tool_use>{"id":"t1","name":"terminal","input":{"command":"pwd"}}</tool_use>'
    )

    result = extract_inline_tool_calls(text)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "t1"
    assert result.content == "Use <tool_use> in docs first.\n"


def test_quoted_tool_use_example_is_not_extracted():
    text = (
        'Example: "<tool_use>{"name":"read_file",'
        '"input":{"path":"README.md"}}</tool_use>"'
    )

    result = extract_inline_tool_calls(text)

    assert result.tool_calls == []
    assert result.parsed_spans == []
    assert result.content == text


def test_backticked_tool_use_example_is_not_extracted():
    text = '`<tool_use>{"name":"read_file","input":{"path":"README.md"}}</tool_use>`'

    result = extract_inline_tool_calls(text)

    assert result.tool_calls == []
    assert result.parsed_spans == []
    assert result.content == text


def test_surrounding_whitespace_is_preserved():
    text = (
        '  before <tool_use>{"name":"terminal",'
        '"input":{"command":"pwd"}}</tool_use> after  '
    )

    result = extract_inline_tool_calls(text)

    assert len(result.tool_calls) == 1
    assert result.content == "  before  after  "


def test_malformed_json_is_not_extracted():
    text = '<tool_use>{"name": </tool_use>'

    result = extract_inline_tool_calls(text)

    assert result.tool_calls == []
    assert result.parsed_spans == []
    assert result.content == text


def test_missing_name_is_not_extracted():
    text = '<tool_use>{"arguments":{"path":"README.md"}}</tool_use>'

    result = extract_inline_tool_calls(text)

    assert result.tool_calls == []
    assert result.parsed_spans == []
    assert result.content == text


def test_missing_id_gets_stable_generated_id():
    text = '<tool_use>{"name":"read_file","arguments":{"path":"README.md"}}</tool_use>'

    first = extract_inline_tool_calls(text)
    second = extract_inline_tool_calls(text)

    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].id.startswith("call_inline_")
    assert first.tool_calls[0].id == second.tool_calls[0].id
