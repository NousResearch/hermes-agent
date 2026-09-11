"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

import pytest

from agent.message_sanitization import _repair_tool_call_arguments
class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"



    # -- Stage 2: Python None literal --



    # -- Stage 3: trailing comma repair --


    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}


    # -- Stage 4: unclosed brackets --

    @pytest.mark.parametrize("raw, tool_name, expected", [
        pytest.param('{"a": [1, 2', "t", {"a": [1, 2]}, id="nested-array"),
        pytest.param('{"items": ["x", "y"', "t", {"items": ["x", "y"]}, id="string-array"),
        pytest.param('{"content": "hello } world', "write_file", {"content": "hello } world"}, id="brace-in-string"),
        pytest.param('{"a": {"b": 1', "t", {"a": {"b": 1}}, id="nested-object-control"),
        pytest.param('{"content": "hello world', "write_file", {"content": "hello world"}, id="open-string"),
        pytest.param(r'{"content": "hello \"} world', "write_file", {"content": 'hello "} world'}, id="escaped-quote"),
        pytest.param('{"content": "hello ' + '\\', "write_file", {"content": "hello \\"}, id="dangling-backslash"),
        pytest.param(r'{"content": "hello \\', "write_file", {"content": "hello \\"}, id="escaped-backslash"),
        pytest.param(r'{"items": ["hello \\"', "t", {"items": ["hello \\"]}, id="closed-string-after-backslash"),
        pytest.param('{"a": [{"b": [1', "t", {"a": [{"b": [1]}]}, id="mixed-nesting"),
        pytest.param('{"a": [{"b": 1}, 2', "t", {"a": [{"b": 1}, 2]}, id="closed-inner-object"),
        pytest.param('{"content": "{[}]"', "write_file", {"content": "{[}]"}, id="closed-string-delimiters"),
        pytest.param('{"items": ["hello\nworld', "t", {"items": ["hello\nworld"]}, id="control-char-fallback"),
    ])
    def test_truncated_arguments_preserve_payload(self, raw, tool_name, expected):
        assert json.loads(_repair_tool_call_arguments(raw, tool_name)) == expected



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # Closing a truncated key cannot recover its missing value.
        assert _repair_tool_call_arguments('{"truncated', "t") == "{}"

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --

