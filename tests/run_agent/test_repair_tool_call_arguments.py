"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

from run_agent import _repair_tool_call_arguments


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

    def test_unclosed_single_brace(self):
        result = _repair_tool_call_arguments('{"a": 1', "t")
        assert json.loads(result) == {"a": 1}

    def test_unclosed_single_bracket(self):
        result = _repair_tool_call_arguments("[1,2", "t")
        assert json.loads(result) == [1, 2]

    def test_unclosed_nested_bracket_closed_lifo(self):
        # Regression (2026-08-03): naive counting appended '}' before ']',
        # producing '{"a": [1,2}]' (invalid) and falling back to '{}'.
        # Closers must be appended in LIFO order — last opened, first closed.
        result = _repair_tool_call_arguments('{"a": [1,2', "t")
        assert result == '{"a": [1,2]}', f"got {result!r}"
        assert json.loads(result) == {"a": [1, 2]}

    def test_unclosed_nested_object_in_array_lifo(self):
        result = _repair_tool_call_arguments('{"a": [1, {"b": 2', "t")
        assert json.loads(result) == {"a": [1, {"b": 2}]}

    def test_lifo_does_not_break_balanced_json(self):
        result = _repair_tool_call_arguments('{"a": [1, 2]}', "t")
        assert json.loads(result) == {"a": [1, 2]}

    def test_string_aware_scan_ignores_bracket_in_string_value(self):
        # Regression (2026-08-03, reviewer finding): a literal '[' inside a
        # quoted string value is NOT a structural delimiter. A string-blind
        # scan pushed ']' for it, leaving '{"a":"[" unterminated and falling
        # back to '{}'. The string/escape-aware scan must preserve it.
        result = _repair_tool_call_arguments('{"a":"[","b":[1,2', "t")
        assert result == '{"a":"[","b":[1,2]}', f"got {result!r}"
        assert json.loads(result) == {"a": "[", "b": [1, 2]}

    def test_string_aware_scan_ignores_brace_in_string_value(self):
        result = _repair_tool_call_arguments('{"a":"{","b":{"x":1', "t")
        assert json.loads(result) == {"a": "{", "b": {"x": 1}}

    def test_string_aware_scan_handles_escaped_quotes(self):
        # A backslash-escaped quote inside a string must not close it.
        result = _repair_tool_call_arguments('{"a":"\\"","b":[1,2', "t")
        assert json.loads(result) == {"a": '"', "b": [1, 2]}

    def test_string_aware_scan_ignores_delimiters_after_closed_string(self):
        result = _repair_tool_call_arguments('{"a":"]","b":{', "t")
        assert json.loads(result) == {"a": "]", "b": {}}



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --


