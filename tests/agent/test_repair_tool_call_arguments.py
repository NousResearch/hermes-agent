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



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    def test_unrepairable_garbage_returns_empty_object(self):
        # No JSON structure to reconstruct: brackets/closing quotes cannot help.
        assert _repair_tool_call_arguments("garbage no json", "t") == "{}"

    def test_braces_inside_string_values_do_not_skew_the_balance(self):
        # A "}" inside a value must not be counted as closing the object: naive counting
        # sees 2 "}"-worth of closes for 1 "{" and drops a repairable call to "{}".
        result = _repair_tool_call_arguments('{"code": "}", "x": 1', "t")
        assert json.loads(result) == {"code": "}", "x": 1}

    def test_truncated_nested_array_closes_in_stack_order(self):
        # {"items": [{"n": 1}, {"n": 2 needs "}]} appended (stack order), not "}}" —
        # count-based appending grouped all braces before all brackets and never parsed.
        result = _repair_tool_call_arguments('{"items": [{"n": 1}, {"n": 2', "t")
        assert json.loads(result) == {"items": [{"n": 1}, {"n": 2}]}

    # -- Balanced but misnested: the "]" of an array of objects dropped, a "}" closing in
    # its place (#115061, deepseek-v4-flash via a portal). Counts balance, so nothing can be
    # appended; the missing closer has to be inserted BEFORE the misplaced one. --

    @pytest.mark.parametrize("raw, expected", [
        ('{"a": [{"b": 1}, {"c": 2}}]}', {"a": [{"b": 1}, {"c": 2}]}),
        ('{"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}}',
         {"edits": [{"path": "a.py", "mode": "w"}, {"path": "b.py", "mode": "w"}]}),
        ('{"tool": "edit", "args": {"items": [{"k": 1}, {"k": 2}}}}',
         {"tool": "edit", "args": {"items": [{"k": 1}, {"k": 2}]}}),
        ('{"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}}',
         {"calls": [{"name": "a", "arguments": {"x": 1}}, {"name": "b", "arguments": {"y": 2}}]}),
        ('{"a": [1, 2}', {"a": [1, 2]}),
    ])
    def test_misnested_closer_is_inserted_before_the_misplaced_one(self, raw, expected):
        assert json.loads(_repair_tool_call_arguments(raw, "t")) == expected

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --


