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

    def test_nested_array_closes_innermost_first(self):
        """#102311: '{"a": [1, 2' must gain ']' before '}'."""
        parsed = json.loads(_repair_tool_call_arguments('{"a": [1, 2', "t"))
        assert parsed == {"a": [1, 2]}

    def test_array_of_strings_closes_innermost_first(self):
        parsed = json.loads(_repair_tool_call_arguments('{"items": ["x", "y"', "t"))
        assert parsed == {"items": ["x", "y"]}

    def test_brace_inside_string_does_not_count(self):
        """#102311: braces inside string literals are not structure."""
        parsed = json.loads(
            _repair_tool_call_arguments('{"content": "hello } world', "write_file")
        )
        assert parsed == {"content": "hello } world"}

    def test_unterminated_trailing_string_is_closed(self):
        parsed = json.loads(
            _repair_tool_call_arguments('{"content": "hello world', "write_file")
        )
        assert parsed == {"content": "hello world"}

    def test_curly_only_still_repairs(self):
        parsed = json.loads(_repair_tool_call_arguments('{"a": {"b": 1', "t"))
        assert parsed == {"a": {"b": 1}}



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object(self):
        # No structure left to complete — bracket closing can't help.
        assert _repair_tool_call_arguments('{"a": }', "t") == "{}"

    def test_truncated_string_value_is_completed_not_dropped(self):
        """Truncated mid-string-value: closing the string preserves the
        received bytes instead of dropping the whole payload to {}."""
        parsed = json.loads(_repair_tool_call_arguments('{"truncated": "val', "t"))
        assert parsed == {"truncated": "val"}

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --


