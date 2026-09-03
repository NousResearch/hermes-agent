"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

from run_agent import _repair_tool_call_arguments


class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"



    # -- Stage 2: Python None literal --


    def test_none_literal_returns_empty_object(self):
        assert _repair_tool_call_arguments("None", "t") == "{}"

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

    # -- Stage 0.5: gateway leading-value prefix (e.g. `{}{"city": "Paris"}`) --

    def test_leading_empty_object_prefix_salvaged(self):
        # one-api/new-api gateways prepend a complete JSON value (usually an
        # empty object) to the real arguments. The remainder must be
        # recovered, not discarded to "{}".
        result = _repair_tool_call_arguments('{}{"city": "Paris"}', "t")
        assert json.loads(result) == {"city": "Paris"}

    def test_leading_empty_object_prefix_with_whitespace_salvaged(self):
        result = _repair_tool_call_arguments('  {}  {"mode": "tail"}', "t")
        assert json.loads(result) == {"mode": "tail"}

    def test_leading_array_prefix_salvaged(self):
        result = _repair_tool_call_arguments('[]{"a": [1, 2]}', "t")
        assert json.loads(result) == {"a": [1, 2]}

    def test_double_empty_object_prefix_unrepairable(self):
        # '{}{}' has no recoverable args — must fall back to empty object.
        assert _repair_tool_call_arguments("{}", "t") == "{}"

    def test_non_object_suffix_not_salvaged(self):
        # Suffix is an array, not a tool-args object — falls to last resort.
        assert _repair_tool_call_arguments("{}[1, 2]", "t") == "{}"

    def test_plain_prefixless_args_passthrough(self):
        # No leading value to strip — original JSON still parses.
        result = _repair_tool_call_arguments('{"city": "Paris"}', "t")
        assert json.loads(result) == {"city": "Paris"}
