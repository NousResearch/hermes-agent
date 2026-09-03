"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json
import logging

import pytest

from agent.message_sanitization import RepairedArguments
from run_agent import _repair_tool_call_arguments


class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == RepairedArguments("{}", True)

    @pytest.mark.parametrize("raw", ["", "  \n\t", "{}"])
    def test_empty_arguments_are_successful(self, raw):
        result = _repair_tool_call_arguments(raw, "t")
        assert result.arguments == "{}"
        assert result.ok is True



    # -- Stage 2: Python None literal --



    # -- Stage 3: trailing comma repair --


    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        assert result.ok is True
        parsed = json.loads(result.arguments)
        assert parsed == {"a": [1, 2]}

    def test_valid_arguments_are_byte_identical(self):
        raw = '{"query":"café","n":1}'
        result = _repair_tool_call_arguments(raw, "t")
        assert result == RepairedArguments(raw, True)


    # -- Stage 4: unclosed brackets --



    # -- Stage 5: excess closing delimiters --



    # -- Stage 6: last resort --


    def test_unrepairable_partial_returns_empty_object_and_warns(self, caplog):
        # Truncated in the middle of a string key — bracket closing won't help
        raw = '{"truncated": "val'
        with caplog.at_level(logging.WARNING, logger="agent.message_sanitization"):
            result = _repair_tool_call_arguments(raw, "t")
        assert result == RepairedArguments("{}", False)
        warnings = [record for record in caplog.records if "Unrepairable" in record.message]
        assert len(warnings) == 1
        assert raw in warnings[0].message

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --
