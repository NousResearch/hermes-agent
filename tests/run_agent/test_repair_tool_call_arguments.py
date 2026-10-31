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

class TestConcatenatedToolCallArguments:
    """Several complete JSON objects glued into one tool call (Gemini native).

    Observed payload: three ``read_file`` calls whose arguments arrived as
    ``{"path": "a"}{"path": "b"}{"path": "c"}``. Every other repair pass fails
    on it, so the call used to collapse to ``{}`` and the turn died with
    "Response truncated due to output length limit". The first object is the
    call the model meant to make; the rest are logged and dropped.
    """

    def test_three_concatenated_objects_keep_the_first(self, caplog):
        raw = (
            '{"limit": 200, "offset": 1, "path": "/a/billing-invoice-email.ts"}'
            '{"limit": 200, "offset": 1, "path": "/a/billing-mail.ts"}'
            '{"limit": 200, "offset": 1, "path": "/a/email-copy.ts"}'
        )
        result = _repair_tool_call_arguments(raw, "read_file")
        assert json.loads(result) == {
            "limit": 200,
            "offset": 1,
            "path": "/a/billing-invoice-email.ts",
        }
        assert "concatenated tool_call arguments" in caplog.text
        assert "dropped 2" in caplog.text

    def test_whitespace_and_newline_separated_objects_split(self):
        raw = '{"path": "/a"}\n\n{"path": "/b"}'
        assert json.loads(_repair_tool_call_arguments(raw, "read_file")) == {"path": "/a"}

    def test_adjacent_braces_inside_a_string_are_not_split(self):
        raw = '{"content": "a}{b", "path": "/x"}'
        assert json.loads(_repair_tool_call_arguments(raw, "write_file")) == {
            "content": "a}{b",
            "path": "/x",
        }

    def test_truncated_second_object_falls_through_to_other_passes(self):
        raw = '{"path": "/a"}{"path": "/b'
        # Must not raise; the single-object repairs take over as before.
        assert isinstance(_repair_tool_call_arguments(raw, "read_file"), str)
