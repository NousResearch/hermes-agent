"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json

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

    # -- Valid JSON passthrough (this path is via except, but still works) --


    # -- Combined repairs --



    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).




    # -- Stage 4: control-char escape fallback --




class TestInvalidEscapeRepair:
    """Fine-grained tool streaming skips server-side JSON validation, so a completed
    tool_use block can carry invalid escapes (``C:\\path``, a regex ``\\d``) beside raw
    control chars. Both must repair to the literal text instead of ``{}``
    (openclaw/openclaw#141323)."""

    def test_invalid_escapes_and_control_chars_repair_to_literal_text(self):
        raw = '{"path":"C:\\path\\dir","x":"a\tb","re":"\\d+ \\u12"}'
        parsed = json.loads(_repair_tool_call_arguments(raw, "write_file"))
        assert parsed == {"path": "C:\\path\\dir", "x": "a\tb", "re": "\\d+ \\u12"}

    def test_valid_escapes_preserved_and_truncation_still_fails_closed(self):
        valid = '{"a":"line\\nnext","b":"\\u00e9","c":"say \\"hi\\""}'
        assert json.loads(_repair_tool_call_arguments(valid, "t")) == {"a": "line\nnext", "b": "é", "c": 'say "hi"'}
        # A cut-off command with a bad escape must never shorten into an executable one.
        assert _repair_tool_call_arguments('{"command":"rm -rf \\d ', "terminal") == "{}"
