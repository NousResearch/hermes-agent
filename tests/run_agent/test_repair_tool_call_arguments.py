"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json
import pytest

from run_agent import _repair_tool_call_arguments


class TestRepairToolCallArguments:
    """Verify each repair stage in the pipeline."""

    # -- Stage 1: empty / whitespace-only --

    def test_empty_string_returns_empty_object(self):
        assert _repair_tool_call_arguments("", "t") == "{}"

    def test_whitespace_only_returns_empty_object(self):
        assert _repair_tool_call_arguments("   \n\t  ", "t") == "{}"

    def test_none_type_returns_empty_object(self):
        """Non-string input (e.g. None from a broken model response)."""
        assert _repair_tool_call_arguments(None, "t") == "{}"

    # -- Stage 2: Python None literal --

    def test_python_none_literal(self):
        assert _repair_tool_call_arguments("None", "t") == "{}"

    def test_python_none_with_whitespace(self):
        assert _repair_tool_call_arguments("  None  ", "t") == "{}"

    # -- Stage 3: trailing comma repair --

    def test_trailing_comma_in_object(self):
        result = _repair_tool_call_arguments('{"key": "value",}', "t")
        assert json.loads(result) == {"key": "value"}

    def test_trailing_comma_in_array(self):
        result = _repair_tool_call_arguments('{"a": [1, 2,]}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": [1, 2]}

    def test_multiple_trailing_commas(self):
        result = _repair_tool_call_arguments('{"a": 1, "b": 2,}', "t")
        parsed = json.loads(result)
        assert parsed["a"] == 1
        assert parsed["b"] == 2

    # -- Stage 4: unclosed brackets --

    def test_unclosed_brace(self):
        result = _repair_tool_call_arguments('{"key": "value"', "t")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_unclosed_bracket_and_brace(self):
        result = _repair_tool_call_arguments('{"a": [1, 2', "t")
        # Bracket counting adds ']' then '}', producing {"a": [1, 2]}
        # which is valid JSON.  But the naive count can't always recover
        # complex nesting — verify we at least get valid JSON.
        json.loads(result)

    # -- Stage 5: excess closing delimiters --

    def test_extra_closing_brace(self):
        result = _repair_tool_call_arguments('{"key": "value"}}', "t")
        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_extra_closing_bracket(self):
        result = _repair_tool_call_arguments('{"a": [1]]}', "t")
        # Should produce valid JSON
        json.loads(result)

    # -- Stage 6: last resort --

    def test_unrepairable_garbage_returns_empty_object(self):
        assert _repair_tool_call_arguments("totally not json", "t") == "{}"

    def test_unrepairable_partial_returns_empty_object(self):
        # Truncated in the middle of a string key — bracket closing won't help
        assert _repair_tool_call_arguments('{"truncated": "val', "t") == "{}"

    # -- Valid JSON passthrough (this path is via except, but still works) --

    def test_already_valid_json_passes_through(self):
        """When json.loads fails for a non-JSON reason (shouldn't normally
        happen), but the repair pipeline still produces valid output."""
        raw = '{"path": "/tmp/foo", "content": "hello"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed["path"] == "/tmp/foo"

    # -- Combined repairs --

    def test_trailing_comma_plus_unclosed_brace(self):
        result = _repair_tool_call_arguments('{"a": 1, "b": 2,', "t")
        # Trailing comma stripped first, then closing brace added.
        # May or may not fully recover — verify valid JSON at minimum.
        json.loads(result)

    def test_real_world_glm_truncation(self):
        """Simulates GLM-5.1 truncating mid-argument."""
        raw = '{"command": "ls -la /tmp", "timeout": 30, "background":'
        result = _repair_tool_call_arguments(raw, "terminal")
        # Should at least be valid JSON, even if background is lost
        json.loads(result)

    # -- Stage 0: strict=False (literal control chars in strings) --
    # llama.cpp backends sometimes emit literal tabs/newlines inside JSON
    # string values. strict=False accepts these; we re-serialise to the
    # canonical wire form (#12068).

    def test_literal_newline_inside_string_value(self):
        raw = '{"summary": "line one\nline two"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"summary": "line one\nline two"}

    def test_literal_tab_inside_string_value(self):
        raw = '{"summary": "col1\tcol2"}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed == {"summary": "col1\tcol2"}

    def test_literal_control_char_reserialised_to_wire_form(self):
        """After repair, the output must parse under strict=True."""
        raw = '{"msg": "has\tliteral\ttabs"}'
        result = _repair_tool_call_arguments(raw, "t")
        # strict=True must now accept this
        parsed = json.loads(result)
        assert parsed["msg"] == "has\tliteral\ttabs"

    # -- Stage 4: control-char escape fallback --

    def test_control_chars_with_trailing_comma(self):
        """strict=False fails due to trailing comma, but brace-count pass
        + control-char escape rescues it."""
        raw = '{"msg": "line\none",}'
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert "line" in parsed["msg"]

    # -- Stage 0.5: concatenated top-level JSON objects (#25333) --
    # gemini-3-flash-preview occasionally emits parallel tool-call args
    # merged into one buffer as `}{` with no delimiter. raw_decode parses
    # the first complete object; we keep that and drop the trailing
    # concatenation with a clear log line so at least one tool call lands.

    def test_two_concatenated_objects_keeps_first(self):
        """Real-world repro from yantrikos/yantrikdb-hermes-plugin#5:
        gemini-3-flash-preview emitted two complete ``yantrikdb_relate``
        argument objects back-to-back with no delimiter."""
        raw = (
            '{"entity": "Don Bowman", "relationship": "ceo_of", "target": "Agilicus"}'
            '{"entity": "Don Bowman", "relationship": "works_at", "target": "Agilicus"}'
        )
        result = _repair_tool_call_arguments(raw, "yantrikdb_relate")
        parsed = json.loads(result)
        # First object preserved, second dropped (caller can still see the
        # warning log line and act on it if needed).
        assert parsed == {
            "entity": "Don Bowman",
            "relationship": "ceo_of",
            "target": "Agilicus",
        }

    def test_concatenated_objects_with_braces_inside_strings(self):
        """The boundary detector must not be confused by `}{` characters
        that appear *inside* a JSON string value."""
        raw = (
            '{"snippet": "if (cfg) { run(); }", "ok": true}'
            '{"snippet": "second call", "ok": false}'
        )
        result = _repair_tool_call_arguments(raw, "t")
        parsed = json.loads(result)
        assert parsed["snippet"] == "if (cfg) { run(); }"
        assert parsed["ok"] is True

    def test_three_concatenated_objects_keeps_only_first(self):
        """When the model glues N>2 objects, still only the first is
        recovered. The remaining N-1 are dropped (with a warning) — a
        higher-fidelity recovery is a fix for the streaming accumulator,
        not this repair pass."""
        raw = '{"i": 1}{"i": 2}{"i": 3}'
        result = _repair_tool_call_arguments(raw, "t")
        assert json.loads(result) == {"i": 1}

    def test_first_object_invalid_then_trailing_object_still_falls_back(self):
        """If the first object is itself malformed, raw_decode fails and
        the trailing-object pass shouldn't claim to recover anything.
        We expect the existing brace-counting heuristic to take over (or
        the final {} fallback if all repair stages fail)."""
        raw = '{"a": 1,}{"b": 2}'
        result = _repair_tool_call_arguments(raw, "t")
        # The downstream trailing-comma + raw_decode flow recovers {"a":1}
        # from the leading object. Either {"a": 1} or {} is acceptable here;
        # what we MUST NOT do is silently emit invalid JSON.
        parsed = json.loads(result)
        assert parsed in ({}, {"a": 1})

    def test_concatenated_at_array_top_level(self):
        """raw_decode works equally well for top-level arrays, though
        tool_call args are usually objects. Cover the case for safety."""
        raw = '[1, 2, 3][4, 5]'
        result = _repair_tool_call_arguments(raw, "t")
        assert json.loads(result) == [1, 2, 3]

    def test_single_object_with_trailing_whitespace_is_not_treated_as_concatenation(self):
        """The raw_decode pass should not log a warning or drop content when
        the only trailing content is whitespace — that's a normal valid
        input, handled by the strict=False pass above."""
        raw = '{"key": "value"}   \n\t'
        result = _repair_tool_call_arguments(raw, "t")
        assert json.loads(result) == {"key": "value"}

