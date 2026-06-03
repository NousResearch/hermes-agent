"""Tests for _repair_tool_call_arguments — malformed JSON repair pipeline."""

import json
import math

from run_agent import (
    _extract_first_json_object,
    _repair_tool_call_arguments,
    _scrub_nonfinite_numbers,
    _split_concatenated_tool_call_arguments,
)


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


class TestSplitConcatenatedToolCallArguments:
    """Recovery for `{"a":1}{"b":2}` concatenated streaming payloads.

    Mirrors the upstream Hermes PR #25346 / #36039 contract: return a
    list of JSON strings only when the entire payload decodes losslessly
    into 2+ complete top-level dicts.
    """

    def test_two_dicts_split(self):
        raw = '{"entity":"Don Bowman"}{"entity":"Agilicus"}'
        result = _split_concatenated_tool_call_arguments(raw)
        assert result == ['{"entity":"Don Bowman"}', '{"entity":"Agilicus"}']

    def test_three_dicts_split(self):
        raw = '{"a":1}{"b":2}{"c":3}'
        result = _split_concatenated_tool_call_arguments(raw)
        assert result == ['{"a":1}', '{"b":2}', '{"c":3}']

    def test_whitespace_between_objects(self):
        raw = '{"a":1}  \n\t  {"b":2}'
        result = _split_concatenated_tool_call_arguments(raw)
        assert result == ['{"a":1}', '{"b":2}']

    def test_partial_tail_returns_none(self):
        """Truncated second object — must not split, existing truncation
        handling stays in control."""
        raw = '{"entity":"Don Bowman"}{"entity":"Agilic'
        assert _split_concatenated_tool_call_arguments(raw) is None

    def test_single_object_returns_none(self):
        """Single object, no concatenation — leave to the normal path."""
        assert _split_concatenated_tool_call_arguments('{"a":1}') is None

    def test_trailing_garbage_returns_none(self):
        """`{"a":1}garbage` is not a clean split — handled by the
        first-JSON-extraction pass instead."""
        assert _split_concatenated_tool_call_arguments('{"a":1}garbage') is None

    def test_array_payload_returns_none(self):
        """The splitter only rescues dicts; arrays belong to the regular
        repair path."""
        assert _split_concatenated_tool_call_arguments('[1,2][3,4]') is None

    def test_non_string_returns_none(self):
        assert _split_concatenated_tool_call_arguments(None) is None
        assert _split_concatenated_tool_call_arguments(123) is None


class TestExtractFirstJsonObject:
    """Recovery for valid JSON buried in surrounding noise.

    Examples: U+2028/U+2029 line separators, BOM markers, model preamble
    or postscript, leading non-breaking space, etc. — all of these
    make `json.loads` reject the whole payload even though there's a
    perfectly valid dict inside.
    """

    def test_bom_at_start(self):
        result = _extract_first_json_object('\ufeff{"a":1}')
        assert result == '{"a":1}'

    def test_bom_at_end(self):
        result = _extract_first_json_object('{"a":1}\ufeff')
        assert result == '{"a":1}'

    def test_line_separator_between_objects(self):
        """U+2028 is not whitespace per the strict JSON spec, so the
        second object must be discarded and the first rescued."""
        result = _extract_first_json_object('{"a":1}\u2028{"b":2}')
        assert result == '{"a":1}'

    def test_paragraph_separator(self):
        result = _extract_first_json_object('{"a":1}\u2029{"b":2}')
        assert result == '{"a":1}'

    def test_leading_nbsp(self):
        result = _extract_first_json_object('\u00a0{"a":1}')
        assert result == '{"a":1}'

    def test_trailing_garbage(self):
        result = _extract_first_json_object('{"a":1}garbage')
        assert result == '{"a":1}'

    def test_preamble_text(self):
        result = _extract_first_json_object('Here is the JSON: {"a":1}')
        assert result == '{"a":1}'

    def test_postscript_text(self):
        result = _extract_first_json_object('{"a":1} Hope this helps!')
        assert result == '{"a":1}'

    def test_extra_quoted_string_after(self):
        result = _extract_first_json_object('{"a":1}"excess"')
        assert result == '{"a":1}'

    def test_extra_boolean_after(self):
        result = _extract_first_json_object('{"a":1} true')
        assert result == '{"a":1}'

    def test_extra_null_after(self):
        result = _extract_first_json_object('{"a":1} null')
        assert result == '{"a":1}'

    def test_nested_object(self):
        result = _extract_first_json_object('noise {"outer": {"inner": 42}}')
        assert json.loads(result) == {"outer": {"inner": 42}}

    def test_no_dict_found_returns_none(self):
        assert _extract_first_json_object('garbage') is None
        assert _extract_first_json_object('') is None
        assert _extract_first_json_object(None) is None

    def test_first_candidate_must_be_dict(self):
        """If the first JSON value is a bare scalar, don't rescue it —
        almost always model noise."""
        assert _extract_first_json_object('42 noise') is None
        assert _extract_first_json_object('"string" noise') is None

    def test_brace_count_balanced_rejects_partial(self):
        """`{"a":` (unclosed) must not be returned — the parser needs a
        complete dict."""
        assert _extract_first_json_object('{"a":') is None


class TestScrubNonfiniteNumbers:
    """Replace NaN/Infinity floats with None so the re-serialised JSON
    is wire-valid for strict-validating providers (Anthropic, AWS
    Bedrock, Google Vertex).  Python's `json.loads(strict=False)`
    accepts NaN/Infinity tokens, but the round-trip is non-RFC-8259.
    """

    def test_nan_in_dict(self):
        result = _scrub_nonfinite_numbers({"a": float("nan")})
        assert result == {"a": None}
        assert json.dumps(result, separators=(",", ":")) == '{"a":null}'

    def test_infinity_in_dict(self):
        result = _scrub_nonfinite_numbers({"a": float("inf")})
        assert result == {"a": None}

    def test_negative_infinity_in_dict(self):
        result = _scrub_nonfinite_numbers({"a": float("-inf")})
        assert result == {"a": None}

    def test_nested_in_list(self):
        result = _scrub_nonfinite_numbers([float("nan"), 1, float("inf")])
        assert result == [None, 1, None]

    def test_deeply_nested(self):
        result = _scrub_nonfinite_numbers({"a": {"b": [float("nan")]}})
        assert result == {"a": {"b": [None]}}

    def test_normal_floats_untouched(self):
        result = _scrub_nonfinite_numbers({"a": 3.14, "b": 0.0, "c": -1.5})
        assert result == {"a": 3.14, "b": 0.0, "c": -1.5}

    def test_non_float_types_untouched(self):
        result = _scrub_nonfinite_numbers({"s": "NaN", "i": 42, "b": True, "n": None})
        assert result == {"s": "NaN", "i": 42, "b": True, "n": None}

    def test_in_place_returns_new_structure(self):
        """Doesn't mutate the input (returns a new dict/list)."""
        original = {"a": float("nan")}
        result = _scrub_nonfinite_numbers(original)
        assert math.isnan(original["a"])  # original untouched
        assert result == {"a": None}


class TestRepairToolCallArgumentsNewRepairs:
    """End-to-end coverage of the new repair stages: concatenated
    payloads, extracted JSON, and NaN/Inf scrubbing."""

    def test_concatenated_payload_returns_first_dict(self):
        """Single-call repair returns the first dict; the streaming
        split-aware path uses the full list."""
        result = _repair_tool_call_arguments('{"a":1}{"b":2}', "t")
        parsed = json.loads(result)
        assert parsed == {"a": 1}

    def test_extracted_json_from_bom(self):
        result = _repair_tool_call_arguments('\ufeff{"a":1}', "t")
        assert json.loads(result) == {"a": 1}

    def test_extracted_json_from_garbage_tail(self):
        result = _repair_tool_call_arguments('{"a":1}garbage', "t")
        assert json.loads(result) == {"a": 1}

    def test_extracted_json_from_preamble(self):
        result = _repair_tool_call_arguments('Here is the call: {"a":1}', "t")
        assert json.loads(result) == {"a": 1}

    def test_nan_scrubbed_to_null(self):
        result = _repair_tool_call_arguments('{"a": NaN}', "t")
        assert json.loads(result) == {"a": None}

    def test_infinity_scrubbed_to_null(self):
        result = _repair_tool_call_arguments('{"a": Infinity}', "t")
        assert json.loads(result) == {"a": None}

    def test_nested_nonfinite_all_scrubbed(self):
        result = _repair_tool_call_arguments(
            '{"a": 1, "b": NaN, "c": [NaN, Infinity]}', "t"
        )
        assert json.loads(result) == {"a": 1, "b": None, "c": [None, None]}

    def test_real_broken_json_still_falls_back(self):
        """Random text in braces (not a JSON-shaped object) must still
        fall through to the `{}` last resort."""
        result = _repair_tool_call_arguments('{not valid}', "t")
        assert result == "{}"

    def test_extracted_then_reserialised_under_strict(self):
        """The extracted output must round-trip under strict=True so it
        doesn't crash the upstream provider."""
        result = _repair_tool_call_arguments('{"a": 1}garbage', "t")
        json.loads(result, strict=True)  # must not raise


