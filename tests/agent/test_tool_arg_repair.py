"""Tests for the conservative tool-argument repair (literal control chars in strings)."""

import json

import pytest

from agent.tool_arg_repair import (
    diagnose_invalid_arguments,
    escape_control_chars_in_strings,
    repair_tool_arguments,
)


def test_literal_newline_inside_a_string_is_escaped():
    raw = '{"goal": "line one\nline two", "n": 1}'
    repaired, _ = repair_tool_arguments(raw)

    assert repaired is not None
    assert json.loads(repaired) == {"goal": "line one\nline two", "n": 1}


def test_tabs_and_carriage_returns_are_escaped():
    raw = '{"context": "a\tb\r\nc"}'
    repaired, _ = repair_tool_arguments(raw)

    assert json.loads(repaired)["context"] == "a\tb\r\nc"


def test_values_and_key_order_are_preserved():
    raw = '{"tasks": [{"goal": "x\ny", "context": "keep \\"quotes\\" and \\\\slashes"}]}'
    repaired, _ = repair_tool_arguments(raw)
    parsed = json.loads(repaired)

    assert list(parsed) == ["tasks"]
    assert parsed["tasks"][0]["context"] == 'keep "quotes" and \\slashes'


def test_valid_json_is_returned_untouched():
    raw = '{"goal": "a\\nb"}'
    repaired, diagnosis = repair_tool_arguments(raw)

    assert repaired == raw
    assert diagnosis is None


def test_control_chars_outside_strings_are_left_alone():
    raw = '{"a": 1,\n  "b": 2}'
    out, count = escape_control_chars_in_strings(raw)

    assert count == 0
    assert out == raw


def test_existing_escapes_are_not_double_escaped():
    raw = '{"a": "1\\n2", "b": "3\n4"}'
    repaired, _ = repair_tool_arguments(raw)

    assert json.loads(repaired) == {"a": "1\n2", "b": "3\n4"}


def test_truncated_object_is_not_guessed_at():
    raw = '{"goal": "unterminated'
    repaired, diagnosis = repair_tool_arguments(raw)

    assert repaired is None
    assert "goal" in diagnosis


def test_diagnosis_names_the_field_and_the_escape_rule():
    bad = '{"tasks": [{"goal": "ok"}, {"goal": "bad'
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad)
    try:
        json.loads(bad)
    except json.JSONDecodeError as exc:
        diagnosis = diagnose_invalid_arguments(bad, exc)

    assert "char" in diagnosis
    assert "goal" in diagnosis
