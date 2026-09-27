"""Regression test for the tool-error log status suffix (#125238)."""
import json

from agent.tool_executor import _error_status_suffix


def test_terminal_result_suffix_exposes_exit_code_and_error():
    serialized = json.dumps({"output": "boom\n" * 100, "exit_code": 1, "error": None})
    assert _error_status_suffix(serialized) == " exit_code=1 error=None"


def test_accepts_dict_result_directly():
    assert _error_status_suffix({"output": "x", "exit_code": 2, "error": "nope"}) == " exit_code=2 error='nope'"


def test_non_object_or_plain_string_contributes_nothing():
    assert _error_status_suffix("plain traceback text") == ""
    assert _error_status_suffix("[1, 2, 3]") == ""
    assert _error_status_suffix(42) == ""


def test_result_without_status_fields_adds_no_suffix():
    assert _error_status_suffix(json.dumps({"output": "x"})) == ""
