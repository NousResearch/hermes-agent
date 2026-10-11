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


def test_error_value_is_bounded_exit_code_stays_intact():
    """A huge ``error`` payload must not re-inflate the log line the 200-char preview
    bounds (#125248 review); ``exit_code`` stays untruncated."""
    huge = "E" * (200 * 1024)
    suffix = _error_status_suffix({"output": "x" * 500, "error": huge, "exit_code": 137})
    assert suffix.startswith(" exit_code=137")
    error_part = suffix.split(" error=", 1)[1]
    assert len(error_part) <= 200 + len("... (+204692 chars elided)") + 2  # repr quotes + marker
    assert error_part.startswith("'")
    assert error_part.endswith(f"... (+{200 * 1024 + 2 - 200} chars elided)'")
