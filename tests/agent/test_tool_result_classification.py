"""Tests for shared tool result classification helpers."""

import json

from agent.tool_result_classification import (
    file_mutation_result_landed,
    file_mutation_validation_failed,
)


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert file_mutation_result_landed("write_file", result) is True


def test_validation_failure_is_landed_but_requires_repair():
    result = json.dumps({
        "error": "VALIDATION FAILED AFTER EDIT",
        "applied": True,
        "validated": False,
    })

    assert file_mutation_result_landed("patch", result) is True
    assert file_mutation_validation_failed("patch", result) is True


def test_partial_v4a_apply_failure_is_landed_but_is_an_ordinary_error():
    result = json.dumps({
        "success": False,
        "files_created": ["first.py"],
        "applied": True,
        "error": (
            "Apply phase failed (state may be inconsistent — run `git diff` "
            "to assess): second.py: file not found"
        ),
    })

    assert file_mutation_result_landed("patch", result) is True
    assert file_mutation_validation_failed("patch", result) is False






def test_side_effect_classification_keeps_session_mutations():
    from agent.tool_result_classification import tool_may_have_side_effect

    assert tool_may_have_side_effect("todo") is True
    assert tool_may_have_side_effect("memory") is True
    assert tool_may_have_side_effect("write_file") is True
    assert tool_may_have_side_effect("mcp_unknown") is True
    assert tool_may_have_side_effect("read_file") is False
    assert tool_may_have_side_effect("web_search") is False
