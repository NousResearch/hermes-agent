"""Tests for shared tool result classification helpers."""

import json

from agent.tool_result_classification import (
    file_mutation_result_landed,
    structured_tool_failure_message,
    unwrap_tool_result_envelope,
)


def test_unwraps_json_application_payload_from_mcp_result_envelope():
    payload = {"ok": False, "error": {"message": "artifact already exists"}}
    wrapped = json.dumps({"result": json.dumps(payload)})

    assert unwrap_tool_result_envelope(wrapped) == payload


def test_structured_content_is_preferred_for_mcp_failure_classification():
    wrapped = {
        "result": "human-readable summary",
        "structuredContent": {
            "success": False,
            "message": "validation failed",
        },
    }

    assert structured_tool_failure_message(wrapped) == "validation failed"


def test_nested_mcp_success_is_not_a_structured_failure():
    wrapped = {"result": json.dumps({"ok": True, "data": "done"})}

    assert structured_tool_failure_message(wrapped) is None


def test_falsey_error_placeholder_is_not_a_structured_failure():
    wrapped = {"result": json.dumps({"success": True, "error": None})}

    assert structured_tool_failure_message(wrapped) is None


def test_concrete_error_outranks_explicit_success_marker():
    wrapped = {
        "result": json.dumps({"success": True, "error": "upstream rejected request"})
    }

    assert structured_tool_failure_message(wrapped) == "upstream rejected request"


def test_unknown_wrapper_metadata_prevents_nested_failure_inference():
    wrapped = {
        "result": json.dumps({"ok": False, "error": "rejected"}),
        "trace_id": "trace-123",
    }

    assert structured_tool_failure_message(wrapped) is None


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert file_mutation_result_landed("write_file", result) is True






def test_side_effect_classification_keeps_session_mutations():
    from agent.tool_result_classification import tool_may_have_side_effect

    assert tool_may_have_side_effect("todo") is True
    assert tool_may_have_side_effect("memory") is True
    assert tool_may_have_side_effect("write_file") is True
    assert tool_may_have_side_effect("mcp_unknown") is True
    assert tool_may_have_side_effect("read_file") is False
    assert tool_may_have_side_effect("web_search") is False
