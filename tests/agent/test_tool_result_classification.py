"""Tests for shared tool result classification helpers."""

import json

import pytest

from agent.tool_result_classification import (
    file_mutation_result_landed,
    tool_nonexecution,
)


@pytest.mark.parametrize("status", ["blocked", "pending_approval"])
def test_legacy_terminal_refusals_are_not_execution_evidence(status):
    payload = json.dumps({"exit_code": -1, "status": status})
    assert tool_nonexecution("terminal", payload)[0] == status
    # A plugin's 'blocked' result might describe a remote response after effects.
    assert tool_nonexecution("mcp_custom", payload) is None


@pytest.mark.parametrize("payload", [
    {"exit_code": -1, "status": "timeout"},
    {"exit_code": 28, "error": "Connection timed out"},
    {"executed": True, "status": "error"},
    ["blocked"],
    "not json",
])
def test_execution_errors_do_not_prove_nonexecution(payload):
    assert tool_nonexecution("terminal", json.dumps(payload)) is None


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
