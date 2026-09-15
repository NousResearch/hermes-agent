"""Tests for shared tool result classification helpers."""

import json

from agent.tool_result_classification import (
    file_mutation_result_landed,
    is_skill_view_dedup_result,
)


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


def test_skill_view_dedup_result_requires_complete_typed_shape():
    payload = {
        "success": False,
        "status": "deduplicated",
        "dedup": True,
        "content_returned": False,
        "error": "already loaded",
    }

    assert is_skill_view_dedup_result("skill_view", payload) is True
    assert is_skill_view_dedup_result("skill_view", json.dumps(payload)) is True
    assert is_skill_view_dedup_result("read_file", payload) is False
    assert is_skill_view_dedup_result(
        "skill_view", {**payload, "status": "unchanged"}
    ) is False
