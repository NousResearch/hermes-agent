"""Tests for shared tool result classification helpers."""

import json
import logging

from agent.tool_result_classification import file_mutation_result_landed


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert file_mutation_result_landed("write_file", result) is True


def test_patch_with_nested_lsp_diagnostics_counts_as_landed():
    result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert file_mutation_result_landed("patch", result) is True


def test_top_level_file_mutation_error_does_not_count_as_landed():
    result = json.dumps({"success": True, "error": "post-write verification failed"})

    assert file_mutation_result_landed("patch", result) is False


def test_non_json_result_returns_false_and_logs_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger="agent.tool_result_classification"):
        assert file_mutation_result_landed("patch", "not json") is False

    records = [
        record
        for record in caplog.records
        if record.name == "agent.tool_result_classification"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "non-JSON" in records[0].getMessage()
