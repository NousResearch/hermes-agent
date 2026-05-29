"""Tests for shared tool result classification helpers."""

import json

from agent.tool_result_classification import (
    FileMutationOutcome,
    classify_file_mutation_result,
    file_mutation_result_landed,
)


def test_write_file_with_nested_lint_error_counts_as_landed():
    result = json.dumps({
        "bytes_written": 12,
        "lint": {"status": "error", "output": "SyntaxError: invalid syntax"},
    })

    assert classify_file_mutation_result("write_file", result) is FileMutationOutcome.LANDED
    assert file_mutation_result_landed("write_file", result) is True


def test_patch_with_nested_lsp_diagnostics_counts_as_landed():
    result = json.dumps({
        "success": True,
        "diff": "--- a/tmp.py\n+++ b/tmp.py\n",
        "lsp_diagnostics": "<diagnostics>ERROR [1:1] type mismatch</diagnostics>",
    })

    assert classify_file_mutation_result("patch", result) is FileMutationOutcome.LANDED
    assert file_mutation_result_landed("patch", result) is True


def test_protected_refusal_metadata_classifies_separately_from_failed_edit():
    result = json.dumps({
        "error": "Refusing to write to sensitive system path: /etc/example.conf",
        "file_mutation_status": "protected_refusal",
        "refusal_reason": "sensitive_system_path",
    })

    assert classify_file_mutation_result("write_file", result) is FileMutationOutcome.PROTECTED_REFUSAL
    assert file_mutation_result_landed("write_file", result) is False


def test_protected_refusal_with_appended_tool_loop_warning_stays_protected():
    result = json.dumps({
        "error": "Refusing to write to sensitive system path: /etc/example.conf",
        "file_mutation_status": "protected_refusal",
        "refusal_reason": "sensitive_system_path",
    }) + "\n\n[Tool loop warning: identical failed call repeated]"

    assert classify_file_mutation_result("write_file", result) is FileMutationOutcome.PROTECTED_REFUSAL
    assert file_mutation_result_landed("write_file", result) is False


def test_top_level_file_mutation_error_does_not_count_as_landed():
    result = json.dumps({"success": True, "error": "post-write verification failed"})

    assert classify_file_mutation_result("patch", result) is FileMutationOutcome.FAILED_EDIT
    assert file_mutation_result_landed("patch", result) is False
