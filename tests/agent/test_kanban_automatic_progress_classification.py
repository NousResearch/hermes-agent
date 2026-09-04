"""Classifier for kanban automatic progress evidence."""

from __future__ import annotations

import json

import pytest

from agent.tool_result_classification import classify_automatic_progress_evidence


def _terminal_result(exit_code: int = 0) -> str:
    return json.dumps({"exit_code": exit_code, "output": "ok"})


@pytest.mark.parametrize(
    ("tool_name", "args", "result"),
    [
        (
            "write_file",
            {"path": "/tmp/secret.py", "content": "x"},
            json.dumps({"bytes_written": 4, "verified": True}),
        ),
        (
            "patch",
            {"mode": "replace", "path": "/tmp/secret.py", "old_string": "a", "new_string": "b"},
            json.dumps({"success": True, "files_modified": ["/tmp/secret.py"]}),
        ),
        ("terminal", {"command": "pytest tests/unit"}, _terminal_result()),
        ("terminal", {"command": "uv run pytest -q"}, _terminal_result()),
        ("terminal", {"command": "python3 -m pytest -q"}, _terminal_result()),
        ("terminal", {"command": "npm run build"}, _terminal_result()),
        ("terminal", {"command": "pnpm run typecheck"}, _terminal_result()),
        ("terminal", {"command": "ruff check ."}, _terminal_result()),
        ("terminal", {"command": "python -m py_compile module.py"}, _terminal_result()),
        ("terminal", {"command": "git commit -m 'msg'"}, _terminal_result()),
        ("terminal", {"command": "graphify update ."}, _terminal_result()),
        ("terminal", {"command": "bash scripts/run_tests.sh"}, _terminal_result()),
    ],
)
def test_classifier_accepts_qualifying_success(tool_name, args, result):
    evidence = classify_automatic_progress_evidence(tool_name, args, result)
    assert evidence is not None
    evidence_type, detail = evidence
    assert evidence_type
    assert detail
    # Payload must never echo tool args or result fields.
    blob = json.dumps({"evidence_type": evidence_type, "detail": detail}).lower()
    assert "/tmp" not in blob
    assert "secret.py" not in blob
    assert "pytest" not in blob
    assert "npm" not in blob
    assert "module.py" not in blob
    assert "msg" not in blob


def test_classifier_uses_public_evidence_type_contract():
    assert classify_automatic_progress_evidence(
        "terminal", {"command": "uv run pytest -q"}, _terminal_result(),
    ) == ("tests_passed", "tests passed")
    assert classify_automatic_progress_evidence(
        "terminal", {"command": "npm run build"}, _terminal_result(),
    ) == ("build_passed", "build passed")
    assert classify_automatic_progress_evidence(
        "terminal", {"command": "git commit -m safe"}, _terminal_result(),
    ) == ("commit_created", "commit created")


@pytest.mark.parametrize(
    ("tool_name", "args", "result"),
    [
        ("read_file", {"path": "/tmp/x"}, json.dumps({"content": "hi"})),
        ("search_files", {"pattern": "foo"}, json.dumps({"matches": []})),
        ("terminal", {"command": "ls -la"}, _terminal_result()),
        ("terminal", {"command": "cat README.md"}, _terminal_result()),
        ("terminal", {"command": "pytest"}, _terminal_result(exit_code=1)),
        (
            "write_file",
            {"path": "/tmp/x", "content": "a"},
            json.dumps({"error": "permission denied"}),
        ),
        (
            "patch",
            {"mode": "replace", "path": "/tmp/x", "old_string": "a", "new_string": "b"},
            json.dumps({"success": True, "no_change": True}),
        ),
        (
            "terminal",
            {"command": "pytest"},
            json.dumps(
                {
                    "error": "Tool execution cancelled by user interrupt",
                    "status": "cancelled",
                }
            ),
        ),
        (
            "terminal",
            {"command": "pytest"},
            json.dumps({"error": "timed out after 30.0s", "status": "timeout"}),
        ),
    ],
)
def test_classifier_rejects_non_qualifying(tool_name, args, result):
    assert classify_automatic_progress_evidence(tool_name, args, result) is None


def test_classifier_rejects_masking_shell_control():
    assert classify_automatic_progress_evidence(
        "terminal",
        {"command": "pytest || true"},
        _terminal_result(),
    ) is None
    assert classify_automatic_progress_evidence(
        "terminal",
        {"command": "pytest | tail"},
        _terminal_result(),
    ) is None
