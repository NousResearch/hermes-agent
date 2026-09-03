"""Phase 2 read-only boundary guards for Task 19."""

from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_IMPORTS = frozenset(
    {
        "subprocess",
        "requests",
        "httpx",
        "urllib",
        "webbrowser",
        "sqlite3",
        "delegate_task",
    }
)

FORBIDDEN_CALL_NAMES = frozenset(
    {
        "atomic_write_json",
        "append_jsonl",
        "append_run_event",
        "append_task_event",
        "apply_task_transition",
        "apply_attempt_transition",
        "record_run_final_closure",
        "complete_run_manually",
        "review_run_manually",
        "plan_run_followup",
        "request_run_execution",
        "execute_run_execution_request",
        "verify_run_execution_result",
        "plan_post_verification_followup",
        "request_post_verification_execution",
        "record_post_verification_execution_result",
        "record_post_verification_execution_verification",
    }
)


def _collect_import_roots(tree: ast.AST) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    return imported


def _collect_called_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


def test_observe_module_import_boundary():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "htr" / "observe.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = _collect_import_roots(tree)
    assert FORBIDDEN_IMPORTS.isdisjoint(imported)
    assert imported <= {"__future__", "htr", "json", "dataclasses", "datetime", "os", "pathlib", "typing"}


def test_observe_module_does_not_call_mutators():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "htr" / "observe.py").read_text(encoding="utf-8")
    called = _collect_called_names(ast.parse(source))
    assert FORBIDDEN_CALL_NAMES.isdisjoint(called)


def test_action_plan_module_import_boundary():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "htr" / "action_plan.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = _collect_import_roots(tree)
    assert FORBIDDEN_IMPORTS.isdisjoint(imported)
    assert imported <= {
        "__future__",
        "htr",
        "hashlib",
        "json",
        "dataclasses",
        "pathlib",
        "typing",
    }


def test_action_plan_module_does_not_call_mutators():
    repo_root = Path(__file__).resolve().parents[2]
    source = (repo_root / "htr" / "action_plan.py").read_text(encoding="utf-8")
    called = _collect_called_names(ast.parse(source))
    assert FORBIDDEN_CALL_NAMES.isdisjoint(called)
