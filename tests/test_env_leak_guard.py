"""Regression guard: no test module may os.environ-copy from the filesystem at import time.

This catches the bug in #80343 where tests/run_agent/test_sequential_chats_live.py
had a module-level _load_user_env() that read ~/.hermes/.env and wrote into
os.environ at *collection* time — before pytest's autouse fixtures (which scrub
credentials) had run.  The result was non-deterministic test failures that
depended on what the developer happened to have in their local .env file.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest


def _find_test_modules(root: Path = Path(__file__).parent) -> list[Path]:
    """Yield all Python files under root that live in a tests/ directory."""
    return sorted(root.glob("**/test_*.py"))


def _module_scope_assigns_read_file(code: str) -> list[tuple[int, str]]:
    """Walk module-level AST and return (lineno, pattern) for any os.environ mutation from file read.

    Tracks direct patterns:
      - os.environ[k] = <file-read>
      - os.environ.setdefault(k, <file-read>)
      - os.environ.update({k: <file-read>})
    Does NOT flag deterministic os.environ["CONSTANT"] = "value" at module scope.
    """
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    violations = []
    for node in ast.walk(tree):
        # Only look at module-level nodes (not inside functions/classes)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue

        # os.environ[<expr>] = <value>   or   os.environ.update({...})
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if _is_environ_getitem(target):
                    val = node.value
                    if _reads_file(val):
                        violations.append((node.lineno, "_environ[key] = <file-read>"))
            continue

        # os.environ.setdefault(k, <value>)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                _is_environ_setdefault(call)
                and call.args
                and _reads_file(call.args[1])
            ):
                violations.append(
                    (node.lineno, "os.environ.setdefault(key, <file-read>)")
                )
            continue

        # os.environ.update({k: <file-read>})
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if _is_environ_update(call):
                for kw in call.keywords:
                    if kw.arg is None:  # **dict form
                        if _dict_value_reads_file(kw.value):
                            violations.append(
                                (node.lineno, "os.environ.update({k: <file-read>})")
                            )
                for arg in call.args:
                    if isinstance(arg, ast.Dict):
                        for k, v in zip(arg.keys, arg.values):
                            if k and isinstance(k, ast.Str | ast.Constant) and _reads_file(v):
                                violations.append(
                                    (node.lineno, "os.environ.update({k: <file-read>})")
                                )

    return violations


# --- helpers ---


def _is_environ_getitem(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "os"
        and node.value.attr == "environ"
    )


def _is_environ_setdefault(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "os"
        and call.func.attr == "setdefault"
    )


def _is_environ_update(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "os"
        and call.func.attr == "update"
    )


def _is_path_read(node: ast.expr) -> bool:
    """True for Path(...).read_text(), open(...).read(), etc."""
    if isinstance(node, ast.Call):
        f = node.func
        # Path(...).read_text() / Path(...).read_bytes()
        if isinstance(f, ast.Attribute) and f.attr in ("read_text", "read_bytes"):
            return True
        # open(...).read() / open(...).readlines()
        if isinstance(f, ast.Name) and f.id == "open":
            return True
    return False


def _reads_file(node: ast.expr) -> bool:
    if _is_path_read(node):
        return True
    if isinstance(node, ast.Call) and len(node.args) == 1:
        return _reads_file(node.args[0])
    return False


def _dict_value_reads_file(dikt: ast.expr) -> bool:
    if isinstance(dikt, ast.Dict):
        return any(_reads_file(v) for v in dikt.values if v is not None)
    return False


# --- the actual test ---


@pytest.mark.parametrize(
    "module",
    _find_test_modules(),
    ids=lambda p: p.relative_to(Path(__file__).parent).as_posix(),
)
def test_no_file_read_to_environ_at_module_scope(module: Path):
    """Verify no test module writes the result of a file read into os.environ at import time."""
    code = module.read_text()
    violations = _module_scope_assigns_read_file(code)
    if violations:
        lines = "\n".join(
            f"  line {ln}: {pattern}" for ln, pattern in violations
        )
        pytest.fail(
            f"{module.relative_to(Path(__file__).parent).as_posix()} "
            f"writes file-read values into os.environ at module scope:\n{lines}\n"
            "This causes non-deterministic test failures because it runs at "
            "pytest collection time — before autouse fixtures (like "
            "_hermetic_environment) have scrubbed developer-local credential "
            "variables from os.environ. Move such code behind a fixture or "
            "inside a test function."
        )
