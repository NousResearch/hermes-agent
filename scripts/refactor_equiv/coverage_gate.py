"""Branch coverage gate for extraction corpora."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Callable


class CoverageGateError(AssertionError):
    """Raised when the corpus leaves an extracted module branch uncovered."""


def require_full_branch_coverage(module_path: str | Path, exercise: Callable[[], object]) -> None:
    path = str(Path(module_path).resolve())
    try:
        import coverage
    except ModuleNotFoundError:
        _stdlib_branch_gate(Path(path), exercise)
        return

    cov = coverage.Coverage(branch=True, include=[path])
    cov.start()
    try:
        exercise()
    finally:
        cov.stop()
        cov.save()
    percent = cov.report(morfs=[path], file=None, show_missing=True)
    if percent < 100.0:
        raise CoverageGateError(f"branch coverage {percent:.2f}% for {path}; required 100%")


def _stdlib_branch_gate(path: Path, exercise: Callable[[], object]) -> None:
    covered: set[int] = set()
    target = str(path)

    def trace(frame, event, arg):
        if event == "line" and frame.f_code.co_filename == target:
            covered.add(frame.f_lineno)
        return trace

    old = sys.gettrace()
    sys.settrace(trace)
    try:
        exercise()
    finally:
        sys.settrace(old)

    missing: list[int] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # Every stdlib-visible branch construct, not just `if` (Greptile P2):
        # if/elif arms + else, while/for bodies + their else clauses, except
        # handlers, and ternary (IfExp) arms. Boolean short-circuit operands
        # share a line and are not separable by a line tracer — the real
        # `coverage` package (branch=True) is the stronger gate when installed.
        if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            if node.body and node.body[0].lineno not in covered:
                missing.append(node.body[0].lineno)
            if node.orelse and node.orelse[0].lineno not in covered:
                missing.append(node.orelse[0].lineno)
        elif isinstance(node, ast.Try):
            for handler in node.handlers:
                if handler.body and handler.body[0].lineno not in covered:
                    missing.append(handler.body[0].lineno)
            if node.orelse and node.orelse[0].lineno not in covered:
                missing.append(node.orelse[0].lineno)
        elif isinstance(node, ast.IfExp):
            for arm in (node.body, node.orelse):
                if arm.lineno not in covered:
                    missing.append(arm.lineno)
    if missing:
        raise CoverageGateError(
            f"branch coverage below 100% for {path}; missing branch lines {sorted(set(missing))}"
        )
