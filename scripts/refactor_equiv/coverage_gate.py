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
        if isinstance(node, ast.If):
            if node.body and node.body[0].lineno not in covered:
                missing.append(node.body[0].lineno)
            if node.orelse and node.orelse[0].lineno not in covered:
                missing.append(node.orelse[0].lineno)
    if missing:
        raise CoverageGateError(
            f"branch coverage below 100% for {path}; missing branch lines {sorted(missing)}"
        )
