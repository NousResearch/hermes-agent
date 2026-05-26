"""Lint regression tests for scripts/contributor_audit.py.

These tests assert that the file stays clean of specific lint rule classes
that the project's default ruff config does not enable. Without these guards,
violations can silently accumulate over time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TARGET = "scripts/contributor_audit.py"


def _ruff_check(select: str) -> subprocess.CompletedProcess:
    """Run ruff check with a specific --select filter on the target file."""
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", TARGET, "--select", select,
         "--output-format=concise"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_contributor_audit_has_zero_f_class_violations() -> None:
    """scripts/contributor_audit.py must have zero F-class violations.

    Covers: F401 (unused imports), F841 (unused variables), and all other
    F-class rules (F811 redefinitions, F821 undefined names, etc.).
    """
    result = _ruff_check("F")
    assert result.returncode == 0, (
        f"{TARGET} has F-class violations:\n{result.stdout}\n{result.stderr}"
    )


def test_contributor_audit_has_zero_ruf100_violations() -> None:
    """scripts/contributor_audit.py must have zero RUF100 (unused noqa) violations."""
    result = _ruff_check("RUF100")
    assert result.returncode == 0, (
        f"{TARGET} has RUF100 violations:\n{result.stdout}\n{result.stderr}"
    )
