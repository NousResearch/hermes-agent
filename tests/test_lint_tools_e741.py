"""Regression test for E741 (ambiguous variable name) in tools/. """

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TARGETS = [
    "tools/fuzzy_match.py",
    "tools/patch_parser.py",
]


class TestToolsE741:
    """Guard against E741 (ambiguous variable name) in tools/ files."""

    def test_tools_fuzzy_match_py_has_zero_e741_violations(self) -> None:
        """tools/fuzzy_match.py must have zero E741 violations."""
        target = REPO_ROOT / "tools" / "fuzzy_match.py"
        assert target.exists(), f"Target file not found: {target}"

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select=E741",
             "--output-format=concise", str(target)],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, (
            f"tools/fuzzy_match.py has E741 violations:\n"
            f"{result.stdout}\n"
        )

    def test_tools_patch_parser_py_has_zero_e741_violations(self) -> None:
        """tools/patch_parser.py must have zero E741 violations."""
        target = REPO_ROOT / "tools" / "patch_parser.py"
        assert target.exists(), f"Target file not found: {target}"

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select=E741",
             "--output-format=concise", str(target)],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, (
            f"tools/patch_parser.py has E741 violations:\n"
            f"{result.stdout}\n"
        )
