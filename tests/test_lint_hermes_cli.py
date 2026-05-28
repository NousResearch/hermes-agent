"""Regression tests for E741 (ambiguous variable name) in hermes_cli/."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestHermesCliE741:
    """Guard against E741 (ambiguous variable name) in hermes_cli/."""

    def test_hermes_cli_doctor_py_has_zero_e741_violations(self) -> None:
        """hermes_cli/doctor.py must have zero E741 violations."""
        target = REPO_ROOT / "hermes_cli" / "doctor.py"
        assert target.exists(), f"Target file not found: {target}"

        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select=E741",
             "--output-format=concise", str(target)],
            capture_output=True, text=True, check=False,
        )

        assert result.returncode == 0, (
            f"hermes_cli/doctor.py has E741 violations:\n"
            f"{result.stdout}\n"
        )
