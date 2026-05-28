"""Regression tests: hermes_cli/tools_config.py lint cleanliness."""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_tools_config_has_zero_e741_violations() -> None:
    """hermes_cli/tools_config.py must have zero E741 (ambiguous variable name) violations."""
    target = REPO_ROOT / "hermes_cli" / "tools_config.py"
    assert target.exists(), f"Target not found: {target}"

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--select=E741",
         "--output-format=concise", str(target)],
        capture_output=True, text=True, check=False,
    )

    assert result.returncode == 0, (
        f"hermes_cli/tools_config.py has E741 violation(s):\n{result.stdout}"
    )
