"""Regression tests for non-interactive `hermes tools --summary`."""

import os
import subprocess
import sys


def test_tools_summary_runs_without_tty(tmp_path):
    """`hermes tools --summary` is documented as print-and-exit, not interactive."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env["NO_COLOR"] = "1"

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "tools", "--summary"],
        cwd=str(tmp_path),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    assert "Tool Summary" in result.stdout
    assert "requires an interactive terminal" not in result.stderr + result.stdout
