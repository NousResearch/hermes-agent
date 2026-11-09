"""``python -m scripts.code_health.ruff_runner`` runs the pyproject-pinned ruff, whatever is on PATH."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.code_health.ruff_runner import pinned_version

REPO = Path(__file__).resolve().parents[2]


def test_module_runs_the_pinned_ruff_with_the_given_arguments() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.code_health.ruff_runner", "--version"],
        cwd=REPO, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["ruff", pinned_version(REPO)]


def test_module_forwards_the_ruff_exit_status(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.code_health.ruff_runner", "check", "--isolated",
         "--no-cache", "--select", "F401", str(bad)],
        cwd=REPO, capture_output=True, text=True, timeout=300, stdin=subprocess.DEVNULL, check=False,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "F401" in proc.stdout
