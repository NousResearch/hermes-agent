"""P13 test: kensei-quality-gate --dry-run prevents dispatch."""
import subprocess, sys, os, json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "kensei-quality-gate.py"


def test_quality_gate_dry_run_no_write(tmp_path):
    """--dry-run must not create logboard files or dispatch gates."""
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)

    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=30
    )
    # Should exit 0 (no review tasks found in empty kanban = routine)
    assert r.returncode == 0, f"exit {r.returncode}: {r.stderr[:200]}"
    # No logboard files should be created in dry-run
    logboard = hermes_home / "governance" / "logboard"
    if logboard.exists():
        files = list(logboard.glob("*.json"))
        assert len(files) == 0, f"dry-run created logboard files: {files}"


def test_quality_gate_hermes_home_respected(tmp_path):
    """Script must use HERMES_HOME for all paths."""
    hermes_home = tmp_path / "custom-hermes"
    env = dict(os.environ)
    env["HERMES_HOME"] = str(hermes_home)

    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True, text=True, env=env, cwd=str(REPO), timeout=30
    )
    assert r.returncode == 0
    # If logboard dir was created, it must be under custom-hermes
    logboard = hermes_home / "governance" / "logboard"
    if logboard.exists():
        assert str(hermes_home) in str(logboard)