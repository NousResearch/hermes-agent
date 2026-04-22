import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_AGENT = REPO_ROOT / "run_agent.py"


def test_run_agent_version_flag_prints_version_without_launching_demo():
    result = subprocess.run(
        [sys.executable, str(RUN_AGENT), "--version"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Hermes Agent v0.10.0 (2026.4.16)" in result.stdout
    assert "User Query:" not in result.stdout


def test_run_agent_version_subcommand_prints_version_without_launching_demo():
    result = subprocess.run(
        [sys.executable, str(RUN_AGENT), "version"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert "Hermes Agent v0.10.0 (2026.4.16)" in result.stdout
    assert "User Query:" not in result.stdout
