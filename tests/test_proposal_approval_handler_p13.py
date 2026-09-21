"""P13 dry-run + HERMES_HOME contract test for proposal_approval_handler.py.

--dry-run must short-circuit before any Discord API poll or kanban creation.
All paths (proposals dir, state, .env) resolve via HERMES_HOME, not hardcoded
/home/kensei/.hermes.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "proposal_approval_handler.py"


def _run(args=None, env_extra=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, str(SCRIPT)]
    if args:
        cmd.extend(args)
    return subprocess.run(
        cmd, capture_output=True, text=True,
        env=env, cwd=str(REPO_ROOT), timeout=15,
    )


def test_dry_run_exits_zero_and_prints_plan(tmp_path):
    """--dry-run prints the plan and exits 0 without polling Discord."""
    env = {"HERMES_HOME": str(tmp_path / "hermes")}
    r = _run(["--dry-run"], env)
    assert r.returncode == 0, r.stderr
    assert "[dry-run] would poll" in r.stdout


def test_dry_run_never_contacts_discord(tmp_path):
    """Dry-run must not attempt any network call."""
    env = {"HERMES_HOME": str(tmp_path / "hermes")}
    r = _run(["--dry-run"], env)
    assert r.returncode == 0, r.stderr
    assert "discord.com" not in r.stdout + r.stderr
    assert "urlopen" not in r.stderr
    assert "Traceback" not in r.stderr


def test_dry_run_flag_exists():
    """The script must accept a --dry-run CLI flag."""
    text = SCRIPT.read_text()
    assert "--dry-run" in text, (
        "proposal_approval_handler.py: no --dry-run argument defined"
    )
    assert "argparse" in text, (
        "proposal_approval_handler.py: no argparse import for --dry-run"
    )


def test_paths_use_hermes_home():
    """All path constants must resolve via HERMES_HOME, not hardcoded."""
    text = SCRIPT.read_text()
    assert "/home/kensei/.hermes" not in text, (
        "proposal_approval_handler.py: hardcoded /home/kensei/.hermes path"
    )
    assert "HERMES_HOME" in text, (
        "proposal_approval_handler.py: does not reference HERMES_HOME"
    )
