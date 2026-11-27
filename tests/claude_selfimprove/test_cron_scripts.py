"""End-to-end smoke tests for the deployed cron entry-point scripts.

These copy the two thin entry scripts and the package itself into a single
directory the way the real install step does (see the pipeline's deploy
step: scripts land flat in ``~/.hermes/scripts/``, the package lands in
``~/.hermes/scripts/claude_selfimprove/`` right next to them), then invoke
each script exactly as cron would: as a subprocess, with no special
PYTHONPATH — only the sibling-package ``sys.path.insert`` each script does
for itself.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "claude_selfimprove"
SCAN_SCRIPT = PACKAGE_DIR / "cron_scripts" / "claude_selfimprove_scan.py"
CONSOLIDATE_SCRIPT = PACKAGE_DIR / "cron_scripts" / "claude_selfimprove_consolidate.py"


@pytest.fixture
def deployed(tmp_path):
    """A tmp dir laid out exactly like the real ~/.hermes/scripts/ deploy."""
    scripts_dir = tmp_path / "deployed_scripts"
    scripts_dir.mkdir()
    shutil.copytree(PACKAGE_DIR, scripts_dir / "claude_selfimprove")
    shutil.copy(SCAN_SCRIPT, scripts_dir / "claude_selfimprove_scan.py")
    shutil.copy(CONSOLIDATE_SCRIPT, scripts_dir / "claude_selfimprove_consolidate.py")

    home = tmp_path / "home"
    hermes_home = tmp_path / "hermes_home"
    home.mkdir()
    hermes_home.mkdir()
    return {"scripts_dir": scripts_dir, "home": home, "hermes_home": hermes_home}


def _run(script_path: Path, deployed: dict) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ)
    env["HOME"] = str(deployed["home"])
    env["HERMES_HOME"] = str(deployed["hermes_home"])
    env.pop("CLAUDE_SELFIMPROVE_CLAUDE_DIR", None)
    env.pop("CLAUDE_SELFIMPROVE_CLAUDE_HERMES_DIR", None)
    return subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, timeout=30, env=env,
    )


def test_scan_script_runs_clean_on_empty_profile(deployed):
    result = _run(deployed["scripts_dir"] / "claude_selfimprove_scan.py", deployed)
    assert result.returncode == 0, result.stderr
    assert "claude-selfimprove scan:" in result.stdout
    assert "0 new turns" in result.stdout


def test_consolidate_script_runs_clean_on_empty_profile(deployed):
    result = _run(deployed["scripts_dir"] / "claude_selfimprove_consolidate.py", deployed)
    assert result.returncode == 0, result.stderr
    assert "claude-selfimprove consolidate:" in result.stdout
    assert "0 pending evaluated" in result.stdout


def test_scan_script_finds_a_real_transcript_turn(deployed):
    import json

    proj_dir = deployed["home"] / ".claude" / "projects" / "some-project"
    proj_dir.mkdir(parents=True)
    (proj_dir / "sess-1.jsonl").write_text(
        json.dumps(
            {"type": "user", "sessionId": "sess-1", "message": {"role": "user", "content": "hello"}}
        )
        + "\n"
    )
    result = _run(deployed["scripts_dir"] / "claude_selfimprove_scan.py", deployed)
    assert result.returncode == 0, result.stderr
    assert "1 new turns" in result.stdout
