from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_top_level_ops_help_delegates_to_ops_cli(tmp_path):
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    repo_root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "ops", "--help"],
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=20,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "eval" in result.stdout
    assert "failures" in result.stdout
    assert "ops_args" not in result.stdout
