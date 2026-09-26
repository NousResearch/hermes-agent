"""Kanban CLI process-level exit-code tests."""

from __future__ import annotations

import os
import subprocess
import sys


def test_nonexistent_board_json_command_exits_nonzero(tmp_path):
    """A nonexistent board must fail the shell command, not just print stderr."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "kanban",
            "--board",
            "definitely-missing-board",
            "list",
            "--json",
        ],
        cwd=os.environ.get("HERMES_REPO_UNDER_TEST", os.getcwd()),
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 1
    assert "kanban: board 'definitely-missing-board' does not exist" in result.stderr
