"""Kanban CLI process-level exit-code tests."""

from __future__ import annotations

import os
import subprocess
import sys


def test_missing_board_json_command_exits_nonzero(tmp_path):
    """A missing --board must fail the shell command, not just print stderr."""
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

    assert result.returncode != 0
    assert result.stdout == ""
    assert "kanban: board 'definitely-missing-board' does not exist" in result.stderr
