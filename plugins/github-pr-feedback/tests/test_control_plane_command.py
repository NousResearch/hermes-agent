"""Control commands must never import an assigned PR's Hermes checkout."""

import shlex
import subprocess

from github_pr_feedback.controller import _governed_command_prefix


def test_control_plane_command_ignores_worktree_module_shadow(tmp_path):
    home = tmp_path / "control"
    home.mkdir()
    workspace = tmp_path / "pr"
    shadow = workspace / "hermes_cli"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("", encoding="utf-8")
    (shadow / "main.py").write_text("print('WRONG_WORKTREE_CONTROL_PLANE')\n", encoding="utf-8")
    command = shlex.split(_governed_command_prefix(home))
    # Exercise the actual CLI launcher without GitHub credentials or a network
    # operation; version-local is sufficient to identify module resolution.
    result = subprocess.run(
        command[:-1] + ["--version-local"], cwd=workspace,
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "WRONG_WORKTREE_CONTROL_PLANE" not in result.stdout
    assert "Hermes Agent" in result.stdout
