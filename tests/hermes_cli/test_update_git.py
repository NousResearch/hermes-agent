import subprocess
from pathlib import Path
from unittest.mock import patch

from hermes_cli.update_git import resolve_update_remote


def test_resolve_update_remote_prefers_main_tracking_remote(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        if "config --get branch.main.remote" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="upstream\n", stderr="")
        if cmd[-1] == "remote":
            return subprocess.CompletedProcess(cmd, 0, stdout="origin\nupstream\n", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    with patch("hermes_cli.update_git.subprocess.run", side_effect=side_effect):
        assert resolve_update_remote(repo_dir) == "upstream"


def test_resolve_update_remote_falls_back_to_origin(tmp_path):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)
        if "config --get branch.main.remote" in joined:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")
        if cmd[-1] == "remote":
            return subprocess.CompletedProcess(cmd, 0, stdout="origin\n", stderr="")
        raise AssertionError(f"Unexpected command: {cmd}")

    with patch("hermes_cli.update_git.subprocess.run", side_effect=side_effect):
        assert resolve_update_remote(repo_dir) == "origin"
