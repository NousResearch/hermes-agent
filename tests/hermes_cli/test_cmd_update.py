"""Focused tests for cmd_update checkout protection."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_update


def _make_run_side_effect(*, branch="main", dirty=False, commit_count="0"):
    recorded = []

    def side_effect(cmd, **kwargs):
        recorded.append(cmd)
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        if "status --porcelain" in joined:
            stdout = " M hermes_cli/main.py\n" if dirty else ""
            return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect, recorded


@pytest.fixture
def mock_args():
    return SimpleNamespace()


@patch("shutil.which", return_value=None)
@patch("subprocess.run")
def test_update_blocks_non_main_checkout(mock_run, _mock_which, mock_args, capsys):
    """cmd_update should refuse feature branches instead of silently switching."""
    mock_run.side_effect, recorded = _make_run_side_effect(branch="feature/safe-branch")

    with pytest.raises(SystemExit, match="1"):
        cmd_update(mock_args)

    commands = [" ".join(str(a) for a in cmd) for cmd in recorded]
    assert all("fetch origin" not in command for command in commands)

    out = capsys.readouterr().out
    assert "Refusing to update this checkout" in out
    assert "feature/safe-branch" in out


@patch("shutil.which", return_value=None)
@patch("subprocess.run")
def test_update_blocks_dirty_checkout(mock_run, _mock_which, mock_args, capsys):
    """Dirty worktrees should fail before any network or reset action."""
    mock_run.side_effect, recorded = _make_run_side_effect(dirty=True)

    with pytest.raises(SystemExit, match="1"):
        cmd_update(mock_args)

    commands = [" ".join(str(a) for a in cmd) for cmd in recorded]
    assert all("fetch origin" not in command for command in commands)

    out = capsys.readouterr().out
    assert "working tree has local changes" in out


@patch("shutil.which", return_value=None)
@patch("subprocess.run")
def test_update_blocks_dev_checkout_env(mock_run, _mock_which, mock_args, capsys, monkeypatch):
    """An explicit dev-checkout env flag should hard-block self-update."""
    monkeypatch.setenv("HERMES_DEV_CHECKOUT", "1")
    mock_run.side_effect, recorded = _make_run_side_effect()

    with pytest.raises(SystemExit, match="1"):
        cmd_update(mock_args)

    commands = [" ".join(str(a) for a in cmd) for cmd in recorded]
    assert all("fetch origin" not in command for command in commands)

    out = capsys.readouterr().out
    assert "protected from self-update" in out
