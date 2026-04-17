"""Tests for cmd_update current-branch rebase behavior."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_update


def _make_run_side_effect(branch="main", commit_count="0", rebase_ok=True):
    """Build a side_effect function for subprocess.run that simulates git commands."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        if cmd == ["git", "rebase", "origin/main"]:
            rc = 0 if rebase_ok else 1
            stderr = "" if rebase_ok else "error: could not apply abc123\n"
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr=stderr)

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


class TestCmdUpdateCurrentBranchRebase:
    """cmd_update rebases the current branch onto origin/main."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_rebases_current_feature_branch(
        self, mock_run, _mock_which, mock_args
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="fix/stoicneko", commit_count="3", rebase_ok=True
        )

        cmd_update(mock_args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert len(rev_list_cmds) == 1
        assert "HEAD..origin/main" in rev_list_cmds[0]

        rebase_cmds = [c for c in commands if c == "git rebase origin/main"]
        assert len(rebase_cmds) == 1

        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 0

        checkout_cmds = [c for c in commands if "checkout" in c]
        assert len(checkout_cmds) == 0

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_already_up_to_date(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="fix/stoicneko", commit_count="0", rebase_ok=True
        )

        cmd_update(mock_args)

        captured = capsys.readouterr()
        assert "Already up to date!" in captured.out

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        rebase_cmds = [c for c in commands if c == "git rebase origin/main"]
        assert len(rebase_cmds) == 0

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_detached_head_exits(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(branch="HEAD", commit_count="3")

        with pytest.raises(SystemExit, match="1"):
            cmd_update(mock_args)

        captured = capsys.readouterr()
        assert "detached HEAD" in captured.out

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        rebase_cmds = [c for c in commands if c == "git rebase origin/main"]
        assert len(rebase_cmds) == 0

    def test_update_non_interactive_skips_migration_prompt(self, mock_args, capsys):
        """When stdin/stdout aren't TTYs, config migration prompt is skipped."""
        with patch("shutil.which", return_value=None), patch(
            "subprocess.run"
        ) as mock_run, patch("builtins.input") as mock_input, patch(
            "hermes_cli.config.get_missing_env_vars", return_value=["MISSING_KEY"]
        ), patch("hermes_cli.config.get_missing_config_fields", return_value=[]), patch(
            "hermes_cli.config.check_config_version", return_value=(1, 2)
        ), patch("hermes_cli.main.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = False
            mock_sys.stdout.isatty.return_value = False
            mock_run.side_effect = _make_run_side_effect(
                branch="main", commit_count="1", rebase_ok=True
            )

            cmd_update(mock_args)

            mock_input.assert_not_called()
            captured = capsys.readouterr()
            assert "Non-interactive session" in captured.out
