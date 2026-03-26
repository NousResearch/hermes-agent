"""Tests for cmd_update — git fallback, Nix detection, and ZIP update paths."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.main import cmd_update, PROJECT_ROOT


def _make_run_side_effect(branch="main", verify_ok=True, commit_count="0"):
    """Build a side_effect function for subprocess.run that simulates git commands."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        # git rev-parse --abbrev-ref HEAD  (get current branch)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        # git rev-parse --verify origin/{branch}  (check remote branch exists)
        if "rev-parse" in joined and "--verify" in joined:
            rc = 0 if verify_ok else 128
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        # git rev-list HEAD..origin/{branch} --count
        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        # Fallback: return a successful CompletedProcess with empty stdout
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


class TestCmdUpdateBranchFallback:
    """cmd_update falls back to main when current branch has no remote counterpart."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_falls_back_to_main_when_branch_not_on_remote(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="fix/stoicneko", verify_ok=False, commit_count="3"
        )

        cmd_update(mock_args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        # rev-list should use origin/main, not origin/fix/stoicneko
        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert len(rev_list_cmds) == 1
        assert "origin/main" in rev_list_cmds[0]
        assert "origin/fix/stoicneko" not in rev_list_cmds[0]

        # pull should use main, not fix/stoicneko
        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 1
        assert "main" in pull_cmds[0]

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_uses_current_branch_when_on_remote(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="2"
        )

        cmd_update(mock_args)

        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]

        rev_list_cmds = [c for c in commands if "rev-list" in c]
        assert len(rev_list_cmds) == 1
        assert "origin/main" in rev_list_cmds[0]

        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 1
        assert "main" in pull_cmds[0]

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_already_up_to_date(
        self, mock_run, _mock_which, mock_args, capsys
    ):
        mock_run.side_effect = _make_run_side_effect(
            branch="main", verify_ok=True, commit_count="0"
        )

        cmd_update(mock_args)

        captured = capsys.readouterr()
        assert "Already up to date!" in captured.out

        # Should NOT have called pull
        commands = [" ".join(str(a) for a in c.args[0]) for c in mock_run.call_args_list]
        pull_cmds = [c for c in commands if "pull" in c]
        assert len(pull_cmds) == 0


class TestCmdUpdateNix:
    """cmd_update detects Nix store and runs the full Nix upgrade path.

    Patches PROJECT_ROOT to a fake /nix/store path so the Nix detection
    branch is triggered without needing a real Nix install.
    """

    def _nix_side_effect(self, cmd, **kwargs):
        """Simulate nix and hermes gateway restart calls."""
        joined = " ".join(str(c) for c in cmd)
        if "nix" in joined and "profile" in joined and "upgrade" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="upgrading hermes-agent...\n")
        if "hermes_cli.main" in joined or ("gateway" in joined and "restart" in joined):
            return subprocess.CompletedProcess(cmd, 0, stdout="✓ User service restarted\n", stderr="")
        if "--version" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="Hermes Agent v0.5.0\n", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def test_nix_update_runs_upgrade_and_restart(self, mock_args, capsys):
        """Happy path: nix upgrade → restart → verify → returns normally."""
        # _update_via_nix does NOT call sys.exit on success — it returns.
        with patch("subprocess.run", side_effect=self._nix_side_effect):
            with patch("hermes_cli.main.PROJECT_ROOT", new=Path("/nix/store/fake-hermes-agent")):
                cmd_update(mock_args)  # should return, not raise

        captured = capsys.readouterr()
        assert "Nix installation detected" in captured.out
        assert "Pulling latest release from Nix store" in captured.out
        assert "Restarting gateway service" in captured.out
        assert "Update complete" in captured.out

    def test_nix_update_with_nix_var_path(self, mock_args, capsys):
        """Also detects /nix/var as a valid Nix path."""
        with patch("subprocess.run", side_effect=self._nix_side_effect):
            with patch("hermes_cli.main.PROJECT_ROOT", new=Path("/nix/var/nix/profiles/per-user/itenev/hermes-agent")):
                cmd_update(mock_args)
        captured = capsys.readouterr()
        assert "Nix installation detected" in captured.out

    def test_nix_update_exits_nonzero_on_nix_failure(self, mock_args, capsys):
        """If nix profile upgrade fails, sys.exit(1) is called."""
        def fail_on_nix(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "nix" in joined and "upgrade" in joined:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error: flake not found\n")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fail_on_nix):
            with patch("hermes_cli.main.sys.exit") as mock_exit:
                with patch("hermes_cli.main.PROJECT_ROOT", new=Path("/nix/store/fake-hermes-agent")):
                    cmd_update(mock_args)
        mock_exit.assert_called_with(1)
        captured = capsys.readouterr()
        assert "nix profile upgrade failed" in captured.out

    def test_nix_update_gateway_restart_nonfatal(self, mock_args, capsys):
        """Gateway restart failure prints a warning but still returns normally."""
        def nix_then_restart_fail(cmd, **kwargs):
            joined = " ".join(str(c) for c in cmd)
            if "nix" in joined and "upgrade" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "restart" in joined:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="service not found\n")
            if "--version" in joined:
                return subprocess.CompletedProcess(cmd, 0, stdout="Hermes Agent v0.5.0\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=nix_then_restart_fail):
            with patch("hermes_cli.main.PROJECT_ROOT", new=Path("/nix/store/fake-hermes-agent")):
                cmd_update(mock_args)  # returns normally, no sys.exit on restart failure
        captured = capsys.readouterr()
        assert "Gateway restart failed" in captured.out
        assert "Update complete" in captured.out
