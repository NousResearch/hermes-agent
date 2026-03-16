"""Tests for cmd_update gateway auto-restart — systemd + launchd coverage.

Ensures ``hermes update`` correctly detects running gateways managed by
systemd (Linux) or launchd (macOS) and restarts/informs the user properly,
rather than leaving zombie processes or telling users to manually restart
when launchd will auto-respawn.
"""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

import hermes_cli.gateway as gateway_cli
from hermes_cli.main import cmd_update


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_side_effect(
    branch="main",
    verify_ok=True,
    commit_count="3",
    systemd_active=False,
    launchctl_loaded=False,
):
    """Build a subprocess.run side_effect that simulates git + service commands."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        # git rev-parse --abbrev-ref HEAD
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{branch}\n", stderr="")

        # git rev-parse --verify origin/{branch}
        if "rev-parse" in joined and "--verify" in joined:
            rc = 0 if verify_ok else 128
            return subprocess.CompletedProcess(cmd, rc, stdout="", stderr="")

        # git rev-list HEAD..origin/{branch} --count
        if "rev-list" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{commit_count}\n", stderr="")

        # systemctl --user is-active
        if "systemctl" in joined and "is-active" in joined:
            if systemd_active:
                return subprocess.CompletedProcess(cmd, 0, stdout="active\n", stderr="")
            return subprocess.CompletedProcess(cmd, 3, stdout="inactive\n", stderr="")

        # systemctl --user restart
        if "systemctl" in joined and "restart" in joined:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        # launchctl list ai.hermes.gateway
        if "launchctl" in joined and "list" in joined:
            if launchctl_loaded:
                return subprocess.CompletedProcess(cmd, 0, stdout="PID\tStatus\tLabel\n123\t0\tai.hermes.gateway\n", stderr="")
            return subprocess.CompletedProcess(cmd, 113, stdout="", stderr="Could not find service")

        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return side_effect


@pytest.fixture
def mock_args():
    return SimpleNamespace()


# ---------------------------------------------------------------------------
# Launchd plist includes --replace
# ---------------------------------------------------------------------------


class TestLaunchdPlistReplace:
    """The generated launchd plist must include --replace so respawned
    gateways kill stale instances."""

    def test_plist_contains_replace_flag(self):
        plist = gateway_cli.generate_launchd_plist()
        assert "--replace" in plist

    def test_plist_program_arguments_order(self):
        """--replace comes after 'run' in the ProgramArguments."""
        plist = gateway_cli.generate_launchd_plist()
        lines = [line.strip() for line in plist.splitlines()]
        # Find 'run' and '--replace' in the string entries
        string_values = [
            line.replace("<string>", "").replace("</string>", "")
            for line in lines
            if "<string>" in line and "</string>" in line
        ]
        assert "run" in string_values
        assert "--replace" in string_values
        run_idx = string_values.index("run")
        replace_idx = string_values.index("--replace")
        assert replace_idx == run_idx + 1


# ---------------------------------------------------------------------------
# cmd_update — macOS launchd detection
# ---------------------------------------------------------------------------


class TestLaunchdPlistRefresh:
    """refresh_launchd_plist_if_needed rewrites stale plists (like systemd's
    refresh_systemd_unit_if_needed)."""

    def test_refresh_rewrites_stale_plist(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old content</plist>")

        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        calls = []
        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        result = gateway_cli.refresh_launchd_plist_if_needed()

        assert result is True
        # Plist should now contain the generated content (which includes --replace)
        assert "--replace" in plist_path.read_text()
        # Should have unloaded then reloaded
        assert any("unload" in str(c) for c in calls)
        assert any("load" in str(c) for c in calls)

    def test_refresh_skips_when_current(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        # Write the current expected content
        plist_path.write_text(gateway_cli.generate_launchd_plist())

        calls = []
        monkeypatch.setattr(
            gateway_cli.subprocess, "run",
            lambda cmd, **kw: calls.append(cmd) or SimpleNamespace(returncode=0),
        )

        result = gateway_cli.refresh_launchd_plist_if_needed()

        assert result is False
        assert len(calls) == 0  # No launchctl calls needed

    def test_refresh_skips_when_no_plist(self, tmp_path, monkeypatch):
        plist_path = tmp_path / "nonexistent.plist"
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        result = gateway_cli.refresh_launchd_plist_if_needed()
        assert result is False

    def test_launchd_start_calls_refresh(self, tmp_path, monkeypatch):
        """launchd_start refreshes the plist before starting."""
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist>old</plist>")
        monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)

        calls = []
        def fake_run(cmd, check=False, **kwargs):
            calls.append(cmd)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(gateway_cli.subprocess, "run", fake_run)

        gateway_cli.launchd_start()

        # First calls should be refresh (unload/load), then start
        cmd_strs = [" ".join(c) for c in calls]
        assert any("unload" in s for s in cmd_strs)
        assert any("start" in s for s in cmd_strs)


class TestCmdUpdateLaunchdRestart:
    """cmd_update correctly detects and handles launchd on macOS."""

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_detects_launchd_and_skips_manual_restart_message(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """When launchd is running the gateway, update should print
        'auto-restart via launchd' instead of 'Restart it with: hermes gateway run'."""
        # Create a fake launchd plist so is_macos + plist.exists() passes
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        plist_path.write_text("<plist/>")

        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: True,
        )
        monkeypatch.setattr(
            gateway_cli, "get_launchd_plist_path", lambda: plist_path,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=True,
        )

        # Mock get_running_pid to return a PID
        with patch("gateway.status.get_running_pid", return_value=12345), \
             patch("gateway.status.remove_pid_file"):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Gateway restarted via launchd" in captured
        assert "Restart it with: hermes gateway run" not in captured
        # Verify launchctl stop + start were called (not manual SIGTERM)
        launchctl_calls = [
            c for c in mock_run.call_args_list
            if len(c.args[0]) > 0 and c.args[0][0] == "launchctl"
        ]
        stop_calls = [c for c in launchctl_calls if "stop" in c.args[0]]
        start_calls = [c for c in launchctl_calls if "start" in c.args[0]]
        assert len(stop_calls) >= 1
        assert len(start_calls) >= 1

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_without_launchd_shows_manual_restart(
        self, mock_run, _mock_which, mock_args, capsys, tmp_path, monkeypatch,
    ):
        """When no service manager is running, update should show the manual restart hint."""
        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: True,
        )
        plist_path = tmp_path / "ai.hermes.gateway.plist"
        # plist does NOT exist — no launchd service
        monkeypatch.setattr(
            gateway_cli, "get_launchd_plist_path", lambda: plist_path,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            launchctl_loaded=False,
        )

        with patch("gateway.status.get_running_pid", return_value=12345), \
             patch("gateway.status.remove_pid_file"), \
             patch("os.kill"):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Restart it with: hermes gateway run" in captured
        assert "Gateway restarted via launchd" not in captured

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_with_systemd_still_restarts_via_systemd(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """On Linux with systemd active, update should restart via systemctl."""
        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: False,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=True,
        )

        with patch("gateway.status.get_running_pid", return_value=12345), \
             patch("gateway.status.remove_pid_file"), \
             patch("os.kill"):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Gateway restarted" in captured
        # Verify systemctl restart was called
        restart_calls = [
            c for c in mock_run.call_args_list
            if "restart" in " ".join(str(a) for a in c.args[0])
            and "systemctl" in " ".join(str(a) for a in c.args[0])
        ]
        assert len(restart_calls) == 1

    @patch("shutil.which", return_value=None)
    @patch("subprocess.run")
    def test_update_no_gateway_running_skips_restart(
        self, mock_run, _mock_which, mock_args, capsys, monkeypatch,
    ):
        """When no gateway is running, update should skip the restart section entirely."""
        monkeypatch.setattr(
            gateway_cli, "is_macos", lambda: False,
        )

        mock_run.side_effect = _make_run_side_effect(
            commit_count="3",
            systemd_active=False,
        )

        with patch("gateway.status.get_running_pid", return_value=None):
            cmd_update(mock_args)

        captured = capsys.readouterr().out
        assert "Stopped gateway" not in captured
        assert "Gateway restarted" not in captured
        assert "Gateway restarted via launchd" not in captured
