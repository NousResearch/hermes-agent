"""Tests for launchd_restart() kickstart flag fix.

Regression test for issue #42446: launchd_restart() used kickstart -k which
races with KeepAlive=true, causing double gateway starts on macOS 26+.

The fix removes the -k flag from kickstart since the preceding code already
handles graceful termination via SIGTERM + wait.
"""

import subprocess
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers – mock the heavy imports inside gateway.py
# ---------------------------------------------------------------------------

def _import_launchd_restart():
    """Import launchd_restart with minimal gateway module side-effects."""
    # gateway.py has heavy imports; patch what we can to avoid full init
    import importlib
    import hermes_cli.gateway as gw_mod
    return gw_mod.launchd_restart


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLaunchdRestartKickstartNoKillFlag:
    """Verify launchd_restart uses plain kickstart (no -k)."""

    @patch("hermes_cli.gateway.subprocess.run")
    @patch("hermes_cli.gateway._launchd_domain", return_value="gui/501")
    @patch("hermes_cli.gateway.get_launchd_label", return_value="com.hermes.gateway")
    @patch("hermes_cli.gateway._get_restart_drain_timeout", return_value=5)
    @patch("gateway.status.get_running_pid", return_value=12345)
    @patch("hermes_cli.gateway._request_gateway_self_restart", return_value=False)
    @patch("hermes_cli.gateway.terminate_pid")
    @patch("hermes_cli.gateway._wait_for_gateway_exit", return_value=True)
    def test_kickstart_uses_no_kill_flag(
        self, mock_wait, mock_terminate, mock_self_restart,
        mock_get_pid, mock_drain_timeout, mock_label,
        mock_domain, mock_subprocess_run
    ):
        """kickstart must NOT include -k flag after SIGTERM + wait.

        The -k flag re-terminates the service and races with KeepAlive=true,
        causing a double-spawn. Since terminate_pid + wait already handle
        graceful shutdown, kickstart should only start (not kill-start).
        """
        launchd_restart = _import_launchd_restart()

        launchd_restart()

        # Find the kickstart call
        calls = mock_subprocess_run.call_args_list
        kickstart_calls = [
            c for c in calls
            if len(c[0]) > 0 and "kickstart" in c[0][0]
        ]
        assert len(kickstart_calls) >= 1, "Expected at least one kickstart call"

        # The primary kickstart call should NOT contain -k
        primary_kickstart = kickstart_calls[0]
        cmd = primary_kickstart[0][0]
        assert cmd == ["launchctl", "kickstart", "gui/501/com.hermes.gateway"], \
            f"Expected plain kickstart without -k, got: {cmd}"

    @patch("hermes_cli.gateway.subprocess.run")
    @patch("hermes_cli.gateway._launchd_domain", return_value="gui/501")
    @patch("hermes_cli.gateway.get_launchd_label", return_value="com.hermes.gateway")
    @patch("hermes_cli.gateway._get_restart_drain_timeout", return_value=5)
    @patch("gateway.status.get_running_pid", return_value=None)
    def test_kickstart_no_kill_flag_when_pid_absent(
        self, mock_get_pid, mock_drain_timeout, mock_label,
        mock_domain, mock_subprocess_run
    ):
        """Even when no PID is found (gateway not running), kickstart
        should not use -k — consistent with launchd_start() pattern."""
        launchd_restart = _import_launchd_restart()

        launchd_restart()

        calls = mock_subprocess_run.call_args_list
        kickstart_calls = [
            c for c in calls
            if len(c[0]) > 0 and "kickstart" in c[0][0]
        ]
        assert len(kickstart_calls) >= 1
        cmd = kickstart_calls[0][0][0]
        assert "-k" not in cmd, f"-k flag should not be present: {cmd}"

    @patch("hermes_cli.gateway.subprocess.run")
    @patch("hermes_cli.gateway._launchd_domain", return_value="gui/501")
    @patch("hermes_cli.gateway.get_launchd_label", return_value="com.hermes.gateway")
    @patch("hermes_cli.gateway._get_restart_drain_timeout", return_value=5)
    @patch("gateway.status.get_running_pid", return_value=12345)
    @patch("hermes_cli.gateway._request_gateway_self_restart", return_value=False)
    @patch("hermes_cli.gateway.terminate_pid")
    @patch("hermes_cli.gateway._wait_for_gateway_exit", return_value=True)
    def test_terminate_called_before_kickstart(
        self, mock_wait, mock_terminate, mock_self_restart,
        mock_get_pid, mock_drain_timeout, mock_label,
        mock_domain, mock_subprocess_run
    ):
        """Verify that terminate_pid and wait are called before kickstart,
        confirming the graceful shutdown happens first."""
        launchd_restart = _import_launchd_restart()

        launchd_restart()

        # terminate_pid should be called with the PID
        mock_terminate.assert_called_once_with(12345, force=False)
        # wait should be called
        mock_wait.assert_called_once()

    @patch("hermes_cli.gateway.subprocess.run")
    @patch("hermes_cli.gateway._launchd_domain", return_value="gui/501")
    @patch("hermes_cli.gateway.get_launchd_label", return_value="com.hermes.gateway")
    @patch("hermes_cli.gateway._get_restart_drain_timeout", return_value=5)
    @patch("gateway.status.get_running_pid", return_value=12345)
    @patch("hermes_cli.gateway._request_gateway_self_restart", return_value=True)
    def test_self_restart_skips_kickstart(
        self, mock_self_restart, mock_get_pid, mock_drain_timeout,
        mock_label, mock_domain, mock_subprocess_run
    ):
        """When self-restart (SIGUSR1) succeeds, kickstart is not called."""
        launchd_restart = _import_launchd_restart()

        launchd_restart()

        # No subprocess calls expected (self-restart handles it)
        kickstart_calls = [
            c for c in mock_subprocess_run.call_args_list
            if len(c[0]) > 0 and "kickstart" in c[0][0]
        ]
        assert len(kickstart_calls) == 0, \
            "kickstart should not be called when self-restart succeeds"
