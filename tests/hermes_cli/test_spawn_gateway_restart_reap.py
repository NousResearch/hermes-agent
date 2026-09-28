"""Tests for _spawn_gateway_restart orphan-reap guard (#77276)."""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_restart_cooldown():
    """Clear the #89034 repeat-restart cooldown between cases.

    ``_spawn_gateway_restart`` now coalesces a second restart request that
    arrives within ``GATEWAY_RESTART_COOLDOWN_SECONDS`` of the last spawn, so
    without this the first case's spawn suppresses the second case's.
    """
    import hermes_cli.web_server as web_server

    web_server._LAST_GATEWAY_RESTART = None
    yield
    web_server._LAST_GATEWAY_RESTART = None


class TestSpawnGatewayRestartReapsOrphans:
    """_spawn_gateway_restart must reap orphaned gateways before spawning."""


    @patch("hermes_cli.web_server_gateway._gateway_subcommand", return_value=["gateway", "restart"])
    @patch("hermes_cli.web_server_gateway._spawn_hermes_action")
    @patch("hermes_cli.web_server_gateway._ACTION_PROCS", {})
    def test_reap_failure_does_not_block_spawn(self, mock_spawn, mock_subcmd):
        """If reap raises, the restart still proceeds."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_spawn.return_value = mock_proc

        from hermes_cli.web_server import _spawn_gateway_restart

        with patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans",
            side_effect=OSError("permission denied"),
        ):
            proc, reused = _spawn_gateway_restart()

        mock_spawn.assert_called_once()
        assert proc is mock_proc

    @pytest.mark.parametrize("child_running", [True, False], ids=["in-flight", "cooldown"])
    @patch("hermes_cli.web_server_gateway._gateway_subcommand", return_value=["gateway", "restart"])
    @patch("hermes_cli.web_server_gateway._spawn_hermes_action")
    def test_reused_restart_is_never_reaped(self, mock_spawn, mock_subcmd, child_running):
        """A repeat request that reuses the last restart must not reap first: until that
        child writes gateway.pid it matches the unsupervised scan, so the reap killed it."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None if child_running else 0
        procs, commands = {}, {}

        def _spawn(subcommand, name):
            procs[name] = mock_proc
            commands[name] = tuple(subcommand)
            return mock_proc

        mock_spawn.side_effect = _spawn
        from hermes_cli.web_server import _spawn_gateway_restart

        with patch("hermes_cli.web_server_gateway._ACTION_PROCS", procs), patch(
            "hermes_cli.web_server_gateway._ACTION_COMMANDS", commands
        ), patch("hermes_cli.gateway._reap_unsupervised_gateway_orphans") as mock_reap:
            first = _spawn_gateway_restart()
            second = _spawn_gateway_restart()

        assert first == (mock_proc, False)
        assert second == (mock_proc, True)
        mock_reap.assert_called_once()
