"""Tests for _spawn_gateway_restart orphan-reap guard (#77276)."""
from __future__ import annotations

import asyncio
import subprocess
from typing import Any, cast
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

    @patch("hermes_cli.web_server._gateway_subcommand", return_value=["gateway", "restart"])
    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
    def test_reap_called_before_spawn(self, mock_spawn, mock_subcmd):
        """Orphan reap runs before the new gateway process is spawned."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_spawn.return_value = mock_proc

        from hermes_cli.web_server import _spawn_gateway_restart

        with patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
        ) as mock_reap:
            proc, reused = _spawn_gateway_restart()

        mock_reap.assert_called_once()
        mock_spawn.assert_called_once()
        assert proc is mock_proc
        assert reused is False

    @patch("hermes_cli.web_server._gateway_subcommand", return_value=["gateway", "restart"])
    @patch("hermes_cli.web_server._spawn_hermes_action")
    @patch("hermes_cli.web_server._ACTION_PROCS", {})
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


class _FinishedProc:
    def __init__(self, pid: int, code: int = 0):
        self.pid = pid
        self._code = code
        self.wait_calls = 0

    def poll(self):
        return self._code

    def wait(self, timeout=None):
        assert timeout == 0
        self.wait_calls += 1
        return self._code


class _RunningThenFinishedProc(_FinishedProc):
    def __init__(self, pid: int, code: int = 0):
        super().__init__(pid, code)
        self._running = True

    def poll(self):
        if self._running:
            return None
        return self._code

    def finish(self):
        self._running = False


@pytest.mark.asyncio
async def test_action_reaper_reaps_completed_registered_action():
    from hermes_cli import web_server

    proc = _FinishedProc(pid=1234, code=7)
    web_server._ACTION_PROCS["unit-action"] = cast(Any, proc)
    web_server._ACTION_COMMANDS["unit-action"] = ("doctor",)
    web_server._ACTION_RESULTS.pop("unit-action", None)

    task = asyncio.create_task(web_server.run_action_reaper(interval=0.001))
    try:
        for _ in range(20):
            await asyncio.sleep(0.01)
            if "unit-action" not in web_server._ACTION_PROCS:
                break
        assert "unit-action" not in web_server._ACTION_PROCS
        assert "unit-action" not in web_server._ACTION_COMMANDS
        assert web_server._ACTION_RESULTS["unit-action"] == {"exit_code": 7, "pid": 1234}
        assert proc.wait_calls == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        web_server._ACTION_PROCS.pop("unit-action", None)
        web_server._ACTION_COMMANDS.pop("unit-action", None)
        web_server._ACTION_RESULTS.pop("unit-action", None)


@pytest.mark.asyncio
async def test_action_reaper_drains_retired_same_name_orphan():
    from hermes_cli import web_server

    proc = _RunningThenFinishedProc(pid=5678)
    with web_server._ACTION_ORPHANS_LOCK:
        web_server._ACTION_ORPHANS.clear()

    web_server._retire_action_proc(proc)
    with web_server._ACTION_ORPHANS_LOCK:
        assert web_server._ACTION_ORPHANS == [proc]

    proc.finish()
    task = asyncio.create_task(web_server.run_action_reaper(interval=0.001))
    try:
        for _ in range(20):
            await asyncio.sleep(0.01)
            with web_server._ACTION_ORPHANS_LOCK:
                if not web_server._ACTION_ORPHANS:
                    break
        with web_server._ACTION_ORPHANS_LOCK:
            assert web_server._ACTION_ORPHANS == []
        assert proc.wait_calls == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with web_server._ACTION_ORPHANS_LOCK:
            web_server._ACTION_ORPHANS.clear()
