"""The dashboard restart action must not latch on a foreground restart child.

On a host with no service manager, ``hermes gateway restart`` falls through to
its manual fallback and runs ``run_gateway()`` in that same process: the child
*becomes* the gateway and never exits — the shape that
``gateway.status.looks_like_gateway_runtime_command_line`` exists for
(#51325/#51468).  The action layer used to read that as "a restart is still in
flight", which pinned ``/api/actions/gateway-restart/status`` at ``running``
for the life of the gateway and coalesced every later restart request onto that
child, turning the dashboard's restart button into a permanent no-op.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_restart_cooldown():
    """Keep the module-level cooldown state out of neighbouring tests."""
    import hermes_cli.web_server as web_server

    web_server._LAST_GATEWAY_RESTART = None
    yield
    web_server._LAST_GATEWAY_RESTART = None


def _live_proc(pid: int) -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = None
    proc.pid = pid
    return proc


def _exited_proc(pid: int) -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.poll.return_value = 0
    proc.pid = pid
    return proc


class TestHostsTheGatewayDetection:
    """Owning the gateway PID record is what distinguishes the two states."""

    def test_child_owning_the_pid_record_is_the_gateway(self):
        from hermes_cli.web_server import _gateway_restart_child_hosts_the_gateway

        with patch("hermes_cli.web_server.get_running_pid", return_value=4242):
            assert _gateway_restart_child_hosts_the_gateway(_live_proc(4242)) is True

    def test_child_that_does_not_own_it_is_still_in_flight(self):
        from hermes_cli.web_server import _gateway_restart_child_hosts_the_gateway

        with patch("hermes_cli.web_server.get_running_pid", return_value=99):
            assert _gateway_restart_child_hosts_the_gateway(_live_proc(4242)) is False

    def test_exited_child_is_never_the_gateway(self):
        """A finished restart is handled by the pre-existing exit-code path."""
        from hermes_cli.web_server import _gateway_restart_child_hosts_the_gateway

        with patch("hermes_cli.web_server.get_running_pid", return_value=4242):
            assert _gateway_restart_child_hosts_the_gateway(_exited_proc(4242)) is False

    def test_probe_failure_keeps_in_flight_semantics(self):
        from hermes_cli.web_server import _gateway_restart_child_hosts_the_gateway

        with patch(
            "hermes_cli.web_server.get_running_pid", side_effect=OSError("no record")
        ):
            assert _gateway_restart_child_hosts_the_gateway(_live_proc(4242)) is False


class TestDetectionAgainstARealPidRecord:
    """The same detection, with the real ``gateway.status`` code and real I/O.

    The whole fix rests on one claim about existing code: a process whose argv
    is ``gateway restart`` and which owns the PID record reads back as the
    running gateway.  That is what ``looks_like_gateway_runtime_command_line``
    exists for, and why the strict ``looks_like_gateway_command_line`` is left
    alone (#51325/#51468).  Mocking ``get_running_pid`` cannot check the claim,
    so check it here.
    """

    RESTART_CMDLINE = "/usr/local/bin/python /usr/local/bin/hermes gateway restart"

    def test_the_runtime_matcher_accepts_a_restart_command_line(self):
        """No mocks at all — this pair of verdicts is the whole premise."""
        from gateway.status import (
            looks_like_gateway_command_line,
            looks_like_gateway_runtime_command_line,
        )

        assert looks_like_gateway_runtime_command_line(self.RESTART_CMDLINE) is True
        # Still not a *management*-command match: #51468 kept this one strict.
        assert looks_like_gateway_command_line(self.RESTART_CMDLINE) is False

    @pytest.fixture
    def gateway_identity_files(self):
        from gateway import status

        yield status
        status.release_gateway_runtime_lock()
        status._get_pid_path().unlink(missing_ok=True)

    def test_a_restart_process_owning_the_record_reads_back_as_the_gateway(
        self, gateway_identity_files, monkeypatch
    ):
        """End-to-end through the real record, under the suite's temp HERMES_HOME.

        ``acquire_gateway_runtime_lock()`` + ``write_pid_file()`` is verbatim what
        ``run_gateway()`` does at startup, so the record on disk is the one a
        no-supervisor ``gateway restart`` leaves behind.  Everything downstream is
        real: lock probe, record parse, liveness, ``start_time`` match, profile
        scoping.

        The single substitution is ``_read_process_cmdline``:
        ``_record_matches_live_gateway_pid`` prefers the *live OS* command line
        over the recorded argv, and a pytest process cannot present
        ``gateway restart`` as its own argv.  The string it returns is the same
        one asserted un-mocked above.
        """
        from hermes_cli.web_server import _gateway_restart_child_hosts_the_gateway

        status = gateway_identity_files
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "restart"])
        monkeypatch.setattr(
            status,
            "_read_process_cmdline",
            lambda pid: self.RESTART_CMDLINE if pid == os.getpid() else None,
        )
        assert status.acquire_gateway_runtime_lock() is True
        status.write_pid_file()

        assert status.get_running_pid(cleanup_stale=False) == os.getpid()
        assert _gateway_restart_child_hosts_the_gateway(_live_proc(os.getpid())) is True

    def test_a_non_gateway_process_owning_the_record_is_not_the_gateway(
        self, gateway_identity_files, monkeypatch
    ):
        """PID reuse guard: same record, but the live process is not a gateway."""
        from hermes_cli.web_server import _gateway_restart_child_hosts_the_gateway

        status = gateway_identity_files
        monkeypatch.setattr(sys, "argv", ["hermes", "gateway", "restart"])
        monkeypatch.setattr(
            status, "_read_process_cmdline", lambda pid: "/usr/bin/s6-log /var/log/x"
        )
        assert status.acquire_gateway_runtime_lock() is True
        status.write_pid_file()

        hosting = _gateway_restart_child_hosts_the_gateway(_live_proc(os.getpid()))
        assert hosting is False


class TestSpawnGatewayRestart:
    """A child that already became the gateway must not block a new restart."""

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    def test_a_hosting_child_does_not_coalesce_the_next_request(
        self, mock_spawn, mock_subcmd
    ):
        from hermes_cli.web_server import _spawn_gateway_restart

        hosting = _live_proc(4242)
        fresh = _live_proc(4243)
        mock_spawn.return_value = fresh

        with patch(
            "hermes_cli.web_server._ACTION_PROCS", {"gateway-restart": hosting}
        ), patch(
            "hermes_cli.web_server._ACTION_COMMANDS",
            {"gateway-restart": ("gateway", "restart")},
        ), patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
        ), patch(
            "hermes_cli.web_server.get_running_pid", return_value=4242
        ):
            proc, reused = _spawn_gateway_restart()

        assert reused is False, (
            "the restart it was handed is already done — the child IS the gateway"
        )
        assert proc is fresh
        mock_spawn.assert_called_once()

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    def test_a_genuinely_in_flight_child_is_still_reused(self, mock_spawn, mock_subcmd):
        """#89034's guard stays: a restart mid-handoff is not restarted again."""
        from hermes_cli.web_server import _spawn_gateway_restart

        in_flight = _live_proc(4242)

        with patch(
            "hermes_cli.web_server._ACTION_PROCS", {"gateway-restart": in_flight}
        ), patch(
            "hermes_cli.web_server._ACTION_COMMANDS",
            {"gateway-restart": ("gateway", "restart")},
        ), patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
        ), patch(
            "hermes_cli.web_server.get_running_pid", return_value=None
        ):
            proc, reused = _spawn_gateway_restart()

        assert proc is in_flight
        assert reused is True
        mock_spawn.assert_not_called()

    @patch(
        "hermes_cli.web_server._gateway_subcommand",
        return_value=["gateway", "restart"],
    )
    @patch("hermes_cli.web_server._spawn_hermes_action")
    def test_the_cooldown_still_absorbs_a_repeat_burst(self, mock_spawn, mock_subcmd):
        """Releasing the latch must not reopen the #89034 restart storm."""
        import hermes_cli.web_server as web_server
        from hermes_cli.web_server import _spawn_gateway_restart

        hosting = _live_proc(4242)
        mock_spawn.return_value = hosting

        with patch(
            "hermes_cli.web_server._ACTION_PROCS", {}
        ), patch(
            "hermes_cli.web_server._ACTION_COMMANDS", {}
        ), patch(
            "hermes_cli.gateway._reap_unsupervised_gateway_orphans"
        ), patch(
            "hermes_cli.web_server.get_running_pid", return_value=4242
        ), patch(
            "hermes_cli.web_server.time.monotonic", side_effect=[100.0, 103.5]
        ):
            _spawn_gateway_restart()
            web_server._ACTION_PROCS["gateway-restart"] = hosting
            web_server._ACTION_COMMANDS["gateway-restart"] = ("gateway", "restart")
            _, reused = _spawn_gateway_restart()

        assert mock_spawn.call_count == 1
        assert reused is True


class TestActionStatus:
    """The status endpoint must not report a completed restart as running."""

    def test_a_hosting_child_is_reported_as_finished(self, tmp_path):
        import hermes_cli.web_server as web_server

        hosting = _live_proc(4242)

        with patch.object(web_server, "_ACTION_LOG_DIR", tmp_path), patch.object(
            web_server, "_ACTION_PROCS", {"gateway-restart": hosting}
        ), patch("hermes_cli.web_server.get_running_pid", return_value=4242):
            result = asyncio.run(web_server.get_action_status("gateway-restart"))
            # The handle must survive: _terminate_desktop_managed_gateway()
            # stops the Desktop-owned gateway through it.
            assert web_server._ACTION_PROCS["gateway-restart"] is hosting

        assert result["running"] is False
        assert result["exit_code"] == 0
        assert result["pid"] == 4242

    def test_a_genuinely_in_flight_child_is_still_reported_running(self, tmp_path):
        import hermes_cli.web_server as web_server

        in_flight = _live_proc(4242)

        with patch.object(web_server, "_ACTION_LOG_DIR", tmp_path), patch.object(
            web_server, "_ACTION_PROCS", {"gateway-restart": in_flight}
        ), patch("hermes_cli.web_server.get_running_pid", return_value=None):
            result = asyncio.run(web_server.get_action_status("gateway-restart"))

        assert result["running"] is True
        assert result["exit_code"] is None

    def test_other_actions_are_untouched(self, tmp_path):
        """The reconciliation is scoped to ``gateway-restart``."""
        import hermes_cli.web_server as web_server

        live = _live_proc(4242)

        with patch.object(web_server, "_ACTION_LOG_DIR", tmp_path), patch.object(
            web_server, "_ACTION_PROCS", {"dump": live}
        ), patch("hermes_cli.web_server.get_running_pid", return_value=4242):
            result = asyncio.run(web_server.get_action_status("dump"))

        assert result["running"] is True
        assert result["exit_code"] is None
