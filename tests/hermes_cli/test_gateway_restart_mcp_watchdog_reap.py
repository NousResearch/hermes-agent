"""Tests for reaping old-gateway MCP watchdog children on `gateway restart` (#87906).

A manual `gateway restart` (the common no-service-supervisor path) killed the
old gateway PID but left its already-spawned `mcp_stdio_watchdog.py` ->
MCP-server pairs running until their own 2s-poll/3s-grace self-heal cycle,
racing the freshly-spawned pairs for the new gateway. These tests cover the
new best-effort reaper: PID matching, the /proc-vs-ps scan, the
SIGTERM-only-no-SIGKILL safety contract, and the manual-restart wiring.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.gateway as gateway_mod

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WATCHDOG_SCRIPT = os.path.join(_REPO_ROOT, "tools", "mcp_stdio_watchdog.py")


# ---------------------------------------------------------------------------
# _cmdline_matches_watchdog_ppid
# ---------------------------------------------------------------------------


class TestCmdlineMatchesWatchdogPpid:
    def test_matches_watchdog_with_correct_ppid(self):
        cmdline = (
            "/usr/bin/python3 /repo/tools/mcp_stdio_watchdog.py --ppid 4242 "
            "-- /bin/sh arr-mcp.sh"
        )
        assert gateway_mod._cmdline_matches_watchdog_ppid(cmdline, 4242) is True

    def test_no_match_when_watchdog_script_absent(self):
        cmdline = "/bin/sh arr-mcp.sh --ppid 4242"
        assert gateway_mod._cmdline_matches_watchdog_ppid(cmdline, 4242) is False

    def test_no_substring_collision_short_ppid_against_long_target(self):
        """--ppid 4242 must NOT match old_gateway_pid=42420 (substring, not token)."""
        cmdline = "python3 tools/mcp_stdio_watchdog.py --ppid 4242 -- real-cmd"
        assert gateway_mod._cmdline_matches_watchdog_ppid(cmdline, 42420) is False

    def test_no_substring_collision_long_ppid_against_short_target(self):
        """--ppid 42420 must NOT match old_gateway_pid=4242 (substring, not token)."""
        cmdline = "python3 tools/mcp_stdio_watchdog.py --ppid 42420 -- real-cmd"
        assert gateway_mod._cmdline_matches_watchdog_ppid(cmdline, 4242) is False

    def test_malformed_ppid_missing_value_does_not_raise(self):
        cmdline = "python3 tools/mcp_stdio_watchdog.py --ppid"
        assert gateway_mod._cmdline_matches_watchdog_ppid(cmdline, 4242) is False

    def test_malformed_ppid_non_integer_value_does_not_raise(self):
        cmdline = "python3 tools/mcp_stdio_watchdog.py --ppid notanumber -- real-cmd"
        assert gateway_mod._cmdline_matches_watchdog_ppid(cmdline, 4242) is False


# ---------------------------------------------------------------------------
# _scan_mcp_watchdog_pids
# ---------------------------------------------------------------------------


def _fake_proc_dir(entries: dict):
    """Mirror the fake-/proc idiom from test_gateway_proc_fallback.py."""

    def _isdir(path):
        return str(path) == "/proc"

    def _listdir(path):
        if str(path) == "/proc":
            return [str(pid) for pid in entries] + ["self", "version"]
        raise FileNotFoundError(path)

    def _open(path, mode="r", **kwargs):
        path_str = str(path)
        if "/cmdline" in path_str:
            pid = int(path_str.split("/proc/")[1].split("/")[0])
            raw = entries.get(pid, "").encode("utf-8").replace(b" ", b"\x00")
            m = MagicMock()
            m.read.return_value = raw
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        raise FileNotFoundError(path)

    return _isdir, _listdir, _open


class TestScanMcpWatchdogPidsProc:
    def test_matches_via_proc(self):
        my_pid = os.getpid()
        watchdog_cmd = "python3 tools/mcp_stdio_watchdog.py --ppid 4242 -- arr-mcp.sh"
        other_ppid_cmd = "python3 tools/mcp_stdio_watchdog.py --ppid 9999 -- arr-mcp.sh"
        unrelated_cmd = "python -m hermes_cli.main gateway run"
        entries = {
            my_pid: "python -m hermes_cli.main",
            111: watchdog_cmd,
            222: other_ppid_cmd,
            333: unrelated_cmd,
        }
        _isdir, _listdir, _open = _fake_proc_dir(entries)

        with (
            patch("os.path.isdir", side_effect=_isdir),
            patch("os.listdir", side_effect=_listdir),
            patch("builtins.open", side_effect=_open),
            patch("subprocess.run") as mock_ps,
        ):
            pids = gateway_mod._scan_mcp_watchdog_pids(4242)

        assert pids == [111]
        mock_ps.assert_not_called()

    def test_non_posix_is_a_noop(self):
        with patch("os.name", "nt"):
            assert gateway_mod._scan_mcp_watchdog_pids(4242) == []


class TestScanMcpWatchdogPidsPsFallback:
    """/proc absent (e.g. macOS, the reporter's platform) -> ps -A eww fallback."""

    def test_matches_via_ps_fallback(self):
        ps_stdout = (
            "  111 python3 tools/mcp_stdio_watchdog.py --ppid 4242 -- arr-mcp.sh\n"
            "  222 python3 tools/mcp_stdio_watchdog.py --ppid 9999 -- arr-mcp.sh\n"
            "  333 python -m hermes_cli.main gateway run\n"
        )
        mock_result = MagicMock(returncode=0, stdout=ps_stdout)

        with (
            patch("os.path.isdir", return_value=False),
            patch("subprocess.run", return_value=mock_result) as mock_ps,
        ):
            pids = gateway_mod._scan_mcp_watchdog_pids(4242)

        assert pids == [111]
        mock_ps.assert_called_once()
        assert mock_ps.call_args.args[0][:2] == ["ps", "-A"]

    def test_ps_nonzero_exit_returns_empty(self):
        mock_result = MagicMock(returncode=1, stdout="")
        with (
            patch("os.path.isdir", return_value=False),
            patch("subprocess.run", return_value=mock_result),
        ):
            assert gateway_mod._scan_mcp_watchdog_pids(4242) == []

    def test_ps_spawn_failure_returns_empty(self):
        with (
            patch("os.path.isdir", return_value=False),
            patch("subprocess.run", side_effect=OSError("no ps")),
        ):
            assert gateway_mod._scan_mcp_watchdog_pids(4242) == []


@pytest.mark.skipif(os.name != "posix", reason="watchdog wrapping is POSIX-only")
class TestScanMcpWatchdogPidsRealPs:
    """Exercises the real ``ps`` binary end to end, unmocked.

    The tests above mock ``subprocess.run`` entirely, so they can't catch a
    broken argv. The original fallback invocation (``ps -A eww -o
    pid=,command=``) was rejected outright by a real macOS ``ps`` — the
    exact platform this scan exists for, since macOS has no /proc — while
    every mocked test still passed. This spawns the real watchdog script
    against a real ``sleep`` and confirms the real scan finds it.
    """

    def test_finds_a_real_watchdog_process(self):
        own_pid = os.getpid()  # the watchdog's actual OS parent once spawned below
        proc = subprocess.Popen(
            [sys.executable, _WATCHDOG_SCRIPT, "--ppid", str(own_pid), "--", "sleep", "5"],
        )
        try:
            deadline = time.monotonic() + 5
            pids: list[int] = []
            while time.monotonic() < deadline:
                pids = gateway_mod._scan_mcp_watchdog_pids(own_pid)
                if pids:
                    break
                time.sleep(0.2)
            assert proc.pid in pids
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)


# ---------------------------------------------------------------------------
# _reap_mcp_watchdog_children
# ---------------------------------------------------------------------------


class TestReapMcpWatchdogChildren:
    def test_non_posix_is_a_noop(self):
        with patch("os.name", "nt"):
            assert gateway_mod._reap_mcp_watchdog_children(4242) == 0

    def test_invalid_pid_is_a_noop(self):
        assert gateway_mod._reap_mcp_watchdog_children(None) == 0
        assert gateway_mod._reap_mcp_watchdog_children(0) == 0
        assert gateway_mod._reap_mcp_watchdog_children(-1) == 0

    def test_no_matching_watchdogs_is_a_noop(self):
        with (
            patch("hermes_cli.gateway._scan_mcp_watchdog_pids", return_value=[]),
            patch("hermes_cli.gateway.os.kill") as mock_kill,
        ):
            assert gateway_mod._reap_mcp_watchdog_children(4242) == 0
        mock_kill.assert_not_called()

    def test_happy_path_sigterm_confirmed_gone(self):
        """Watchdog is SIGTERM'd and reported gone -> counted as reaped."""
        with (
            patch(
                "hermes_cli.gateway._scan_mcp_watchdog_pids", return_value=[111]
            ),
            patch("hermes_cli.gateway.os.kill") as mock_kill,
            patch("gateway.status._pid_exists", return_value=False),
            patch("hermes_cli.gateway.time.sleep") as mock_sleep,
        ):
            reaped = gateway_mod._reap_mcp_watchdog_children(4242, timeout=5.0)

        assert reaped == 1
        mock_kill.assert_called_once_with(111, signal.SIGTERM)
        mock_sleep.assert_not_called()  # gone on first check, no poll wait needed

    def test_multiple_watchdogs_sigtermed_up_front_batched(self):
        """All matched PIDs get SIGTERM before any _pid_exists poll — not
        sequentially with a check-and-wait between each individual kill (a
        user with 5 configured stdio MCP servers must not pay a 5x wait)."""
        call_order = []

        def _record_kill(pid, sig):
            call_order.append(("kill", pid, sig))

        def _record_pid_exists(pid):
            call_order.append(("pid_exists", pid))
            return False  # all confirmed gone on the very first poll

        def _record_sleep(seconds):
            call_order.append(("sleep", seconds))

        with (
            patch(
                "hermes_cli.gateway._scan_mcp_watchdog_pids",
                return_value=[111, 222, 333],
            ),
            patch("hermes_cli.gateway.os.kill", side_effect=_record_kill) as mock_kill,
            patch("gateway.status._pid_exists", side_effect=_record_pid_exists),
            patch("hermes_cli.gateway.time.sleep", side_effect=_record_sleep),
        ):
            reaped = gateway_mod._reap_mcp_watchdog_children(4242, timeout=5.0)

        assert reaped == 3
        assert mock_kill.call_count == 3
        assert {c[1] for c in call_order if c[0] == "kill"} == {111, 222, 333}

        kill_indices = [i for i, c in enumerate(call_order) if c[0] == "kill"]
        other_indices = [i for i, c in enumerate(call_order) if c[0] != "kill"]
        # All three SIGTERMs happen before the first _pid_exists/sleep check —
        # a batched up-front signal, not one-kill-then-wait repeated 3x.
        assert other_indices, "test setup should have produced pid_exists calls"
        assert max(kill_indices) < min(other_indices)

    def test_unresponsive_watchdog_never_escalates_to_sigkill(self):
        """CRITICAL regression guard: an unresponsive watchdog gets exactly one
        os.kill call total (SIGTERM) — no SIGKILL escalation — and a warning
        is logged. SIGKILL-ing the watchdog would bypass its shutdown handler
        and orphan the real MCP child (the exact bug #87906 exists to fix)."""
        with (
            patch(
                "hermes_cli.gateway._scan_mcp_watchdog_pids", return_value=[111]
            ),
            patch("hermes_cli.gateway.os.kill") as mock_kill,
            patch("gateway.status._pid_exists", return_value=True),
            patch("hermes_cli.gateway.time.sleep"),
            patch.object(gateway_mod.logger, "warning") as mock_warn,
        ):
            reaped = gateway_mod._reap_mcp_watchdog_children(4242, timeout=0.05)

        assert mock_kill.call_count == 1
        mock_kill.assert_called_once_with(111, signal.SIGTERM)
        sigkill_calls = [
            c for c in mock_kill.call_args_list if signal.SIGTERM not in c.args
        ]
        assert sigkill_calls == []
        assert reaped == 0
        mock_warn.assert_called_once()
        warn_args = mock_warn.call_args.args
        assert 111 in warn_args[1]
        assert warn_args[2] == 4242


# ---------------------------------------------------------------------------
# Wiring: manual-restart branch calls the reaper with the captured old PID,
# before run_gateway(), and a reap failure does not block the restart.
# ---------------------------------------------------------------------------


class TestManualRestartWiresWatchdogReap:
    def _make_args(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            gateway_command="restart",
            all=False,
            system=False,
        )

    def test_reap_called_with_captured_old_pid_before_run_gateway(self):
        call_order = []

        with (
            patch.object(gateway_mod, "supports_systemd_services", return_value=False),
            patch.object(gateway_mod, "is_macos", return_value=False),
            patch.object(gateway_mod, "is_windows", return_value=False),
            patch("gateway.status.get_running_pid", return_value=5555),
            patch.object(
                gateway_mod,
                "stop_profile_gateway",
                side_effect=lambda: call_order.append("stop") or True,
            ),
            patch.object(
                gateway_mod,
                "_wait_for_gateway_exit",
                side_effect=lambda **_kw: call_order.append("wait") or True,
            ),
            patch.object(
                gateway_mod,
                "_reap_mcp_watchdog_children",
                side_effect=lambda pid, **_kw: call_order.append(("reap", pid)) or 2,
            ) as mock_reap,
            patch.object(
                gateway_mod,
                "run_gateway",
                side_effect=lambda **_kw: call_order.append("run") or None,
            ) as mock_run,
        ):
            gateway_mod._gateway_command_inner(self._make_args())

        mock_reap.assert_called_once_with(5555)
        mock_run.assert_called_once()
        assert call_order == ["stop", "wait", ("reap", 5555), "run"]

    def test_reap_failure_does_not_block_restart(self):
        with (
            patch.object(gateway_mod, "supports_systemd_services", return_value=False),
            patch.object(gateway_mod, "is_macos", return_value=False),
            patch.object(gateway_mod, "is_windows", return_value=False),
            patch("gateway.status.get_running_pid", return_value=5555),
            patch.object(gateway_mod, "stop_profile_gateway", return_value=True),
            patch.object(gateway_mod, "_wait_for_gateway_exit", return_value=True),
            patch.object(
                gateway_mod,
                "_reap_mcp_watchdog_children",
                side_effect=OSError("permission denied"),
            ),
            patch.object(gateway_mod, "run_gateway") as mock_run,
        ):
            gateway_mod._gateway_command_inner(self._make_args())

        mock_run.assert_called_once()

    def test_no_old_pid_skips_reap(self):
        with (
            patch.object(gateway_mod, "supports_systemd_services", return_value=False),
            patch.object(gateway_mod, "is_macos", return_value=False),
            patch.object(gateway_mod, "is_windows", return_value=False),
            patch("gateway.status.get_running_pid", return_value=None),
            patch.object(gateway_mod, "stop_profile_gateway", return_value=False),
            patch.object(gateway_mod, "_wait_for_gateway_exit", return_value=True),
            patch.object(gateway_mod, "_reap_mcp_watchdog_children") as mock_reap,
            patch.object(gateway_mod, "run_gateway") as mock_run,
        ):
            gateway_mod._gateway_command_inner(self._make_args())

        mock_reap.assert_not_called()
        mock_run.assert_called_once()
