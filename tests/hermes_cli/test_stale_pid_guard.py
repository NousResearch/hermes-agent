# -*- coding: utf-8 -*-
"""Regression tests for the fail-closed PID-ownership guard.

Refs #90471 / #89614.  The three patched Windows ``taskkill`` boundaries:

- ``hermes_cli/_subprocess_compat.pid_is_hermes`` / ``kill_process_tree``
- ``hermes_cli/dashboard_procs._kill_stale_dashboard_processes`` (win32)
- ``hermes_cli/update_cmd._stop_process_trees``

Acceptance from #90471:
1. missing / unreadable / non-matching identity fails closed -> no taskkill
2. a recycled or foreign PID control process remains untouched
3. probe failure or timeout is never converted into permission to kill
"""
import subprocess
import sys
from unittest import mock

import pytest

from hermes_cli import _subprocess_compat
from hermes_cli import dashboard_procs
from hermes_cli import update_cmd


def _probe_stdout(value: str) -> mock.Mock:
    return mock.Mock(stdout=value)


class TestPidIsHermes:
    """The shared identity probe must fail closed on every ambiguity."""

    def test_non_windows_is_unconditional_pass(self):
        # Non-Windows callers have no taskkill path at all.
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", False):
            assert _subprocess_compat.pid_is_hermes(1234) is True

    def test_invalid_pid_inputs_do_not_crash(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True):
            assert _subprocess_compat.pid_is_hermes(-1) is True
            assert _subprocess_compat.pid_is_hermes(0) is True
            assert _subprocess_compat.pid_is_hermes("not-a-pid") is True

    def test_probe_matches_hermes_like_process(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat.subprocess, "run", return_value=_probe_stdout("1\n")
        ) as run:
            assert _subprocess_compat.pid_is_hermes(1234) is True
            # the probe is a powershell identity query, never a taskkill
            argv = run.call_args.args[0]
            assert argv[0] == "powershell"
            assert "taskkill" not in argv

    def test_probe_rejects_foreign_process(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat.subprocess, "run", return_value=_probe_stdout("0\n")
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    def test_probe_blank_stdout_fails_closed(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat.subprocess, "run", return_value=_probe_stdout("")
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    def test_probe_timeout_fails_closed(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("powershell", 5),
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    def test_probe_oserror_fails_closed(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat.subprocess, "run", side_effect=OSError("broken pipe")
        ):
            assert _subprocess_compat.pid_is_hermes(1234) is False

    @pytest.mark.skipif(sys.platform != "win32", reason="real probe is windows-only")
    def test_missing_pid_real_probe_fails_closed(self):
        # A PID that cannot exist must never be judged Hermes-owned.
        assert _subprocess_compat.pid_is_hermes(2**24) is False


class TestKillProcessTree:
    """kill_process_tree must never taskkill a PID the probe rejects."""

    def _proc(self, pid=4321):
        return mock.Mock(pid=pid)

    def test_foreign_pid_never_taskkilled(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "pid_is_hermes", return_value=False
        ) as guard, mock.patch.object(_subprocess_compat.subprocess, "run") as run:
            _subprocess_compat.kill_process_tree(self._proc())
            guard.assert_called_once_with(4321)
            run.assert_not_called()

    def test_probe_error_never_taskkilled(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "pid_is_hermes", return_value=False
        ), mock.patch.object(_subprocess_compat.subprocess, "run") as run:
            _subprocess_compat.kill_process_tree(self._proc())
            run.assert_not_called()

    def test_hermes_pid_still_taskkilled(self):
        with mock.patch.object(_subprocess_compat, "IS_WINDOWS", True), mock.patch.object(
            _subprocess_compat, "pid_is_hermes", return_value=True
        ), mock.patch.object(_subprocess_compat.subprocess, "run") as run:
            _subprocess_compat.kill_process_tree(self._proc())
            run.assert_called_once()
            argv = run.call_args.args[0]
            assert argv[0] == "taskkill"
            assert "/PID" in argv
            assert str(4321) in argv


class TestStopProcessTrees:
    """update_cmd._stop_process_trees guard behaviour."""

    def test_foreign_pids_only_probed(self):
        with mock.patch.object(
            update_cmd.subprocess, "run", return_value=_probe_stdout("0\n")
        ) as run:
            update_cmd._stop_process_trees([1111, 2222])
        # two probes, zero taskkill
        assert len(run.call_args_list) == 2
        for call in run.call_args_list:
            assert call.args[0][0] != "taskkill"

    def test_hermes_pid_probed_then_taskkilled(self):
        calls = [_probe_stdout("1\n"), mock.Mock(returncode=0)]
        with mock.patch.object(update_cmd.subprocess, "run", side_effect=calls) as run:
            update_cmd._stop_process_trees([1111])
        assert len(run.call_args_list) == 2
        assert run.call_args_list[0].args[0][0] == "powershell"
        assert run.call_args_list[1].args[0][0] == "taskkill"

    def test_probe_timeout_skips_taskkill(self):
        with mock.patch.object(
            update_cmd.subprocess, "run",
            side_effect=subprocess.TimeoutExpired("powershell", 5),
        ) as run:
            update_cmd._stop_process_trees([1111, 2222])  # must not raise
        assert len(run.call_args_list) == 2  # probe attempted per pid, never taskkill
        for call in run.call_args_list:
            assert call.args[0][0] != "taskkill"


class TestKillStaleDashboardProcesses:
    """dashboard_procs win32 kill branch guard behaviour."""

    def _fake_m(self, pids=(12345,)):
        m = mock.Mock()
        m._find_stale_dashboard_pids.return_value = list(pids)
        return m

    def test_foreign_pid_reported_not_killed(self):
        with mock.patch.object(dashboard_procs, "_m", return_value=self._fake_m()), mock.patch.object(
            dashboard_procs.sys, "platform", "win32"
        ), mock.patch.object(
            dashboard_procs.subprocess, "run", return_value=_probe_stdout("0\n")
        ) as run:
            result = dashboard_procs._kill_stale_dashboard_processes()
        assert result["killed"] == []
        assert result["failed"] == [(12345, "not hermes-owned (USER PATCH guard)")]
        for call in run.call_args_list:
            assert call.args[0][0] != "taskkill"

    def test_hermes_pid_killed(self):
        calls = [_probe_stdout("1\n"), mock.Mock(returncode=0, stderr="", stdout="")]
        with mock.patch.object(dashboard_procs, "_m", return_value=self._fake_m()), mock.patch.object(
            dashboard_procs.sys, "platform", "win32"
        ), mock.patch.object(dashboard_procs.subprocess, "run", side_effect=calls) as run:
            result = dashboard_procs._kill_stale_dashboard_processes()
        taskkill_calls = [c for c in run.call_args_list if c.args[0][0] == "taskkill"]
        assert len(taskkill_calls) == 1
        assert result["killed"] == [12345]
        assert result["failed"] == []
