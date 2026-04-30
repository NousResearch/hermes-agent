"""Tests for the stale-dashboard handling run at the end of ``hermes update``.

``hermes update`` detects ``hermes dashboard`` processes left over from the
previous version and kills them (SIGTERM + SIGKILL grace, or ``taskkill /F``
on Windows).  Without this, the running backend silently serves stale Python
against a freshly-updated JS bundle, producing 401s / empty data.

History:
- #16872 introduced the warn-only helper (``_warn_stale_dashboard_processes``).
- #17049 fixed a Windows wmic UnicodeDecodeError crash on non-UTF-8 locales.
- This file now also covers the kill semantics that replaced the warning.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock, call

import pytest

from hermes_cli.main import (
    _find_stale_dashboard_pids,
    _kill_stale_dashboard_processes,
    _warn_stale_dashboard_processes,  # back-compat alias
)


def _ps_line(pid: int, cmd: str) -> str:
    """Format a line as it would appear in ``ps -A -o pid=,command=`` output."""
    return f"{pid:>7} {cmd}"


def _ps_runner(stdout: str):
    """Build a subprocess.run side_effect that only stubs ps -A calls.

    Any other subprocess.run invocation (e.g. taskkill on Windows) is
    handed back as a successful no-op.  This lets tests exercise the real
    scan path without having to re-stub every unrelated subprocess call
    made later in ``_kill_stale_dashboard_processes``.
    """
    def _side_effect(args, *a, **kw):
        if isinstance(args, (list, tuple)) and args and args[0] == "ps":
            return MagicMock(returncode=0, stdout=stdout, stderr="")
        # Any other subprocess.run (e.g. taskkill) — benign success stub.
        return MagicMock(returncode=0, stdout="", stderr="")
    return _side_effect


class TestFindStaleDashboardPids:
    """Unit tests for the ps/wmic-based detection step."""

    def test_no_matches_returns_empty(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(111, "/usr/bin/python3 -m some.other.module")
                + "\n"
                + _ps_line(222, "/usr/bin/bash")
                + "\n",
                stderr="",
            )
            assert _find_stale_dashboard_pids() == []

    def test_matches_running_dashboard(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119") + "\n",
                stderr="",
            )
            assert _find_stale_dashboard_pids() == [12345]

    def test_multiple_matches(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119"),
                    _ps_line(12346, "hermes dashboard --port 9120 --no-open"),
                    _ps_line(12347, "python /home/x/hermes_cli/main.py dashboard"),
                ]) + "\n",
                stderr="",
            )
            assert sorted(_find_stale_dashboard_pids()) == [12345, 12346, 12347]

    def test_self_pid_excluded(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(os.getpid(), "python3 -m hermes_cli.main dashboard"),
                    _ps_line(12345, "hermes dashboard --port 9119"),
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert os.getpid() not in pids
        assert 12345 in pids

    def test_ps_not_found_returns_empty(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _find_stale_dashboard_pids() == []

    def test_ps_timeout_returns_empty(self):
        import subprocess as sp
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ps", 10)):
            assert _find_stale_dashboard_pids() == []

    def test_unrelated_process_containing_word_dashboard_not_matched(self):
        """Guards against greedy pgrep-style matching catching chat sessions
        or unrelated processes whose cmdline happens to contain 'dashboard'.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119"),
                    _ps_line(22222, "python3 -m hermes_cli.main chat -q 'rewrite my dashboard'"),
                    _ps_line(33333, "node /opt/grafana/dashboard-server.js"),
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert pids == [12345]

    def test_grep_lines_ignored(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(99999, "grep hermes dashboard"),
                    _ps_line(12345, "hermes dashboard --port 9119"),
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert 99999 not in pids
        assert 12345 in pids

    def test_invalid_pid_lines_skipped(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    "notapid hermes dashboard --bad",
                    _ps_line(12345, "hermes dashboard --port 9119"),
                    "   ",
                ]) + "\n",
                stderr="",
            )
            pids = _find_stale_dashboard_pids()
        assert pids == [12345]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX kill semantics")
class TestKillStaleDashboardPosix:
    """Kill path on Linux / macOS: SIGTERM then SIGKILL any survivors."""

    def test_no_stale_processes_is_a_noop(self, capsys):
        with patch("hermes_cli.main._find_stale_dashboard_pids", return_value=[]):
            _kill_stale_dashboard_processes()
        assert capsys.readouterr().out == ""

    def test_sigterm_graceful_exit(self, capsys):
        """Processes that exit on SIGTERM (the probe gets ProcessLookupError)
        are reported as stopped and SIGKILL is never sent."""
        import signal as _signal

        killed_signals: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            killed_signals.append((pid, sig))
            if sig == 0:
                # Probe after SIGTERM → "process gone".
                raise ProcessLookupError
            # SIGTERM itself: succeed silently.

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345, 12346]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes()

        # Both got SIGTERM.
        sigterms = [pid for pid, sig in killed_signals if sig == _signal.SIGTERM]
        assert sorted(sigterms) == [12345, 12346]
        # No SIGKILL was needed.
        assert not any(sig == _signal.SIGKILL for _, sig in killed_signals)

        out = capsys.readouterr().out
        assert "Stopping 2 dashboard" in out
        assert "✓ stopped PID 12345" in out
        assert "✓ stopped PID 12346" in out
        assert "Restart the dashboard" in out

    def test_sigkill_fallback_for_survivors(self, capsys):
        """If a process survives SIGTERM + the grace window, SIGKILL is sent."""
        import signal as _signal

        sent: list[tuple[int, int]] = []

        def fake_kill(pid, sig):
            sent.append((pid, sig))
            # Simulate stubborn process: probe (sig 0) always succeeds,
            # SIGTERM does nothing, SIGKILL is where it "dies".
            if sig in (_signal.SIGTERM, 0, _signal.SIGKILL):
                return
            # Any other signal — also fine.

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[99999]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"), \
             patch("time.monotonic", side_effect=[0.0] + [10.0] * 20):
            # monotonic jumps past the 3s deadline on the second read so the
            # grace loop exits immediately after one iteration.
            _kill_stale_dashboard_processes()

        signals_sent = [sig for _, sig in sent]
        assert _signal.SIGTERM in signals_sent
        assert _signal.SIGKILL in signals_sent

        out = capsys.readouterr().out
        assert "✓ stopped PID 99999" in out

    def test_permission_error_is_reported_not_raised(self, capsys):
        """os.kill raising PermissionError (e.g. another user's process)
        must not abort hermes update — it's reported as a failure and we
        move on."""
        def fake_kill(pid, sig):
            raise PermissionError("Operation not permitted")

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes()  # must not raise

        out = capsys.readouterr().out
        assert "✗ failed to stop PID 12345" in out
        assert "Operation not permitted" in out

    def test_process_already_gone_counts_as_stopped(self, capsys):
        """ProcessLookupError on the initial SIGTERM means the process
        already exited between detection and the kill — treat as success."""
        def fake_kill(pid, sig):
            raise ProcessLookupError

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345]), \
             patch("os.kill", side_effect=fake_kill), \
             patch("time.sleep"):
            _kill_stale_dashboard_processes()

        out = capsys.readouterr().out
        assert "✓ stopped PID 12345" in out
        assert "failed to stop" not in out


class TestKillStaleDashboardWindows:
    """Kill path on Windows: taskkill /F."""

    def test_taskkill_invoked_for_each_pid(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "win32")

        def fake_run(args, *a, **kw):
            # taskkill returns 0 on success
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345, 12346]), \
             patch("subprocess.run", side_effect=fake_run) as mock_run:
            _kill_stale_dashboard_processes()

        # Each PID triggered a taskkill /PID <n> /F invocation.
        taskkill_calls = [
            c for c in mock_run.call_args_list
            if c.args and isinstance(c.args[0], list) and c.args[0][:1] == ["taskkill"]
        ]
        assert len(taskkill_calls) == 2
        assert ["taskkill", "/PID", "12345", "/F"] in [c.args[0] for c in taskkill_calls]
        assert ["taskkill", "/PID", "12346", "/F"] in [c.args[0] for c in taskkill_calls]

        out = capsys.readouterr().out
        assert "✓ stopped PID 12345" in out
        assert "✓ stopped PID 12346" in out

    def test_taskkill_failure_is_reported(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "win32")

        def fake_run(args, *a, **kw):
            return MagicMock(returncode=128, stdout="",
                             stderr="ERROR: Access is denied.")

        with patch("hermes_cli.main._find_stale_dashboard_pids",
                   return_value=[12345]), \
             patch("subprocess.run", side_effect=fake_run):
            _kill_stale_dashboard_processes()  # must not raise

        out = capsys.readouterr().out
        assert "✗ failed to stop PID 12345" in out
        assert "Access is denied" in out


class TestBackCompatAlias:
    """``_warn_stale_dashboard_processes`` is kept as an alias for the
    new kill function so old imports don't break."""

    def test_alias_is_the_kill_function(self):
        assert _warn_stale_dashboard_processes is _kill_stale_dashboard_processes


class TestWindowsWmicEncoding:
    """Regression tests for #17049 — the Windows wmic branch must not crash
    `hermes update` on non-UTF-8 system locales (e.g. cp936 on zh-CN).
    """

    def test_wmic_invoked_with_utf8_ignore_errors(self, monkeypatch):
        """The wmic subprocess.run call must pass encoding='utf-8' and
        errors='ignore' so the subprocess reader thread cannot raise
        UnicodeDecodeError on non-UTF-8 wmic output."""
        monkeypatch.setattr(sys, "platform", "win32")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "CommandLine=python -m hermes_cli.main dashboard\n"
                    "ProcessId=12345\n"
                ),
                stderr="",
            )
            _find_stale_dashboard_pids()

        # The wmic call is the first subprocess.run invocation.
        assert mock_run.called, "subprocess.run was not invoked"
        wmic_call = mock_run.call_args_list[0]
        kwargs = wmic_call.kwargs
        assert kwargs.get("encoding") == "utf-8", (
            "encoding kwarg must be 'utf-8' so wmic output is decoded "
            "deterministically rather than via the implicit reader-thread "
            "default that crashes on non-UTF-8 locales (#17049)."
        )
        assert kwargs.get("errors") == "ignore", (
            "errors kwarg must be 'ignore' so undecodable bytes don't take "
            "down the reader thread (#17049)."
        )

    def test_wmic_returns_none_stdout_does_not_crash(self, monkeypatch):
        """If subprocess.run returns successfully but stdout is None — which
        is what Python 3.11 leaves behind when the reader thread silently
        crashed on UnicodeDecodeError before this fix landed — detection
        must short-circuit instead of raising AttributeError on
        ``None.split('\\n')`` and aborting `hermes update` (#17049)."""
        monkeypatch.setattr(sys, "platform", "win32")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=None, stderr=""
            )
            # Must not raise.
            assert _find_stale_dashboard_pids() == []
