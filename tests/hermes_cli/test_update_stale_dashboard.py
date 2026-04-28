"""Tests for _warn_stale_dashboard_processes — stale dashboard detection.

Ensures ``hermes update`` warns the user when dashboard processes from a
previous version are still running after files on disk have been replaced.
See #16872.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.main import _warn_stale_dashboard_processes


def _ps_line(pid: int, cmd: str) -> str:
    """Format a line as it would appear in ``ps -A -o pid=,command=`` output."""
    return f"{pid:>7} {cmd}"


class TestWarnStaleDashboardProcesses:
    """Unit tests for the stale dashboard process warning."""

    def test_no_warning_when_no_dashboard_running(self, capsys):
        """ps returns no matching processes — no warning should be printed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(111, "/usr/bin/python3 -m some.other.module")
                + "\n"
                + _ps_line(222, "/usr/bin/bash")
                + "\n",
                stderr="",
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "dashboard process" not in output

    def test_warning_printed_for_running_dashboard(self, capsys):
        """ps finds a dashboard PID — warning with PID should appear."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=_ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119") + "\n",
                stderr="",
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "1 dashboard process" in output
        assert "PID 12345" in output
        assert "kill <pid>" in output

    def test_multiple_dashboard_pids(self, capsys):
        """Multiple dashboard processes — all PIDs listed."""
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
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "3 dashboard process" in output
        assert "PID 12345" in output
        assert "PID 12346" in output
        assert "PID 12347" in output

    def test_self_pid_excluded(self, capsys):
        """The current process PID should not be reported."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(os.getpid(), "python3 -m hermes_cli.main dashboard"),
                    _ps_line(12345, "hermes dashboard --port 9119"),
                ]) + "\n",
                stderr="",
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        # The self PID may still appear inside an unrelated context, so anchor
        # the check to "PID <self>" which is how the warning prints.
        assert f"PID {os.getpid()}" not in output
        assert "PID 12345" in output

    def test_ps_not_found_silently_ignored(self, capsys):
        """If ps is missing (FileNotFoundError), no crash, no warning."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert output == ""

    def test_ps_timeout_silently_ignored(self, capsys):
        """If ps times out, no crash, no warning."""
        import subprocess as sp

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ps", 10)):
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert output == ""

    def test_empty_ps_output_no_warning(self, capsys):
        """ps returns 0 but empty stdout — no warning."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="\n", stderr=""
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "dashboard process" not in output

    def test_invalid_pid_lines_skipped(self, capsys):
        """Malformed ps lines should be skipped gracefully."""
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
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "PID 12345" in output
        assert "1 dashboard process" in output

    def test_unrelated_process_containing_word_dashboard_not_matched(self, capsys):
        """A process whose cmdline contains 'dashboard' but isn't a hermes
        dashboard process must NOT be flagged.  This guards against the old
        ``pgrep -f "hermes.*dashboard"`` greedy regex that matched e.g. a
        chat session argv containing both words.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    # Legitimate dashboard — should match.
                    _ps_line(12345, "python3 -m hermes_cli.main dashboard --port 9119"),
                    # hermes running something else, with "dashboard" as a
                    # substring of an unrelated arg — should NOT match.
                    _ps_line(22222, "python3 -m hermes_cli.main chat -q 'rewrite my dashboard'"),
                    # Completely unrelated process mentioning dashboard.
                    _ps_line(33333, "node /opt/grafana/dashboard-server.js"),
                ]) + "\n",
                stderr="",
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "1 dashboard process" in output
        assert "PID 12345" in output
        assert "PID 22222" not in output
        assert "PID 33333" not in output

    def test_grep_lines_ignored(self, capsys):
        """Lines containing 'grep' (from a pipe in ps output) are ignored."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="\n".join([
                    _ps_line(99999, "grep hermes dashboard"),
                    _ps_line(12345, "hermes dashboard --port 9119"),
                ]) + "\n",
                stderr="",
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "PID 99999" not in output
        assert "PID 12345" in output


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
            # Provide a minimal valid wmic /FORMAT:LIST response.
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=(
                    "CommandLine=python -m hermes_cli.main dashboard\n"
                    "ProcessId=12345\n"
                ),
                stderr="",
            )
            _warn_stale_dashboard_processes()

        assert mock_run.called, "subprocess.run was not invoked"
        kwargs = mock_run.call_args.kwargs
        assert kwargs.get("encoding") == "utf-8", (
            "encoding kwarg must be 'utf-8' so wmic output is decoded "
            "deterministically rather than via the implicit reader-thread "
            "default that crashes on non-UTF-8 locales (#17049)."
        )
        assert kwargs.get("errors") == "ignore", (
            "errors kwarg must be 'ignore' so undecodable bytes don't take "
            "down the reader thread (#17049)."
        )

    def test_wmic_returns_none_stdout_does_not_crash(self, monkeypatch, capsys):
        """If subprocess.run returns successfully but stdout is None — which
        is what Python 3.11 leaves behind when the reader thread silently
        crashed on UnicodeDecodeError before this fix landed — the warning
        must short-circuit instead of raising AttributeError on
        ``None.split('\\n')`` and aborting `hermes update` (#17049)."""
        monkeypatch.setattr(sys, "platform", "win32")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=None, stderr=""
            )
            # Must not raise.
            _warn_stale_dashboard_processes()

        output = capsys.readouterr().out
        assert "dashboard process" not in output
