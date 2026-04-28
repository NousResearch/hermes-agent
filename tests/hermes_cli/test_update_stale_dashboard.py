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


class TestWarnStaleDashboardProcesses:
    """Unit tests for the stale dashboard process warning."""

    def test_no_warning_when_no_dashboard_running(self, capsys):
        """pgrep finds nothing — no warning should be printed."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr=""
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "dashboard process" not in output

    def test_warning_printed_for_running_dashboard(self, capsys):
        """pgrep finds a dashboard PID — warning with PID should appear."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="12345\n", stderr=""
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
                returncode=0, stdout="12345\n12346\n12347\n", stderr=""
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
            # Return the current process PID
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=f"{os.getpid()}\n12345\n",
                stderr="",
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert str(os.getpid()) not in output
        assert "PID 12345" in output

    def test_pgrep_not_found_silently_ignored(self, capsys):
        """If pgrep is missing (FileNotFoundError), no crash, no warning."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert output == ""

    def test_pgrep_timeout_silently_ignored(self, capsys):
        """If pgrep times out, no crash, no warning."""
        import subprocess as sp

        with patch("subprocess.run", side_effect=sp.TimeoutExpired("pgrep", 5)):
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert output == ""

    def test_empty_pgrep_output_no_warning(self, capsys):
        """pgrep returns 0 but empty stdout — no warning."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="\n", stderr=""
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "dashboard process" not in output

    def test_invalid_pid_lines_skipped(self, capsys):
        """Non-numeric lines from pgrep should be skipped gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="notapid\n12345\nalso_bad\n", stderr=""
            )
            _warn_stale_dashboard_processes()
        output = capsys.readouterr().out
        assert "PID 12345" in output
        assert "1 dashboard process" in output
