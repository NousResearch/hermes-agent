"""Regression tests for profile-aware stale dashboard detection.

Verifies that ``_find_stale_dashboard_pids`` correctly identifies dashboard
processes launched with ``--profile <name>`` or ``-p <name>`` flags between
the binary name and the subcommand (issue #56717).
"""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_ps_output(lines: list[tuple[int, str]]) -> str:
    """Build a fake ``ps -A -o pid=,command=`` output string."""
    return "\n".join(f"{pid} {cmd}" for pid, cmd in lines)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProfileAwareDashboardDetection:
    """Non-default profile dashboards should be found even when
    ``--profile <name>`` sits between the binary and the subcommand."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only ps test")
    def test_profile_flag_between_binary_and_subcommand(self):
        """``hermes --profile bruce dashboard --isolated`` should match."""
        from hermes_cli.main import _find_stale_dashboard_pids

        fake_ps = _fake_ps_output([
            (9001, "/venv/bin/hermes --profile bruce dashboard --isolated"),
            (9002, "/venv/bin/python -m hermes_cli.main --profile bruce dashboard --isolated"),
            (9003, "/venv/bin/python -m hermes_cli.main -p coder serve --port 9119"),
            (9999, "/usr/bin/vim somefile.py"),  # unrelated — must NOT match
        ])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=fake_ps,
            )
            pids = _find_stale_dashboard_pids()

        assert 9001 in pids, "hermes --profile bruce dashboard should be detected"
        assert 9002 in pids, "hermes_cli.main --profile bruce dashboard should be detected"
        assert 9003 in pids, "hermes_cli.main -p coder serve should be detected"
        assert 9999 not in pids, "unrelated process must not match"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only ps test")
    def test_default_profile_still_detected(self):
        """Existing non-profile dashboards must continue to match."""
        from hermes_cli.main import _find_stale_dashboard_pids

        fake_ps = _fake_ps_output([
            (8001, "/venv/bin/hermes dashboard --isolated"),
            (8002, "/venv/bin/python -m hermes_cli.main dashboard --isolated"),
        ])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=fake_ps,
            )
            pids = _find_stale_dashboard_pids()

        assert 8001 in pids
        assert 8002 in pids

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only ps test")
    def test_short_profile_flag_p(self):
        """``-p bruce`` should also be normalised away."""
        from hermes_cli.main import _find_stale_dashboard_pids

        fake_ps = _fake_ps_output([
            (7001, "/venv/bin/hermes -p bruce dashboard --isolated"),
        ])
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout=fake_ps,
            )
            pids = _find_stale_dashboard_pids()

        assert 7001 in pids
