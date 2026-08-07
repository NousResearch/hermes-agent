"""Tests for ``hermes_cli.dashboard_procs._is_hermes_dashboard_or_serve_cmd``.

Verifies that dashboard/serve process detection works correctly when
profile arguments appear between the executable and subcommand.
Fixes #75791.
"""

from __future__ import annotations

import pytest

from hermes_cli.dashboard_procs import _is_hermes_dashboard_or_serve_cmd


class TestIsHermesDashboardOrServeCmd:
    """Token-based command-line matching for dashboard/serve processes."""

    # --- Should match (True) ---

    @pytest.mark.parametrize("cmd", [
        "hermes dashboard --no-open --skip-build",
        "hermes_cli.main dashboard --port 9119",
        "python -m hermes_cli.main dashboard --no-open",
        "hermes serve --port 8080",
        "hermes_cli.main serve --port 8080",
        "python -m hermes_cli.main serve --port 8080",
        "hermes_cli/main.py dashboard --port 9119",
        "hermes_cli/main.py serve --port 8080",
    ])
    def test_basic_invocations(self, cmd):
        assert _is_hermes_dashboard_or_serve_cmd(cmd) is True

    @pytest.mark.parametrize("cmd", [
        "hermes --profile myprofile dashboard --no-open",
        "hermes -p production dashboard --port 8080",
        "hermes --profile myprofile serve --port 8080",
        "hermes --profile=prod dashboard",
        "hermes -p=prod dashboard",
        "python -m hermes_cli --profile myprofile dashboard",
        "python -m hermes_cli.main --profile work dashboard --port 9119",
    ])
    def test_profile_args_between_exe_and_subcommand(self, cmd):
        """Primary fix target: profile args no longer prevent matching."""
        assert _is_hermes_dashboard_or_serve_cmd(cmd) is True

    @pytest.mark.parametrize("cmd", [
        '"C:\\Program Files\\hermes\\hermes.exe" --profile work dashboard --port 9119',
        "hermes_cli.main dashboard --port 9119",
    ])
    def test_windows_paths(self, cmd):
        assert _is_hermes_dashboard_or_serve_cmd(cmd) is True

    # --- Should NOT match (False) ---

    @pytest.mark.parametrize("cmd", [
        None,
        "",
        "vim hermes dashboard",
        "grep hermes dashboard",
        "node server.js --port 9119",
        "hermes update",
        "hermes status",
        "hermes gateway status",
        "hermes_cli.main gateway run",
        "hermes -p dash dashbo",  # "dash" profile, "dashbo" ≠ dashboard/serve
    ])
    def test_non_matching_invocations(self, cmd):
        assert _is_hermes_dashboard_or_serve_cmd(cmd) is False

    def test_profile_named_dashboard_with_serve_subcommand(self):
        """Profile value 'dashboard' stripped, then 'serve' matches."""
        assert _is_hermes_dashboard_or_serve_cmd(
            "hermes --profile dashboard serve"
        ) is True
