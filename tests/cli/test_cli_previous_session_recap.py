"""Tests for _show_previous_session_recap() — the one-line "last time you
were doing X, N ago" reminder shown on a fresh (non-resumed) CLI start.

Distinct from _display_resumed_history() (tests/cli/test_resume_display.py),
which shows a full multi-exchange recap when /resume reattaches to the SAME
session. This is a one-liner about the PREVIOUS, already-ended session,
shown when starting a brand new one.
"""

from unittest.mock import MagicMock

import pytest

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj._session_db = MagicMock()
    return cli_obj


class TestShowPreviousSessionRecap:
    def test_shows_title_and_relative_time(self, capsys):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_001", "title": "fixing the memory leak", "preview": "", "last_active": None},
        ])

        cli_obj._show_previous_session_recap()
        output = capsys.readouterr().out

        assert '"fixing the memory leak"' in output
        assert "Last time:" in output

    def test_falls_back_to_preview_when_untitled(self, capsys):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_001", "title": "", "preview": "help me debug this crash", "last_active": None},
        ])

        cli_obj._show_previous_session_recap()
        output = capsys.readouterr().out

        assert '"help me debug this crash"' in output

    def test_no_prior_sessions_prints_nothing(self, capsys):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[])

        cli_obj._show_previous_session_recap()
        output = capsys.readouterr().out

        assert output == ""

    def test_untitled_and_no_preview_prints_nothing(self, capsys):
        """A session with neither a title nor a preview (e.g. never got a
        user message) has nothing worth recapping."""
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_001", "title": "", "preview": "", "last_active": None},
        ])

        cli_obj._show_previous_session_recap()
        output = capsys.readouterr().out

        assert output == ""

    def test_long_label_is_truncated(self, capsys):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_001", "title": "x" * 200, "preview": "", "last_active": None},
        ])

        cli_obj._show_previous_session_recap()
        output = capsys.readouterr().out

        assert "x" * 200 not in output
        assert "…" in output

    def test_escape_sequences_in_label_are_stripped(self, capsys):
        """Stored session titles/previews are untrusted for display — a
        title carrying terminal escape sequences must not reach the
        terminal raw (mirrors the same threat model _display_resumed_history
        and session_recap.py guard against)."""
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_001", "title": "\x1b[2J\x1b]0;pwned\x07evil title", "preview": "", "last_active": None},
        ])

        cli_obj._show_previous_session_recap()
        output = capsys.readouterr().out

        assert "\x1b" not in output
        assert "evil title" in output

    def test_never_raises_when_list_recent_sessions_fails(self):
        """Best-effort: this must never block CLI startup."""
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(side_effect=RuntimeError("db locked"))

        cli_obj._show_previous_session_recap()  # must not raise

    def test_only_looks_at_the_single_most_recent_session(self):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[])

        cli_obj._show_previous_session_recap()

        cli_obj._list_recent_sessions.assert_called_once_with(limit=1)


class _StopRun(Exception):
    """Raised by the stubbed ``_console_print`` to abort ``run()`` right after
    the startup dispatch branch, before it reaches the interactive input loop
    (which would otherwise block forever in a test)."""


def _make_run_ready_cli(*, resumed):
    """A HermesCLI wired to run real dispatch logic in HermesCLI.run(), with
    everything before and after the branch under test stubbed out."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._resumed = resumed
    cli_obj._claim_active_session = MagicMock(return_value=True)
    cli_obj.show_banner = MagicMock()
    cli_obj._show_security_advisories = MagicMock()
    cli_obj._preload_resumed_session = MagicMock(return_value=True)
    cli_obj._display_resumed_history = MagicMock()
    cli_obj._show_previous_session_recap = MagicMock()
    # First real print after the dispatch branch (the welcome banner) aborts
    # the run — we only want to exercise the branch itself.
    cli_obj._console_print = MagicMock(side_effect=_StopRun)
    return cli_obj


class TestRunStartupDispatch:
    """Runtime coverage for the wiring in HermesCLI.run(): the previous-session
    recap and the resumed-history recap are mutually exclusive, driven by
    self._resumed. Executes the real run() dispatch rather than reading its
    source (AGENTS.md bans source-inspection tests: they pass on broken wiring
    and fail on harmless refactors)."""

    def test_fresh_start_shows_recap_not_resumed_history(self):
        cli_obj = _make_run_ready_cli(resumed=False)

        with pytest.raises(_StopRun):
            cli_obj.run()

        cli_obj._show_previous_session_recap.assert_called_once()
        cli_obj._display_resumed_history.assert_not_called()

    def test_resumed_start_shows_history_not_recap(self):
        cli_obj = _make_run_ready_cli(resumed=True)

        with pytest.raises(_StopRun):
            cli_obj.run()

        cli_obj._display_resumed_history.assert_called_once()
        cli_obj._show_previous_session_recap.assert_not_called()

    def test_resumed_but_preload_failure_shows_neither(self):
        """A resumed session whose history fails to preload shows no recap at
        all rather than falling back to the fresh-start one."""
        cli_obj = _make_run_ready_cli(resumed=True)
        cli_obj._preload_resumed_session.return_value = False

        with pytest.raises(_StopRun):
            cli_obj.run()

        cli_obj._display_resumed_history.assert_not_called()
        cli_obj._show_previous_session_recap.assert_not_called()
