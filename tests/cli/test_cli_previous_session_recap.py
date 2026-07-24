"""Tests for _show_previous_session_recap() — the one-line "last time you
were doing X, N ago" reminder shown on a fresh (non-resumed) CLI start.

Distinct from _display_resumed_history() (tests/cli/test_resume_display.py),
which shows a full multi-exchange recap when /resume reattaches to the SAME
session. This is a one-liner about the PREVIOUS, already-ended session,
shown when starting a brand new one.
"""

from unittest.mock import MagicMock

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


def test_run_calls_recap_only_on_the_non_resumed_branch():
    """Source-level regression guard for the wiring in HermesCLI.run():
    _show_previous_session_recap() must sit in the same if/else as
    _display_resumed_history() (mutually exclusive), not run unconditionally
    or alongside it — otherwise /resume would show both recaps at once.
    """
    import inspect

    src = inspect.getsource(HermesCLI.run)
    resumed_branch, _, rest = src.partition("if self._resumed:")
    assert resumed_branch != src, "run() no longer branches on self._resumed"
    else_branch = rest.split("else:", 1)[1]
    # Only look at the immediate else: block, not the rest of run().
    else_block = else_branch.split("\n\n", 1)[0]
    assert "_show_previous_session_recap()" in else_block
    assert "_display_resumed_history()" not in else_block
