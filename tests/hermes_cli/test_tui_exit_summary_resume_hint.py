"""TUI exit summary resume-hint tests: profile-flag parity with the classic CLI path.

The TUI launcher's ``_print_tui_exit_summary`` is a separate entry point from
``cli_session_mixin.py::_print_exit_summary`` (made profile-aware in #30444);
these tests pin the same behavior for the TUI sibling (#125078).
"""

from unittest.mock import MagicMock, patch

from hermes_cli.main_tui_launch import _print_tui_exit_summary

SESSION_ID = "20260927_000001_abc123"


def _make_session_db(*_args, title: str | None = "My TUI Session", **_kwargs):
    db = MagicMock()
    session = {"message_count": 3, "input_tokens": 10, "output_tokens": 5,
               "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0}
    db.get_session.return_value = session
    db.get_session_title.return_value = title
    return db


def _no_title_session_db(**_kwargs):
    return _make_session_db(title=None)


class TestTuiExitSummaryResumeHint:
    """``hermes --tui`` exit hints must include ``-p <profile>`` for named profiles —
    sessions live under ``~/.hermes/profiles/<profile>/``, so a hint copied without
    ``-p`` from a non-default profile resolves against the default profile and fails.
    """

    def test_tui_resume_hint_no_profile_flag_on_default(self, capsys):
        with patch("hermes_state.SessionDB", _make_session_db), patch(
            "hermes_cli.profiles.get_active_profile_name", return_value="default"
        ):
            _print_tui_exit_summary(SESSION_ID)
        out = capsys.readouterr().out
        assert f"hermes --tui --resume {SESSION_ID}" in out
        assert " -p " not in out

    def test_tui_resume_hint_no_profile_flag_on_custom(self, capsys):
        with patch("hermes_state.SessionDB", _make_session_db), patch(
            "hermes_cli.profiles.get_active_profile_name", return_value="custom"
        ):
            _print_tui_exit_summary(SESSION_ID)
        out = capsys.readouterr().out
        assert f"hermes --tui --resume {SESSION_ID}" in out
        assert " -p " not in out

    def test_tui_resume_hint_includes_profile_flag_for_named_profile(self, capsys):
        with patch("hermes_state.SessionDB", _make_session_db), patch(
            "hermes_cli.profiles.get_active_profile_name", return_value="work"
        ):
            _print_tui_exit_summary(SESSION_ID)
        out = capsys.readouterr().out
        assert f"hermes --tui --resume {SESSION_ID} -p work" in out

    def test_tui_title_hint_includes_profile_flag_too(self, capsys):
        with patch("hermes_state.SessionDB", _make_session_db), patch(
            "hermes_cli.profiles.get_active_profile_name", return_value="work"
        ):
            _print_tui_exit_summary(SESSION_ID)
        out = capsys.readouterr().out
        assert 'hermes --tui -c "My TUI Session" -p work' in out
        assert f"hermes --tui --resume {SESSION_ID} -p work" in out

    def test_tui_resume_hint_falls_back_when_profile_lookup_fails(self, capsys):
        """If get_active_profile_name raises, print the hint without -p, not a crash."""
        with patch("hermes_state.SessionDB", _make_session_db), patch(
            "hermes_cli.profiles.get_active_profile_name",
            side_effect=RuntimeError("profiles unavailable"),
        ):
            _print_tui_exit_summary(SESSION_ID)
        out = capsys.readouterr().out
        assert f"hermes --tui --resume {SESSION_ID}" in out
        assert " -p " not in out

    def test_tui_no_title_line_when_session_has_no_title(self, capsys):
        with patch("hermes_state.SessionDB", _no_title_session_db), patch(
            "hermes_cli.profiles.get_active_profile_name", return_value="work"
        ):
            _print_tui_exit_summary(SESSION_ID)
        out = capsys.readouterr().out
        assert f"hermes --tui --resume {SESSION_ID} -p work" in out
        assert "--tui -c" not in out
