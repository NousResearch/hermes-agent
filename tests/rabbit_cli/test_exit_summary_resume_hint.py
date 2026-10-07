"""Tests for the CLI exit summary's resume hint, including profile-flag support."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from cli import RabbitCLI
from rabbit_cli.main_tui_launch import _print_tui_exit_summary

import rabbit_cli.main  # noqa: F401 — _print_tui_exit_summary imports it lazily; load it at collection time


def _make_cli(session_id="20260524_000001_abc123"):
    cli_obj = RabbitCLI.__new__(RabbitCLI)
    cli_obj.session_id = session_id
    # _print_exit_summary requires a populated conversation history (msg_count > 0)
    # to print the resume hint at all. One synthetic user turn is enough.
    cli_obj.conversation_history = [{"role": "user", "content": "hi"}]
    cli_obj.agent = None
    cli_obj._session_db = None
    cli_obj.session_start = datetime.now()
    return cli_obj


class TestExitSummaryResumeHint:
    """The exit-line ``Resume this session with:`` hint must include the
    active profile (`-p <name>`) so session IDs round-trip across
    profile boundaries — sessions live under `~/.rabbit-profiles/<profile>/`,
    so a hint copied without `-p` from a non-default profile won't find
    the session.
    """

    def test_resume_hint_no_profile_flag_on_default(self, capsys):
        cli_obj = _make_cli()
        with patch("rabbit_cli.profiles.get_active_profile_name", return_value="default"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        # No `-p` for the default profile.
        assert "rabbit --resume 20260524_000001_abc123" in out
        assert " -p " not in out

    def test_resume_hint_no_profile_flag_on_custom(self, capsys):
        cli_obj = _make_cli()
        with patch("rabbit_cli.profiles.get_active_profile_name", return_value="custom"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        # "custom" is the standard RABBIT_HOME indicator — no -p needed.
        assert "rabbit --resume 20260524_000001_abc123" in out
        assert " -p " not in out

    def test_resume_hint_includes_profile_flag_for_named_profile(self, capsys):
        cli_obj = _make_cli()
        with patch("rabbit_cli.profiles.get_active_profile_name", return_value="dev"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        assert "rabbit --resume 20260524_000001_abc123 -p dev" in out

    def test_resume_hint_includes_profile_flag_on_title_hint_too(self, capsys, tmp_path):
        """When a session title is available, the `rabbit -c "title"` hint
        must also include the `-p` flag for non-default profiles.
        """
        cli_obj = _make_cli()
        fake_db = MagicMock()
        fake_db.get_session_title.return_value = "My Cool Session"
        cli_obj._session_db = fake_db

        with patch("rabbit_cli.profiles.get_active_profile_name", return_value="dev"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        assert 'rabbit -c "My Cool Session" -p dev' in out
        assert "rabbit --resume 20260524_000001_abc123 -p dev" in out

    def test_resume_hint_falls_back_when_profile_lookup_fails(self, capsys):
        """If `get_active_profile_name` raises (e.g. profiles module
        missing during ``rabbit update`` mid-flight), fall back to no
        flag rather than crashing the exit summary.
        """
        cli_obj = _make_cli()
        with patch(
            "rabbit_cli.profiles.get_active_profile_name",
            side_effect=RuntimeError("profiles unavailable"),
        ):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        # Resume hint still printed without -p.
        assert "rabbit --resume 20260524_000001_abc123" in out
        assert " -p " not in out


def _tui_session_db(*_args, **_kwargs):
    db = MagicMock()
    db.get_session.return_value = {"message_count": 3}
    db.get_session_title.return_value = "My TUI Session"
    return db


class TestTuiExitSummaryResumeHint:
    """``_print_tui_exit_summary`` (rabbit_cli/main_tui_launch.py) is a separate entry
    point from the classic CLI summary above and must carry the same ``-p`` flag (#125078)."""

    def test_tui_hints_include_profile_flag_for_named_profile(self, capsys):
        with patch("rabbit_state.SessionDB", _tui_session_db), patch(
            "rabbit_cli.profiles.get_active_profile_name", return_value="dev"
        ):
            _print_tui_exit_summary("20260524_000001_abc123")
        out = capsys.readouterr().out
        assert "rabbit --tui --resume 20260524_000001_abc123 -p dev" in out
        assert 'rabbit --tui -c "My TUI Session" -p dev' in out

    def test_tui_hints_no_profile_flag_on_default(self, capsys):
        with patch("rabbit_state.SessionDB", _tui_session_db), patch(
            "rabbit_cli.profiles.get_active_profile_name", return_value="default"
        ):
            _print_tui_exit_summary("20260524_000001_abc123")
        out = capsys.readouterr().out
        assert "rabbit --tui --resume 20260524_000001_abc123" in out
        assert " -p " not in out
