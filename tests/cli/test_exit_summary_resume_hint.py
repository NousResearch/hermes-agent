"""Tests for the CLI exit summary's resume hint, including profile-flag support."""

from datetime import datetime
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli(session_id="20260524_000001_abc123"):
    cli_obj = HermesCLI.__new__(HermesCLI)
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
    profile boundaries — sessions live under `~/.hermes-profiles/<profile>/`,
    so a hint copied without `-p` from a non-default profile won't find
    the session.
    """

    def test_resume_hint_no_profile_flag_on_default(self, capsys):
        cli_obj = _make_cli()
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        # No `-p` for the default profile.
        assert "ht --resume 20260524_000001_abc123" in out
        assert " -p " not in out

    def test_resume_hint_no_profile_flag_on_custom(self, capsys):
        cli_obj = _make_cli()
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="custom"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        # "custom" is the standard HERMES_HOME indicator — no -p needed.
        assert "ht --resume 20260524_000001_abc123" in out
        assert " -p " not in out

    def test_resume_hint_includes_profile_flag_for_named_profile(self, capsys):
        cli_obj = _make_cli()
        with patch("hermes_cli.profiles.get_active_profile_name", return_value="dev"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        assert "ht --resume 20260524_000001_abc123 -p dev" in out

    def test_resume_hint_includes_profile_flag_on_title_hint_too(self, capsys, tmp_path):
        """When a session title is available, the `hermes -c "title"` hint
        must also include the `-p` flag for non-default profiles.
        """
        cli_obj = _make_cli()
        fake_db = MagicMock()
        fake_db.get_session_title.return_value = "My Cool Session"
        cli_obj._session_db = fake_db

        with patch("hermes_cli.profiles.get_active_profile_name", return_value="dev"):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        assert 'ht -c "My Cool Session" -p dev' in out
        assert "ht --resume 20260524_000001_abc123 -p dev" in out

    def test_resume_hint_falls_back_when_profile_lookup_fails(self, capsys):
        """If `get_active_profile_name` raises (e.g. profiles module
        missing during ``hermes update`` mid-flight), fall back to no
        flag rather than crashing the exit summary.
        """
        cli_obj = _make_cli()
        with patch(
            "hermes_cli.profiles.get_active_profile_name",
            side_effect=RuntimeError("profiles unavailable"),
        ):
            cli_obj._print_exit_summary()
        out = capsys.readouterr().out
        # Resume hint still printed without -p.
        assert "ht --resume 20260524_000001_abc123" in out
        assert " -p " not in out
