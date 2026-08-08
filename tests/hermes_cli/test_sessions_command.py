"""Behavioral tests for interactive `/sessions` session management."""

from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "current-session"
    cli.session_title = None
    cli._pending_resume_sessions = None
    cli._session_db = MagicMock()
    cli._session_db.resolve_session_id.side_effect = lambda target: {
        "archived": "archived-session",
        "current": "current-session",
    }.get(target)
    return cli


def _run(cli, command):
    with (
        patch("cli._cprint"),
        patch("cli._DIM", "", create=True),
        patch("cli._RST", "", create=True),
    ):
        cli._handle_sessions_command(command)


class TestSessionsList:
    def test_list_arms_displayed_twentieth_session(self):
        cli = _make_cli()
        rows = [
            {
                "id": f"session-{index}",
                "title": f"Session {index}",
                "preview": "",
                "last_active": None,
            }
            for index in range(1, 21)
        ]
        cli._list_recent_sessions = MagicMock(return_value=rows)

        with patch("hermes_cli.main._relative_time", return_value="now"):
            _run(cli, "/sessions")

        cli._list_recent_sessions.assert_called_once_with(limit=20)
        assert cli._pending_resume_sessions == rows
        assert cli._resolve_sessions_target("20") == "session-20"


class TestSessionsDelete:
    def test_delete_strips_confirmation_flag_before_resolving_target(self):
        cli = _make_cli()
        cli._confirm_destructive_slash = MagicMock(return_value=True)
        cli._session_db.get_session.return_value = {"title": "Archived"}
        cli._session_db.delete_session.return_value = True

        with patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            _run(cli, "/sessions delete archived --yes")

        cli._session_db.resolve_session_id.assert_called_once_with("archived")
        cli._session_db.delete_session.assert_called_once()

    def test_delete_refuses_current_session(self):
        cli = _make_cli()

        _run(cli, "/sessions delete current")

        cli._session_db.delete_session.assert_not_called()


class TestSessionsRename:
    def test_rename_current_session_updates_in_memory_title(self):
        cli = _make_cli()

        _run(cli, "/sessions rename current Renamed session")

        cli._session_db.set_session_title.assert_called_once_with(
            "current-session", "Renamed session"
        )
        assert cli.session_title == "Renamed session"


class TestSessionsPrune:
    def test_prune_parses_days_before_inline_confirmation(self):
        cli = _make_cli()
        cli._confirm_destructive_slash = MagicMock(return_value=True)
        cli._session_db.prune_sessions.return_value = 3

        with patch("hermes_constants.get_hermes_home", return_value=MagicMock()):
            _run(cli, "/sessions prune --days 30 --yes")

        assert cli._session_db.prune_sessions.call_args.kwargs["older_than_days"] == 30
        assert cli._confirm_destructive_slash.call_args.kwargs["cmd_original"].endswith(
            "--yes"
        )

    def test_prune_rejects_negative_days_without_confirmation(self):
        cli = _make_cli()
        cli._confirm_destructive_slash = MagicMock(return_value=True)

        _run(cli, "/sessions prune --days -1")

        cli._confirm_destructive_slash.assert_not_called()
        cli._session_db.prune_sessions.assert_not_called()

    def test_prune_rejects_unknown_argument_grammar(self):
        cli = _make_cli()
        cli._confirm_destructive_slash = MagicMock(return_value=True)

        _run(cli, "/sessions prune 30")

        cli._confirm_destructive_slash.assert_not_called()
        cli._session_db.prune_sessions.assert_not_called()
