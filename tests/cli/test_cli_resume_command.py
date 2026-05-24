from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj._resumed = False
    cli_obj._pending_title = None
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._session_db = MagicMock()
    # _handle_resume_command now triggers _display_resumed_history (#31695),
    # which reads self.resume_display. "minimal" short-circuits the recap so
    # the test only exercises session-switch behavior.
    cli_obj.resume_display = "minimal"
    return cli_obj


class TestCliResumeCommand:
    def test_show_recent_sessions_includes_indexes_and_resume_hint(self, capsys):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_002", "title": "Coding", "preview": "build feature", "last_active": None},
            {"id": "sess_001", "title": "Research", "preview": "read docs", "last_active": None},
        ])

        shown = cli_obj._show_recent_sessions(reason="resume")
        output = capsys.readouterr().out

        assert shown is True
        assert "1" in output
        assert "2" in output
        assert "Coding" in output
        assert "Research" in output
        assert "/resume 2" in output
        assert "/resume <session title>" in output

    def test_handle_resume_by_index_switches_to_numbered_session(self):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_002", "title": "Coding"},
            {"id": "sess_001", "title": "Research"},
        ])
        cli_obj._session_db.get_session.return_value = {"id": "sess_001", "title": "Research"}
        cli_obj._session_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        # resolve_resume_session_id passes the id through when no compression chain.
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_001"

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint") as mock_cprint,
        ):
            cli_obj._handle_resume_command("/resume 2")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert cli_obj.session_id == "sess_001"
        assert "Resumed session sess_001" in printed
        assert "Research" in printed

    def test_handle_resume_by_index_out_of_range(self):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_002", "title": "Coding"},
        ])

        with patch("cli._cprint") as mock_cprint:
            cli_obj._handle_resume_command("/resume 9")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "out of range" in printed.lower()
        assert "/resume" in printed
        assert cli_obj.session_id == "current_session"
