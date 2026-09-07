import os
from types import SimpleNamespace
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
    cli_obj._pending_resume_sessions = None
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

    def test_show_recent_sessions_uses_prompt_toolkit_safe_print(self):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_002", "title": "Coding", "preview": "build feature", "last_active": None},
        ])

        running_app = SimpleNamespace(_is_running=True)
        with (
            patch("prompt_toolkit.application.get_app_or_none", return_value=running_app),
            patch("cli._cprint") as mock_cprint,
        ):
            shown = cli_obj._show_recent_sessions(reason="sessions")

        assert shown is True
        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Recent sessions" in printed
        assert "Coding" in printed


    def test_handle_resume_by_index_switches_to_numbered_session(self):
        cli_obj = _make_cli()
        cli_obj._list_recent_sessions = MagicMock(return_value=[
            {"id": "sess_002", "title": "Coding"},
            {"id": "sess_001", "title": "Research"},
        ])
        cli_obj._session_db.get_session.return_value = {"id": "sess_001", "title": "Research"}
        cli_obj._session_db.get_resume_conversations.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ], [
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




class TestCliResumeRestoresCwd:
    """Mid-chat /resume must retarget the working directory to where the
    session was started — the same contract as a startup ``hermes -c`` /
    ``--resume``.

    Regression coverage for #38562: ``_restore_session_cwd()`` was wired into
    the startup resume paths but not into ``_handle_resume_command()``, so an
    interactive ``/resume`` (and ``/sessions <id>``, which delegates here) left
    the process + ``TERMINAL_CWD`` pointing at whatever directory the user had
    cd'd into — so the terminal/code-exec tools and relative paths ran in the
    wrong repo.
    """

    def _resumable_cli(self, session_meta):
        cli_obj = _make_cli()
        cli_obj._session_db.get_session.return_value = session_meta
        cli_obj._session_db.get_resume_conversations.return_value = [
            {"role": "user", "content": "hello"},
        ], [
            {"role": "user", "content": "hello"},
        ]
        cli_obj._session_db.resolve_resume_session_id.return_value = session_meta["id"]
        return cli_obj

    def test_handle_resume_restores_recorded_cwd(self, tmp_path):
        recorded = str(tmp_path)
        cli_obj = self._resumable_cli({"id": "sess_dir", "title": "Dir", "cwd": recorded})

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="sess_dir"),
            patch("cli._cprint"),
            patch.object(cli_obj, "_console_print"),
            patch("os.chdir") as mock_chdir,
            patch.dict(os.environ, {}, clear=False),
        ):
            cli_obj._handle_resume_command("/resume Dir")
            # Assert inside the patch.dict scope — it restores os.environ on exit.
            assert os.environ.get("TERMINAL_CWD") == recorded

        mock_chdir.assert_called_once_with(recorded)


    def test_sessions_command_restores_recorded_cwd(self, tmp_path):
        # /sessions <id> delegates to the resume flow, so it restores cwd too.
        recorded = str(tmp_path)
        cli_obj = self._resumable_cli({"id": "sess_dir", "title": "Dir", "cwd": recorded})

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="sess_dir"),
            patch("cli._cprint"),
            patch.object(cli_obj, "_console_print"),
            patch("os.chdir") as mock_chdir,
            patch.dict(os.environ, {}, clear=False),
        ):
            cli_obj._handle_sessions_command("/sessions Dir")
            # Assert inside the patch.dict scope — it restores os.environ on exit.
            assert os.environ.get("TERMINAL_CWD") == recorded

        mock_chdir.assert_called_once_with(recorded)


class TestPendingResumeNumberedSelection:
    """Bare `/resume` arms a one-shot prompt so the next bare number resumes.

    Regression coverage for #34584: previously, running `/resume` (no args)
    printed the recent-sessions list but left no selection state armed, so
    typing just `3` on the next line was sent to the agent as chat instead of
    resuming session #3.
    """

    def test_bare_resume_arms_pending_selection(self):
        cli_obj = _make_cli()
        sessions = [
            {"id": "sess_002", "title": "Coding"},
            {"id": "sess_001", "title": "Research"},
        ]
        cli_obj._list_recent_sessions = MagicMock(return_value=sessions)
        cli_obj._show_recent_sessions = MagicMock(return_value=True)

        with patch("cli._cprint"):
            cli_obj._handle_resume_command("/resume")

        assert cli_obj._pending_resume_sessions == sessions


    def test_pending_number_resumes_selected_session(self):
        cli_obj = _make_cli()
        sessions = [
            {"id": "sess_002", "title": "Coding"},
            {"id": "sess_001", "title": "Research"},
        ]
        cli_obj._pending_resume_sessions = sessions
        # _handle_resume_command("/resume 2") re-resolves the index via
        # _list_recent_sessions, so it must return the same list.
        cli_obj._list_recent_sessions = MagicMock(return_value=sessions)
        cli_obj._session_db.get_session.return_value = {"id": "sess_001", "title": "Research"}
        cli_obj._session_db.get_resume_conversations.return_value = [
            {"role": "user", "content": "hello"},
        ], [
            {"role": "user", "content": "hello"},
        ]
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_001"

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint"),
        ):
            consumed = cli_obj._consume_pending_resume_selection("2")

        assert consumed is True
        assert cli_obj.session_id == "sess_001"
        # One-shot: prompt is disarmed after consuming.
        assert cli_obj._pending_resume_sessions is None




    def test_pending_disarmed_by_other_command(self):
        cli_obj = _make_cli()
        cli_obj._pending_resume_sessions = [{"id": "sess_002", "title": "Coding"}]
        # Stub out the help handler so process_command("/help") is cheap.
        cli_obj.show_help = MagicMock()

        cli_obj.process_command("/help")

        # A non-resume command disarms the one-shot prompt (#34584).
        assert cli_obj._pending_resume_sessions is None




class TestResumeFlushesBeforeEndSession:
    """Regression for #47202: /resume must flush un-persisted messages to
    the session DB before ending the old session, just like /new and
    compress_context() already do."""

    def test_resume_flushes_when_agent_present(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        agent = MagicMock()
        cli_obj.agent = agent

        cli_obj._session_db.get_session.return_value = {"id": "target", "title": "T"}
        cli_obj._session_db.get_resume_conversations.return_value = ([], [])
        cli_obj._session_db.resolve_resume_session_id.return_value = "target"

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="target"),
            patch("cli._cprint"),
        ):
            cli_obj._handle_resume_command("/resume target")

        agent._flush_messages_to_session_db.assert_called_once_with(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            conversation_history=[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        )
        cli_obj._session_db.end_session.assert_called_once()
