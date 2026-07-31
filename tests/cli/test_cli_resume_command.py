import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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


class TestResumeArmingRealDb:
    """Real-DB resume arming: bare-number selection after `/resume list
    <page>` must resolve against the real session store — no MagicMock
    session-listing fakes (review F7).

    Regression coverage: with offset > 0 the old bounds check compared a
    global rank against the page length, so every valid selection on
    page 2+ was rejected.
    """

    @pytest.fixture
    def cli_db(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        base = 1_700_000_000.0
        for i in range(15):
            sid = f"sess_{i:03d}"
            db.create_session(sid, "cli")
            db.set_session_title(sid, f"Session {i:02d}")
            db.append_message(sid, "user", f"opener {i}", timestamp=base + i * 60.0)
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (base + i * 60.0, sid),
            )
        db._conn.commit()
        cli_obj = HermesCLI.__new__(HermesCLI)
        cli_obj.session_id = "current_session"
        cli_obj._resumed = False
        cli_obj._pending_title = None
        cli_obj.conversation_history = []
        cli_obj.agent = None
        cli_obj._session_db = db
        cli_obj._pending_resume_sessions = None
        cli_obj.resume_display = "minimal"
        yield cli_obj, db
        db.close()

    def test_bare_number_after_list_page_two_resolves(self, cli_db):
        cli_obj, db = cli_db
        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint"),
        ):
            cli_obj._handle_resume_command("/resume list 2")
            assert cli_obj._pending_resume_offset == 10
            # Page 2 of 15 sessions = ranks 11..15.
            assert len(cli_obj._pending_resume_sessions) == 5
            consumed = cli_obj._consume_pending_resume_selection("12")
        assert consumed is True
        # 12th most recent session in the canonical list (sess_014 is #1).
        assert cli_obj.session_id == "sess_003"

    def test_bare_number_below_page_two_window_out_of_range(self, cli_db):
        cli_obj, db = cli_db
        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint") as mock_cprint,
        ):
            cli_obj._handle_resume_command("/resume list 2")
            consumed = cli_obj._consume_pending_resume_selection("5")
        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert consumed is True
        assert "out of range" in printed.lower()
        assert cli_obj.session_id == "current_session"

    def test_bare_number_after_page_one_still_resolves(self, cli_db):
        cli_obj, db = cli_db
        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint"),
        ):
            cli_obj._handle_resume_command("/resume")
            consumed = cli_obj._consume_pending_resume_selection("2")
        assert consumed is True
        assert cli_obj.session_id == "sess_013"

    def test_sessions_list_page_two_delegates_and_pages(self, cli_db):
        """`/sessions list 2` (with a page) falls through to `/resume
        list 2` — the dead page-parse block never ran — and still pages
        to ranks 11..15 with a working bare-number selection (F8).
        """
        cli_obj, db = cli_db
        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint"),
        ):
            cli_obj._handle_sessions_command("/sessions list 2")
            assert cli_obj._pending_resume_offset == 10
            assert len(cli_obj._pending_resume_sessions) == 5
            consumed = cli_obj._consume_pending_resume_selection("14")
        assert consumed is True
        assert cli_obj.session_id == "sess_001"

    def test_invalid_limit_warns_and_uses_default(self, cli_db):
        """`--limit abc` must warn instead of failing silently, keep the
        default page size, and still consume the flag+value tokens (F12)."""
        cli_obj, db = cli_db
        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint") as mock_cprint,
        ):
            cli_obj._handle_resume_command("/resume --limit abc")
        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "Invalid --limit" in printed
        assert "abc" in printed
        # Flag consumed; the empty target armed the default page-1 listing.
        assert cli_obj._pending_resume_limit == 10
        assert len(cli_obj._pending_resume_sessions) == 10


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
        assert "Use /resume <number>" in output
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

    def test_pending_out_of_range_consumed_with_message(self):
        cli_obj = _make_cli()
        cli_obj._pending_resume_sessions = [{"id": "sess_002", "title": "Coding"}]

        with patch("cli._cprint") as mock_cprint:
            consumed = cli_obj._consume_pending_resume_selection("9")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        # An out-of-range number is still consumed (not sent to the agent),
        # and the prompt is disarmed.
        assert consumed is True
        assert "out of range" in printed.lower()
        assert cli_obj.session_id == "current_session"
        assert cli_obj._pending_resume_sessions is None

    def test_pending_number_below_paginated_window_is_out_of_range(self):
        """A bare number not on the displayed window (e.g. `5` while the
        prompt shows ranks 11..20) is rejected instead of silently resolving
        to a different window's row. The bounds check is offset-aware so
        page-2 selections keep working when pagination is active."""
        cli_obj = _make_cli()
        sessions = [
            {"id": f"sess_{11 + i}", "title": f"Session {11 + i}"} for i in range(10)
        ]
        cli_obj._pending_resume_sessions = sessions
        cli_obj._pending_resume_offset = 10

        with patch("cli._cprint") as mock_cprint:
            consumed = cli_obj._consume_pending_resume_selection("5")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert consumed is True
        assert "out of range" in printed.lower()
        assert cli_obj.session_id == "current_session"

    def test_pending_number_above_paginated_window_is_out_of_range(self):
        """The prompt shows ranks 11..20; `21` is not on screen and must fail."""
        cli_obj = _make_cli()
        sessions = [
            {"id": f"sess_{11 + i}", "title": f"Session {11 + i}"} for i in range(10)
        ]
        cli_obj._pending_resume_sessions = sessions
        cli_obj._pending_resume_offset = 10

        with patch("cli._cprint") as mock_cprint:
            consumed = cli_obj._consume_pending_resume_selection("21")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert consumed is True
        assert "out of range" in printed.lower()
        assert cli_obj.session_id == "current_session"

    def test_pending_non_numeric_falls_through_and_disarms(self):
        cli_obj = _make_cli()
        cli_obj._pending_resume_sessions = [{"id": "sess_002", "title": "Coding"}]

        with patch("cli._cprint"):
            consumed = cli_obj._consume_pending_resume_selection("hello there")

        # Free text is NOT consumed (caller treats it as chat), but the
        # one-shot prompt is disarmed so a later number isn't hijacked.
        assert consumed is False
        assert cli_obj._pending_resume_sessions is None

    def test_no_pending_returns_false(self):
        cli_obj = _make_cli()
        assert cli_obj._pending_resume_sessions is None
        assert cli_obj._consume_pending_resume_selection("3") is False

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
