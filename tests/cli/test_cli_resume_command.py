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

    def test_show_history_uses_prompt_toolkit_safe_print(self):
        cli_obj = _make_cli()
        cli_obj.conversation_history = [{"role": "user", "content": "Hello"}]

        running_app = SimpleNamespace(_is_running=True)
        with (
            patch("prompt_toolkit.application.get_app_or_none", return_value=running_app),
            patch("cli._cprint") as mock_cprint,
        ):
            cli_obj.show_history()

        printed = "\n".join(call.args[0] for call in mock_cprint.call_args_list)
        assert "Conversation History" in printed
        assert "Hello" in printed

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

    def test_handle_resume_strips_outer_brackets(self):
        """Users copy `<session_id>` from the usage hint literally.

        Strip outer ``<>``, ``[]``, ``""``, and ``''`` before lookup so
        ``/resume <abc123>`` works the same as ``/resume abc123``.
        """
        cli_obj = _make_cli()
        cli_obj._session_db.get_session.return_value = {"id": "sess_alpha", "title": "Alpha"}
        cli_obj._session_db.get_messages_as_conversation.return_value = []
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_alpha"

        for raw in ("<sess_alpha>", "[sess_alpha]", '"sess_alpha"', "'sess_alpha'"):
            cli_obj.session_id = "current_session"
            with (
                patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="sess_alpha"),
                patch("cli._cprint"),
            ):
                cli_obj._handle_resume_command(f"/resume {raw}")
            assert cli_obj.session_id == "sess_alpha", (
                f"bracket-stripping failed for {raw!r}: session_id stayed {cli_obj.session_id}"
            )

    def test_handle_resume_does_not_strip_partial_brackets(self):
        """Mismatched or single brackets must pass through unmodified.

        ``"<half`` (just an open angle) is not a wrapping pair, so the
        lookup should treat it verbatim — preserving the existing
        not-found error path instead of mangling the input.
        """
        cli_obj = _make_cli()
        cli_obj._session_db.get_session.return_value = None

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
            patch("cli._cprint") as mock_cprint,
        ):
            cli_obj._handle_resume_command("/resume <half")

        printed = " ".join(str(call) for call in mock_cprint.call_args_list)
        assert "<half" in printed


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

    def test_bare_resume_no_sessions_does_not_arm(self):
        cli_obj = _make_cli()
        cli_obj._show_recent_sessions = MagicMock(return_value=False)
        cli_obj._list_recent_sessions = MagicMock(return_value=[])

        with patch("cli._cprint"):
            cli_obj._handle_resume_command("/resume")

        assert cli_obj._pending_resume_sessions is None

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
        cli_obj._session_db.get_messages_as_conversation.return_value = [
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


class TestRestoreSessionCwdMarkup:
    """Regression: _restore_session_cwd must not crash with Rich MarkupError.

    Lines that used ``[{_DIM}]`` inside Rich markup triggered
    ``rich.errors.MarkupError: closing tag [/] at position N has nothing to
    close`` because ``_DIM`` is an ANSI escape (``\\x1b[2;3m``), not a valid
    Rich tag.  The fix replaces ``[{_DIM}]`` with Rich's native ``[dim]`` tag.
    See: https://github.com/NousResearch/hermes-agent/issues/39469
    """

    def test_missing_dir_does_not_raise_markup_error(self):
        """Session cwd gone → dim warning, no MarkupError."""
        cli_obj = _make_cli()
        console = MagicMock()
        cli_obj._output_console = MagicMock(return_value=console)

        # Use a path that definitely does not exist.
        cli_obj._restore_session_cwd({"cwd": "/nonexistent/path/to/nowhere"})

        # Should have printed a warning via console.print, not crashed.
        assert console.print.called
        printed = str(console.print.call_args)
        assert "Working directory is gone" in printed or "gone" in printed.lower()

    def test_chdir_failure_does_not_raise_markup_error(self, tmp_path):
        """os.chdir fails → dim warning, no MarkupError."""
        import os
        cli_obj = _make_cli()
        console = MagicMock()
        cli_obj._output_console = MagicMock(return_value=console)

        # Create a directory, then make it unreadable (simulate chdir failure).
        target = tmp_path / "locked"
        target.mkdir()

        # Patch os.chdir to raise OSError for our target path.
        original_chdir = os.chdir
        def fake_chdir(path):
            if str(path) == str(target):
                raise OSError("Permission denied")
            return original_chdir(path)

        with patch("os.chdir", side_effect=fake_chdir):
            cli_obj._restore_session_cwd({"cwd": str(target)})

        assert console.print.called
        printed = str(console.print.call_args)
        assert "Could not enter" in printed or "permission" in printed.lower()

    def test_success_path_does_not_raise_markup_error(self, tmp_path):
        """Successful cwd switch → dim info, no MarkupError."""
        import os
        cli_obj = _make_cli()
        console = MagicMock()
        cli_obj._output_console = MagicMock(return_value=console)

        original_cwd = os.getcwd()
        try:
            cli_obj._restore_session_cwd({"cwd": str(tmp_path)})
            assert console.print.called
            printed = str(console.print.call_args)
            assert "Working directory" in printed or "working" in printed.lower()
        finally:
            os.chdir(original_cwd)


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
        cli_obj._session_db.get_messages_as_conversation.return_value = []
        cli_obj._session_db.resolve_resume_session_id.return_value = "target"

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="target"),
            patch("cli._cprint"),
        ):
            cli_obj._handle_resume_command("/resume target")

        agent._flush_messages_to_session_db.assert_called_once_with(
            [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        )
        cli_obj._session_db.end_session.assert_called_once()


class TestListRecentSessionsSourceScope:
    """_list_recent_sessions must surface sessions from every source, not just
    'cli'. A profile used only via the desktop app stores sessions with
    source='desktop'; the old source='cli' filter hid them from /sessions and
    /resume (reported as 'No previous sessions yet'). See fix in cli.py
    _list_recent_sessions."""

    def test_list_recent_sessions_queries_all_sources(self):
        from unittest.mock import patch

        from hermes_cli.session_listing import query_session_listing

        cli_obj = _make_cli()
        captured = {}

        def fake_query(db, *, source, current_session_id, include_all_sources,
                       include_unnamed, limit, exclude_sources):
            captured.update(
                source=source, include_all_sources=include_all_sources,
                exclude_sources=exclude_sources,
            )
            return [{"id": "desktop_sess", "source": "desktop", "title": "Coding"}]

        with patch(
            "hermes_cli.session_listing.query_session_listing", side_effect=fake_query
        ):
            rows = cli_obj._list_recent_sessions(limit=10)

        # The real method must now query across ALL sources (desktop included).
        assert captured["source"] is None
        assert captured["include_all_sources"] is True
        assert captured["exclude_sources"] == ["tool"]
        assert rows and rows[0]["source"] == "desktop"

        # And document that the *old* filter would have dropped it.
        old_rows = query_session_listing(
            cli_obj._session_db, source="cli",
            current_session_id=cli_obj.session_id, include_all_sources=False,
            include_unnamed=True, limit=10, exclude_sources=["tool"],
        )
        assert old_rows == []
