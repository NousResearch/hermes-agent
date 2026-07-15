"""Tests for _handle_sessions_command and _resolve_sessions_target in CLICommandsMixin."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_mixin():
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    obj = object.__new__(CLICommandsMixin)

    obj.session_id = "current-session-id"
    obj.session_title = None
    obj._pending_resume_sessions = None

    db = MagicMock()
    db.get_session.return_value = {"id": "abc123", "title": "My Session"}
    db.resolve_session_id.return_value = "abc123"
    db.set_session_title.return_value = True
    db.delete_session.return_value = True
    db.prune_sessions.return_value = 5
    obj._session_db = db

    return obj


def _run(obj, cmd, printed=None):
    lines = []

    def fake_cprint(msg=""):
        lines.append(str(msg))

    def fake_dim():
        return ""

    def fake_rst():
        return ""

    with (
        patch("hermes_cli.cli_commands_mixin._cprint", side_effect=fake_cprint),
        patch("cli._cprint", side_effect=fake_cprint, create=True),
        patch("cli._DIM", "", create=True),
        patch("cli._RST", "", create=True),
    ):
        import hermes_cli.cli_commands_mixin as mod
        orig = getattr(mod, "_cprint", None)
        mod._cprint = fake_cprint
        try:
            obj._handle_sessions_command(cmd)
        finally:
            if orig is not None:
                mod._cprint = orig
            else:
                del mod._cprint

    return lines


class TestHandleSessionsCommandList:
    def test_bare_sessions_shows_list(self):
        obj = _make_mixin()
        sessions = [
            {"id": "abc123", "title": "My Session", "preview": "hello world", "last_active": None},
            {"id": "def456", "title": "Other Session", "preview": "foo bar", "last_active": None},
        ]
        obj._list_recent_sessions = MagicMock(return_value=sessions)
        obj._show_recent_sessions = MagicMock()

        with patch("hermes_cli.cli_commands_mixin._relative_time", return_value="2d ago", create=True):
            with patch("hermes_cli.main._relative_time", return_value="2d ago"):
                with patch("cli._cprint", create=True), patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
                    obj._handle_sessions_command("/sessions")

        assert obj._pending_resume_sessions == sessions

    def test_sessions_list_subcommand(self):
        obj = _make_mixin()
        sessions = [{"id": "abc123", "title": "T", "preview": "p", "last_active": None}]
        obj._list_recent_sessions = MagicMock(return_value=sessions)

        with patch("hermes_cli.main._relative_time", return_value="1h ago"):
            with patch("cli._cprint", create=True), patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
                obj._handle_sessions_command("/sessions list")

        assert obj._pending_resume_sessions == sessions

    def test_no_sessions_prints_message(self):
        obj = _make_mixin()
        obj._list_recent_sessions = MagicMock(return_value=[])
        lines = []

        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions")

        assert any("No sessions" in l for l in lines)
        assert obj._pending_resume_sessions is None

    def test_no_session_db(self):
        obj = _make_mixin()
        obj._session_db = None
        lines = []

        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True), \
             patch("hermes_state.format_session_db_unavailable", return_value="DB unavailable", create=True):
            obj._handle_sessions_command("/sessions")

        assert any("unavailable" in l.lower() or "DB" in l for l in lines)


class TestHandleSessionsCommandDelete:
    def test_delete_by_id(self):
        obj = _make_mixin()
        obj._list_recent_sessions = MagicMock(return_value=[])
        obj._confirm_destructive_slash = MagicMock(return_value=True)

        with patch("cli._cprint", create=True), patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock(__truediv__=lambda s, x: MagicMock()), create=True):
            obj._handle_sessions_command("/sessions delete abc123")

        obj._session_db.delete_session.assert_called_once()

    def test_delete_current_session_blocked(self):
        obj = _make_mixin()
        obj._list_recent_sessions = MagicMock(return_value=[])
        obj._session_db.resolve_session_id.return_value = "current-session-id"
        lines = []

        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions delete current-session-id")

        assert any("Cannot delete" in l for l in lines)
        obj._session_db.delete_session.assert_not_called()

    def test_delete_missing_target(self):
        lines = []
        obj = _make_mixin()
        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions delete")
        assert any("Usage" in l for l in lines)


class TestHandleSessionsCommandRename:
    def test_rename_by_id(self):
        obj = _make_mixin()
        obj._list_recent_sessions = MagicMock(return_value=[])
        lines = []

        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions rename abc123 New Title")

        obj._session_db.set_session_title.assert_called_once_with("abc123", "New Title")
        assert any("Renamed" in l for l in lines)

    def test_rename_current_session_updates_local_title(self):
        obj = _make_mixin()
        obj._session_db.resolve_session_id.return_value = "current-session-id"
        obj._list_recent_sessions = MagicMock(return_value=[])

        with patch("cli._cprint", create=True), patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions rename current-session-id Updated")

        assert obj.session_title == "Updated"

    def test_rename_missing_title(self):
        lines = []
        obj = _make_mixin()
        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions rename abc123")
        assert any("Usage" in l for l in lines)


class TestHandleSessionsCommandPrune:
    def test_prune_default_days(self):
        obj = _make_mixin()
        obj._confirm_destructive_slash = MagicMock(return_value=True)
        lines = []

        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock(__truediv__=lambda s, x: MagicMock()), create=True):
            obj._handle_sessions_command("/sessions prune")

        obj._session_db.prune_sessions.assert_called_once()
        call_kwargs = obj._session_db.prune_sessions.call_args[1]
        assert call_kwargs.get("older_than_days", 90) == 90
        assert any("5" in l for l in lines)

    def test_prune_custom_days(self):
        obj = _make_mixin()
        obj._confirm_destructive_slash = MagicMock(return_value=True)

        with patch("cli._cprint", create=True), patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock(__truediv__=lambda s, x: MagicMock()), create=True):
            obj._handle_sessions_command("/sessions prune --days 30")

        call_kwargs = obj._session_db.prune_sessions.call_args[1]
        assert call_kwargs.get("older_than_days") == 30

    def test_prune_days_with_case_insensitive_yes_parses_days_before_confirmation(self):
        obj = _make_mixin()
        obj._confirm_destructive_slash = MagicMock(return_value=True)

        with patch("cli._cprint", create=True), patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True), \
             patch("hermes_constants.get_hermes_home", return_value=MagicMock(__truediv__=lambda s, x: MagicMock()), create=True):
            obj._handle_sessions_command("/sessions prune --days 30 --YES")

        assert obj._session_db.prune_sessions.call_args[1]["older_than_days"] == 30
        assert obj._confirm_destructive_slash.call_args.kwargs["cmd_original"].endswith("--YES")

    def test_prune_rejects_negative_days(self):
        obj = _make_mixin()
        lines = []

        with patch("cli._cprint", side_effect=lambda m="": lines.append(m), create=True), \
             patch("cli._DIM", "", create=True), patch("cli._RST", "", create=True):
            obj._handle_sessions_command("/sessions prune --days -1")

        assert any("non-negative" in line for line in lines)
        obj._session_db.prune_sessions.assert_not_called()


class TestResolveSessionsTarget:
    def test_resolve_by_number(self):
        obj = _make_mixin()
        sessions = [
            {"id": "first-id"},
            {"id": "second-id"},
        ]
        obj._list_recent_sessions = MagicMock(return_value=sessions)

        result = obj._resolve_sessions_target("2")
        assert result == "second-id"

    def test_resolve_number_out_of_range(self):
        obj = _make_mixin()
        obj._list_recent_sessions = MagicMock(return_value=[{"id": "only"}])
        result = obj._resolve_sessions_target("99")
        assert result is None

    def test_resolve_by_session_id(self):
        obj = _make_mixin()
        obj._session_db.resolve_session_id.return_value = "abc123"
        result = obj._resolve_sessions_target("abc123")
        assert result == "abc123"

    def test_resolve_falls_back_to_name_lookup(self):
        obj = _make_mixin()
        obj._session_db.resolve_session_id.return_value = None

        with patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="found-id"):
            result = obj._resolve_sessions_target("My Session Title")

        assert result == "found-id"
