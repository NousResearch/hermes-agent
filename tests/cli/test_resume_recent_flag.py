"""Tests for --recent N flag on resume.

When ``--recent N`` is passed alongside ``--resume <id>``, only the last N
messages are loaded from the session DB instead of the full transcript.
This is useful for quick check-ins on long sessions without paying the
full token cost.
"""

import pytest
from unittest.mock import MagicMock, patch


def _make_cli():
    from cli import HermesCLI

    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "current_session"
    cli_obj._resumed = False
    cli_obj._pending_title = None
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._session_db = MagicMock()
    cli_obj._pending_resume_sessions = None
    cli_obj.resume_display = "minimal"
    cli_obj._recent_limit = None
    return cli_obj


class TestGetMessagesAsConversationLimit:
    """Unit tests for the limit parameter on get_messages_as_conversation."""

    def test_limit_returns_only_last_n_messages(self, tmp_path):
        """When limit=N, only the last N rows should be returned."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "test.db")
        db.create_session(session_id="sess_test", source="cli")
        for i in range(10):
            db._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, id, active) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                ("sess_test", "user" if i % 2 == 0 else "assistant", f"msg_{i}", i, i),
            )
        db._conn.commit()

        # Full load
        full = db.get_messages_as_conversation("sess_test")
        assert len(full) == 10

        # Limited load — last 3
        limited = db.get_messages_as_conversation("sess_test", limit=3)
        assert len(limited) == 3
        # Should be the last 3 by timestamp+id order
        assert limited[0]["content"] == "msg_7"
        assert limited[1]["content"] == "msg_8"
        assert limited[2]["content"] == "msg_9"

        # Limit larger than available — returns all
        over = db.get_messages_as_conversation("sess_test", limit=100)
        assert len(over) == 10

        # Limit of None — returns all (no limit requested)
        no_limit = db.get_messages_as_conversation("sess_test", limit=None)
        assert len(no_limit) == 10

        # Limit of 0 — returns all (guard skips non-positive values)
        zero_limit = db.get_messages_as_conversation("sess_test", limit=0)
        assert len(zero_limit) == 10

        db.close()

    def test_limit_preserves_message_structure(self, tmp_path):
        """Limited messages should have the same structure as full messages."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "test2.db")
        db.create_session(session_id="sess_test", source="cli")

        for i in range(5):
            db._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp, id, active) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                ("sess_test", "user", f"hello_{i}", i, i),
            )
        db._conn.commit()

        limited = db.get_messages_as_conversation("sess_test", limit=2)
        assert len(limited) == 2
        assert limited[0]["role"] == "user"
        assert limited[0]["content"] == "hello_3"
        assert limited[1]["content"] == "hello_4"

        db.close()


class TestRecentFlagInResumeCommand:
    """Tests for --recent flag integration in the /resume command path."""

    def test_resume_with_recent_limit_passes_limit_to_db(self):
        """When _recent_limit is set, /resume <id> passes limit to
        get_messages_as_conversation."""
        cli_obj = _make_cli()
        cli_obj._recent_limit = 5
        cli_obj._session_db.get_session.return_value = {
            "id": "sess_002",
            "title": "Coding",
        }
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_002"
        cli_obj._session_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="sess_002"),
            patch("cli._cprint"),
            patch("cli._sync_process_session_id"),
        ):
            cli_obj._handle_resume_command("/resume sess_002")

        # Verify limit was passed
        call_args = cli_obj._session_db.get_messages_as_conversation.call_args
        assert call_args.kwargs.get("limit") == 5
        assert cli_obj.session_id == "sess_002"

    def test_resume_without_recent_passes_none_limit(self):
        """When _recent_limit is not set, limit=None is passed (full load)."""
        cli_obj = _make_cli()
        cli_obj._recent_limit = None
        cli_obj._session_db.get_session.return_value = {
            "id": "sess_002",
            "title": "Coding",
        }
        cli_obj._session_db.resolve_resume_session_id.return_value = "sess_002"
        cli_obj._session_db.get_messages_as_conversation.return_value = [
            {"role": "user", "content": "hello"},
        ]

        with (
            patch("hermes_cli.main._resolve_session_by_name_or_id", return_value="sess_002"),
            patch("cli._cprint"),
            patch("cli._sync_process_session_id"),
        ):
            cli_obj._handle_resume_command("/resume sess_002")

        call_args = cli_obj._session_db.get_messages_as_conversation.call_args
        assert call_args.kwargs.get("limit") is None


class TestParserRecentFlag:
    """Tests that --recent is parsed correctly by the argument parser."""

    def test_top_level_parser_accepts_recent(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _, _ = build_top_level_parser()
        args = parser.parse_args(["--resume", "sess_123", "--recent", "10"])
        assert args.recent == 10
        assert args.resume == "sess_123"

    def test_top_level_parser_recent_defaults_to_none(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _, _ = build_top_level_parser()
        args = parser.parse_args(["--resume", "sess_123"])
        assert args.recent is None

    def test_chat_parser_accepts_recent(self):
        from hermes_cli._parser import build_top_level_parser

        parser, _, chat_parser = build_top_level_parser()
        args = parser.parse_args(["chat", "--resume", "sess_123", "--recent", "20"])
        assert getattr(args, "recent", None) == 20
        assert args.resume == "sess_123"

    def test_parser_rejects_zero_recent(self):
        """--recent 0 should be rejected at parse time."""
        import argparse
        from hermes_cli._parser import build_top_level_parser

        parser, _, _ = build_top_level_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--resume", "sess_123", "--recent", "0"])

    def test_parser_rejects_negative_recent(self):
        """--recent -5 should be rejected at parse time."""
        from hermes_cli._parser import build_top_level_parser

        parser, _, _ = build_top_level_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--resume", "sess_123", "--recent", "-5"])