"""Tests for the /rewind slash command — pick a previous user message and re-prompt.

These tests exercise ``HermesCLI._handle_rewind_command`` against a real
:class:`SessionDB`, with the curses picker patched to return a known index.
"""

import os
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def session_db(tmp_path):
    """Create a real SessionDB for testing."""
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    os.makedirs(tmp_path / ".hermes", exist_ok=True)
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / ".hermes" / "test_rewind.db")
    yield db
    db.close()


@pytest.fixture
def cli_instance(tmp_path, session_db):
    """Minimal HermesCLI-like MagicMock for testing _handle_rewind_command."""
    cli = MagicMock()
    cli._session_db = session_db
    cli.session_id = "20260510_120000_rewind"
    cli.model = "anthropic/claude-sonnet-4.6"
    cli.session_start = datetime.now()
    cli._app = None  # No prompt_toolkit app — exercise the print-fallback path.
    cli.agent = None

    # Seed: 3 user turns w/ assistant responses.
    session_db.create_session(
        session_id=cli.session_id,
        source="cli",
        model=cli.model,
    )
    session_db.append_message(cli.session_id, role="user", content="first thing")
    session_db.append_message(cli.session_id, role="assistant", content="reply 1")
    session_db.append_message(cli.session_id, role="user", content="second thing")
    session_db.append_message(cli.session_id, role="assistant", content="reply 2")
    session_db.append_message(cli.session_id, role="user", content="third thing")
    session_db.append_message(cli.session_id, role="assistant", content="reply 3")

    cli.conversation_history = session_db.get_messages_as_conversation(cli.session_id)
    return cli


class TestRewindHandler:
    """Test _handle_rewind_command CLI handler behaviour."""

    def test_rewind_truncates_history(self, cli_instance, session_db):
        """Picking a user message rewinds the DB and reloads conversation_history."""
        from cli import HermesCLI

        # list_recent_user_messages returns newest-first → index 0 is "third",
        # index 1 is "second", index 2 is "first". Pick "second".
        cli_instance._run_curses_picker = MagicMock(return_value=1)
        HermesCLI._handle_rewind_command(cli_instance, "/rewind")

        contents = [m["content"] for m in cli_instance.conversation_history]
        assert contents == ["first thing", "reply 1"]

    def test_rewind_cancel_returns_silently(self, cli_instance, session_db):
        """Picker returning None (cancel) leaves history untouched."""
        from cli import HermesCLI
        before = list(cli_instance.conversation_history)

        cli_instance._run_curses_picker = MagicMock(return_value=None)
        HermesCLI._handle_rewind_command(cli_instance, "/rewind")

        assert cli_instance.conversation_history == before
        # No rewind row flipped in DB either
        all_rows = session_db.get_messages(cli_instance.session_id, include_inactive=True)
        active_rows = session_db.get_messages(cli_instance.session_id)
        assert len(all_rows) == len(active_rows)

    def test_rewind_handler_no_session_db(self, cli_instance):
        """Without a SessionDB, the handler bails gracefully and leaves history intact."""
        from cli import HermesCLI
        cli_instance._session_db = None
        before = list(cli_instance.conversation_history)
        cli_instance._run_curses_picker = MagicMock()

        # Should not raise, should not call the picker either.
        HermesCLI._handle_rewind_command(cli_instance, "/rewind")
        assert cli_instance._run_curses_picker.call_count == 0
        assert cli_instance.conversation_history == before

    def test_rewind_handler_empty_history(self, cli_instance):
        """Empty conversation_history bails before touching the DB."""
        from cli import HermesCLI
        cli_instance.conversation_history = []
        cli_instance._run_curses_picker = MagicMock()

        HermesCLI._handle_rewind_command(cli_instance, "/rewind")
        assert cli_instance._run_curses_picker.call_count == 0

    def test_memory_provider_notified_on_rewind(self, cli_instance, session_db):
        """Memory manager hook fires with rewound=True after a successful rewind."""
        from cli import HermesCLI

        mm = MagicMock()
        agent = MagicMock()
        agent._memory_manager = mm
        # Mirror the hasattr checks in the handler — make these accessor-friendly.
        agent._last_flushed_db_idx = 0
        cli_instance.agent = agent
        cli_instance._run_curses_picker = MagicMock(return_value=0)

        HermesCLI._handle_rewind_command(cli_instance, "/rewind")

        assert mm.on_session_switch.call_count == 1
        args, kwargs = mm.on_session_switch.call_args
        assert args[0] == cli_instance.session_id
        assert kwargs["rewound"] is True
        assert kwargs["reset"] is False
        # parent_session_id is empty for a same-session rewind
        assert kwargs["parent_session_id"] == ""

    def test_rewind_no_user_messages(self, tmp_path, session_db):
        """Session with only system/assistant messages → 'no user messages' bail."""
        from cli import HermesCLI

        cli = MagicMock()
        cli._session_db = session_db
        cli.session_id = "20260510_120001_empty"
        cli._app = None
        cli.agent = None
        session_db.create_session(session_id=cli.session_id, source="cli")
        session_db.append_message(cli.session_id, role="assistant", content="greetings")
        cli.conversation_history = session_db.get_messages_as_conversation(cli.session_id)
        cli._run_curses_picker = MagicMock()

        HermesCLI._handle_rewind_command(cli, "/rewind")
        assert cli._run_curses_picker.call_count == 0  # bailed before opening picker

    def test_rewind_prefills_prompt_buffer_when_app_present(
        self, cli_instance, session_db
    ):
        """When ``self._app`` is wired, the chosen message lands in current_buffer."""
        from cli import HermesCLI

        buf = SimpleNamespace(text="", cursor_position=0)
        cli_instance._app = SimpleNamespace(current_buffer=buf)
        cli_instance._run_curses_picker = MagicMock(return_value=1)

        HermesCLI._handle_rewind_command(cli_instance, "/rewind")

        assert buf.text == "second thing"
        assert buf.cursor_position == len("second thing")

    def test_rewind_invokes_agent_state_reset(self, cli_instance, session_db):
        """Successful rewind calls reset_session_state and updates _last_flushed_db_idx."""
        from cli import HermesCLI

        agent = MagicMock()
        agent._last_flushed_db_idx = 999
        cli_instance.agent = agent
        cli_instance._run_curses_picker = MagicMock(return_value=0)

        HermesCLI._handle_rewind_command(cli_instance, "/rewind")

        assert agent.reset_session_state.called
        assert agent._invalidate_system_prompt.called
        # After rewinding to the newest user message (index 0), only that single
        # user message is left active, so _last_flushed_db_idx should reflect 0
        # (rewind soft-deletes the target message itself per spec).
        assert agent._last_flushed_db_idx == len(cli_instance.conversation_history)


class TestRewindCommandDef:
    """The CommandDef registration for /rewind."""

    def test_rewind_in_registry(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        names = [c.name for c in COMMAND_REGISTRY]
        assert "rewind" in names

    def test_rewind_in_session_category(self):
        from hermes_cli.commands import COMMAND_REGISTRY
        rewind = next(c for c in COMMAND_REGISTRY if c.name == "rewind")
        assert rewind.category == "Session"

    def test_rewind_resolves(self):
        from hermes_cli.commands import resolve_command
        result = resolve_command("rewind")
        assert result is not None
        assert result.name == "rewind"
