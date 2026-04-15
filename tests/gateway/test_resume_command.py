"""Tests for /resume gateway slash command.

Tests the _handle_resume_command handler (switch to a previously-named session)
across gateway messenger platforms.
"""

from unittest.mock import MagicMock, AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, build_session_key


def _make_event(text="/resume", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _session_key_for_event(event):
    """Get the session key that build_session_key produces for an event."""
    return build_session_key(event.source)


def _make_runner(session_db=None, current_session_id="current_session_001",
                 event=None):
    """Create a bare GatewayRunner with a mock session_store and optional session_db."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = session_db
    runner._running_agents = {}

    # Compute the real session key if an event is provided
    session_key = build_session_key(event.source) if event else "agent:main:telegram:dm"

    # Mock session_store that returns a session entry with a known session_id
    mock_session_entry = MagicMock()
    mock_session_entry.session_id = current_session_id
    mock_session_entry.session_key = session_key
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    mock_store.load_transcript.return_value = []
    mock_store.switch_session.return_value = mock_session_entry
    runner.session_store = mock_store

    # Stub out memory flushing
    runner._async_flush_memories = AsyncMock()

    return runner


# ---------------------------------------------------------------------------
# _handle_resume_command
# ---------------------------------------------------------------------------


class TestHandleResumeCommand:
    """Tests for GatewayRunner._handle_resume_command."""

    @pytest.mark.asyncio
    async def test_no_session_db(self):
        """Returns error when session database is unavailable."""
        runner = _make_runner(session_db=None)
        event = _make_event(text="/resume My Project")
        result = await runner._handle_resume_command(event)
        assert "not available" in result.lower()

    @pytest.mark.asyncio
    async def test_list_sessions_when_no_arg_uses_plaintext_fallback(self, tmp_path):
        """With no argument, fallback output is deterministic plaintext."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.create_session("sess_002", "telegram")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)

        assert result == (
            "Resume Sessions:\n"
            "1. sess_002 | Coding\n"
            "2. sess_001 | Research\n"
            "Use: /resume <session name or id>"
        )
        db.close()

    @pytest.mark.asyncio
    async def test_telegram_no_arg_uses_inline_menu_when_available(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.create_session("sess_002", "telegram")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        tg_adapter = MagicMock()
        tg_adapter.send_resume_menu = AsyncMock(return_value=MagicMock(success=True, message_id="99"))
        runner.adapters[Platform.TELEGRAM] = tg_adapter

        result = await runner._handle_resume_command(event)

        assert result == ""
        tg_adapter.send_resume_menu.assert_called_once()
        kwargs = tg_adapter.send_resume_menu.call_args.kwargs
        assert kwargs["chat_id"] == "67890"
        assert kwargs["session_key"] == "agent:main:telegram:dm:67890"
        assert [s["title"] for s in kwargs["sessions"]] == ["Coding", "Research"]
        db.close()

    @pytest.mark.asyncio
    async def test_telegram_no_arg_keeps_compressed_root_as_menu_anchor(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "telegram")
        db.set_session_title("sess_root", "Research")
        db.append_message("sess_root", "user", "old part")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "telegram", parent_session_id="sess_root")
        db.append_message("sess_child", "user", "latest part")

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        tg_adapter = MagicMock()
        tg_adapter.send_resume_menu = AsyncMock(return_value=MagicMock(success=True, message_id="99"))
        runner.adapters[Platform.TELEGRAM] = tg_adapter

        result = await runner._handle_resume_command(event)

        assert result == ""
        kwargs = tg_adapter.send_resume_menu.call_args.kwargs
        assert kwargs["sessions"][0]["id"] == "sess_root"
        assert kwargs["sessions"][0]["title"] == "Research"
        db.close()

    @pytest.mark.asyncio
    async def test_list_includes_untitled_sessions_with_preview_or_id_fallback(self, tmp_path):
        """With no arg, untitled root sessions are still listed deterministically."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_preview", "discord")
        db.append_message("sess_preview", "user", "latest untitled session preview")
        db.create_session("sess_empty", "discord")

        event = _make_event(text="/resume", platform=Platform.DISCORD)
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)

        assert result == (
            "Resume Sessions:\n"
            "1. sess_empty\n"
            "2. sess_preview | latest untitled session preview\n"
            "Use: /resume <session name or id>"
        )
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_name(self, tmp_path):
        """Resolves a title and switches to that session."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        assert "My Project" in result
        # Verify switch_session was called with the old session ID
        runner.session_store.switch_session.assert_called_once()
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "old_session_abc"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_nonexistent_name(self, tmp_path):
        """Returns error for unknown session name."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume Nonexistent Session")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "No session found" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_already_on_session(self, tmp_path):
        """Returns friendly message when already on the requested session."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("current_session_001", "telegram")
        db.set_session_title("current_session_001", "Active Project")

        event = _make_event(text="/resume Active Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)
        assert "Already on session" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_auto_lineage(self, tmp_path):
        """Asking for 'My Project' when 'My Project #2' exists gets the latest."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_v1", "telegram")
        db.set_session_title("sess_v1", "My Project")
        db.create_session("sess_v2", "telegram")
        db.set_session_title("sess_v2", "My Project #2")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        # Should resolve to #2 (latest in lineage)
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_v2"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_title_uses_anchor_session_without_auto_follow(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "telegram")
        db.set_session_title("sess_root", "My Project")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "telegram", parent_session_id="sess_root")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume My Project")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_root"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_session_id_uses_anchor_session_without_auto_follow(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "telegram")
        db.set_session_title("sess_root", "My Project")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "telegram", parent_session_id="sess_root")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume sess_root")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_root"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_title_with_last_follows_latest_compression_child(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "telegram")
        db.set_session_title("sess_root", "My Project")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "telegram", parent_session_id="sess_root")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume My Project --last")
        runner = _make_runner(session_db=db, current_session_id="current_session_001", event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_child"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_session_id_with_last_follows_latest_compression_child(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "telegram")
        db.set_session_title("sess_root", "My Project")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "telegram", parent_session_id="sess_root")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume sess_root --last")
        runner = _make_runner(session_db=db, current_session_id="current_session_001", event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_child"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_title_with_list_on_telegram_opens_chain_menu(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "telegram")
        db.set_session_title("sess_root", "My Project")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "telegram", parent_session_id="sess_root")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume My Project --list")
        runner = _make_runner(session_db=db, current_session_id="current_session_001", event=event)
        tg_adapter = MagicMock()
        tg_adapter.send_resume_menu = AsyncMock(return_value=MagicMock(success=True, message_id="99"))
        runner.adapters[Platform.TELEGRAM] = tg_adapter

        result = await runner._handle_resume_command(event)

        assert result == ""
        kwargs = tg_adapter.send_resume_menu.call_args.kwargs
        assert [s["id"] for s in kwargs["sessions"]] == ["sess_root", "sess_child"]
        assert kwargs["metadata"]["resume_menu_mode"] == "chain"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_title_with_list_uses_plaintext_fallback_on_non_telegram(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_root", "discord")
        db.set_session_title("sess_root", "My Project")
        db.end_session("sess_root", "compression")
        db.create_session("sess_child", "discord", parent_session_id="sess_root")
        db.set_session_title("sess_child", "My Project #2")
        db.create_session("current_session_001", "discord")

        event = _make_event(text="/resume sess_root --list", platform=Platform.DISCORD)
        runner = _make_runner(session_db=db, current_session_id="current_session_001", event=event)
        result = await runner._handle_resume_command(event)

        assert result == (
            "Resume Points:\n"
            "1. sess_root | My Project\n"
            "2. sess_child | My Project #2\n"
            "Use: /resume <session id>"
        )
        db.close()

    @pytest.mark.asyncio
    async def test_resume_clears_running_agent(self, tmp_path):
        """Switching sessions clears any cached running agent."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session", "telegram")
        db.set_session_title("old_session", "Old Work")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume Old Work")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        # Simulate a running agent using the real session key
        real_key = _session_key_for_event(event)
        runner._running_agents[real_key] = MagicMock()

        await runner._handle_resume_command(event)

        assert real_key not in runner._running_agents
        db.close()

    @pytest.mark.asyncio
    async def test_resume_flushes_memories(self, tmp_path):
        """Resume should flush memories from the current session before switching."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session", "telegram")
        db.set_session_title("old_session", "Old Work")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume Old Work")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )

        await runner._handle_resume_command(event)

        runner._async_flush_memories.assert_called_once_with(
            "current_session_001",
            "agent:main:telegram:dm:67890",
        )
        db.close()
