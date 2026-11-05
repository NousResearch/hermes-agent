"""Tests for /title gateway slash command.

Tests the _handle_title_command handler (set/show session titles)
across all gateway messenger platforms.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource
from tests.fixtures.storage_seed import seed_session

def _make_event(text="/title", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)

def _runner_class():
    from gateway.run import GatewayRunner

    class _Runner(GatewayRunner):
        persistence = None  # the storage interface handle the slash handlers use, set by the test

    return _Runner


def _make_runner(session_db=None):
    """Create a bare GatewayRunner with a mock session_store and an optional storage interface handle."""
    runner = object.__new__(_runner_class())
    runner.adapters = {}
    runner._voice_mode = {}
    runner.persistence = session_db

    # Mock session_store that returns a session entry with a known session_id
    mock_session_entry = MagicMock()
    mock_session_entry.session_id = "test_session_123"
    mock_session_entry.session_key = "telegram:12345:67890"
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    runner.session_store = mock_store

    return runner

# ---------------------------------------------------------------------------
# _handle_title_command
# ---------------------------------------------------------------------------

class TestHandleTitleCommand:
    """Tests for GatewayRunner._handle_title_command."""

    @pytest.mark.asyncio
    async def test_title_conflict(self, tmp_path):
        """Setting a title already used by another session returns error."""
        from persistence_assembly import open_persistence, profile_scope
        db = open_persistence(profile_scope(home=tmp_path))
        seed_session(db, "other_session", "telegram", title="Taken Title")
        seed_session(db, "test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title Taken Title")
        result = await runner._handle_title_command(event)
        assert "already in use" in result
        db.close()

    @pytest.mark.asyncio
    async def test_title_control_chars_sanitized(self, tmp_path):
        """Control characters are stripped and sanitized title is stored."""
        from persistence_assembly import open_persistence, profile_scope
        db = open_persistence(profile_scope(home=tmp_path))
        seed_session(db, "test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title hello\x00world")
        result = await runner._handle_title_command(event)
        assert "helloworld" in result
        assert db.sessions.get_title("test_session_123") == "helloworld"
        db.close()

    @pytest.mark.asyncio
    async def test_set_title_propagates_to_telegram_topic_rename(self, tmp_path):
        """/title <name> also renames the visible Telegram topic, not just the DB."""
        from persistence_assembly import open_persistence, profile_scope
        db = open_persistence(profile_scope(home=tmp_path))
        seed_session(db, "test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        runner._schedule_telegram_topic_title_rename = MagicMock()

        event = _make_event(text="/title My Topic Name")
        result = await runner._handle_title_command(event)

        assert "My Topic Name" in result
        runner._schedule_telegram_topic_title_rename.assert_called_once_with(
            event.source, "test_session_123", "My Topic Name"
        )
        db.close()

    @pytest.mark.asyncio
    async def test_show_title_does_not_rename_topic(self, tmp_path):
        """Showing the title (no arg) must not trigger a topic rename."""
        from persistence_assembly import open_persistence, profile_scope
        db = open_persistence(profile_scope(home=tmp_path))
        seed_session(db, "test_session_123", "telegram", title="Existing Title")

        runner = _make_runner(session_db=db)
        runner._schedule_telegram_topic_title_rename = MagicMock()

        event = _make_event(text="/title")
        await runner._handle_title_command(event)

        runner._schedule_telegram_topic_title_rename.assert_not_called()
        db.close()

# ---------------------------------------------------------------------------
# /title in help and known_commands
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# /new with title
# ---------------------------------------------------------------------------

class TestResetCommandWithTitle:
    """Tests for GatewayRunner._handle_reset_command with a title argument."""

    @pytest.mark.asyncio
    async def test_reset_command_duplicate_title_surfaces_warning(self):
        """/new <title> with an already-in-use title returns a warning in the reply."""
        from datetime import datetime

        from gateway.session import SessionEntry, SessionSource, build_session_key

        runner = object.__new__(_runner_class())
        runner.config = GatewayConfig(
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
        )
        adapter = MagicMock()
        adapter.send = AsyncMock()
        runner.adapters = {Platform.TELEGRAM: adapter}
        runner._voice_mode = {}
        runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
        runner._session_model_overrides = {}
        runner._pending_model_notes = {}
        runner._background_tasks = set()

        source = SessionSource(
            platform=Platform.TELEGRAM,
            user_id="12345",
            chat_id="67890",
            user_name="testuser",
        )
        session_key = build_session_key(source)
        new_session_entry = SessionEntry(
            session_key=session_key,
            session_id="sess-new",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            platform=Platform.TELEGRAM,
            chat_type="dm",
        )
        runner.session_store = MagicMock()
        runner.session_store.get_or_create_session.return_value = new_session_entry
        runner.session_store.reset_session.return_value = new_session_entry
        runner.session_store._entries = {session_key: new_session_entry}
        runner.session_store._generate_session_key.return_value = session_key
        runner._running_agents = {}
        runner._pending_messages = {}
        runner._pending_approvals = {}
        runner._session_db = AsyncMock()
        runner.persistence = MagicMock()
        runner.persistence.aio.sessions.set_title = AsyncMock(side_effect=ValueError(
            "Title 'Dup' is already in use by session abc-123"
        ))
        runner._agent_cache = {}
        runner._agent_cache_lock = None
        runner._is_user_authorized = lambda _source: True
        runner._format_session_info = lambda: ""

        event = _make_event(text="/new Dup")
        result = await runner._handle_reset_command(event)

        runner.persistence.aio.sessions.set_title.assert_called_once()
        reply = str(result)
        assert "already in use" in reply
        assert "session started untitled" in reply
        # Header must NOT claim the rejected title as the session name
        assert "New session started: Dup" not in reply

# ---------------------------------------------------------------------------
# /new in help output
# ---------------------------------------------------------------------------
