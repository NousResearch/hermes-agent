"""Tests for gateway session-title flows.

Tests the /title handler plus native gateway session-title propagation
for manual and auto-generated titles.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/title", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890", thread_id=None):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        thread_id=thread_id,
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_db=None):
    """Create a bare GatewayRunner with a mock session_store and optional session_db."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_db = session_db
    runner._background_tasks = set()

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
    async def test_set_title(self, tmp_path):
        """Setting a title returns confirmation."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title My Research Project")
        result = await runner._handle_title_command(event)
        assert "My Research Project" in result
        assert "✏️" in result

        # Verify in DB
        assert db.get_session_title("test_session_123") == "My Research Project"
        db.close()

    @pytest.mark.asyncio
    async def test_set_title_renames_telegram_topic_when_in_thread(self, tmp_path):
        """Telegram /title should schedule thread-title sync via the native callback path."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        event = _make_event(text="/title Indicative Topic", thread_id="470094")
        result = await runner._handle_title_command(event)

        adapter.register_post_delivery_callback.assert_called_once()
        callback = adapter.register_post_delivery_callback.call_args.args[1]
        callback()
        await asyncio.sleep(0)
        adapter.update_thread_title.assert_awaited_once_with("67890", "470094", "Indicative Topic")
        assert "Telegram topic renamed too" not in result
        assert db.get_session_title("test_session_123") == "Indicative Topic"
        db.close()

    @pytest.mark.asyncio
    async def test_set_title_renames_telegram_general_topic_when_thread_is_one(self, tmp_path):
        """Telegram General topic thread_id=1 should also use the deferred sync path."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        event = _make_event(text="/title Lobby", thread_id="1")
        result = await runner._handle_title_command(event)

        adapter.register_post_delivery_callback.assert_called_once()
        callback = adapter.register_post_delivery_callback.call_args.args[1]
        callback()
        await asyncio.sleep(0)
        adapter.update_thread_title.assert_awaited_once_with("67890", "1", "Lobby")
        assert "Telegram topic renamed too" not in result
        assert db.get_session_title("test_session_123") == "Lobby"
        db.close()

    @pytest.mark.asyncio
    async def test_set_title_skips_telegram_topic_rename_without_thread(self, tmp_path):
        """Telegram chats without a thread_id keep the existing DB-only behavior."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        event = _make_event(text="/title Plain Chat Title")
        result = await runner._handle_title_command(event)

        adapter.register_post_delivery_callback.assert_not_called()
        assert db.get_session_title("test_session_123") == "Plain Chat Title"
        db.close()


class TestGatewayAutoTitleSync:
    @pytest.mark.asyncio
    async def test_auto_title_flow_uses_same_session_title_path(self, tmp_path):
        """Gateway auto-title should persist title and sync Telegram thread title."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        source = _make_event(thread_id="470094").source

        with patch("agent.title_generator.generate_title_if_missing", return_value="Auto Topic"):
            await runner._auto_title_gateway_session(
                session_id="test_session_123",
                session_key="telegram:12345:67890",
                source=source,
                user_message="hello",
                assistant_response="hi there",
            )

        assert db.get_session_title("test_session_123") == "Auto Topic"
        adapter.update_thread_title.assert_awaited_once_with("67890", "470094", "Auto Topic")
        db.close()

    @pytest.mark.asyncio
    async def test_auto_title_skips_platform_sync_when_no_thread(self, tmp_path):
        """Gateway auto-title without a thread should remain DB-only."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        source = _make_event().source

        with patch("agent.title_generator.generate_title_if_missing", return_value="Auto Session"):
            await runner._auto_title_gateway_session(
                session_id="test_session_123",
                session_key="telegram:12345:67890",
                source=source,
                user_message="hello",
                assistant_response="hi there",
            )

        assert db.get_session_title("test_session_123") == "Auto Session"
        adapter.update_thread_title.assert_not_called()
        db.close()

    @pytest.mark.asyncio
    async def test_auto_title_skips_overwriting_existing_manual_title(self, tmp_path):
        """Gateway auto-title should not clobber a title set while generation was in flight."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.set_session_title("test_session_123", "Manual Title")

        runner = _make_runner(session_db=db)
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter
        source = _make_event(thread_id="470094").source

        with patch("agent.title_generator.generate_title_if_missing", return_value="Auto Topic"):
            await runner._auto_title_gateway_session(
                session_id="test_session_123",
                session_key="telegram:12345:67890",
                source=source,
                user_message="hello",
                assistant_response="hi there",
            )

        assert db.get_session_title("test_session_123") == "Manual Title"
        adapter.update_thread_title.assert_not_called()
        db.close()


class TestGatewayTitleHelpers:
    @pytest.mark.asyncio
    async def test_sync_session_title_to_source_returns_false_without_adapter_or_updater(self):
        runner = _make_runner()
        source = _make_event(thread_id="470094").source

        assert await runner._sync_session_title_to_source(source, "Title") is False

        runner.adapters[Platform.TELEGRAM] = MagicMock()
        assert await runner._sync_session_title_to_source(source, "Title") is False

    @pytest.mark.asyncio
    async def test_sync_session_title_to_source_handles_updater_errors(self):
        runner = _make_runner()
        source = _make_event(thread_id="470094").source
        adapter = MagicMock()
        adapter.update_thread_title = AsyncMock(side_effect=RuntimeError("boom"))
        runner.adapters[Platform.TELEGRAM] = adapter

        assert await runner._sync_session_title_to_source(source, "Title") is False

    @pytest.mark.asyncio
    async def test_schedule_session_title_sync_after_delivery_falls_back_to_adapter_dict(self):
        runner = _make_runner()
        source = _make_event(thread_id="470094").source
        adapter = MagicMock()
        adapter._post_delivery_callbacks = {}
        del adapter.register_post_delivery_callback
        adapter.update_thread_title = AsyncMock(return_value=True)
        runner.adapters[Platform.TELEGRAM] = adapter

        scheduled = runner._schedule_session_title_sync_after_delivery(
            session_key="telegram:12345:67890",
            source=source,
            title="Title",
        )

        assert scheduled is True
        callback = adapter._post_delivery_callbacks["telegram:12345:67890"]
        callback()
        await asyncio.sleep(0)
        adapter.update_thread_title.assert_awaited_once_with("67890", "470094", "Title")

    @pytest.mark.asyncio
    async def test_apply_session_title_uses_only_if_missing_path(self, tmp_path):
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        source = _make_event(thread_id="470094").source

        changed = await runner._apply_session_title(
            session_id="test_session_123",
            source=source,
            title="Initial",
            only_if_missing=True,
        )
        unchanged = await runner._apply_session_title(
            session_id="test_session_123",
            source=source,
            title="Second",
            only_if_missing=True,
        )

        assert changed is True
        assert unchanged is False
        assert db.get_session_title("test_session_123") == "Initial"
        db.close()

    @pytest.mark.asyncio
    async def test_maybe_schedule_gateway_auto_title_registers_post_delivery_callback(self):
        runner = _make_runner(session_db=MagicMock())
        source = _make_event(thread_id="470094").source
        adapter = MagicMock()
        runner.adapters[Platform.TELEGRAM] = adapter

        with patch("agent.title_generator.should_auto_title", return_value=True):
            runner._maybe_schedule_gateway_auto_title(
                session_id="test_session_123",
                session_key="telegram:12345:67890",
                source=source,
                user_message="hello",
                assistant_response="hi",
                conversation_history=[{"role": "user", "content": "hello"}],
                generation=5,
            )

        adapter.register_post_delivery_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_maybe_schedule_gateway_auto_title_launches_immediately_without_adapter_registration(self):
        runner = _make_runner(session_db=MagicMock())
        source = _make_event(thread_id="470094").source

        with patch("agent.title_generator.should_auto_title", return_value=True), \
             patch.object(runner, "_auto_title_gateway_session", AsyncMock()) as auto_title:
            runner._maybe_schedule_gateway_auto_title(
                session_id="test_session_123",
                session_key=None,
                source=source,
                user_message="hello",
                assistant_response="hi",
                conversation_history=[{"role": "user", "content": "hello"}],
            )
            await asyncio.sleep(0)

        auto_title.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_show_title_when_set(self, tmp_path):
        """Showing title when one is set returns the title."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.set_session_title("test_session_123", "Existing Title")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title")
        result = await runner._handle_title_command(event)
        assert "Existing Title" in result
        assert "📌" in result
        db.close()

    @pytest.mark.asyncio
    async def test_show_title_when_not_set(self, tmp_path):
        """Showing title when none is set returns usage hint."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title")
        result = await runner._handle_title_command(event)
        assert "No title set" in result
        assert "/title" in result
        db.close()

    @pytest.mark.asyncio
    async def test_title_conflict(self, tmp_path):
        """Setting a title already used by another session returns error."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("other_session", "telegram")
        db.set_session_title("other_session", "Taken Title")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title Taken Title")
        result = await runner._handle_title_command(event)
        assert "already in use" in result
        assert "⚠️" in result
        db.close()

    @pytest.mark.asyncio
    async def test_no_session_db(self):
        """Returns error when session database is not available."""
        runner = _make_runner(session_db=None)
        event = _make_event(text="/title My Title")
        result = await runner._handle_title_command(event)
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_title_too_long(self, tmp_path):
        """Setting a title that exceeds max length returns error."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        long_title = "A" * 150
        event = _make_event(text=f"/title {long_title}")
        result = await runner._handle_title_command(event)
        assert "too long" in result
        assert "⚠️" in result
        db.close()

    @pytest.mark.asyncio
    async def test_title_control_chars_sanitized(self, tmp_path):
        """Control characters are stripped and sanitized title is stored."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title hello\x00world")
        result = await runner._handle_title_command(event)
        assert "helloworld" in result
        assert db.get_session_title("test_session_123") == "helloworld"
        db.close()

    @pytest.mark.asyncio
    async def test_title_only_control_chars(self, tmp_path):
        """Title with only control chars returns empty error."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/title \x00\x01\x02")
        result = await runner._handle_title_command(event)
        assert "empty after cleanup" in result
        db.close()

    @pytest.mark.asyncio
    async def test_works_across_platforms(self, tmp_path):
        """The /title command works for Discord, Slack, and WhatsApp too."""
        from hermes_state import SessionDB
        for platform in [Platform.DISCORD, Platform.TELEGRAM]:
            db = SessionDB(db_path=tmp_path / f"state_{platform.value}.db")
            db.create_session("test_session_123", platform.value)

            runner = _make_runner(session_db=db)
            event = _make_event(text="/title Cross-Platform Test", platform=platform)
            result = await runner._handle_title_command(event)
            assert "Cross-Platform Test" in result
            assert db.get_session_title("test_session_123") == "Cross-Platform Test"
            db.close()


# ---------------------------------------------------------------------------
# /title in help and known_commands
# ---------------------------------------------------------------------------


class TestTitleInHelp:
    """Verify /title appears in help text and known commands."""

    @pytest.mark.asyncio
    async def test_title_in_help_output(self):
        """The /help output includes /title."""
        runner = _make_runner()
        event = _make_event(text="/help")
        # Need hooks for help command
        from gateway.hooks import HookRegistry
        runner.hooks = HookRegistry()
        result = await runner._handle_help_command(event)
        assert "/title" in result

    def test_title_is_known_command(self):
        """The /title command is in the _known_commands set."""
        from gateway.run import GatewayRunner
        import inspect
        source = inspect.getsource(GatewayRunner._handle_message)
        assert '"title"' in source
