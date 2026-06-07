"""Tests for /resume gateway slash command.

Tests the _handle_resume_command handler (switch to a previously-named session)
across gateway messenger platforms.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


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

    return runner


def _make_dispatch_runner(session_db=None, current_session_id="current_session_001"):
    """Create a GatewayRunner ready to dispatch slash commands via _handle_message."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(
        emit=AsyncMock(),
        emit_collect=AsyncMock(return_value=[]),
        loaded_hooks=False,
    )

    source = SessionSource(
        platform=Platform.TELEGRAM,
        user_id="12345",
        chat_id="67890",
        user_name="testuser",
        chat_type="dm",
    )
    session_entry = SessionEntry(
        session_key=build_session_key(source),
        session_id=current_session_id,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
    )
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner.session_store.append_to_transcript = MagicMock()
    runner.session_store.rewrite_transcript = MagicMock()
    runner.session_store.update_session = MagicMock()
    runner._session_db = session_db
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._agent_cache = {}
    runner._agent_cache_lock = None
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._busy_input_mode = "interrupt"
    runner._draining = False
    runner._restart_requested = False
    runner._shutdown_event = MagicMock()
    runner._running_agents_generation = {}
    runner._session_model_overrides = {}
    runner._failed_platforms = {}
    runner._exit_cleanly = False
    runner._exit_reason = None
    runner._active_sessions = {}
    runner._session_tasks = {}
    runner._pending_restart_inputs = {}
    runner._pending_drain_events = {}
    runner._status_verbosity = "normal"
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    runner._release_running_agent_state = MagicMock()
    runner._clear_session_boundary_security_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner._queue_or_replace_pending_event = MagicMock()
    runner._agent_has_active_subagents = MagicMock(return_value=False)
    runner._check_slash_access = MagicMock(return_value=None)
    runner._handle_message_with_agent = AsyncMock(
        side_effect=AssertionError("/sessions must not fall through to the agent")
    )
    runner._run_agent = AsyncMock(
        side_effect=AssertionError("/sessions must not call _run_agent")
    )

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
    async def test_list_named_sessions_when_no_arg(self, tmp_path):
        """With no argument, lists recently titled sessions."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.create_session("sess_002", "telegram")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "Research" in result
        assert "Coding" in result
        assert "Named Sessions" in result
        assert "1." in result
        assert "2." in result
        assert "/resume 1" in result
        db.close()

    @pytest.mark.asyncio
    async def test_list_shows_usage_when_no_titled(self, tmp_path):
        """With no arg and no titled sessions, shows instructions."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")  # No title

        event = _make_event(text="/resume")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_resume_command(event)
        assert "No named sessions" in result
        assert "/title" in result
        db.close()

    @pytest.mark.asyncio
    async def test_resume_by_index(self, tmp_path):
        """Numeric argument resumes the indexed titled session from the list."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.create_session("sess_002", "telegram")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume 2")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "Resumed" in result
        runner.session_store.switch_session.assert_called_once()
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "sess_001"
        db.close()

    @pytest.mark.asyncio
    async def test_resume_index_out_of_range(self, tmp_path):
        """Out-of-range numeric arguments show a helpful error."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.set_session_title("sess_001", "Research")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume 9")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        result = await runner._handle_resume_command(event)

        assert "out of range" in result.lower()
        assert "/resume" in result
        runner.session_store.switch_session.assert_not_called()
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
    async def test_resume_follows_compression_continuation(self, tmp_path):
        """Gateway /resume should reopen the live descendant after compression."""
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("compressed_root", "telegram")
        db.set_session_title("compressed_root", "Compressed Work")
        db.end_session("compressed_root", "compression")
        db.create_session("compressed_child", "telegram", parent_session_id="compressed_root")
        db.append_message("compressed_child", "user", "hello from continuation")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume Compressed Work")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )
        runner.session_store.load_transcript.side_effect = (
            lambda session_id: [{"role": "user", "content": "hello from continuation"}]
            if session_id == "compressed_child"
            else []
        )

        result = await runner._handle_resume_command(event)

        assert "Resumed session" in result
        assert "(1 message)" in result
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "compressed_child"
        runner.session_store.load_transcript.assert_called_with("compressed_child")
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
    async def test_resume_evicts_cached_agent(self, tmp_path):
        """Gateway /resume evicts the cached AIAgent so the next message
        rebuilds with the correct session_id end-to-end — mirrors /branch
        and /reset. Without this, the cached agent's memory provider keeps
        writing into the wrong session. See #6672.
        """
        import threading
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session", "telegram")
        db.set_session_title("old_session", "Old Work")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume Old Work")
        runner = _make_runner(session_db=db, current_session_id="current_session_001",
                              event=event)
        # Seed the cache with a fake agent
        real_key = _session_key_for_event(event)
        runner._agent_cache = {real_key: (MagicMock(), object())}
        runner._agent_cache_lock = threading.RLock()

        await runner._handle_resume_command(event)

        assert real_key not in runner._agent_cache
        db.close()

    @pytest.mark.asyncio
    async def test_resume_strips_outer_brackets(self, tmp_path):
        """Users may copy `<session_id>` from the usage hint literally.

        The gateway should strip outer ``<>``, ``[]``, ``""``, and ``''``
        before lookup so ``/resume <abc123>`` works the same as
        ``/resume abc123``.
        """
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("abc123", "telegram")
        db.set_session_title("abc123", "Bracketed")
        db.create_session("current_session_001", "telegram")

        for raw in ("<abc123>", "[abc123]", '"abc123"', "'abc123'"):
            event = _make_event(text=f"/resume {raw}")
            runner = _make_runner(
                session_db=db,
                current_session_id="current_session_001",
                event=event,
            )
            result = await runner._handle_resume_command(event)
            # Either the session was resumed (and we get a "Resumed" / "Already on" reply)
            # or it was found-then-redirected. Failure mode = "No session found matching '<abc123>'".
            assert "abc123" not in str(result) or "not found" not in str(result).lower(), (
                f"bracket stripping failed for {raw!r}: gateway returned {result!r}"
            )
        db.close()

    @pytest.mark.asyncio
    async def test_resume_resolves_by_session_id(self, tmp_path):
        """The gateway should accept a bare session ID, not just a title.

        Before this fix, /resume in the gateway only called
        ``resolve_session_by_title``, so ``/resume <session_id>`` always
        returned "Session not found" even for valid IDs.
        """
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("unnamed_session_xyz", "telegram")
        # Deliberately no title set — this session can ONLY be resolved by ID.
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/resume unnamed_session_xyz")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )
        result = await runner._handle_resume_command(event)

        # Should NOT be the not-found error.
        assert "not found" not in str(result).lower(), (
            f"session-id lookup failed: {result!r}"
        )
        db.close()


class TestHandleSessionsCommand:
    """Tests for native gateway /sessions support."""

    @pytest.mark.asyncio
    async def test_no_args_lists_recent_sessions(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.create_session("sess_002", "telegram")
        db.set_session_title("sess_001", "Research")
        db.set_session_title("sess_002", "Coding")

        event = _make_event(text="/sessions")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "Named Sessions" in result
        assert "Research" in result
        assert "Coding" in result
        assert "/resume 1" in result
        db.close()

    @pytest.mark.asyncio
    async def test_list_alias_uses_recent_sessions_view(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.set_session_title("sess_001", "Research")

        event = _make_event(text="/sessions list")
        runner = _make_runner(session_db=db, event=event)
        result = await runner._handle_sessions_command(event)

        assert "Named Sessions" in result
        assert "Research" in result
        db.close()

    @pytest.mark.asyncio
    async def test_target_delegates_to_resume_behavior(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("old_session_abc", "telegram")
        db.set_session_title("old_session_abc", "My Project")
        db.create_session("current_session_001", "telegram")

        event = _make_event(text="/sessions My Project")
        runner = _make_runner(
            session_db=db,
            current_session_id="current_session_001",
            event=event,
        )
        result = await runner._handle_sessions_command(event)

        assert "Resumed" in result
        runner.session_store.switch_session.assert_called_once()
        call_args = runner.session_store.switch_session.call_args
        assert call_args[0][1] == "old_session_abc"
        db.close()

    @pytest.mark.asyncio
    async def test_dispatch_handles_sessions_natively(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("sess_001", "telegram")
        db.set_session_title("sess_001", "Research")

        runner = _make_dispatch_runner(session_db=db)
        result = await runner._handle_message(_make_event("/sessions"))

        assert result is not None
        assert "Named Sessions" in result
        assert "Research" in result
        runner._run_agent.assert_not_called()
        runner._handle_message_with_agent.assert_not_called()
        db.close()
