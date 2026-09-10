"""Tests for /retitle gateway slash command.

Tests the ``_handle_retitle_command`` handler that regenerates a session's
title from the last N conversation turns. Mirrors the fixture patterns used
in ``test_title_command.py``.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _make_event(text="/retitle", platform=Platform.TELEGRAM,
                user_id="12345", chat_id="67890"):
    """Build a MessageEvent for testing."""
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
    )
    return MessageEvent(text=text, source=source)


def _make_runner(session_db=None):
    """Create a bare GatewayRunner with a mock session_store and optional session_db."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    if session_db is not None:
        from hermes_state import AsyncSessionDB
        session_db = AsyncSessionDB(session_db)
    runner._session_db = session_db

    mock_session_entry = MagicMock()
    mock_session_entry.session_id = "test_session_123"
    mock_session_entry.session_key = "telegram:12345:67890"
    mock_store = MagicMock()
    mock_store.get_or_create_session.return_value = mock_session_entry
    runner.session_store = mock_store

    return runner


def _default_cfg(**overrides):
    """Default retitle config used by the handler."""
    cfg = {
        "enabled": True,
        "turns_window": 10,
        "touch_platform_names": False,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# _handle_retitle_command
# ---------------------------------------------------------------------------


class TestHandleRetitleCommand:
    """Tests for GatewayRunner._handle_retitle_command."""

    @pytest.mark.asyncio
    async def test_returns_done_message_on_success(self, tmp_path):
        """Successful retitle returns the done string with the new title."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.set_session_title("test_session_123", "Old Title")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        with patch("agent.title_generator._retitle_config", return_value=_default_cfg()), \
             patch("agent.title_generator.retitle_session", return_value="Debugging Postgres pool"):
            # Populate history so `if not history` doesn't short-circuit.
            db.append_message("test_session_123", "user", "hey")
            result = await runner._handle_retitle_command(event)

        assert "Debugging Postgres pool" in result
        assert "Retitled" in result
        db.close()

    @pytest.mark.asyncio
    async def test_returns_no_change_when_retitle_returns_none(self, tmp_path):
        """When retitle_session returns None (and title source is not 'user'), reply is 'no_change'."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.append_message("test_session_123", "user", "hey")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        with patch("agent.title_generator._retitle_config", return_value=_default_cfg()), \
             patch("agent.title_generator.retitle_session", return_value=None):
            result = await runner._handle_retitle_command(event)

        assert "couldn't" in result.lower() or "skipped" in result.lower()
        db.close()

    @pytest.mark.asyncio
    async def test_returns_user_held_when_title_is_user_set(self, tmp_path):
        """When the title is user-set and retitle returns None, reply asks to pass 'force'."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.set_session_title("test_session_123", "My Manual Title")
        # Force provenance to 'user' — the retitle worker's guard checks this.
        db.set_session_title_source("test_session_123", "user")
        db.append_message("test_session_123", "user", "hey")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        with patch("agent.title_generator._retitle_config", return_value=_default_cfg()), \
             patch("agent.title_generator.retitle_session", return_value=None):
            result = await runner._handle_retitle_command(event)

        assert "manual title" in result.lower()
        assert "force" in result.lower()
        db.close()

    @pytest.mark.asyncio
    async def test_passes_force_when_arg_is_force(self, tmp_path):
        """`/retitle force` passes force=True through to retitle_session."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.append_message("test_session_123", "user", "hey")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle force")

        with patch("agent.title_generator._retitle_config", return_value=_default_cfg()), \
             patch("agent.title_generator.retitle_session", return_value="New Title") as mock_retitle:
            await runner._handle_retitle_command(event)

        assert mock_retitle.called
        _, kwargs = mock_retitle.call_args
        assert kwargs.get("force") is True
        db.close()

    @pytest.mark.asyncio
    async def test_reads_turns_window_from_config(self, tmp_path):
        """Config.turns_window drives both get_messages(limit=) and retitle_session(turns_window=)."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        for i in range(30):
            db.append_message("test_session_123", "user", f"msg-{i}")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        cfg = _default_cfg(turns_window=15)
        with patch("agent.title_generator._retitle_config", return_value=cfg), \
             patch("agent.title_generator.retitle_session", return_value="X") as mock_retitle, \
             patch.object(db, "get_messages", wraps=db.get_messages) as mock_get:
            await runner._handle_retitle_command(event)

        # get_messages should be called with limit=turns_window*2 == 30
        _, kwargs = mock_get.call_args
        assert kwargs.get("limit") == 30
        assert kwargs.get("latest") is True

        _, r_kwargs = mock_retitle.call_args
        assert r_kwargs.get("turns_window") == 15
        db.close()

    @pytest.mark.asyncio
    async def test_passes_touch_platform_names_from_config(self, tmp_path):
        """Config.touch_platform_names is forwarded to retitle_session."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")
        db.append_message("test_session_123", "user", "hey")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        cfg = _default_cfg(touch_platform_names=True)
        with patch("agent.title_generator._retitle_config", return_value=cfg), \
             patch("agent.title_generator.retitle_session", return_value="X") as mock_retitle:
            await runner._handle_retitle_command(event)

        _, kwargs = mock_retitle.call_args
        assert kwargs.get("touch_platform_names") is True
        db.close()

    @pytest.mark.asyncio
    async def test_returns_disabled_when_config_disabled(self, tmp_path):
        """When retitle.enabled is False, handler refuses with the disabled string."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        with patch("agent.title_generator._retitle_config", return_value={"enabled": False}):
            result = await runner._handle_retitle_command(event)

        assert "disabled" in result.lower()
        db.close()

    @pytest.mark.asyncio
    async def test_returns_no_history_when_empty(self, tmp_path):
        """Empty session history returns the no_history string."""
        from hermes_state import SessionDB
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("test_session_123", "telegram")

        runner = _make_runner(session_db=db)
        event = _make_event(text="/retitle")

        with patch("agent.title_generator._retitle_config", return_value=_default_cfg()):
            result = await runner._handle_retitle_command(event)

        assert "no conversation history" in result.lower()
        db.close()

    @pytest.mark.asyncio
    async def test_returns_db_unavailable_when_no_session_db(self):
        """Without a session_db handle, the handler returns the db_unavailable string."""
        runner = _make_runner(session_db=None)
        event = _make_event(text="/retitle")

        result = await runner._handle_retitle_command(event)
        assert "not available" in result.lower() or "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# /retitle in the command registry
# ---------------------------------------------------------------------------


class TestRetitleInRegistry:
    """Verify /retitle is registered and dispatchable."""

    def test_retitle_in_command_registry(self):
        """The /retitle command is registered with busy_policy='dispatch'."""
        from hermes_cli.commands import resolve_command
        cmd = resolve_command("retitle")
        assert cmd is not None
        assert cmd.name == "retitle"
        assert cmd.busy_policy == "dispatch"
        assert cmd.category == "Session"
