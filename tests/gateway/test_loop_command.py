"""Tests for /loop gateway command — LoopManager-based same-session continuation.

Verifies that the gateway handler correctly dispatches subcommands,
parses interval prefixes, and integrates with LoopManager.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=_make_source(),
        message_id="m1",
    )


def _make_runner():
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._background_tasks = set()
    runner._session_db = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._queued_events = {}
    return runner


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so SessionDB.state_meta writes don't clobber the real one."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import loop

    loop._DB_CACHE.clear()
    yield home
    loop._DB_CACHE.clear()


# ------------------------------------------------------------------
# Gateway /loop handler tests
# ------------------------------------------------------------------

class TestLoopCommandGateway:

    @pytest.mark.asyncio
    async def test_no_args_shows_status_when_none(self, hermes_home):
        """/loop with no args shows 'No active loop' when none set."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-1"
        runner.session_store.get_or_create_session.return_value = session_entry

        result = await runner._handle_loop_command(_make_event("/loop"))

        assert "No active loop" in result

    @pytest.mark.asyncio
    async def test_set_prompt(self, hermes_home):
        """/loop <prompt> creates a loop and kicks off immediately."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-2"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._dispatch_loop_prompt = MagicMock()
        runner._session_key_for_source = lambda _s: "key-2"

        result = await runner._handle_loop_command(_make_event("/loop check deployment"))

        assert "Loop set" in result
        assert "check deployment" in result
        assert "300s interval" in result  # default
        runner._dispatch_loop_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_with_interval(self, hermes_home):
        """/loop 5m check deployment parses interval prefix."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-3"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._dispatch_loop_prompt = MagicMock()
        runner._session_key_for_source = lambda _s: "key-3"

        result = await runner._handle_loop_command(_make_event("/loop 5m check deployment"))

        assert "Loop set" in result
        assert "300s interval" in result
        assert "check deployment" in result
        runner._dispatch_loop_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_every_syntax(self, hermes_home):
        """/loop every 30m summarize news parses 'every' prefix."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-4"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._dispatch_loop_prompt = MagicMock()
        runner._session_key_for_source = lambda _s: "key-4"

        result = await runner._handle_loop_command(_make_event("/loop every 30m summarize news"))

        assert "Loop set" in result
        assert "1800s interval" in result
        assert "summarize news" in result
        runner._dispatch_loop_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_empty_prompt_error(self, hermes_home):
        """/loop 5m with no prompt returns usage."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-5"
        runner.session_store.get_or_create_session.return_value = session_entry

        result = await runner._handle_loop_command(_make_event("/loop 5m"))

        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_status_shows_active_loop(self, hermes_home):
        """/loop status shows active loop details."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-6"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-6"

        await runner._handle_loop_command(_make_event("/loop check deployment"))
        result = await runner._handle_loop_command(_make_event("/loop status"))

        assert "active" in result.lower()
        assert "check deployment" in result

    @pytest.mark.asyncio
    async def test_pause_subcommand(self, hermes_home):
        """/loop pause pauses the active loop."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-7"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-7"

        await runner._handle_loop_command(_make_event("/loop check deployment"))
        result = await runner._handle_loop_command(_make_event("/loop pause"))

        assert "Loop paused" in result
        assert "check deployment" in result

    @pytest.mark.asyncio
    async def test_resume_subcommand(self, hermes_home):
        """/loop resume resumes a paused loop."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-8"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-8"

        await runner._handle_loop_command(_make_event("/loop check deployment"))
        await runner._handle_loop_command(_make_event("/loop pause"))
        result = await runner._handle_loop_command(_make_event("/loop resume"))

        assert "Loop resumed" in result
        assert "check deployment" in result

    @pytest.mark.asyncio
    async def test_clear_subcommand(self, hermes_home):
        """/loop clear clears the active loop."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-9"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-9"

        await runner._handle_loop_command(_make_event("/loop check deployment"))
        result = await runner._handle_loop_command(_make_event("/loop clear"))

        assert "Loop cleared" in result

    @pytest.mark.asyncio
    async def test_clear_when_no_loop(self, hermes_home):
        """/loop clear when no loop shows friendly message."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-10"
        runner.session_store.get_or_create_session.return_value = session_entry

        result = await runner._handle_loop_command(_make_event("/loop clear"))

        assert "No active loop" in result

    @pytest.mark.asyncio
    async def test_pause_when_no_loop(self, hermes_home):
        """/loop pause when no loop shows friendly message."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-11"
        runner.session_store.get_or_create_session.return_value = session_entry

        result = await runner._handle_loop_command(_make_event("/loop pause"))

        assert "No loop set" in result

    @pytest.mark.asyncio
    async def test_resume_when_no_loop(self, hermes_home):
        """/loop resume when no loop shows friendly message."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "gw-sid-12"
        runner.session_store.get_or_create_session.return_value = session_entry

        result = await runner._handle_loop_command(_make_event("/loop resume"))

        assert "No loop to resume" in result

    @pytest.mark.asyncio
    async def test_gateway_known_command_auto_registered(self):
        """Adding CommandDef to registry auto-registers loop as gateway-known."""
        from hermes_cli.commands import is_gateway_known_command
        assert is_gateway_known_command("loop") is True


# ------------------------------------------------------------------
# _post_turn_loop_continuation hook tests
# ------------------------------------------------------------------

class TestLoopContinuationHook:
    """_post_turn_loop_continuation is now a no-op — loop is driven by
    the LoopScheduler background daemon thread, not a gateway hook."""

    @pytest.mark.asyncio
    async def test_no_op_never_enqueues(self, hermes_home):
        """No-op hook never calls _enqueue_fifo."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "hook-gw-sid-1"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-hook-1"

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="hook-gw-sid-1")
        loop_mgr.set("check deployment")

        await runner._post_turn_loop_continuation(
            session_entry=session_entry,
            source=_make_source(),
            final_response="all good",
        )

        runner._enqueue_fifo.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_does_not_error(self, hermes_home):
        """No-op hook doesn't raise with empty session."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "hook-gw-sid-2"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-hook-2"

        # No loop set — should still not error
        await runner._post_turn_loop_continuation(
            session_entry=session_entry,
            source=_make_source(),
            final_response="all good",
        )

        runner._enqueue_fifo.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_op_with_goal_active(self, hermes_home):
        """No-op hook doesn't error with active goal."""
        runner = _make_runner()
        session_entry = MagicMock()
        session_entry.session_id = "hook-gw-sid-3"
        runner.session_store.get_or_create_session.return_value = session_entry
        runner._enqueue_fifo = MagicMock()
        runner._session_key_for_source = lambda _s: "key-hook-3"

        from hermes_cli.goals import GoalManager
        goal_mgr = GoalManager(session_id="hook-gw-sid-3")
        goal_mgr.set("do the thing")

        from hermes_cli.loop import LoopManager, save_loop
        loop_mgr = LoopManager(session_id="hook-gw-sid-3")
        loop_mgr.set("check deployment", interval_seconds=300)
        loop_mgr.state.last_fired_at = __import__("time").time()
        save_loop("hook-gw-sid-3", loop_mgr.state)

        await runner._post_turn_loop_continuation(
            session_entry=session_entry,
            source=_make_source(),
            final_response="all good",
        )

        runner._enqueue_fifo.assert_not_called()


# ------------------------------------------------------------------
# Synthetic event detection tests
# ------------------------------------------------------------------

class TestLoopSyntheticEventDetection:

    def test_is_loop_continuation_event_matches(self):
        """_is_loop_continuation_event returns True for loop continuations."""
        from gateway.run import GatewayRunner
        event = MagicMock()
        event.text = "[Loop check] check deployment"
        assert GatewayRunner._is_loop_continuation_event(event) is True

    def test_is_loop_continuation_event_no_match(self):
        """_is_loop_continuation_event returns False for regular messages."""
        from gateway.run import GatewayRunner
        event = MagicMock()
        event.text = "hello world"
        assert GatewayRunner._is_loop_continuation_event(event) is False

    def test_clear_loop_pending_continuations_removes_loop_events(self):
        """_clear_loop_pending_continuations removes only loop continuation events."""
        from gateway.run import GatewayRunner
        runner = _make_runner()

        loop_event = MagicMock()
        loop_event.text = "[Loop check] check deployment"
        normal_event = MagicMock()
        normal_event.text = "hello world"

        adapter = runner.adapters[Platform.TELEGRAM]
        adapter._pending_messages = {"key-1": loop_event}
        runner._queued_events = {"key-1": [loop_event, normal_event]}

        removed = runner._clear_loop_pending_continuations("key-1", adapter)
        assert removed == 2
        assert "key-1" not in adapter._pending_messages
        assert runner._queued_events["key-1"] == [normal_event]


# ------------------------------------------------------------------
# Stale continuation discard tests
# ------------------------------------------------------------------

class TestLoopStaleContinuation:

    def test_loop_still_active_for_session_true(self, hermes_home):
        """_loop_still_active_for_session returns True when loop is active."""
        from gateway.run import GatewayRunner
        runner = _make_runner()

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="stale-sid-1")
        loop_mgr.set("check deployment")

        assert runner._loop_still_active_for_session("stale-sid-1") is True

    def test_loop_still_active_for_session_false(self, hermes_home):
        """_loop_still_active_for_session returns False when loop is paused."""
        from gateway.run import GatewayRunner
        runner = _make_runner()

        from hermes_cli.loop import LoopManager
        loop_mgr = LoopManager(session_id="stale-sid-2")
        loop_mgr.set("check deployment")
        loop_mgr.pause()

        assert runner._loop_still_active_for_session("stale-sid-2") is False

    def test_loop_still_active_for_session_none(self):
        """_loop_still_active_for_session returns False when no loop exists."""
        from gateway.run import GatewayRunner
        runner = _make_runner()

        assert runner._loop_still_active_for_session("stale-sid-none") is False
