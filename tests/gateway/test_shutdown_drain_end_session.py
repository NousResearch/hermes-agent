"""Test that drain timeout path calls end_session to prevent session leak.

When the gateway shuts down and the drain times out (agents don't finish
within the drain window), the interrupted sessions must be closed in
state.db. Without this, every restart cycle creates a new session row
with ended_at=NULL, causing unbounded session accumulation.

Regression test for: https://github.com/NousResearch/hermes-agent/issues/XXXXX
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import gateway.run as gateway_run
from gateway.config import Platform, PlatformConfig
from gateway.session import SessionEntry, SessionSource, _now
from tests.gateway.restart_test_helpers import RestartTestAdapter, make_restart_source


def _make_entry(session_key: str, session_id: str) -> SessionEntry:
    """Build a minimal SessionEntry for the routing store."""
    now = _now()
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="123456",
        user_id="u1",
        chat_type="dm",
        thread_id=None,
        chat_name="Test User",
        parent_chat_id=None,
    )
    return SessionEntry(
        session_key=session_key,
        session_id=session_id,
        created_at=now,
        updated_at=now,
        origin=source,
        display_name="Test User",
        platform=source.platform,
        chat_type=source.chat_type,
        was_auto_reset=False,
        auto_reset_reason=None,
        reset_had_activity=False,
    )


@pytest.mark.asyncio
async def test_drain_timeout_calls_end_session_on_interrupted_agents(tmp_path):
    """When drain times out, end_session(session_id, 'agent_close') is called.

    This is the regression test for the session-leak bug: every restart
    cycle that times out during drain would leave ended_at=NULL in
    state.db, causing sessions to accumulate indefinitely.
    """

    # Use the existing test helpers to build a real-enough runner.
    adapter = RestartTestAdapter()
    runner, _adapter = _make_runner(adapter)

    # Force drain timeout: agents will still be running when stop() returns.
    runner._restart_drain_timeout = 0.05
    runner._restart_requested = True  # restart (not plain shutdown)

    session_key = "telegram:dm:123456:"
    session_id = "20250713_120000_abc12345"

    # Set up a running agent (simulates mid-turn agent that won't finish).
    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}

    # Seed the routing store entry so we have a session_id to close.
    entry = _make_entry(session_key, session_id)
    runner.async_session_store._entries[session_key] = entry

    # Mock _session_db.end_session (the actual fix target).
    end_session_mock = AsyncMock()
    mock_session_db = MagicMock()
    mock_session_db.end_session = end_session_mock
    runner._session_db = mock_session_db

    with (
        patch.object(gateway_run, "_hermes_home", tmp_path),
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
    ):
        await runner.stop()

    # The agent must have been interrupted.
    running_agent.interrupt.assert_called()

    # end_session MUST have been called with session_id and "agent_close".
    end_session_mock.assert_awaited_once_with(session_id, "agent_close")


@pytest.mark.asyncio
async def test_drain_completion_does_not_call_end_session():
    """When drain completes gracefully, end_session is NOT called.

    Agents that finish within the drain window complete their own turn and
    call end_session() at turn end. The shutdown path should NOT duplicate
    that call.
    """
    adapter = RestartTestAdapter()
    runner, _adapter = _make_runner(adapter)

    runner._restart_drain_timeout = 5.0  # generous window
    runner._restart_requested = False  # plain shutdown

    session_key = "telegram:dm:123456:"
    session_id = "20250713_120000_graceful1"

    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}

    entry = _make_entry(session_key, session_id)
    runner.async_session_store._entries[session_key] = entry

    end_session_mock = AsyncMock()
    mock_session_db = MagicMock()
    mock_session_db.end_session = end_session_mock
    runner._session_db = mock_session_db

    # Agent finishes immediately so drain completes gracefully.
    async def finish_soon():
        await asyncio.sleep(0.05)
        runner._running_agents.clear()

    asyncio.create_task(finish_soon())

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
    ):
        await runner.stop()

    # Agent finished cleanly — interrupt was NOT called.
    running_agent.interrupt.assert_not_called()

    # end_session was NOT called from the shutdown path.
    # (The agent's own turn-end would call it; we test the shutdown path here.)
    end_session_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_drain_timeout_skips_none_session_id_entries():
    """Entries without a session_id (not yet persisted) must not crash end_session."""
    adapter = RestartTestAdapter()
    runner, _adapter = _make_runner(adapter)

    runner._restart_drain_timeout = 0.05
    runner._restart_requested = True

    session_key = "telegram:dm:123456:"
    # Entry with no session_id (brand new, not yet written to DB).
    entry = _make_entry(session_key, session_id="")
    runner.async_session_store._entries[session_key] = entry

    running_agent = MagicMock()
    runner._running_agents = {session_key: running_agent}

    end_session_mock = AsyncMock()
    mock_session_db = MagicMock()
    mock_session_db.end_session = end_session_mock
    runner._session_db = mock_session_db

    with (
        patch("gateway.status.remove_pid_file"),
        patch("gateway.status.write_runtime_status"),
    ):
        # Must not raise — session_id is empty string.
        await runner.stop()

    # end_session was never called because session_id was falsy.
    end_session_mock.assert_not_awaited()
    running_agent.interrupt.assert_called()


# -------------------------------------------------------------------------- #
# Test fixture helpers                                                       #
# -------------------------------------------------------------------------- #

from collections import OrderedDict

from gateway.config import GatewayConfig
from gateway.run import GatewayRunner


def _make_runner(adapter: RestartTestAdapter):
    """Minimal GatewayRunner fixture for shutdown tests.

    This duplicates the minimum surface of make_restart_runner() from
    restart_test_helpers.py so the test file is self-contained and does
    not depend on helper internals that may change.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    runner._running = True
    runner._shutdown_event = asyncio.Event()
    runner._exit_reason = None
    runner._exit_code = None
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._pending_model_notes = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._signal_initiated_shutdown = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._detached_restart_helper_started = False
    runner._restart_command_source = None
    runner._restart_drain_timeout = gateway_run.DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT
    runner._stop_task = None
    runner._busy_input_mode = "interrupt"
    runner._update_prompt_pending = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._session_sources = OrderedDict()
    runner._session_sources_max = 512
    runner._shutdown_all_gateway_honcho = lambda: None
    runner._update_runtime_status = MagicMock()
    runner._queue_or_replace_pending_event = GatewayRunner._queue_or_replace_pending_event.__get__(
        runner, GatewayRunner
    )
    runner._session_key_for_source = GatewayRunner._session_key_for_source.__get__(
        runner, GatewayRunner
    )
    runner._handle_active_session_busy_message = GatewayRunner._handle_active_session_busy_message.__get__(
        runner, GatewayRunner
    )
    runner._handle_restart_command = GatewayRunner._handle_restart_command.__get__(
        runner, GatewayRunner
    )
    runner._handle_set_home_command = GatewayRunner._handle_set_home_command.__get__(
        runner, GatewayRunner
    )
    runner._send_restart_notification = GatewayRunner._send_restart_notification.__get__(
        runner, GatewayRunner
    )
    runner._send_home_channel_startup_notifications = GatewayRunner._send_home_channel_startup_notifications.__get__(
        runner, GatewayRunner
    )
    runner._status_action_label = GatewayRunner._status_action_label.__get__(
        runner, GatewayRunner
    )
    runner._status_action_gerund = GatewayRunner._status_action_gerund.__get__(
        runner, GatewayRunner
    )
    runner._queue_during_drain_enabled = GatewayRunner._queue_during_drain_enabled.__get__(
        runner, GatewayRunner
    )
    runner._running_agent_count = GatewayRunner._running_agent_count.__get__(
        runner, GatewayRunner
    )
    runner._snapshot_running_agents = GatewayRunner._snapshot_running_agents.__get__(
        runner, GatewayRunner
    )
    runner._notify_active_sessions_of_shutdown = GatewayRunner._notify_active_sessions_of_shutdown.__get__(
        runner, GatewayRunner
    )
    runner._cache_session_source = GatewayRunner._cache_session_source.__get__(
        runner, GatewayRunner
    )
    runner._get_cached_session_source = GatewayRunner._get_cached_session_source.__get__(
        runner, GatewayRunner
    )
    runner._launch_detached_restart_command = GatewayRunner._launch_detached_restart_command.__get__(
        runner, GatewayRunner
    )
    runner.request_restart = GatewayRunner.request_restart.__get__(runner, GatewayRunner)
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner.session_store._entries = {}
    runner.delivery_router = MagicMock()

    platform_adapter = adapter
    platform_adapter.set_message_handler(AsyncMock(return_value=None))
    platform_adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    runner.adapters = {Platform.TELEGRAM: platform_adapter}
    return runner, platform_adapter
