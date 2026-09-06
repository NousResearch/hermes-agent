"""Regression tests for the durable-reaped session guard in _handle_message (#99106).

A session whose durable row was ended in state.db (``ws_orphan_reap`` /
``agent_close``) while the gateway process stayed alive keeps its in-memory
turn slot (``_is_session_running`` stays True). Before the guard, the next
inbound message took the PRIORITY fast-path and was interrupt()-delivered into
the dead runtime — silently dropped, never reaching the
``get_or_create_session`` routing self-heal (#54878). The guard evicts the
stale slot at routing time so the message falls through to the cold path.

Salvaged from PR #99183 (@Finn763).
"""

import time

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import (
    GatewayConfig,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
)
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore


class _FakeAdapter:
    def __init__(self):
        self._pending_messages = {}
        self._active_sessions = {}

    async def send(self, *args, **kwargs):
        pass


class _DeadReapedAgent:
    """Runtime whose turn was reaped: interrupt() lands nowhere."""

    def __init__(self):
        self.interrupts = []

    def interrupt(self, text):
        self.interrupts.append(text)

    def get_activity_summary(self):
        # Recently active — the pre-existing idle-staleness eviction must NOT
        # fire, so only the durable-reaped guard can heal this shape.
        return {
            "seconds_since_activity": 0,
            "last_activity_desc": "tool",
            "api_call_count": 1,
            "max_iterations": 50,
        }


def _make_runner(store) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="x")}
    )
    runner.adapters = {Platform.TELEGRAM: _FakeAdapter()}
    runner._pending_messages = {}
    runner._voice_mode = {}
    runner._background_tasks = set()
    runner._draining = False
    runner._restart_requested = False
    runner._restart_task_started = False
    runner._restart_detached = False
    runner._restart_via_service = False
    runner._restart_drain_timeout = 0.0
    runner._stop_task = None
    runner._exit_code = None
    runner._update_runtime_status = MagicMock()
    runner._is_user_authorized = lambda _source: True
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.session_store = store
    runner.delivery_router = MagicMock()
    return runner


def _source(chat_id="555001") -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=chat_id,
    )


def _store(tmp_path) -> SessionStore:
    config = GatewayConfig(
        default_reset_policy=SessionResetPolicy(mode="none")
    )
    return SessionStore(sessions_dir=tmp_path, config=config)


def _occupy_turn_slot(runner, key, agent):
    state = runner._session_state(key)
    state.turn.agent = agent
    # Turn started 2 minutes ago — outside the Telegram follow-up grace
    # window, inside the idle-staleness threshold.
    state.turn.started_ts = time.time() - 120


@pytest.mark.asyncio
async def test_reaped_session_message_reaches_cold_path(tmp_path):
    """DB-ended session + live turn slot: next message must heal, not drop."""
    store = _store(tmp_path)
    src = _source()
    entry = store.get_or_create_session(src)
    key = store._generate_session_key(src)

    runner = _make_runner(store)
    agent = _DeadReapedAgent()
    _occupy_turn_slot(runner, key, agent)

    # The durable row is ended out from under the live slot (the reap).
    store._db.end_session(entry.session_id, "ws_orphan_reap")
    assert store._is_session_ended_in_db(entry.session_id) is True

    event = MessageEvent(
        text="hello, are you there?",
        message_type=MessageType.TEXT,
        source=src,
    )

    cold_path = AsyncMock(return_value="COLD_PATH_REPLY")
    with (
        patch.object(GatewayRunner, "_handle_message_with_agent", cold_path),
        patch.object(GatewayRunner, "_run_post_turn_hooks", AsyncMock()),
        patch.object(GatewayRunner, "_clear_durable_active_turn", AsyncMock()),
        patch.object(GatewayRunner, "_persist_active_agents", lambda self: None),
    ):
        result = await runner._handle_message(event)

    assert result == "COLD_PATH_REPLY"
    assert cold_path.await_count == 1
    # Nothing was interrupt()-delivered into the dead runtime.
    assert agent.interrupts == []


@pytest.mark.asyncio
async def test_alive_session_keeps_priority_interrupt_path(tmp_path):
    """Control: with the durable row still open, the PRIORITY interrupt
    fast-path must behave exactly as before (no eviction)."""
    store = _store(tmp_path)
    src = _source(chat_id="555002")
    store.get_or_create_session(src)
    key = store._generate_session_key(src)

    runner = _make_runner(store)
    agent = _DeadReapedAgent()
    _occupy_turn_slot(runner, key, agent)

    event = MessageEvent(
        text="follow-up while running",
        message_type=MessageType.TEXT,
        source=src,
    )

    cold_path = AsyncMock(return_value="COLD_PATH_REPLY")
    with (
        patch.object(GatewayRunner, "_handle_message_with_agent", cold_path),
        patch.object(GatewayRunner, "_agent_has_active_subagents", lambda self, a: False),
        patch.object(
            GatewayRunner,
            "_session_has_compression_in_flight",
            AsyncMock(return_value=False),
        ),
    ):
        result = await runner._handle_message(event)

    assert result is None
    assert cold_path.await_count == 0
    assert agent.interrupts == ["follow-up while running"]


@pytest.mark.asyncio
async def test_guard_is_inert_with_stubbed_session_store(tmp_path):
    """Bare test runners stub session_store with MagicMock; the guard must
    not evict (peek_session_id returns a Mock, not a str) and must not raise."""
    store = MagicMock()
    src = _source(chat_id="555003")

    runner = _make_runner(store)
    key = runner._session_key_for_source(src)
    agent = _DeadReapedAgent()
    _occupy_turn_slot(runner, key, agent)

    event = MessageEvent(
        text="stub store follow-up",
        message_type=MessageType.TEXT,
        source=src,
    )

    with (
        patch.object(GatewayRunner, "_agent_has_active_subagents", lambda self, a: False),
        patch.object(
            GatewayRunner,
            "_session_has_compression_in_flight",
            AsyncMock(return_value=False),
        ),
    ):
        result = await runner._handle_message(event)

    # Slot untouched; the message took the normal busy path.
    assert runner._is_session_running(key) is True
    assert result is None
    assert agent.interrupts == ["stub store follow-up"]
