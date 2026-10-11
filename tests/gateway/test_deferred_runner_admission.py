"""Adapter replacement must not steer a deferred event into a running turn."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.pairing import PairingStore
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore
from plugins.platforms.matrix.adapter import MatrixAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["steer", "interrupt"])
async def test_replaced_adapter_defers_reactions_in_runner_fifo(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("MATRIX_ALLOW_ALL_USERS", "true")
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = SessionStore(tmp_path / "sessions", runner.config)
    runner.pairing_store = PairingStore()
    runner.hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]))
    runner._draining = False
    runner._busy_input_mode = mode
    original = MatrixAdapter(PlatformConfig(enabled=True))
    replacement = MatrixAdapter(PlatformConfig(enabled=True))
    runner.adapters = {Platform.MATRIX: original}
    source = SessionSource(platform=Platform.MATRIX, chat_id="!room:test",
                           chat_type="group", user_id="@alice:test", thread_id="$thread")
    entry = runner.session_store.get_or_create_session(source)
    key = runner._session_key_for_source(source)
    active = SimpleNamespace(steer=Mock(return_value=True), interrupt=Mock(),
                             _active_children=[], _supports_active_turn_redirect=False)
    runner._running_agents = {key: active}
    original._active_sessions[key] = asyncio.Event()
    earlier = MessageEvent(text="Earlier queued request", source=source, message_id="$earlier")
    original._pending_messages[key] = earlier
    older_overflow = MessageEvent(text="Earlier overflow request", source=source, message_id="$overflow")
    runner._enqueue_fifo(key, older_overflow, original)
    runner.adapters[Platform.MATRIX] = replacement
    replacement.gateway_runner = runner
    replacement.set_session_store(runner.session_store)
    replacement.set_message_handler(runner._handle_message)
    runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
    runner._hm_evict_idle_stale_agent = lambda _key: None
    runner._hm_evict_reaped_agent = lambda _key: None
    runner._session_has_compression_in_flight = AsyncMock(return_value=False)
    events = [MessageEvent(
        text="Matrix reaction by @alice:test: 👍", source=source,
        message_id=f"$reaction{index}", allow_gateway_control=False, defer_until_idle=True,
        metadata={"gateway_session_key": key, "gateway_session_id": entry.session_id,
                  "gateway_session_strict": True},
    ) for index in range(2)]
    assert replacement._active_sessions == {}
    for event in events:
        await runner._handle_message(event)
    assert (active.steer.call_args_list, active.interrupt.call_args_list) == ([], [])
    overflow = runner._overflow_queue(key)
    assert overflow is not None
    assert [replacement._pending_messages[key], *overflow] == [older_overflow, *events]
    assert [event._gateway_accepted for event in events] == [True, True]
    assert runner._peek_session_state(key).turn.agent is active
    assert not original._active_sessions[key].is_set()
    drained = []
    for _ in [earlier, older_overflow, *events]:
        pending, _ = await runner._run_agent_drain_pending(
            {"final_response": "Active answer"}, original, source, key,
        )
        drained.append(pending)
    assert drained == [earlier, older_overflow, *events]
    assert original._pending_messages == {}
    assert replacement._pending_messages == {}
    assert runner._overflow_queue(key) == []
