"""Regression test: /stop must cancel the WHOLE queue, not just the head.

When messages pile up behind a running turn (explicit ``/queue`` or
``busy_input_mode: queue``), the adapter holds exactly one in its single
pending slot; everything behind it lives in ``GatewayRunner._queued_events``.
``/stop`` routes through ``_interrupt_and_clear_session``, which used to drain
only the adapter slot — so ``/stop`` dropped the queue head and let the tail
(``B``, ``C``, …) promote and run on its own, one turn after another, while the
user had been told ``⚡ Stopped.`` (#73060).

The fix clears the session's ``conversation.queued_events`` alongside the slot
(surfaced here through the ``_queued_events`` compat view), so a ``/stop`` leaves
the session with an empty queue and nothing left to drain.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource, build_session_key


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _session_entry() -> SessionEntry:
    return SessionEntry(
        session_key=build_session_key(_make_source()),
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        total_tokens=0,
    )


def _make_runner(session_entry: SessionEntry):
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")}
    )
    adapter = MagicMock()
    adapter.send = AsyncMock()
    adapter._pending_messages = {}
    adapter._active_sessions = {}
    adapter.interrupt_session_activity = AsyncMock()
    # Real single-slot semantics: get_pending_message pops the one slot.
    adapter.get_pending_message = lambda sk: adapter._pending_messages.pop(sk, None)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._voice_mode = {}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = session_entry
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_any_sessions.return_value = True
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._queued_events = {}
    runner._pending_approvals = {}
    runner._session_db = MagicMock()
    runner._session_db.get_session_title.return_value = None
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._show_reasoning = False
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._should_send_voice_reply = lambda *_args, **_kwargs: False
    runner._send_voice_reply = AsyncMock()
    runner._capture_gateway_honcho_if_configured = lambda *args, **kwargs: None
    runner._emit_gateway_run_progress = AsyncMock()
    return runner, adapter


async def _queue(runner, text: str, message_id: str) -> None:
    await runner._handle_message(
        MessageEvent(text=f"/queue {text}", source=_make_source(), message_id=message_id)
    )


@pytest.mark.asyncio
async def test_stop_cancels_the_entire_queue_not_just_the_head():
    runner, adapter = _make_runner(_session_entry())
    sk = build_session_key(_make_source())
    runner._running_agents[sk] = MagicMock()  # a turn is in flight

    for i, text in enumerate(["A", "B", "C"]):
        await _queue(runner, text, f"q{i}")

    # Head sits in the adapter slot; B and C overflow into _queued_events.
    assert getattr(adapter._pending_messages.get(sk), "text", None) == "A"
    assert [e.text for e in runner._queued_events.get(sk, [])] == ["B", "C"]
    assert runner._queue_depth(sk, adapter=adapter) == 3

    reply = await runner._handle_message(
        MessageEvent(text="/stop", source=_make_source(), message_id="s1")
    )
    assert reply is not None

    # /stop must leave the queue empty — slot and overflow both gone.
    assert adapter._pending_messages.get(sk) is None
    assert runner._queued_events.get(sk) in (None, [])
    assert runner._queue_depth(sk, adapter=adapter) == 0

    # And there is nothing left for the drain to promote and run.
    promoted = runner._promote_queued_event(sk, adapter, None)
    assert promoted is None
