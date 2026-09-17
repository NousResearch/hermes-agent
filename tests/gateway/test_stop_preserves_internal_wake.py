"""#114456: /stop on a busy session must preserve a parked internal wake.

``_interrupt_and_clear_session`` consumes-and-discards the adapter's pending
slot; without the invalidation_reason gate a parked async-delegation notice
never reached the post-command drain and the idle session stalled until the
next user message. One invariant runs the real /stop busy path with a parked
internal notice and asserts it reaches ``_start_session_processing``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource


def _make_notice() -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="u1"
    )
    return MessageEvent(
        text="[ASYNC DELEGATION BATCH COMPLETE — deleg_1]",
        message_type=MessageType.TEXT,
        source=source,
        message_id="msg-notice",
        internal=True,
    )


@pytest.mark.asyncio
async def test_stop_busy_path_restarts_parked_internal_notice(monkeypatch):
    """Parked internal wake survives the runner cleanup and the adapter drain
    hands it to a fresh processing task. Red on main (discarded mid-path)."""
    import threading

    from gateway.run import GatewayRunner

    session_key = "agent:main:telegram:dm:12345"
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm")
    notice = _make_notice()

    adapter = SimpleNamespace(_pending_messages={session_key: notice}, _active_sessions={})
    for _name in ("get_pending_message", "_flush_text_debounce_now", "_release_session_guard", "_text_debounce_store"):
        setattr(adapter, _name, getattr(BasePlatformAdapter, _name).__get__(adapter))

    class _RecordingAgent:
        def interrupt(self, reason=None):
            pass

    agent = _RecordingAgent()
    agent._gateway_turn_process_task_id = "session-123"
    agent._gateway_turn_process_baseline = frozenset()

    runner = object.__new__(GatewayRunner)
    runner._running_agents = {session_key: agent}
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner.adapters = {}
    runner._pending_messages = {}
    runner._invalidate_session_run_generation = lambda key, reason=None: None
    runner._release_running_agent_state = lambda key, **kw: None
    runner._adapter_for_source = lambda src: adapter
    from tools.process_registry import process_registry

    monkeypatch.setattr(process_registry, "kill_started_since", lambda *a, **k: 1)

    # Real /stop busy path: runner cleanup runs first (used to discard the slot).
    await runner._busy_stop_command(MagicMock(), session_key, source)

    # ...then the adapter's post-command drain restarts the preserved notice.
    guard = asyncio.Event()
    adapter._active_sessions[session_key] = guard
    mock_start = MagicMock(return_value=True)
    adapter._start_session_processing = mock_start
    await BasePlatformAdapter._drain_pending_after_session_command(
        adapter, session_key, guard
    )
    mock_start.assert_called_once_with(notice, session_key)
