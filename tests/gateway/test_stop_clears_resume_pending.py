"""Regression: /stop retires the restart-recovery marker.

A gateway restart that interrupts a turn marks the session ``resume_pending``; only a turn that
completes successfully clears it. A /stop ends the (often resumed) turn as "interrupted", so
without this the marker survived and the NEXT restart auto-resumed the work the user had
explicitly stopped. Both /stop routes converge on ``_interrupt_and_clear_session``, which is
driven for real here (stubs as in test_agent_loop_stopped_hook.py).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key


class _Store:
    def __init__(self, session_key):
        self._key = session_key
        self.cleared = []

    def get_or_create_session(self, source):
        return SimpleNamespace(session_key=self._key)

    def clear_resume_pending(self, session_key):
        self.cleared.append(session_key)
        return True


def _setup():
    from gateway.run import GatewayRunner

    source = SessionSource(platform=Platform.TELEGRAM, user_id="u1", chat_id="c1", user_name="t", chat_type="dm")
    key = build_session_key(source)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="***")})
    runner.adapters = {Platform.TELEGRAM: SimpleNamespace(send=AsyncMock())}
    runner.hooks = SimpleNamespace(emit=AsyncMock(), loaded_hooks=False)
    runner._running_agents = {key: MagicMock()}
    runner._pending_messages = {}
    runner._invalidate_session_run_generation = lambda *a, **kw: None
    runner._release_running_agent_state = lambda *a, **kw: None
    runner._is_user_authorized_for_source = lambda source, **kw: True
    runner.session_store = _Store(key)
    return runner, source, key


@pytest.mark.asyncio
async def test_busy_path_stop_clears_resume_pending():
    """/stop while the agent is running: the busy-path handler (the common case)."""
    runner, source, key = _setup()
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    await runner._busy_stop_command(event, key, source)
    assert runner.session_store.cleared == [key]


@pytest.mark.asyncio
async def test_idle_path_stop_clears_resume_pending():
    runner, source, key = _setup()
    await runner._handle_stop_command(MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source))
    assert runner.session_store.cleared == [key]


@pytest.mark.asyncio
async def test_other_interrupts_keep_the_marker():
    """Only a user's stop retires it; e.g. the /new fast path or a shutdown interrupt must not."""
    runner, source, key = _setup()
    await runner._interrupt_and_clear_session(key, source, interrupt_reason="some other reason",
                                              invalidation_reason="test")
    assert runner.session_store.cleared == []


@pytest.mark.asyncio
async def test_stop_survives_a_failing_clear():
    runner, source, key = _setup()

    def _boom(session_key):
        raise RuntimeError("store unavailable")

    runner.session_store.clear_resume_pending = _boom
    event = MessageEvent(text="/stop", message_type=MessageType.TEXT, source=source)
    assert await runner._busy_stop_command(event, key, source)
