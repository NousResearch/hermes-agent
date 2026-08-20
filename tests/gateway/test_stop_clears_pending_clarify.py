"""Regression test: /stop must clear a pending clarify so the next
non-command message is not swallowed as the interrupted turn's answer.

When a turn is interrupted with /stop while the agent is blocked in a
clarify (waiting for the user's reply), ``_interrupt_and_clear_session``
used to leave the clarify entry registered.  The next non-command message
in the session was then intercepted as the clarify response and dropped —
the turn was already interrupted, so nothing was emitted and no new turn
started (log fingerprint: ``Gateway intercepted clarify text response``
followed by ``Turn ended: reason=interrupted_by_user ... response_len=0``).
"""

from unittest.mock import MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource

SESSION_KEY = "agent:main:slack:dm:D123:1111.2222"


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.SLACK)

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="m1")


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


def _make_runner():
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    # No adapter — the test only asserts clarify cleanup, not adapter I/O.
    runner._adapter_for_source = lambda source: None
    # Mock the state-heavy internals so the test focuses on clarify cleanup.
    runner._peek_session_state = MagicMock(return_value=None)
    runner._invalidate_session_run_generation = MagicMock(return_value=2)
    runner._release_running_agent_state = MagicMock()
    runner._evict_cached_agent = MagicMock()
    return runner


def _source():
    return SessionSource(
        platform=Platform.SLACK,
        chat_id="D123",
        chat_type="dm",
        user_id="U1",
        thread_id="1111.2222",
    )


@pytest.mark.asyncio
async def test_stop_interrupt_clears_open_ended_clarify():
    """/stop must clear a pending open-ended clarify (awaiting_text=True)."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _make_runner()
    entry = cm.register("cl-stop-open", SESSION_KEY, "need a token", None)
    assert entry.awaiting_text is True
    assert cm.get_pending_for_session(SESSION_KEY, include_choice_prompts=True) is not None

    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="Stop requested",
        invalidation_reason="stop_command",
    )

    # The pending clarify must be gone, and the blocked wait must have been
    # unblocked with the empty-string sentinel (so the agent thread drains).
    assert cm.get_pending_for_session(SESSION_KEY, include_choice_prompts=True) is None
    assert entry.event.is_set()
    assert entry.response == ""
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_stop_interrupt_clears_choice_clarify():
    """/stop must clear a pending native multi-choice clarify too."""
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    runner = _make_runner()
    entry = cm.register("cl-stop-choice", SESSION_KEY, "pick one", ["a", "b"])
    assert entry.awaiting_text is False
    assert cm.get_pending_for_session(SESSION_KEY, include_choice_prompts=True) is not None

    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="Stop requested",
        invalidation_reason="stop_command",
    )

    assert cm.get_pending_for_session(SESSION_KEY, include_choice_prompts=True) is None
    assert entry.event.is_set()
    assert entry.response == ""
    _clear_clarify_state()


@pytest.mark.asyncio
async def test_interrupt_with_no_pending_clarify_is_noop():
    """/stop with no clarify registered must still complete cleanly."""
    _clear_clarify_state()

    runner = _make_runner()
    await runner._interrupt_and_clear_session(
        SESSION_KEY,
        _source(),
        interrupt_reason="Stop requested",
        invalidation_reason="stop_command",
    )
    # No exception raised and no clarify left behind.
    _clear_clarify_state()
