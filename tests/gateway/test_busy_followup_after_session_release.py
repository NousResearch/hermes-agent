"""Follow-ups routed while a session finishes must not be stranded (#121393).

``handle_message`` decides "session busy" under the guard, then
``_handle_message_while_active`` awaits the busy-session handler before it
queues the event.  The running turn can finish inside that yield: its cleanup
releases the guard and performs the final pending-drain.  Resuming the routing
afterwards used to queue the event into ``_pending_messages``/the text-debounce
store anyway — with no owner task left, nothing ever drains it.  The update is
already acked on the wire, so the message is lost silently (no log, no reply),
which is exactly the Telegram DM dropped right after ``response ready``
(#121393).

The admission must re-check ownership after the last await: guard gone means
the session went idle mid-route, so the event starts a fresh turn instead of
queueing behind a task that no longer exists.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent
from gateway.session import build_session_key
from tests.gateway.test_active_session_text_merge import _make_adapter, _make_event


class _TurnSim:
    """Fake session owner: records fresh-turn spawns instead of running the
    full background pipeline, and lets the test release the guard the way a
    finishing turn's cleanup does."""

    def __init__(self, adapter: BasePlatformAdapter, session_key: str):
        self.adapter = adapter
        self.session_key = session_key
        self.dispatched: list[str] = []
        self._hold = asyncio.Event()
        adapter._process_message_background = self._run  # type: ignore[method-assign]

    async def _run(self, event: MessageEvent, session_key: str) -> None:
        self.dispatched.append(event.text or "")
        await self._hold.wait()

    def finish_current_turn(self) -> None:
        """Mirror `_cleanup_finished_session_task` + `_finish_session_task`
        with nothing pending: guard released, owner entry dropped."""
        self.adapter._active_sessions.pop(self.session_key, None)
        self.adapter._session_tasks.pop(self.session_key, None)


async def _wait_until(predicate, timeout: float = 1.5) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.02)
    return predicate()


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_text_mode", ["queue", ""], ids=["debounce", "direct_merge"])
async def test_followup_started_as_fresh_turn_when_session_released_mid_route(busy_text_mode):
    """Text that yields in the busy handler across cleanup must start a turn,
    in both debounce (queue) and direct-merge ('') modes (#121393)."""
    adapter = _make_adapter()
    adapter._busy_text_mode = busy_text_mode
    followup = _make_event("did you get that?")
    session_key = build_session_key(followup.source)
    adapter._active_sessions[session_key] = asyncio.Event()  # seeded running turn
    turn = _TurnSim(adapter, session_key)

    routing_gate = asyncio.Event()
    handler_entered = asyncio.Event()

    async def _busy_handler(event, key):
        handler_entered.set()
        # Yield: inside this await the running turn finishes and cleanup
        # releases the guard (its final pending-drain saw nothing).
        await routing_gate.wait()
        turn.finish_current_turn()
        return False

    adapter.set_busy_session_handler(_busy_handler)

    admission = asyncio.create_task(adapter.handle_message(followup))
    await handler_entered.wait()
    routing_gate.set()
    await admission
    # Post-fix: the guard was gone when admission resumed, so the event became
    # a fresh turn instead of queueing behind a task that no longer exists.
    assert await _wait_until(lambda: bool(turn.dispatched)), (
        "follow-up was stranded: dispatched=%r pending=%r debounce=%r active=%r tasks=%r"
        % (
            turn.dispatched,
            dict(adapter._pending_messages),
            dict(adapter._text_debounce),
            dict(adapter._active_sessions),
            {k: t.done() for k, t in adapter._session_tasks.items()},
        )
    )
    assert turn.dispatched[0] == "did you get that?"
    assert followup._gateway_accepted is True
    # Nothing left holding the event.
    assert session_key not in adapter._pending_messages
    assert session_key not in adapter._text_debounce


@pytest.mark.asyncio
async def test_followup_still_queues_when_the_session_stays_active():
    """No regression: a live guard keeps the queue-behind behavior (#121393)."""
    adapter = _make_adapter()
    adapter._busy_text_mode = ""
    followup = _make_event("still busy here")
    session_key = build_session_key(followup.source)
    adapter._active_sessions[session_key] = asyncio.Event()
    turn = _TurnSim(adapter, session_key)

    routing_gate = asyncio.Event()
    handler_entered = asyncio.Event()

    async def _busy_handler(event, key):
        handler_entered.set()
        await routing_gate.wait()
        return False  # session untouched — still active

    adapter.set_busy_session_handler(_busy_handler)

    admission = asyncio.create_task(adapter.handle_message(followup))
    await handler_entered.wait()
    routing_gate.set()
    await admission
    assert routing_gate.is_set()
    await asyncio.sleep(0)

    assert turn.dispatched == []  # no fresh turn spawned
    assert session_key in adapter._active_sessions  # guard untouched
    pending = adapter._pending_messages.get(session_key)
    assert pending is not None and pending.text == "still busy here"
