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
from gateway.run_busy import GatewayBusySessionMixin
from gateway.run_inbound import GatewayInboundMixin
from gateway.session import build_session_key
from gateway.session_state import SessionState
from tests.gateway.test_active_session_text_merge import (
    _make_adapter, _make_event, _make_initialized_adapter,
)


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


class _QueueRunner(GatewayBusySessionMixin):
    """The runner's FIFO, its post-turn chain and its idle-path orphan rescue."""

    _BUSY_QUEUE_MAX_PENDING = 32
    _hm_rescue_orphaned_fifo = GatewayInboundMixin._hm_rescue_orphaned_fifo

    def __init__(self, adapter: BasePlatformAdapter):
        self.adapter = adapter
        self.states: dict[str, SessionState] = {}

    def _delivery_adapter_for(self, source):
        return self.adapter

    def _session_state(self, key):
        return self.states.setdefault(key, SessionState())

    def _peek_session_state(self, key):
        return self.states.get(key)


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_text_mode", ["interrupt", "queue"])
async def test_followups_arriving_during_final_delivery_keep_arrival_order(busy_text_mode):
    """The runner has drained its queue and the adapter is still delivering the answer when two
    people send follow-ups. The adapter's post-turn drain starts the first; the second must not
    overtake it through the runner's orphan rescue."""
    adapter = _make_initialized_adapter()
    adapter._busy_text_mode = busy_text_mode
    adapter._busy_text_debounce_seconds = 5.0
    adapter._busy_text_hard_cap_seconds = 10.0
    runner = _QueueRunner(adapter)
    adapter.gateway_runner = runner
    adapter._event_session_key = lambda event: "shared"  # type: ignore[method-assign]
    delivering, all_ran = asyncio.Event(), asyncio.Event()
    ran: list[str] = []

    async def runner_handle_message(event):
        event, _, _ = runner._hm_rescue_orphaned_fifo(event, event.source, False, "shared")
        ran.append(event.text)
        while (queued := runner._promote_queued_event(
                "shared", adapter, adapter._pending_messages.pop("shared", None))) is not None:
            ran.append(queued.text)
        if event.text == "first":
            await delivering.wait()
        if len(ran) == 3:
            all_ran.set()
        return None

    async def busy_handler(event, session_key):
        if busy_text_mode == "queue":
            return False
        runner._queue_or_replace_pending_event(session_key, event)
        return True

    adapter.set_message_handler(runner_handle_message)
    adapter.set_busy_session_handler(busy_handler)
    await adapter.handle_message(_make_event("first", chat_type="group", user_id="alice"))
    await asyncio.sleep(0)
    for text, sender in [("from bob", "bob"), ("from carol", "carol")]:
        await adapter.handle_message(_make_event(text, chat_type="group", user_id=sender))
    delivering.set()
    await asyncio.wait_for(all_ran.wait(), 2.0)
    await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 2.0)

    assert ran == ["first", "from bob", "from carol"]
