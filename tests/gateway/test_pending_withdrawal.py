"""A message that its sender deletes before its turn starts must never reach the agent.

Covers ``BasePlatformAdapter.withdraw_pending_message`` over the adapter's input buffers and
pending slot, and the runner's FIFO overflow and interrupt fallback behind them.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

ROOM = "!room:example.org"
ALICE = "@alice:example.org"


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.MATRIX)
        self._message_handler = AsyncMock(return_value=None)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}


def _event(adapter, message_id, *, photo=False):
    return MessageEvent(
        text=f"text of {message_id}",
        message_type=MessageType.PHOTO if photo else MessageType.TEXT,
        source=adapter.build_source(chat_id=ROOM, chat_type="dm", user_id=ALICE),
        message_id=message_id,
        media_urls=[f"/cache/{message_id}.png"] if photo else [],
        media_types=["image/png"] if photo else [],
        timestamp=datetime(2026, 9, 30),
    )


async def _receive(adapter, store, message_ids):
    """Feed ``message_ids`` into ``store`` and return a reader for the pending event in it."""
    photo = store == "photo_burst"
    events = [_event(adapter, message_id, photo=photo) for message_id in message_ids]
    key = adapter._event_session_key(events[0])
    if store == "text_batch":
        adapter._text_batch_delay_seconds = 60
        for event in events:
            adapter._enqueue_text_event(event)
        return lambda: adapter._pending_text_batches.get(key)
    adapter._active_sessions[key] = asyncio.Event()
    if store == "busy_debounce":
        adapter._busy_text_mode = "queue"
        adapter._busy_text_debounce_seconds = adapter._busy_text_hard_cap_seconds = 60
    for event in events:
        await adapter.handle_message(event)
    if store == "busy_debounce":
        return lambda: getattr(adapter._text_debounce.get(key), "event", None)
    return lambda: adapter._pending_messages.get(key)


def _stop_timers(*adapters):
    for adapter in adapters:
        for task in adapter._pending_text_batch_tasks.values():
            task.cancel()
        for state in adapter._text_debounce.values():
            state.cancel_timer()


async def _assert_remaining_turn(store, withdrawn):
    """Withdrawing messages from a merged pending turn replays the merges of the remaining
    messages in their original order, and leaves nothing when none remain. For a burst of one
    kind, as here, that is the turn that the remaining messages would have produced alone."""
    received, control = _Adapter(), _Adapter()
    pending = await _receive(received, store, ["$a", "$b", "$c"])
    remaining = [message_id for message_id in ["$a", "$b", "$c"] if message_id not in withdrawn]
    expected = (await _receive(control, store, remaining))() if remaining else None

    found = [received.withdraw_pending_message(message_id, chat_id=ROOM, sender_id=ALICE)
             for message_id in withdrawn]
    _stop_timers(received, control)

    assert (found, pending()) == ([True] * len(withdrawn), expected)



def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._draining = False
    return runner


async def _assert_withdrawal_scope(platform, chat_id, sender_id):
    """The runner's FIFO overflow contains every adapter's follow-ups, so a withdrawal must
    leave a message with the same IDs from another sender, chat or platform queued."""
    runner, adapter = _runner(), _Adapter()
    adapter.set_queued_withdrawal_handler(runner._withdraw_queued_followups)
    queued = MessageEvent(text="queued", message_id="$a", source=SessionSource(
        platform=platform, chat_id=chat_id, chat_type="dm", user_id=sender_id))
    overflow = runner._session_state("other-session").conversation.queued_events
    overflow.append(queued)

    found = adapter.withdraw_pending_message("$a", chat_id=ROOM, sender_id=ALICE)

    assert (found, overflow) == (False, [queued])



async def _assert_followup_order(withdrawn, late, expected_turns):
    """Busy follow-ups queue behind a running turn (slot plus FIFO overflow), and each one
    interrupts it. After withdrawals, the drain runs the remaining follow-ups in arrival order,
    including one that arrives later (``late``), and never falls back to the withdrawn text
    that the agent kept as its interrupt message."""
    runner, adapter = _runner(), _Adapter()
    adapter.set_queued_withdrawal_handler(runner._withdraw_queued_followups)
    events = [_event(adapter, message_id) for message_id in ["$a", "$b", "$c"]]
    key = adapter._event_session_key(events[0])
    runner._session_state(key).turn.agent = MagicMock()
    for event in events:
        runner._enqueue_fifo(key, event, adapter)

    for message_id in withdrawn:
        adapter.withdraw_pending_message(message_id, chat_id=ROOM, sender_id=ALICE)
    if late:
        runner._enqueue_fifo(key, _event(adapter, "$d"), adapter)
    result = {"interrupted": True, "interrupt_message": events[-1].text}
    turns = []
    while True:
        pending_event, pending = await runner._run_agent_drain_pending(result, adapter, events[0].source, key)
        if pending_event is None and pending is None:
            break
        turns.append(pending_event.message_id)
        result = {"final_response": "done"}

    assert turns == expected_turns


@pytest.mark.asyncio
@pytest.mark.parametrize("check, args", [
    *[(_assert_remaining_turn, (store, removed))
      for store in ("text_batch", "busy_debounce", "pending_text", "photo_burst")
      for removed in (["$b"], ["$a"], ["$a", "$c"], ["$a", "$b", "$c"])],
    *[(_assert_withdrawal_scope, (platform, room, sender))
      for platform, room, sender in (
          (Platform.MATRIX, ROOM, "@mallory:example.org"),
          (Platform.MATRIX, "!elsewhere:example.org", ALICE),
          (Platform.DISCORD, ROOM, ALICE),
      )],
    *[(_assert_followup_order, args) for args in (
        (["$a"], True, ["$b", "$c", "$d"]),
        (["$b"], True, ["$a", "$c", "$d"]),
        (["$a", "$b", "$c"], False, []),
    )],
])
async def test_withdrawal_preserves_the_other_queued_input(check, args):
    await check(*args)


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", ["first", "middle", "last", "all"])
async def test_withdrawal_preserves_independent_debounce_events(removed):
    from gateway.platforms.base_text_debounce import TextDebounceState

    adapter = _Adapter()
    events = [_event(adapter, name) for name in ("first", "middle", "last")]
    timer = asyncio.create_task(asyncio.sleep(60))
    state = TextDebounceState(events[-1], timer, 1.0, 2.0, events[:-1])
    adapter._text_debounce_store()["key"] = state
    ids = [event.message_id for event in events] if removed == "all" else [removed]
    try:
        found = [adapter.withdraw_pending_message(mid, chat_id=ROOM, sender_id=ALICE) for mid in ids]
        remaining = [event for event in events if event.message_id not in ids]
        current = adapter._text_debounce_store().get("key")
        actual = (found, [*current.earlier_events, current.event] if current else [], current.task if current else None)
        assert actual == ([True] * len(ids), remaining, timer if remaining else None)
    finally:
        timer.cancel()
        await asyncio.gather(timer, return_exceptions=True)

