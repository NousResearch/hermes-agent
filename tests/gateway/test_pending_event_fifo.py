"""Queued events preserve arrival order, attribution and reply context."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.event import MessageType
from tests.gateway.test_active_session_text_merge import (
    _make_event,
    _make_initialized_adapter,
)
from tests.gateway.test_busy_followup_after_session_release import _QueueRunner


def _queue(adapter, runner, key):
    head = adapter._pending_messages.get(key)
    return ([head] if head else []) + list(runner._overflow_queue(key) or ())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario,kind",
    [
        ("intervening", MessageType.TEXT),
        ("intervening", MessageType.PHOTO),
        ("reply", MessageType.TEXT),
        ("reply", MessageType.PHOTO),
        ("flush-control", MessageType.TEXT),
        ("flush-plugin", MessageType.TEXT),
        ("buffer-control", MessageType.TEXT),
        ("buffer-plugin", MessageType.TEXT),
        ("buffer-reply", MessageType.TEXT),
        ("runnerless-control", MessageType.TEXT),
        ("runnerless-plugin", MessageType.TEXT),
        ("runnerless-sender", MessageType.TEXT),
        ("runnerless-reply", MessageType.TEXT),
    ],
)
async def test_pending_events_preserve_arrival_order_and_context(
    scenario, kind, monkeypatch: pytest.MonkeyPatch
):
    adapter = _make_initialized_adapter()
    runner = _QueueRunner(adapter)
    runnerless = scenario.startswith("runnerless-")
    adapter.gateway_runner = None if runnerless else runner
    adapter._active_sessions["shared"] = asyncio.Event()
    adapter._busy_text_mode = "queue"
    monkeypatch.setattr(adapter, "_event_session_key", lambda event: "shared")
    adapter.set_message_handler(AsyncMock(return_value=None))
    first = _make_event("first", chat_type="group", user_id="alice")
    second = _make_event("second", chat_type="group", user_id="alice")
    if scenario in {"intervening", "reply", "flush-control"}:
        first.message_type = MessageType.PHOTO
        first.media_urls = ["/tmp/first.jpg"]
        first.media_types = ["image/jpeg"]
        first.media_text_inlined = [False]
    second.message_type = kind
    if kind == MessageType.PHOTO:
        second.media_urls = ["/tmp/second.jpg"]
        second.media_types = ["image/jpeg"]
        second.media_text_inlined = [False]
    if scenario == "reply" or scenario.endswith("-reply"):
        first.reply_to_message_id, first.reply_to_text = "quote-a", "quoted A"
        second.reply_to_message_id, second.reply_to_text = "quote-b", "quoted B"
    if scenario.endswith("-control"):
        second.allow_gateway_control = False
        if scenario == "flush-control":
            first.text, second.text = "", "/new"
    if scenario.endswith("-plugin"):
        second.metadata = {"hermes_plugin_id": "example"}
    if scenario.endswith("-sender"):
        second.source.user_id = "bob"

    events = [first, second]
    if scenario == "intervening":
        events.insert(1, _make_event("between", chat_type="group", user_id="bob"))
    if runnerless:
        events.insert(0, _make_event("earlier", chat_type="group", user_id="carol"))
    expected = deepcopy(events)
    try:
        if scenario in {"intervening", "reply"}:
            for event in events:
                runner._queue_or_replace_pending_event("shared", event)
        elif scenario.startswith("flush-"):
            adapter._pending_messages["shared"] = first
            await adapter._queue_text_debounce("shared", second)
            await adapter._flush_text_debounce_now("shared")
        else:
            if runnerless:
                adapter._pending_messages["shared"] = events[0]
            await adapter.handle_message(first)
            await adapter.handle_message(second)
            await adapter._flush_text_debounce_now("shared")
        actual = _queue(adapter, runner, "shared")
        state = adapter._text_debounce.get("shared")
        if state is not None:
            actual.extend([*state.earlier_events, state.event])
        assert actual == expected
    finally:
        adapter._discard_text_debounce("shared")


@pytest.mark.asyncio
@pytest.mark.parametrize("busy_dispatches", [1, 2])
@pytest.mark.parametrize("identity", ["original", "rewrite", "idless-rewrite"])
async def test_requeued_deferred_turn_stays_before_later_fifo_items(
    busy_dispatches, identity
):
    adapter = _make_initialized_adapter()
    runner = _QueueRunner(adapter)
    adapter.gateway_runner = runner
    events = [
        _make_event(text, chat_type="group", user_id=sender)
        for text, sender in [("first", "alice"), ("second", "bob"), ("third", "carol")]
    ]
    if identity == "idless-rewrite":
        events[0].message_id = None
    for event in events:
        runner._queue_or_replace_pending_event("shared", event)
    guard = asyncio.Event()
    adapter._active_sessions["shared"] = guard
    attempts = []
    processed = []
    done = asyncio.Event()

    async def handler(event):
        attempts.append(event.text)
        if len(attempts) <= busy_dispatches:
            queued = (
                replace(event, text=event.text) if identity != "original" else event
            )
            runner._queue_or_replace_pending_event("shared", queued)
            return None
        processed.append(event.text)
        if len(processed) == 3:
            done.set()
        return None

    adapter.set_message_handler(handler)
    await adapter._drain_pending_after_session_command("shared", guard)
    try:
        await asyncio.wait_for(done.wait(), 2)
        await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 2)
        assert processed == ["first", "second", "third"]
    finally:
        await adapter.cancel_background_tasks()
