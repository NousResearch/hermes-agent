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
    "scenario,kind,busy_dispatches,identity",
    [
        pytest.param(
            "intervening",
            MessageType.TEXT,
            0,
            "original",
            id="intervening-MessageType.TEXT",
        ),
        pytest.param(
            "intervening",
            MessageType.PHOTO,
            0,
            "original",
            id="intervening-MessageType.PHOTO",
        ),
        pytest.param(
            "reply", MessageType.TEXT, 0, "original", id="reply-MessageType.TEXT"
        ),
        pytest.param(
            "reply", MessageType.PHOTO, 0, "original", id="reply-MessageType.PHOTO"
        ),
        pytest.param(
            "flush-control",
            MessageType.TEXT,
            0,
            "original",
            id="flush-control-MessageType.TEXT",
        ),
        pytest.param(
            "flush-plugin",
            MessageType.TEXT,
            0,
            "original",
            id="flush-plugin-MessageType.TEXT",
        ),
        pytest.param(
            "buffer-control",
            MessageType.TEXT,
            0,
            "original",
            id="buffer-control-MessageType.TEXT",
        ),
        pytest.param(
            "buffer-plugin",
            MessageType.TEXT,
            0,
            "original",
            id="buffer-plugin-MessageType.TEXT",
        ),
        pytest.param(
            "buffer-reply",
            MessageType.TEXT,
            0,
            "original",
            id="buffer-reply-MessageType.TEXT",
        ),
        pytest.param(
            "runnerless-control",
            MessageType.TEXT,
            0,
            "original",
            id="runnerless-control-MessageType.TEXT",
        ),
        pytest.param(
            "runnerless-plugin",
            MessageType.TEXT,
            0,
            "original",
            id="runnerless-plugin-MessageType.TEXT",
        ),
        pytest.param(
            "runnerless-sender",
            MessageType.TEXT,
            0,
            "original",
            id="runnerless-sender-MessageType.TEXT",
        ),
        pytest.param(
            "runnerless-reply",
            MessageType.TEXT,
            0,
            "original",
            id="runnerless-reply-MessageType.TEXT",
        ),
        *[
            pytest.param(
                "redispatch",
                MessageType.TEXT,
                count,
                identity,
                id=f"{identity}-{count}",
            )
            for identity in ("original", "rewrite", "idless-rewrite")
            for count in (1, 2)
        ],
        pytest.param(
            "delivery-interrupt", MessageType.TEXT, 0, "original", id="interrupt"
        ),
        pytest.param("delivery-queue", MessageType.TEXT, 0, "original", id="queue"),
    ],
)
async def test_pending_events_preserve_arrival_order_and_context(
    scenario, kind, busy_dispatches, identity, monkeypatch: pytest.MonkeyPatch
):
    adapter = _make_initialized_adapter()
    runner = _QueueRunner(adapter)
    if scenario == "redispatch":
        adapter.gateway_runner = runner
        events = [
            _make_event(text, chat_type="group", user_id=sender)
            for text, sender in [
                ("first", "alice"),
                ("second", "bob"),
                ("third", "carol"),
            ]
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
        return

    if scenario.startswith("delivery-"):
        busy_text_mode = scenario.removeprefix("delivery-")
        adapter._busy_text_mode = busy_text_mode
        adapter._busy_text_debounce_seconds = 5.0
        adapter._busy_text_hard_cap_seconds = 10.0
        adapter.gateway_runner = runner
        monkeypatch.setattr(adapter, "_event_session_key", lambda event: "shared")
        delivering, all_ran = asyncio.Event(), asyncio.Event()
        ran: list[str] = []

        async def runner_handle_message(event):
            event, _, _ = runner._hm_rescue_orphaned_fifo(
                event, event.source, False, "shared"
            )
            ran.append(event.text)
            while (
                queued := runner._promote_queued_event(
                    "shared", adapter, adapter._pending_messages.pop("shared", None)
                )
            ) is not None:
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
        await adapter.handle_message(
            _make_event("first", chat_type="group", user_id="alice")
        )
        await asyncio.sleep(0)
        for text, sender in [("from bob", "bob"), ("from carol", "carol")]:
            await adapter.handle_message(
                _make_event(text, chat_type="group", user_id=sender)
            )
        delivering.set()
        await asyncio.wait_for(all_ran.wait(), 2.0)
        await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 2.0)

        assert ran == ["first", "from bob", "from carol"]
        return

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
