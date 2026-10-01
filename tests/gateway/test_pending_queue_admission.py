"""Bounded queue admission preserves older complete events and reports refusal."""

import asyncio
import time
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base_pending import pending_dispatch_scope
from gateway.platforms.event import MessageType
from gateway.run import _AGENT_PENDING_SENTINEL
from gateway.run_inbound import GatewayInboundMixin
from gateway.wake import WakeNotAccepted, admit_internal_event
from tests.gateway.test_active_session_text_merge import _make_event, _make_initialized_adapter
from tests.gateway.test_busy_followup_after_session_release import _QueueRunner


class _AdmissionRunner(GatewayInboundMixin, _QueueRunner):
    def __init__(self, adapter):
        super().__init__(adapter)
        self._draining = False
        self._is_user_authorized_for_source = lambda source: True
        self._admit_bot_message_for_source = lambda source: True
        self._effective_busy_input_mode = lambda source: "queue"
        self._effective_busy_text_mode = lambda source: "interrupt"
        self._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
        self._resolve_busy_steer_or_redirect = AsyncMock()
        self._resolve_busy_steer_or_redirect.return_value.effective_mode = "queue"
        self._resolve_busy_steer_or_redirect.return_value.steered = False
        self._resolve_busy_steer_or_redirect.return_value.redirected = False
        self._send_busy_reply = AsyncMock()


def _events(adapter, runner):
    head = adapter._pending_messages.get("shared")
    buffered = adapter._text_debounce.get("shared")
    return (([head] if head else []) + list(runner._overflow_queue("shared") or [])
            + ([*buffered.earlier_events, buffered.event] if buffered else []))


def _setup(depth):
    adapter = _make_initialized_adapter()
    runner = _AdmissionRunner(adapter)
    adapter.gateway_runner = runner
    adapter._active_sessions["shared"] = asyncio.Event()
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 10
    adapter._event_session_key = lambda event: "shared"
    adapter.set_message_handler(AsyncMock(return_value=None))
    events = [_make_event(str(i), chat_type="group", user_id=f"user-{i}") for i in range(depth)]
    if events:
        events[0].message_type = MessageType.PHOTO
        events[0].media_urls, events[0].media_types, events[0].media_text_inlined = ["/tmp/first.jpg"], ["image/jpeg"], [False]
        events[0].reply_to_message_id, events[0].reply_to_text = "quote", "quoted text"
        events[0].reply_to_author_id, events[0].reply_to_author_name = "author", "Quoted author"
        events[0].reply_to_is_own_message = True
        events[0].metadata, events[0].allow_gateway_control = {"hermes_plugin_id": "first"}, False
    for event in events:
        runner._enqueue_fifo("shared", event, adapter)
    return adapter, runner, deepcopy(events)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "fifo", "normal", "queue", "steer", "grace", "debounce", "reserved", "redispatch",
])
async def test_admission_at_capacity_preserves_the_complete_fifo(path, monkeypatch):
    adapter, runner, expected = _setup(31 if path == "reserved" else 32)
    incoming = _make_event("later", chat_type="group", user_id="new-user")
    reply = None
    arrival_accepted = None
    try:
        if path == "reserved":
            buffered = _make_event("reserved", chat_type="group", user_id="buffered-user")
            assert await adapter._queue_text_debounce("shared", buffered)
            expected.append(deepcopy(buffered))
        if path in {"fifo", "reserved"}:
            runner._enqueue_fifo("shared", incoming, adapter)
        if path == "normal":
            runner._queue_or_replace_pending_event("shared", incoming)
        if path == "queue":
            incoming.text = "/queue later"
            reply = await runner._busy_queue_command(incoming, "shared", incoming.source)
        if path == "steer":
            incoming.text = "/steer later"
            runner._session_state("shared").turn.agent = _AGENT_PENDING_SENTINEL
            reply = await runner._busy_steer_command(incoming, "shared", incoming.source)
        if path == "grace":
            runner._session_state("shared").turn.started_ts = time.time()
            monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "10000")
            runner._hm_busy_telegram_grace_queue(incoming, incoming.source, "shared", "queue")
        if path == "debounce":
            await adapter.handle_message(incoming)
        if path == "redispatch":
            attempted = adapter._pending_messages.pop("shared")
            adapter._pending_messages["shared"] = runner._overflow_queue("shared").pop(0)
            with pending_dispatch_scope(adapter, "shared", attempted):
                runner._queue_or_replace_pending_event("shared", attempted)
            incoming = attempted
        assert (_events(adapter, runner), incoming._gateway_accepted,
                "full" in reply.lower() if reply else None, arrival_accepted) == (
                    expected, path.startswith("redispatch"),
                    True if path in {"queue", "steer"} else None,
                    False if path == "redispatch-arrival" else None)
    finally:
        adapter._discard_text_debounce("shared")
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["external", "internal", "debounce", "queue-copy", "steer-copy", "buffered-order", "buffered-command", "buffered-priority"])
async def test_queue_receipts_and_refusals_preserve_event_context(path):
    copying = path.endswith("-copy")
    adapter, runner, expected = _setup(1 if path.startswith("buffered-") else 0 if copying else 32)
    event = _make_event("later", chat_type="group", user_id="new-user")
    event.metadata = {"hermes_plugin_id": "example", "routing": "preserved"}
    event.reply_to_message_id, event.reply_to_text = "quote", "quoted text"
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        if path.startswith("buffered-"):
            buffered = _make_event("buffered", chat_type="group", user_id="buffered-user")
            assert await adapter._queue_text_debounce("shared", buffered)
            incoming = deepcopy(adapter._pending_messages["shared"])
            incoming.text, incoming.message_id = "later", "later"
            expected.extend(deepcopy([buffered, incoming]))
            if path == "buffered-order":
                runner._queue_or_replace_pending_event("shared", incoming)
            elif path == "buffered-priority":
                runner._hm_merge_pending_for_source(incoming.source, "shared", incoming)
            else:
                runner._enqueue_fifo("shared", incoming, adapter)
            assert _events(adapter, runner) == expected
            return
        if copying:
            event.text = "/queue later" if path == "queue-copy" else "/steer later"
            event._bot_loop_admitted = True
            runner._session_state("shared").turn.agent = _AGENT_PENDING_SENTINEL
            command = runner._busy_queue_command if path == "queue-copy" else runner._busy_steer_command
            await command(event, "shared", event.source)
            assert (_events(adapter, runner), getattr(_events(adapter, runner)[0], "_bot_loop_admitted", False)) == (
                [replace(event, text="later", message_type=MessageType.TEXT)], True)
            return
        if path == "internal":
            event.internal = True
            with pytest.raises(WakeNotAccepted):
                await admit_internal_event(adapter, event)
        else:
            if path == "debounce":
                runner._effective_busy_text_mode = lambda source: "queue"
            await adapter.handle_message(event)
        notices = runner._send_busy_reply.await_args_list
        assert (_events(adapter, runner), event._gateway_accepted,
                len(notices), "full" in notices[0].args[2].lower() if notices else None) == (
                    expected, False, 0 if path == "internal" else 1,
                    None if path == "internal" else True)
    finally:
        adapter._discard_text_debounce("shared")
