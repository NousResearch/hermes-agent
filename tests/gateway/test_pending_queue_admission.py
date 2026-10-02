"""Bounded queue admission preserves older complete events and reports refusal."""

import asyncio
import time
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from gateway.platforms.base_pending import (
    pending_dispatch_scope,
    release_pending_dispatch,
    reserve_pending_dispatch,
)
from gateway.platforms.event import MessageType
from gateway.run import _AGENT_PENDING_SENTINEL
from gateway.session import SessionSource
from gateway.wake import WakeNotAccepted, admit_internal_event
from tests.gateway.test_active_session_text_merge import (
    _make_event,
    _make_initialized_adapter,
)
from tests.gateway.test_busy_followup_after_session_release import _QueueRunner
from gateway.session import SessionStore
from gateway.config import GatewayConfig
from gateway.input_owner import gateway_input_owner
from gateway.platforms.base_pending import bind_pending_dispatch_input


class _AdmissionRunner(_QueueRunner):
    def __init__(self, adapter):
        super().__init__(adapter)
        self._draining = False
        self._busy_text_mode = "interrupt"
        self._route_plaintext_approval_while_busy = AsyncMock(return_value=False)
        self._resolve_busy_steer_or_redirect = AsyncMock()
        self._resolve_busy_steer_or_redirect.return_value.effective_mode = "queue"
        self._resolve_busy_steer_or_redirect.return_value.steered = False
        self._resolve_busy_steer_or_redirect.return_value.redirected = False
        self._send_busy_reply = AsyncMock()

    def _is_user_authorized_for_source(
        self, source: SessionSource, *, allow_adapter_delegation: bool = True
    ) -> bool:
        return True

    def _admit_bot_message_for_source(self, source: SessionSource) -> bool:
        return True

    def _effective_busy_input_mode(self, source: SessionSource) -> str:
        return "queue"

    def _effective_busy_text_mode(self, source: SessionSource) -> str:
        return self._busy_text_mode


def _events(adapter, runner):
    head = adapter._pending_messages.get("shared")
    buffered = adapter._text_debounce.get("shared")
    return (
        ([head] if head else [])
        + list(runner._overflow_queue("shared") or [])
        + ([*buffered.earlier_events, buffered.event] if buffered else [])
    )


def _setup(depth, monkeypatch: pytest.MonkeyPatch):
    adapter = _make_initialized_adapter()
    runner = _AdmissionRunner(adapter)
    adapter.gateway_runner = runner
    adapter._active_sessions["shared"] = asyncio.Event()
    adapter._busy_text_mode = "queue"
    adapter._busy_text_debounce_seconds = 10
    monkeypatch.setattr(adapter, "_event_session_key", lambda event: "shared")
    adapter.set_message_handler(AsyncMock(return_value=None))
    events = [
        _make_event(str(i), chat_type="group", user_id=f"user-{i}")
        for i in range(depth)
    ]
    if events:
        events[0].message_type = MessageType.PHOTO
        events[0].media_urls, events[0].media_types = ["/tmp/first.jpg"], ["image/jpeg"]
        events[0].media_text_inlined = [False]
        events[0].reply_to_message_id, events[0].reply_to_text = "quote", "quoted text"
        events[0].reply_to_author_id, events[0].reply_to_author_name = (
            "author",
            "Quoted author",
        )
        events[0].reply_to_is_own_message = True
        events[0].metadata, events[0].allow_gateway_control = (
            {"hermes_plugin_id": "first"},
            False,
        )
    for event in events:
        runner._enqueue_fifo("shared", event, adapter)
    return adapter, runner, deepcopy(events)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "fifo",
        "normal",
        "queue",
        "steer",
        "grace",
        "debounce",
        "reserved",
        "redispatch",
        "redispatch-arrival",
        "redispatch-rewrite",
        "redispatch-idless",
        "cancel-before",
        "cancel-admission",
        "cancel-claimed",
        "cancel-claim-race",
        "cancel-claim-complete",
        "cancel-claim-replaced",
        "reservation-replaced",
    ],
)
async def test_admission_at_capacity_preserves_the_complete_fifo(path, monkeypatch):
    adapter, runner, expected = _setup(31 if path == "reserved" else 32, monkeypatch)
    incoming = _make_event("later", chat_type="group", user_id="new-user")
    reply = None
    arrival_accepted = None
    try:
        if path.startswith("cancel-") or path == "reservation-replaced":
            attempted = adapter._pending_messages.pop("shared")
            adapter._stage_next_queued_event("shared", attempted)
            if path == "reservation-replaced":
                reserve_pending_dispatch(adapter, "shared", incoming)
                with pending_dispatch_scope(adapter, "shared", attempted):
                    release_pending_dispatch(adapter, "shared", attempted)
                assert {
                    key: reservation.event
                    for key, reservation in adapter._pending_dispatch_reservations.items()
                } == {"shared": incoming}
                return
            entered = asyncio.Event()

            async def handler(event):
                if path == "cancel-claimed":
                    release_pending_dispatch(adapter, "shared", event, claimed=True)
                entered.set()
                await asyncio.Event().wait()

            if path in {
                "cancel-claim-race",
                "cancel-claim-complete",
                "cancel-claim-replaced",
            }:

                async def admit(event):
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        if path == "cancel-claim-replaced":
                            reserve_pending_dispatch(adapter, "shared", incoming)
                        return event, event.source, True

                runner._hm_admit_event = admit
                runner._hm_estop_gate = lambda *args: None
                runner._session_key_for_source = lambda source: "shared"
                runner._hm_pending_reply_intercepts = AsyncMock(return_value=None)
                runner._hm_evict_idle_stale_agent = lambda key: None
                runner._is_session_running = lambda key: False
                runner._hm_dispatch_idle_commands = AsyncMock(
                    return_value=(False, None)
                )
                runner._claim_active_session_slot = lambda *args: (None, None)
                runner._hm_rescue_orphaned_fifo = lambda event, source, internal, key: (
                    event,
                    source,
                    internal,
                )
                runner._persist_active_agents = lambda: None
                runner._begin_session_run_generation = lambda key: 1
                async def persist_claimed_input(event, *args):
                    entry = runner.session_store.get_or_create_session(event.source)
                    owner = gateway_input_owner(event, event.source)
                    bind_pending_dispatch_input(entry.session_id, owner)
                    runner.session_store.append_to_transcript(entry.session_id, {
                        "role": "user", "content": event.text,
                        "display_metadata": {"gateway_input_owner": owner},
                    })
                    assert runner.session_store.has_input_owner(entry.session_id, owner)
                    if path != "cancel-claim-complete":
                        raise asyncio.CancelledError
                    return None

                runner._handle_message_with_agent = persist_claimed_input
                runner._run_post_turn_hooks = AsyncMock()
                runner._restore_pending_one_turn_model_override = lambda *args: None
                runner._clear_durable_active_turn = AsyncMock()
                runner._release_running_agent_state = lambda *args, **kwargs: None
                runner._release_turn_lease = lambda *args: None
                handler = runner._handle_message

            adapter.set_message_handler(handler)
            adapter._start_session_processing(attempted, "shared")
            if path != "cancel-before":
                await asyncio.wait_for(entered.wait(), 2)
            await adapter.cancel_session_processing("shared", discard_pending=False)
            assert (
                _events(adapter, runner),
                {
                    key: record.event
                    for key, record in adapter._pending_dispatch_reservations.items()
                },
            ) == (
                expected[1:]
                if path
                in {
                    "cancel-claimed",
                    "cancel-claim-race",
                    "cancel-claim-complete",
                    "cancel-claim-replaced",
                }
                else expected,
                {"shared": incoming} if path == "cancel-claim-replaced" else {},
            )
            return
        if path == "reserved":
            buffered = _make_event(
                "reserved", chat_type="group", user_id="buffered-user"
            )
            assert await adapter._queue_text_debounce("shared", buffered)
            expected.append(deepcopy(buffered))
        if path in {"fifo", "reserved"}:
            runner._enqueue_fifo("shared", incoming, adapter)
        if path == "normal":
            runner._queue_or_replace_pending_event("shared", incoming)
        if path == "queue":
            incoming.text = "/queue later"
            reply = await runner._busy_queue_command(
                incoming, "shared", incoming.source
            )
        if path == "steer":
            incoming.text = "/steer later"
            runner._session_state("shared").turn.agent = _AGENT_PENDING_SENTINEL
            reply = await runner._busy_steer_command(
                incoming, "shared", incoming.source
            )
        if path == "grace":
            runner._session_state("shared").turn.started_ts = time.time()
            monkeypatch.setenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "10000")
            runner._hm_busy_telegram_grace_queue(
                incoming, incoming.source, "shared", "queue"
            )
        if path == "debounce":
            await adapter.handle_message(incoming)
        if path.startswith("redispatch"):
            attempted = adapter._pending_messages.pop("shared")
            if path == "redispatch-idless":
                attempted.message_id = None
                expected[0].message_id = None
            adapter._stage_next_queued_event("shared", attempted)
            if path == "redispatch-arrival":

                async def arrive():
                    runner._enqueue_fifo("shared", incoming, adapter)

                await asyncio.create_task(arrive())
                arrival_accepted = incoming._gateway_accepted
            with pending_dispatch_scope(adapter, "shared", attempted):
                dispatched = (
                    replace(attempted)
                    if path in {"redispatch-rewrite", "redispatch-idless"}
                    else attempted
                )
                runner._queue_or_replace_pending_event("shared", dispatched)
            assert adapter._pending_dispatch_reservations == {}
            incoming = attempted
        assert (
            _events(adapter, runner),
            incoming._gateway_accepted,
            "full" in reply.lower() if reply else None,
            arrival_accepted,
        ) == (
            expected,
            path.startswith("redispatch"),
            True if path in {"queue", "steer"} else None,
            False if path == "redispatch-arrival" else None,
        )
    finally:
        adapter._discard_text_debounce("shared")
        await adapter.cancel_background_tasks()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "external",
        "internal",
        "debounce",
        "queue-copy",
        "steer-copy",
        "buffered-order",
        "buffered-command",
        "buffered-priority",
    ],
)
async def test_queue_receipts_and_refusals_preserve_event_context(
    path, monkeypatch: pytest.MonkeyPatch
):
    copying = path.endswith("-copy")
    adapter, runner, expected = _setup(
        1 if path.startswith("buffered-") else 0 if copying else 32, monkeypatch
    )
    event = _make_event("later", chat_type="group", user_id="new-user")
    event.metadata = {"hermes_plugin_id": "example", "routing": "preserved"}
    event.reply_to_message_id, event.reply_to_text = "quote", "quoted text"
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        if path.startswith("buffered-"):
            buffered = _make_event(
                "buffered", chat_type="group", user_id="buffered-user"
            )
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
            monkeypatch.setattr(event, "_bot_loop_admitted", True, raising=False)
            runner._session_state("shared").turn.agent = _AGENT_PENDING_SENTINEL
            command = (
                runner._busy_queue_command
                if path == "queue-copy"
                else runner._busy_steer_command
            )
            await command(event, "shared", event.source)
            assert (
                _events(adapter, runner),
                getattr(_events(adapter, runner)[0], "_bot_loop_admitted", False),
            ) == ([replace(event, text="later", message_type=MessageType.TEXT)], True)
            return
        if path == "internal":
            event.internal = True
            with pytest.raises(WakeNotAccepted):
                await admit_internal_event(adapter, event)
        else:
            if path == "debounce":
                runner._busy_text_mode = "queue"
            await adapter.handle_message(event)
        notices = runner._send_busy_reply.await_args_list
        assert (
            _events(adapter, runner),
            event._gateway_accepted,
            len(notices),
            "full" in notices[0].args[2].lower() if notices else None,
        ) == (
            expected,
            False,
            0 if path == "internal" else 1,
            None if path == "internal" else True,
        )
    finally:
        adapter._discard_text_debounce("shared")


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_cancel_timeout_preserves_only_non_durable_input_on_completion(
    tmp_path, monkeypatch, durable
):
    from gateway.config import GatewayConfig
    from gateway.platforms import base
    from gateway.platforms.base_pending import bind_pending_dispatch_input
    from gateway.session import SessionStore
    from hermes_state import SessionDB

    adapter, runner, expected = _setup(3, monkeypatch)

    class _CancellationRunner(_AdmissionRunner):
        def __init__(self, adapter, states):
            super().__init__(adapter)
            self.states = states
            self.session_store = SessionStore(
                tmp_path / "sessions", GatewayConfig(multiplex_profiles=False)
            )

    runner = _CancellationRunner(adapter, runner.states)
    adapter.gateway_runner = runner
    db = SessionDB()
    db.create_session("straggler", "gateway")
    attempted = adapter._pending_messages.pop("shared")
    adapter._stage_next_queued_event("shared", attempted)
    entered, cancelled, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    wait_for = asyncio.wait_for

    async def bounded_wait(awaitable, timeout):
        return await wait_for(awaitable, 2 if timeout == 5.0 else timeout)

    monkeypatch.setattr(base.asyncio, "wait_for", bounded_wait)

    async def handler(event):
        bind_pending_dispatch_input("straggler", "straggler-owner")
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await finish.wait()

    adapter.set_message_handler(handler)
    assert adapter._start_session_processing(attempted, "shared")
    task = adapter._session_tasks["shared"]
    try:
        await wait_for(entered.wait(), 5)
        await adapter.cancel_session_processing("shared", discard_pending=False)
        assert (cancelled.is_set(), task.done(), _events(adapter, runner)) == (
            True,
            False,
            expected[1:],
        )
        if durable:
            db.append_message(
                "straggler",
                "user",
                attempted.text,
                display_metadata={"gateway_input_owner": "straggler-owner"},
            )
        finish.set()
        await wait_for(asyncio.shield(task), 5)
        assert (
            _events(adapter, runner),
            adapter._pending_dispatch_reservations,
            task.done(),
        ) == (expected[1:] if durable else expected, {}, True)
    finally:
        finish.set()
        await wait_for(asyncio.shield(task), 5)
        await adapter.cancel_background_tasks()
        runner.session_store.close_all_db_handles()
        db.close()
