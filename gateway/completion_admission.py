"""Execution accounting for durable, gateway-owned delegation completions."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
import logging

from hermes_constants import get_hermes_home, reset_hermes_home_override, set_hermes_home_override

logger = logging.getLogger("gateway.run")
_completion_turn = ContextVar("gateway_completion_turn", default=None)


def completion_receipts(event):
    if getattr(event, "internal", False) is not True:
        return ()
    receipts = getattr(event, "_gateway_completion_receipts", ())
    return receipts if isinstance(receipts, tuple) else ()


def must_use_cold_handler(event=None):
    # The recursive queued path bypasses pinned-parent resolution and per-event accounting.
    return _completion_turn.get() is not None or bool(completion_receipts(event))


@contextmanager
def _receipt_home(event):
    token = set_hermes_home_override(event._gateway_completion_home)
    try:
        yield
    finally:
        reset_hermes_home_override(token)


def attach_completion_receipts(event, receipts, events, adapter):
    event._gateway_completion_receipts = tuple(receipts)
    event._gateway_completion_events = tuple(events)
    event._gateway_completion_home = str(get_hermes_home())
    event._gateway_completion_adapter = adapter


def begin_completion_event(event):
    receipts = completion_receipts(event)
    if not receipts:
        return True
    from tools.async_delegation_admission import begin_queued_completion_deliveries
    with _receipt_home(event):
        started = begin_queued_completion_deliveries(receipts)
    event._gateway_completion_processing_started = started
    event._gateway_completion_stale = not started
    return started


async def completion_owner_is_current(runner, event, session_key, run_generation):
    if not completion_receipts(event):
        return True
    if not runner._is_session_run_current(session_key, run_generation):
        discard_completion_event(event)
        event._gateway_completion_stale = True
        return False
    current = runner.session_store.lookup_by_session_key(session_key)
    if current is None or current.suspended:
        return False
    expected = event._gateway_completion_resolved_session_id
    resolved = current.session_id
    if resolved != expected:
        def compression_tip():
            return runner.session_store._db_for_key(session_key).get_compression_tip(expected)
        if await runner._run_in_executor_with_context(compression_tip) != resolved:
            return False
    current = runner.session_store.lookup_by_session_key(session_key)
    return (current is not None and not current.suspended and current.session_id == resolved
            and runner._is_session_run_current(session_key, run_generation))


def discard_completion_event(event):
    receipts = completion_receipts(event)
    if receipts:
        from tools.async_delegation_admission import settle_queued_completion_deliveries
        with _receipt_home(event):
            settle_queued_completion_deliveries(receipts, "dropped")


def retry_completion_event(runner, event, *, refund=False):
    receipts = completion_receipts(event)
    if not receipts:
        return
    from tools.async_delegation import get_durable_delegation
    from tools.async_delegation_admission import settle_queued_completion_deliveries
    from tools.process_registry import process_registry
    with _receipt_home(event):
        changed = settle_queued_completion_deliveries(receipts, "pending" if refund else "retry")
    if not changed:
        return
    for original in event._gateway_completion_events:
        identity = runner._completion_delivery_identity(original)
        with runner._completion_delivery_lock:
            runner._completion_deliveries_inflight.discard(identity)
            runner._completion_deliveries_delivered.pop(identity, None)
        with _receipt_home(event):
            row = get_durable_delegation(original["delegation_id"])
        if row and row["delivery_state"] == "pending":
            process_registry.completion_queue.put(original)


def release_adapter_completions(adapter):
    """Return only this transport's unstarted work before replacement clears its queues."""
    runner = adapter.gateway_runner
    for task in tuple(adapter._background_tasks):
        event = getattr(task, "_gateway_completion_event", None)
        if getattr(event, "_gateway_completion_adapter", None) is adapter:
            retry_completion_event(runner, event, refund=True)
    for event in tuple(adapter._pending_messages.values()):
        retry_completion_event(runner, event, refund=True)
    for events in getattr(runner, "_queued_events", {}).values():
        if not isinstance(events, list):
            continue
        for event in tuple(events):
            if getattr(event, "_gateway_completion_adapter", None) is adapter:
                retry_completion_event(runner, event, refund=True)
                events[:] = [item for item in events if item is not event]


def _still_queued(runner, event):
    original = event._gateway_completion_events[0]
    key = str(original.get("session_key") or "")
    adapter = runner._adapter_for_source(event.source)
    if adapter is not None and getattr(adapter, "_pending_messages", {}).get(key) is event:
        return True
    state = runner._peek_session_state(key)
    return state is not None and any(item is event for item in state.conversation.queued_events)


async def _settle_event(runner, event):
    from tools.async_delegation_admission import settle_queued_completion_deliveries
    receipts = completion_receipts(event)
    if getattr(event, "_gateway_completion_stale", False):
        return
    if getattr(event, "_gateway_completion_processing_started", False):
        outcome = "delivered" if getattr(event, "_gateway_completion_processing_ok", False) else "unknown"
        with _receipt_home(event):
            changed = settle_queued_completion_deliveries(receipts, outcome)
        if changed and outcome == "unknown":
            logger.warning("Async completion handling did not finish; retaining %d result(s) without automatic replay", changed)
        return
    if _still_queued(runner, event):
        return
    original = event._gateway_completion_events[0]
    parent = str(original.get("parent_session_id") or "")
    outcome = "pending" if getattr(event, "_gateway_completion_refused", False) else "retry"
    if parent:
        async with runner._completion_event_scope(original):
            if await runner._classify_completion_target(parent) == "terminal":
                outcome = "dropped"
    if outcome == "dropped":
        discard_completion_event(event)
        return
    retry_completion_event(runner, event, refund=outcome == "pending")


async def handle_completion_event(runner, handler, event, *, start_hook=None):
    if not completion_receipts(event) or _completion_turn.get() is event:
        if start_hook is not None:
            await start_hook()
        return await handler(event)
    token = _completion_turn.set(event)
    task = asyncio.current_task()
    previous_event = getattr(task, "_gateway_completion_event", None)
    if task is not None:
        setattr(task, "_gateway_completion_event", event)
    try:
        if start_hook is not None:
            await start_hook()
        return await handler(event)
    except asyncio.CancelledError:
        if not getattr(event, "_gateway_completion_processing_started", False):
            if getattr(asyncio.current_task(), "_gateway_completion_user_cancelled", False):
                discard_completion_event(event)
                event._gateway_completion_stale = True
            else:
                event._gateway_completion_refused = True
        raise
    finally:
        try:
            await _settle_event(runner, event)
        finally:
            if task is not None:
                setattr(task, "_gateway_completion_event", previous_event)
            _completion_turn.reset(token)
