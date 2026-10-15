"""Busy same-sender bursts become one shared-runtime admission.

Main merged a busy photo burst into the pending slot and debounced queue-mode text from one sender
(``BasePlatformAdapter._queue_text_debounce``) so a thought split over two messages ran as one
turn. The authority FIFO admits each input as it arrives, so the burst is held here for main's
bounded window first, then committed once through ``dispatch_shared_busy``. Every absorbed caller
returns only after that commit, with its acceptance copied from it (committed before ACK), and
input that arrives later never commits before the burst.
"""
import asyncio
from dataclasses import dataclass, field
from functools import partial
import time

from gateway.platforms.event import MessageEvent, MessageType


@dataclass(eq=False)
class _Burst:
    head: MessageEvent
    done: asyncio.Future
    first_ts: float
    last_ts: float
    timer: asyncio.Task | None = None
    commit: asyncio.Task | None = None
    absorbed: list = field(default_factory=list)


def _kind(adapter, event):
    if event.message_type == MessageType.PHOTO and not event.internal:
        return 'photo'
    return 'text' if adapter._is_queue_text_debounce_candidate(event) else None


def _state(adapter, name):
    return adapter.__dict__.setdefault(name, {})


def holds_busy_burst(adapter, session_key):
    """A burst is held or committing: later input for the route must queue behind it."""
    return any(session_key in adapter.__dict__.get(name, ()) for name in ('_shared_bursts', '_shared_burst_commits'))


async def dispatch_busy_burst(adapter, event, session_key):
    """``dispatch_shared_busy`` for one busy input, merging a same-sender PHOTO+PHOTO or
    queue-mode TEXT+TEXT burst first."""
    from gateway.platforms.base import merge_pending_message_event
    from gateway.session_ingress import dispatch_shared_busy
    kind, bursts = _kind(adapter, event), _state(adapter, '_shared_bursts')
    while (burst := bursts.get(session_key)) is not None:
        if kind is not None and kind == _kind(adapter, burst.head) and adapter._can_merge_text_debounce_events(
                burst.head, event):
            merge_pending_message_event({session_key: burst.head}, session_key, event, merge_text=kind == 'text')
            burst.absorbed.append(event)
            burst.last_ts = time.monotonic()
            _arm(adapter, session_key, burst)
            return await _accepted(burst, event)
        await asyncio.shield(_flush(adapter, session_key, burst))  # a different sender or kind: in order
    previous = _state(adapter, '_shared_burst_commits').get(session_key)
    if kind is None:
        if previous is not None:
            await asyncio.wait({previous})
        return await dispatch_shared_busy(adapter, event, session_key)
    now = time.monotonic()
    burst = bursts[session_key] = _Burst(event, asyncio.get_running_loop().create_future(), now, now)
    _arm(adapter, session_key, burst)
    return await _accepted(burst, event)


async def _accepted(burst, event):
    await asyncio.shield(burst.done)  # raises the merged admission's own refusal
    event._gateway_accepted = burst.head._gateway_accepted


def _track(adapter, task):
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    return task


def _arm(adapter, session_key, burst):
    if burst.timer is not None:
        burst.timer.cancel()
    delay = max(0.0, min(burst.last_ts + adapter._busy_text_debounce_seconds,
                         burst.first_ts + adapter._busy_text_hard_cap_seconds) - time.monotonic())

    async def fire():
        await asyncio.sleep(delay)
        await asyncio.shield(_flush(adapter, session_key, burst))
    burst.timer = _track(adapter, asyncio.create_task(fire()))


def _flush(adapter, session_key, burst):
    """Start committing ``burst`` (once), after any earlier commit for the route."""
    if burst.commit is None:
        bursts, tails = _state(adapter, '_shared_bursts'), _state(adapter, '_shared_burst_commits')
        if bursts.get(session_key) is burst:
            del bursts[session_key]
        if burst.timer is not asyncio.current_task():
            burst.timer.cancel()
        burst.commit = tails[session_key] = _track(adapter, asyncio.create_task(
            _commit(adapter, session_key, burst, tails.get(session_key))))
        burst.commit.add_done_callback(lambda task: tails.get(session_key) is task and tails.pop(session_key))
        burst.commit.add_done_callback(partial(_settle, burst))
    return burst.commit


async def _commit(adapter, session_key, burst, previous):
    from gateway.session_ingress import dispatch_shared_busy
    if previous is not None:
        await asyncio.wait({previous})
    await dispatch_shared_busy(adapter, burst.head, session_key)


def _settle(burst, task):
    """Every absorbed caller sees the merged admission's outcome, refusal or cancellation."""
    if burst.done.done():
        return
    if task.cancelled():
        burst.done.cancel()
    elif task.exception() is not None:
        burst.done.set_exception(task.exception())
    else:
        burst.done.set_result(None)


def discard_busy_burst(adapter, session_key):
    """``/stop`` / ``/new`` drop a held burst as main dropped its debounce buffer; nothing was ACKed."""
    burst = adapter.__dict__.get('_shared_bursts', {}).pop(session_key, None)
    if burst is not None:
        burst.timer.cancel()
        burst.done.done() or burst.done.set_result(None)
