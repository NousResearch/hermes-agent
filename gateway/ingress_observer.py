"""``gateway_ingress_observer``: an observer-only audit stream of what an adapter received.

Fire sites number each event of a connection (an *epoch*) gap-free and hand it to one
process-wide queue without waiting; one daemon thread delivers it to the plugin callbacks in the
profile scope captured when the epoch opened, each callback getting its own read-only copy. The
gateway never waits for, retries or changes behaviour because of an observer, and every structure
this path owns is bounded: the queue (events and accounted bytes), each connection's fetch
association map, the per-epoch seal state (open epochs plus a few closed ones) and the
constant-size fault records.

Each delivery carries ``prev``, the outcome of the previous *delivered* event of its epoch
(an event's own outcome is only ever published by its successor), and the epoch's sticky
``fault`` record. A number that never arrives is an event the observer did not receive.
"""

from __future__ import annotations

import contextvars
import logging
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

HOOK = "gateway_ingress_observer"
MAX_CONTENT_BYTES = 64 * 1024  # per event; above it content is omitted, never truncated
_MAX_QUEUED_EVENTS = 1024
_MAX_QUEUED_BYTES = 4 * 1024 * 1024
_EVENT_BYTES = 256  # accounted per event on top of what its builder reports
_MAX_ASSOCIATIONS = 4096  # per connection
_MAX_CLOSED_EPOCHS = 16
_COUNTER_CAP = 2**31 - 1  # counters saturate here instead of wrapping
_LATE_AFTER_SECONDS = 0.25
_IDLE_EXIT_SECONDS = 60.0  # an idle delivery thread exits; the next event starts it again

Build = Callable[..., tuple[dict[str, Any], int]]


@dataclass(frozen=True, eq=False, slots=True)
class _Stream:
    """All delivery needs of an epoch: its identity and the owning profile's plugins and scope.
    Queued events hold only this, never the connection's producer state."""

    platform: str
    bot_id: str
    epoch_id: str
    plugins: Any
    context: contextvars.Context


@dataclass(frozen=True, slots=True)
class _Fault:
    """Producer-side drop record; replaced, never mutated, so an event can carry it by reference."""

    first_event_no: int
    kinds: frozenset[str]
    dropped_count: int


@dataclass(slots=True)
class _Association:
    fetch_event_no: int
    raw_sha256: str
    refetches: int = 0
    conflict: bool = False


@dataclass(frozen=True, slots=True)
class _Event:
    stream: _Stream
    event_no: int
    kind: str
    created_at: float
    generation: int
    fields: dict[str, Any]
    fault: Optional[_Fault]
    size: int


@dataclass(slots=True)
class _Seal:
    """Dispatcher state of one epoch: the pending seal and the dispatcher's own faults."""

    prev: Optional[tuple[int, str]] = None  # (event_no, outcome) of the last dispatched event
    first_event_no: Optional[int] = None
    kinds: frozenset[str] = frozenset()
    closed: bool = False


class _ReadOnlyDict(dict):
    """A dict no callback can edit: every payload field is a statement by the gateway."""

    def _refuse(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(f"{HOOK} payloads are read-only")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _refuse

    def __reduce__(self):
        return type(self), (dict(self),)


def _read_only(value: Any) -> Any:
    """A deep read-only copy of JSON-shaped data; built per callback, so none shares an object."""
    if isinstance(value, dict):
        return _ReadOnlyDict({key: _read_only(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_read_only(item) for item in value)
    return value


def _no_fields(epoch: IngressEpoch, event_no: int) -> tuple[dict[str, Any], int]:
    return {}, 0


def _end_fields(epoch: IngressEpoch, event_no: int, steps_clean: bool) -> tuple[dict[str, Any], int]:
    return {"steps_clean": steps_clean}, 0


class IngressEpoch:
    """Producer side of one adapter connection. Fire sites run on the adapter's event loop and
    never block or raise; a closed epoch refuses every later event."""

    def __init__(self, platform: str, bot_id: str) -> None:
        from hermes_cli.plugins import get_plugin_manager

        # The owning profile's plugins and scope, resolved where connect() runs.
        self.stream = _Stream(platform, bot_id, secrets.token_hex(8), get_plugin_manager(), contextvars.copy_context())
        self.closed = False
        self._event_no = 0
        self._fault: Optional[_Fault] = None
        self._associations: OrderedDict[int, _Association] = OrderedDict()
        self._dispatcher = _dispatcher
        self._dispatcher.open(self.stream)

    def emit(self, kind: str, generation: int, build: Build = _no_fields, *args: Any) -> None:
        """Number one event and enqueue it. ``build(epoch, event_no, *args)`` returns the
        kind-specific fields and their accounted size; it runs only when a callback exists."""
        if self.closed:
            return
        # Numbered even with no observer, so a callback registered mid-epoch sees the gap.
        self._event_no += 1
        try:
            if not self.stream.plugins.has_hook(HOOK):
                return
            fields, size = build(self, self._event_no, *args)
            event = _Event(self.stream, self._event_no, kind, time.time(), generation, fields, self._fault,
                           _EVENT_BYTES + size)
            if self._dispatcher.put(event):
                return
        except Exception:
            logger.debug("%s: %s event %d not captured", HOOK, kind, self._event_no, exc_info=True)
        self._record_drop()

    def end(self, generation: int, steps_clean: bool) -> None:
        """Emit the epoch's last event; called once the transport's stop steps are over."""
        self.emit("end", generation, _end_fields, steps_clean)
        self.close()

    def close(self) -> None:
        """Refuse every later event, drop the associations and retire the epoch's seal state
        under the closed-epoch bound. Idempotent."""
        self.closed = True
        self._associations.clear()
        self._dispatcher.close(self.stream)

    def associate(self, update_id: int, event_no: int, raw_sha256: str) -> None:
        """Pin a fetched update to its first unresolved fetch; an identical refetch only counts."""
        entry = self._associations.get(update_id)
        if entry is None:
            self._associations[update_id] = _Association(event_no, raw_sha256)
            if len(self._associations) > _MAX_ASSOCIATIONS:
                self._associations.popitem(last=False)
        elif entry.raw_sha256 != raw_sha256:
            entry.conflict = True
        elif entry.refetches < _COUNTER_CAP:
            entry.refetches += 1

    def resolve(self, update_id: int) -> dict[str, Any]:
        """Retire an update's fetch association once it is observed."""
        entry = self._associations.pop(update_id, None)
        if entry is None:
            return {"fetch_event_no": None, "raw_sha256": None, "refetches": 0, "association": "missing"}
        return {"fetch_event_no": entry.fetch_event_no, "raw_sha256": entry.raw_sha256, "refetches": entry.refetches,
                "association": "conflict" if entry.conflict else "ok"}

    def _record_drop(self) -> None:
        fault = self._fault
        if fault is None:
            fault = _Fault(self._event_no, frozenset(), 0)
        elif fault.dropped_count == _COUNTER_CAP:
            return
        dropped = fault.dropped_count + 1
        kinds = fault.kinds | ({"dropped", "saturated"} if dropped == _COUNTER_CAP else {"dropped"})
        self._fault = _Fault(fault.first_event_no, kinds, dropped)


def _fault_record(fault: Optional[_Fault], seal: Optional[_Seal]) -> Optional[dict[str, Any]]:
    """Merge the producer record an event carries with the dispatcher's own record of its epoch."""
    first = fault.first_event_no if fault else None
    kinds = fault.kinds if fault else frozenset()
    if seal is not None and seal.kinds:
        kinds = kinds | seal.kinds
        first = seal.first_event_no if first is None else min(first, seal.first_event_no)
    if not kinds:
        return None
    return {"first_event_no": first, "kinds": sorted(kinds), "dropped_count": fault.dropped_count if fault else 0}


def _deliver(manager: Any, payload: dict[str, Any]) -> tuple[str, Optional[BaseException]]:
    """Run every registered callback on its own read-only copy of ``payload``. The outcome is
    ``failed`` if any raised or was cancelled (or none is registered), ``late`` if together
    they took longer than ``_LATE_AFTER_SECONDS``, else ``ok``; plus the first error."""
    callbacks = manager.iter_hook_callbacks(HOOK)
    if not callbacks:
        return "failed", None
    error = None
    started = time.monotonic()
    for callback in callbacks:
        try:
            manager._invoke_hook_callback(callback, _read_only(payload))
        except BaseException as exc:  # health: allow BLE001 -- plugin boundary: an exception, exit or cancellation fails the event, never the delivery thread
            error = error or exc
    if error is not None:
        return "failed", error
    return ("late" if time.monotonic() - started > _LATE_AFTER_SECONDS else "ok"), None


class _Dispatcher:
    """The process-wide bounded queue, its delivery thread (at most one, started on demand) and
    the seal state of every open epoch plus at most ``_MAX_CLOSED_EPOCHS`` closed ones. Shutdown
    never waits for it."""

    def __init__(self) -> None:
        self._ready = threading.Condition()
        self._events: deque[_Event] = deque()
        self._bytes = 0
        self._thread: Optional[threading.Thread] = None
        self._seals: dict[_Stream, _Seal] = {}

    def open(self, stream: _Stream) -> None:
        with self._ready:
            self._seals[stream] = _Seal()

    def close(self, stream: _Stream) -> None:
        """Mark an epoch closed and evict the oldest closed epochs beyond the bound; their queued
        events are then delivered as ``evicted`` and can never produce a clean end."""
        with self._ready:
            seal = self._seals.get(stream)
            if seal is None:
                return
            seal.closed = True
            closed = [known for known, state in self._seals.items() if state.closed]
            for old in closed[:-_MAX_CLOSED_EPOCHS]:
                del self._seals[old]

    def put(self, event: _Event) -> bool:
        with self._ready:
            if len(self._events) >= _MAX_QUEUED_EVENTS or self._bytes + event.size > _MAX_QUEUED_BYTES:
                return False
            self._events.append(event)
            self._bytes += event.size
            if self._thread is None:
                # health: allow HX012 -- shared by every profile; each delivery runs in its epoch's captured context
                self._thread = threading.Thread(target=self._run, name="gateway-ingress-observer", daemon=True)
                self._thread.start()
            self._ready.notify()
        return True

    def _run(self) -> None:
        while True:
            with self._ready:
                if not self._ready.wait_for(lambda: self._events, _IDLE_EXIT_SECONDS):
                    self._thread = None
                    return
                event = self._events.popleft()
                self._bytes -= event.size
            self._dispatch(event)

    def _dispatch(self, event: _Event) -> None:
        stream = event.stream
        with self._ready:
            seal = self._seals.get(stream)
            prev = None if seal is None else seal.prev
            fault = _fault_record(event.fault, seal)
        payload = {
            "platform": stream.platform, "bot_id": stream.bot_id, "epoch": stream.epoch_id,
            "event_no": event.event_no, "kind": event.kind, "created_at": event.created_at,
            "generation": event.generation,
            # An evicted epoch fails closed: no seal, and it can never produce a clean end.
            "epoch_state": "evicted" if seal is None else "live",
            "prev": None if prev is None else {"event_no": prev[0], "outcome": prev[1]},
            "fault": fault, **event.fields,
        }
        outcome, error = stream.context.run(_deliver, stream.plugins, payload)
        with self._ready:
            current = seal is not None and self._seals.get(stream) is seal  # not evicted before or during delivery
            first_failure = current and outcome == "failed" and "failed" not in seal.kinds
            if current and event.kind == "end":
                del self._seals[stream]
            elif current:
                seal.prev = (event.event_no, outcome)
                if outcome != "ok":
                    seal.kinds |= {outcome}
                    if seal.first_event_no is None:
                        seal.first_event_no = event.event_no
        if error is not None:
            # Warn once per epoch; the fault record carries every later failure.
            logger.log(logging.WARNING if first_failure else logging.DEBUG,
                       "%s callback failed (%s bot %s, epoch %s, event %d)", HOOK, stream.platform, stream.bot_id,
                       stream.epoch_id, event.event_no, exc_info=error)


_dispatcher = _Dispatcher()
