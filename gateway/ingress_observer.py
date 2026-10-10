"""``gateway_ingress_observer``: an observer-only audit stream of what an adapter received.

Fire sites number each event of a connection (an *epoch*) gap-free and hand it to one
process-wide queue without waiting; a single daemon thread delivers it to the plugin callbacks
in the profile scope captured when the epoch opened. The gateway never waits for, retries or
changes behaviour because of an observer, and every structure this path owns is bounded: the
queue (events and accounted bytes), each connection's fetch association map, the per-epoch seal
state (live epochs plus a few closed ones) and the constant-size fault records.

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

Build = Callable[..., tuple[dict[str, Any], int]]


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


@dataclass(slots=True)
class _Event:
    epoch: IngressEpoch
    event_no: int
    kind: str
    created_at: float
    generation: int
    fields: dict[str, Any]
    fault: Optional[_Fault]
    size: int


@dataclass(slots=True)
class _Seal:
    """Delivery-thread state of one epoch: the pending seal and the dispatcher's own faults."""

    prev: Optional[dict[str, Any]] = None
    first_event_no: Optional[int] = None
    kinds: frozenset[str] = frozenset()


def _no_fields(epoch: IngressEpoch, event_no: int) -> tuple[dict[str, Any], int]:
    return {}, 0


def _end_fields(epoch: IngressEpoch, event_no: int, steps_clean: bool) -> tuple[dict[str, Any], int]:
    return {"steps_clean": steps_clean}, 0


class IngressEpoch:
    """Producer side of one adapter connection. Fire sites run on the adapter's event loop and
    never block or raise; a closed epoch refuses every later event."""

    def __init__(self, platform: str, bot_id: str) -> None:
        from hermes_cli.plugins import get_plugin_manager

        self.platform = platform
        self.bot_id = bot_id
        self.epoch_id = secrets.token_hex(8)
        self.closed = False
        self.evicted = False  # written and read by the delivery thread only
        # The owning profile's plugins and scope, resolved where connect() runs.
        self.plugins = get_plugin_manager()
        self.context = contextvars.copy_context()
        self._event_no = 0
        self._fault: Optional[_Fault] = None
        self._associations: OrderedDict[int, _Association] = OrderedDict()
        self._dispatcher = _dispatcher

    def emit(self, kind: str, generation: int, build: Build = _no_fields, *args: Any) -> None:
        """Number one event and enqueue it. ``build(epoch, event_no, *args)`` returns the
        kind-specific fields and their accounted size; it runs only when a callback exists."""
        if self.closed:
            return
        # Numbered even with no observer, so a callback registered mid-epoch sees the gap.
        self._event_no += 1
        try:
            if not self.plugins.has_hook(HOOK):
                return
            fields, size = build(self, self._event_no, *args)
            event = _Event(self, self._event_no, kind, time.time(), generation, fields, self._fault,
                           _EVENT_BYTES + size)
            if self._dispatcher.put(event):
                return
        except Exception:
            logger.debug("%s: %s event %d not captured", HOOK, kind, self._event_no, exc_info=True)
        self._record_drop()

    def end(self, generation: int, steps_clean: bool) -> None:
        """Emit the epoch's last event; called once the transport has been stopped."""
        self.emit("end", generation, _end_fields, steps_clean)
        self.close()

    def close(self) -> None:
        """Refuse every later event; nothing can be observed any more, so drop the associations."""
        self.closed = True
        self._associations.clear()

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


def _deliver(manager: Any, payload: dict[str, Any]) -> str:
    """Run every registered callback; the outcome is ``failed`` if any raised (or none exist),
    ``late`` if they took longer than ``_LATE_AFTER_SECONDS``, else ``ok``."""
    callbacks = manager.iter_hook_callbacks(HOOK)
    outcome = "ok" if callbacks else "failed"
    started = time.monotonic()
    for callback in callbacks:
        try:
            manager._invoke_hook_callback(callback, payload)
        except (Exception, SystemExit) as exc:  # health: allow BLE001 -- plugin boundary; reported warn-once
            manager._report_hook_failure(HOOK, callback, payload, exc)
            outcome = "failed"
    if outcome == "ok" and time.monotonic() - started > _LATE_AFTER_SECONDS:
        return "late"
    return outcome


class _Dispatcher:
    """The process-wide bounded queue and its one delivery thread, started on first use and
    reused for every epoch. Shutdown never waits for it."""

    def __init__(self) -> None:
        self._ready = threading.Condition()
        self._events: deque[_Event] = deque()
        self._bytes = 0
        self._thread: Optional[threading.Thread] = None
        self._seals: dict[IngressEpoch, _Seal] = {}  # delivery thread only

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
                while not self._events:
                    self._ready.wait()
                event = self._events.popleft()
                self._bytes -= event.size
            self._dispatch(event)

    def _dispatch(self, event: _Event) -> None:
        epoch = event.epoch
        seal = self._seals.get(epoch)
        if seal is None and not epoch.evicted:
            seal = self._admit(epoch)
        payload = {
            "platform": epoch.platform, "bot_id": epoch.bot_id, "epoch": epoch.epoch_id, "event_no": event.event_no,
            "kind": event.kind, "created_at": event.created_at, "generation": event.generation,
            # An evicted epoch fails closed: no seal, and it can never produce a clean end.
            "epoch_state": "evicted" if seal is None else "live", "prev": None if seal is None else seal.prev,
            "fault": _fault_record(event.fault, seal), **event.fields,
        }
        outcome = epoch.context.run(_deliver, epoch.plugins, payload)
        if seal is None:
            return
        if event.kind == "end":
            del self._seals[epoch]
            return
        seal.prev = {"event_no": event.event_no, "outcome": outcome}
        if outcome != "ok":
            seal.kinds |= {outcome}
            if seal.first_event_no is None:
                seal.first_event_no = event.event_no

    def _admit(self, epoch: IngressEpoch) -> _Seal:
        """Start an epoch's seal state, evicting the oldest closed epochs beyond the retained few."""
        seal = self._seals[epoch] = _Seal()
        closed = [known for known in self._seals if known.closed]
        for old in closed[:-_MAX_CLOSED_EPOCHS]:
            del self._seals[old]
            old.evicted = True
        return seal


_dispatcher = _Dispatcher()
