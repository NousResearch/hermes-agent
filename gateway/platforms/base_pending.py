"""Pending attribution, dispatch ownership and withdrawal before a turn starts.

Before a message reaches the agent, it can wait in one of the adapter's buffers: a text batch, the
busy-text debounce buffer or the pending slot (with the runner's FIFO overflow behind the slot).
Buffers merge several messages into one event. ``merge_recorded`` records the parts of a merged
event, so that ``withdraw_from_event`` can rebuild it without one message by replaying the merges
of the remaining parts in their original order. The replay reuses each recorded merge function and
does not repeat the choice that selected it. A photo followed by two texts merges into one event
in the pending slot, and withdrawing the photo leaves one event with both texts, although the two
texts alone would have queued as two turns.

``PendingWithdrawalMixin`` declares the attributes that its host (``BasePlatformAdapter``)
provides.

No imports from ``gateway.platforms.base``: it imports this module.
"""

from __future__ import annotations

import asyncio
import logging
import copy
import itertools
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Tuple

from gateway.native_message_deletion import NativeMessageDeletion
from gateway.pending_native import PendingNativeInput
from gateway.platforms.event import MessageEvent
from gateway.session_identity import identity_of


logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gateway.config import Platform


_SECURITY_METADATA_KEYS = (
    "hermes_plugin_id",
    "hermes_plugin_injection",
    "gateway_session_key",
    "gateway_session_id",
    "gateway_session_strict",
    "notification_category",
)

_INGRESS_SEQUENCE = itertools.count()


def ingress_order(event: MessageEvent) -> int:
    if event._ingress_order is None:
        event._ingress_order = next(_INGRESS_SEQUENCE)
    return event._ingress_order


def _sender_identity(event: MessageEvent) -> tuple[str, ...] | None:
    source = event.source
    if source is None:
        return None
    platform = str(getattr(source.platform, "value", source.platform) or "").lower()
    sender = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
    if sender:
        return (platform, str(sender))
    if source.chat_type in {"dm", "private"} and source.chat_id:
        return (platform, "dm", str(source.chat_id))
    return None


def same_message_sender(first: MessageEvent, second: MessageEvent) -> bool:
    """Whether two events have the same known platform user or private chat."""
    sender = _sender_identity(first)
    return sender is not None and sender == _sender_identity(second)


def _same_pending_security_context(first: MessageEvent, second: MessageEvent) -> bool:
    return (
        first.internal == second.internal
        and first.allow_gateway_control == second.allow_gateway_control
        and all(
            (first.metadata or {}).get(key) == (second.metadata or {}).get(key)
            for key in _SECURITY_METADATA_KEYS
        )
    )


def can_join_pending_event(first: MessageEvent, second: MessageEvent) -> bool:
    """Whether coalescing preserves sender, control permissions and reply context."""
    return (
        same_message_sender(first, second)
        and not first._queue_at_turn_boundary
        and not second._queue_at_turn_boundary
        and _same_pending_security_context(first, second)
        and not first.reply_context_conflicts(second)
    )


@dataclass
class _PendingDispatchReservation:
    event: MessageEvent
    claimed: bool = False
    preserve_on_completion: bool = False
    withdrawal_closed: bool = False
    input_session_id: str | None = None
    input_owner: str | None = None
    withdrawn: bool = False
    revision: int = 0
    accepted: bool = True
    task: asyncio.Task | None = None
    previous: _PendingDispatchReservation | None = None
    aliases: list[MessageEvent] = field(default_factory=list)

    def includes(self, event: MessageEvent) -> bool:
        return self.event is event or any(alias is event for alias in self.aliases)

    def bind(self, event: MessageEvent) -> None:
        if self.includes(event):
            return
        event._merged_parts = self.event._merged_parts
        event._pending_native_input = self.event._pending_native_input
        for attr in ("_pending_snapshot_uid", "_gateway_input_owner", "_gateway_pending_stt_text", "_gateway_pending_stt_transcripts",
                     "_gateway_pending_stt_clips", "_gateway_pending_stt_input", "_gateway_pending_stt_echoed_paths"):
            if hasattr(self.event, attr):
                setattr(event, attr, getattr(self.event, attr))
        self.aliases.append(event)

    def withdraw(self, withdraw: Withdraw) -> bool:
        if self.claimed or self.withdrawn or self.withdrawal_closed:
            return False
        matched, remaining = withdraw(self.event)
        if not matched:
            return False
        self.revision += 1
        if remaining is None:
            self.withdrawn = True
            return True
        # Preparers share this event while awaiting provider work. Replacing its
        # identity would leave those callers with the removed contribution.
        echoed = set(getattr(remaining, "_gateway_pending_stt_echoed_paths", ()))
        for event in (self.event, *self.aliases):
            for item in fields(event):
                if item.init:
                    setattr(event, item.name, getattr(remaining, item.name))
            event._merged_parts = remaining._merged_parts
            event._prepared_inbound = None
            event._pending_native_input = remaining._pending_native_input
            for attr in ("_gateway_pending_stt_text", "_gateway_pending_stt_transcripts", "_gateway_pending_stt_clips", "_gateway_pending_stt_input"):
                if hasattr(event, attr):
                    delattr(event, attr)
            setattr(event, "_gateway_pending_stt_echoed_paths", echoed.intersection(event.media_urls))
        return True


@dataclass(frozen=True)
class _PendingDispatch:
    adapter: object
    session_key: str
    event: MessageEvent
    task: asyncio.Task | None
    reservation: _PendingDispatchReservation | None


_dispatch: ContextVar[_PendingDispatch | None] = ContextVar(
    "pending_dispatch", default=None
)


def pending_dispatch_records(adapter: object, session_key: str) -> list[_PendingDispatchReservation]:
    reservations = getattr(adapter, "_pending_dispatch_reservations", {})
    record = reservations.get(session_key)
    records = []
    while record is not None:
        records.append(record)
        record = record.previous
    return list(reversed(records))


def pending_dispatch_record(adapter: object, session_key: str,
                            event: MessageEvent) -> _PendingDispatchReservation | None:
    for record in pending_dispatch_records(adapter, session_key):
        if record.includes(event):
            return record
    dispatch = _dispatch.get()
    if (dispatch is not None and dispatch.adapter is adapter
            and dispatch.session_key == session_key and dispatch.task is asyncio.current_task()
            and (dispatch.event is event or (
                dispatch.event.source == event.source and dispatch.event.message_id == event.message_id
                and (bool(event.message_id) or dispatch.event.timestamp == event.timestamp)))):
        return dispatch.reservation
    return None


def reserve_pending_dispatch(adapter: object, session_key: str, event: MessageEvent, *,
                             accepted: bool = True) -> _PendingDispatchReservation:
    ingress_order(event)
    for record in pending_dispatch_records(adapter, session_key):
        if record.includes(event):
            record.accepted = record.accepted or accepted
            return record
    reservations = getattr(adapter, "_pending_dispatch_reservations", None)
    if reservations is None:
        reservations = {}
        setattr(adapter, "_pending_dispatch_reservations", reservations)
    record = _PendingDispatchReservation(event, accepted=accepted, previous=reservations.get(session_key))
    reservations[session_key] = record
    return record


def release_pending_dispatch(
    adapter: object, session_key: str, event: MessageEvent, *, claimed: bool = False
) -> None:
    reservations = getattr(adapter, "_pending_dispatch_reservations", None)
    if not isinstance(reservations, dict):
        return
    record = next((candidate for candidate in pending_dispatch_records(adapter, session_key)
                   if candidate.includes(event)), None)
    dispatch = _dispatch.get()
    if (record is None and dispatch is not None and dispatch.adapter is adapter
            and dispatch.session_key == session_key and dispatch.task is asyncio.current_task()):
        record = dispatch.reservation
    if record is None:
        return
    release_pending_dispatch_record(adapter, session_key, record, claimed=claimed)


def release_pending_dispatch_record(adapter: object, session_key: str,
                                    record: _PendingDispatchReservation, *, claimed: bool = False) -> None:
    record.claimed = record.claimed or claimed
    if record.preserve_on_completion and not record.claimed:
        return
    reservations = getattr(adapter, "_pending_dispatch_reservations", {})
    reserved = reservations.get(session_key)
    if reserved is record:
        if record.previous is None:
            reservations.pop(session_key, None)
        else:
            reservations[session_key] = record.previous
        return
    for newer in pending_dispatch_records(adapter, session_key):
        if newer.previous is record:
            newer.previous = record.previous
            return


def pending_dispatch_withdrawn(adapter: object, session_key: str, event: MessageEvent) -> bool:
    record = pending_dispatch_record(adapter, session_key, event)
    return record is not None and record.withdrawn


def pending_dispatch_revision(adapter: object, session_key: str, event: MessageEvent) -> int:
    record = pending_dispatch_record(adapter, session_key, event)
    return record.revision if record is not None else 0


def close_pending_dispatch_withdrawal(adapter: object, session_key: str, event: MessageEvent) -> None:
    record = pending_dispatch_record(adapter, session_key, event)
    if record is not None:
        record.withdrawal_closed = True


def bind_pending_dispatch_input(session_id: str, owner: str) -> None:
    """Associate the current provisional dispatch with its transcript input."""
    dispatch = _dispatch.get()
    if dispatch is None or dispatch.task is not asyncio.current_task():
        return
    if dispatch.reservation is None:
        return
    dispatch.reservation.input_session_id = session_id
    dispatch.reservation.input_owner = owner


def pending_dispatch_needs_snapshot(adapter: object, reserved: _PendingDispatchReservation) -> bool:
    """Whether this provisional input still needs shutdown preservation."""
    if reserved.claimed or reserved.withdrawn:
        return False
    if not reserved.input_session_id or not reserved.input_owner:
        return True
    runner = getattr(adapter, "gateway_runner", None)
    if runner is None:
        return True
    scope = getattr(runner, "_profile_scope_for_source", None)
    try:
        with scope(reserved.event.source) if callable(scope) else nullcontext():
            return not runner.session_store.has_input_owner(
                reserved.input_session_id, reserved.input_owner,
            )
    except Exception:
        logger.warning("Could not verify durable pending input; preserving the event", exc_info=True)
        return True


@contextmanager
def pending_dispatch_scope(adapter: object, session_key: str,
                           event: MessageEvent) -> Iterator[None]:
    reservation = next((record for record in pending_dispatch_records(adapter, session_key)
                        if record.includes(event)), None)
    if reservation is not None and reservation.task is None:
        reservation.task = asyncio.current_task()
    token = _dispatch.set(_PendingDispatch(adapter, session_key, event, asyncio.current_task(), reservation))
    try:
        yield
    finally:
        _dispatch.reset(token)


def is_pending_redispatch(
    adapter: object, session_key: str, event: MessageEvent
) -> bool:
    dispatch = _dispatch.get()
    if (
        dispatch is None
        or dispatch.adapter is not adapter
        or dispatch.session_key != session_key
        or dispatch.task is not asyncio.current_task()
    ):
        return False
    if dispatch.reservation is None or not dispatch.reservation.accepted:
        return False
    original = dispatch.event
    return (
        original.source == event.source
        and original.message_id == event.message_id
        and (bool(event.message_id) or original.timestamp == event.timestamp)
    )


Merge = Callable[[MessageEvent, MessageEvent], None]
# Applied to one pending value: (whether a message matched, what remains or None).
Withdraw = Callable[[Any], Tuple[bool, Any]]


def pending_part(event: MessageEvent) -> MessageEvent:
    """Copy ``event`` so that later merges into the original leave the copy unchanged."""
    part = copy.copy(event)
    part.media_urls, part.media_types = list(event.media_urls), list(event.media_types)
    part.media_text_inlined = list(event.media_text_inlined)
    part.merged_message_ids = list(event.merged_message_ids)
    part._merged_parts = list(event._merged_parts)
    return part


def merge_recorded(existing: MessageEvent, event: MessageEvent, merge: Merge) -> None:
    """Merge ``event`` into ``existing`` with ``merge`` and record both as parts of the result.
    The first part is a copy taken before any merge changes ``existing``."""
    if not existing._merged_parts:
        existing._merged_parts = [(pending_part(existing), None)]
    existing._merged_parts.append((event, merge))
    echoed: set[str] = set(getattr(existing, "_gateway_pending_stt_echoed_paths", ()))
    echoed.update(getattr(event, "_gateway_pending_stt_echoed_paths", ()))
    merge(existing, event)
    if echoed:
        existing._gateway_pending_stt_echoed_paths = echoed.intersection(
            existing.media_urls
        )


def withdraw_from_event(
    event: Any, matches: Callable[[MessageEvent], bool]
) -> Tuple[bool, Any]:
    """Remove the messages selected by ``matches`` from the pending value ``event``.

    Returns whether any message matched, and the pending value that remains: ``event`` unchanged
    when nothing matched, None when no message remains, or else an event rebuilt by replaying the
    remaining parts with the functions that originally merged them."""
    if not isinstance(event, MessageEvent):
        return False, event
    if not event._merged_parts:
        return (True, None) if matches(event) else (False, event)
    found, remaining = False, []
    for part, merge in event._merged_parts:
        part_found, rest = withdraw_from_event(part, matches)
        found = found or part_found
        if rest is not None:
            remaining.append((rest, merge))
    if not found:
        return False, event
    if not remaining:
        return True, None
    rebuilt = pending_part(remaining[0][0])
    for attr in ("_pending_snapshot_uid", "_gateway_input_owner"):
        if hasattr(event, attr):
            setattr(rebuilt, attr, getattr(event, attr))
    for part, merge in remaining[1:]:
        merge_recorded(rebuilt, part, merge)
    echoed: set[str] = set(getattr(event, "_gateway_pending_stt_echoed_paths", ()))
    if echoed:
        rebuilt._gateway_pending_stt_echoed_paths = echoed.intersection(
            rebuilt.media_urls
        )
    return True, rebuilt


class PendingWithdrawalMixin:
    """``withdraw_pending_message`` for ``BasePlatformAdapter``."""

    def pending_native_input(self, event: MessageEvent) -> PendingNativeInput | None:
        return None

    platform: Platform
    _pending_messages: Dict[str, MessageEvent]
    _pending_text_batches: Dict[str, MessageEvent]
    _pending_text_batch_tasks: Dict[str, asyncio.Task]
    _pop_text_batch: Callable[[str], Optional[MessageEvent]]
    _text_debounce_store: Callable[[], Dict[str, Any]]
    # ``handler(adapter, withdraw)``, installed by the runner; see set_queued_withdrawal_handler.
    _queued_withdrawal_handler: Optional[Callable[[Any, Withdraw], bool]] = None

    def set_queued_withdrawal_handler(
        self, handler: Optional[Callable[[Any, Withdraw], bool]]
    ) -> None:
        """Install the runner's handler for queued follow-ups. ``withdraw_pending_message`` calls
        ``handler(adapter, withdraw)`` for the pending slots and the runner's FIFO overflow behind
        them, and the handler returns whether anything was withdrawn. Without a handler, only the
        pending slots are searched."""
        self._queued_withdrawal_handler = handler

    def withdraw_pending_message(
        self, message_id: str, *, chat_id: str, sender_id: str
    ) -> bool:
        """Remove a message that its sender deleted before its turn started, so that it never
        reaches the agent. A merged pending turn keeps its other messages. A message matches
        only if it has the ID ``message_id``, is in ``chat_id`` and was sent by ``sender_id``.
        Returns whether the message was found.

        A turn that has already started is not changed. Preparations remain searchable while
        authorization, steering or transcription awaits. Platform adapters call this when they
        observe the deletion.
        A deletion event does not identify a session, so every buffer is searched."""

        def matches(event: MessageEvent) -> bool:
            source = event.source
            return (
                event.message_id == message_id
                and source is not None
                and source.platform == self.platform
                and source.chat_id == chat_id
                and source.user_id == sender_id
            )

        return self._withdraw_pending_where(matches)

    def withdraw_native_messages(self, deletion: NativeMessageDeletion) -> bool:
        """Withdraw native input received by this adapter before its turn started."""
        if deletion.platform != self.platform:
            return False

        def matches(event: MessageEvent) -> bool:
            if not deletion.matches(event):
                return False
            identity = identity_of(event.source)
            if identity is not None:
                return identity.adapter() is self
            transport = getattr(event.source, "_transport_adapter_ref", None)
            return callable(transport) and transport() is self

        return self._withdraw_pending_where(matches)

    def _withdraw_pending_where(self, matches: Callable[[MessageEvent], bool]) -> bool:
        def withdraw(event: Any) -> Tuple[bool, Any]:
            return withdraw_from_event(event, matches)

        found = False
        for key in list(getattr(self, "_pending_dispatch_reservations", {})):
            for record in pending_dispatch_records(self, key):
                if pending_dispatch_needs_snapshot(self, record):
                    found = record.withdraw(withdraw) or found
        for key, event in list(self._pending_text_batches.items()):
            matched, rest = withdraw(event)
            if not matched:
                continue
            found = True
            if rest is not None:
                self._pending_text_batches[key] = rest
                continue
            self._pop_text_batch(key)
            task = self._pending_text_batch_tasks.pop(key, None)
            if task is not None and not task.done():
                task.cancel()
        debounce = self._text_debounce_store()
        for key, state in list(debounce.items()):
            remaining = []
            matched = False
            for event in (*state.earlier_events, state.event):
                removed, rest = withdraw(event)
                matched = removed or matched
                if rest is not None:
                    remaining.append(rest)
            if not matched:
                continue
            found = True
            if remaining:
                state.earlier_events = remaining[:-1]
                state.event = remaining[-1]
                continue
            state.cancel_timer()
            debounce.pop(key, None)
        handler = self._queued_withdrawal_handler
        if handler is not None:
            return handler(self, withdraw) or found
        return self.withdraw_from_pending_slots(withdraw) or found

    def withdraw_from_pending_slots(self, withdraw: Withdraw) -> bool:
        """Apply ``withdraw`` to every pending slot, removing a slot that it empties. Returns
        whether anything was withdrawn."""
        found = False
        for key, event in list(self._pending_messages.items()):
            matched, rest = withdraw(event)
            if not matched:
                continue
            found = True
            if rest is None:
                self._pending_messages.pop(key, None)
            else:
                self._pending_messages[key] = rest
        return found
