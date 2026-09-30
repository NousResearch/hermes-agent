"""Pending-event attribution and dispatch ownership for gateway adapters."""

import asyncio
import logging
import copy
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional, Tuple

from gateway.platforms.event import MessageEvent


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
        and _same_pending_security_context(first, second)
        and not first.reply_context_conflicts(second)
    )


@dataclass
class _PendingDispatchReservation:
    event: MessageEvent
    claimed: bool = False
    preserve_on_completion: bool = False
    input_session_id: str | None = None
    input_owner: str | None = None


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


def reserve_pending_dispatch(
    adapter: object, session_key: str, event: MessageEvent
) -> None:
    reservations = getattr(adapter, "_pending_dispatch_reservations", None)
    if reservations is None:
        reservations = {}
        setattr(adapter, "_pending_dispatch_reservations", reservations)
    reservations[session_key] = _PendingDispatchReservation(event)


def release_pending_dispatch(
    adapter: object, session_key: str, event: MessageEvent, *, claimed: bool = False
) -> None:
    reservations = getattr(adapter, "_pending_dispatch_reservations", None)
    if not isinstance(reservations, dict):
        return
    reserved = reservations.get(session_key)
    dispatch = _dispatch.get()
    owning_dispatch = (
        dispatch is not None
        and dispatch.adapter is adapter
        and dispatch.session_key == session_key
        and dispatch.task is asyncio.current_task()
    )
    record = reserved if reserved is not None and reserved.event is event else None
    if record is None and owning_dispatch:
        record = dispatch.reservation
    if record is None:
        return
    record.claimed = record.claimed or claimed
    if record.preserve_on_completion and not record.claimed:
        return
    if reserved is record:
        reservations.pop(session_key, None)


def bind_pending_dispatch_input(session_id: str, owner: str) -> None:
    """Associate the current provisional dispatch with its transcript input."""
    dispatch = _dispatch.get()
    if dispatch is None or dispatch.task is not asyncio.current_task():
        return
    if dispatch.reservation is None:
        return
    dispatch.reservation.input_session_id = session_id
    dispatch.reservation.input_owner = owner


def pending_dispatch_needs_snapshot(
    adapter: object, reserved: _PendingDispatchReservation
) -> bool:
    """Whether this provisional input still needs shutdown preservation."""
    if reserved.claimed:
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
                reserved.input_session_id,
                reserved.input_owner,
            )
    except Exception:
        logger.warning(
            "Could not verify durable pending input; preserving the event",
            exc_info=True,
        )
        return True


@contextmanager
def pending_dispatch_scope(
    adapter: object, session_key: str, event: MessageEvent
) -> Iterator[None]:
    reservations = getattr(adapter, "_pending_dispatch_reservations", {})
    reservation = (
        reservations.get(session_key) if isinstance(reservations, dict) else None
    )
    if reservation is not None and reservation.event is not event:
        reservation = None
    token = _dispatch.set(
        _PendingDispatch(
            adapter, session_key, event, asyncio.current_task(), reservation
        )
    )
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
    merge(existing, event)


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
    for part, merge in remaining[1:]:
        merge_recorded(rebuilt, part, merge)
    return True, rebuilt


class PendingWithdrawalMixin:
    """``withdraw_pending_message`` for ``BasePlatformAdapter``."""

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

        A turn that has already started is not changed. Platform adapters call this when they
        observe the deletion. A deletion event does not identify a session, so every buffer is
        searched."""

        def matches(event: MessageEvent) -> bool:
            source = event.source
            return (
                event.message_id == message_id
                and source is not None
                and source.chat_id == chat_id
                and source.user_id == sender_id
            )

        def withdraw(event: Any) -> Tuple[bool, Any]:
            return withdraw_from_event(event, matches)

        found = False
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
            matched, rest = withdraw(state.event)
            if not matched:
                continue
            found = True
            if rest is not None:
                state.event = rest
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
