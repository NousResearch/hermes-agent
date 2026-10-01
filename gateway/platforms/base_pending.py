"""Pending-event attribution and dispatch ownership for gateway adapters."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from gateway.platforms.event import MessageEvent


_SECURITY_METADATA_KEYS = (
    "hermes_plugin_id", "hermes_plugin_injection", "gateway_session_key",
    "gateway_session_id", "gateway_session_strict", "notification_category",
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
        and all((first.metadata or {}).get(key) == (second.metadata or {}).get(key)
                for key in _SECURITY_METADATA_KEYS)
    )


def _can_join_pending_event(first: MessageEvent, second: MessageEvent) -> bool:
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


@dataclass(frozen=True)
class _PendingDispatch:
    adapter: object
    session_key: str
    event: MessageEvent
    task: asyncio.Task | None
    reservation: _PendingDispatchReservation | None


_dispatch: ContextVar[_PendingDispatch | None] = ContextVar("pending_dispatch", default=None)


def reserve_pending_dispatch(adapter: object, session_key: str, event: MessageEvent) -> None:
    reservations = getattr(adapter, "_pending_dispatch_reservations", None)
    if reservations is None:
        reservations = {}
        setattr(adapter, "_pending_dispatch_reservations", reservations)
    reservations[session_key] = _PendingDispatchReservation(event)


def release_pending_dispatch(adapter: object, session_key: str, event: MessageEvent, *,
                             claimed: bool = False) -> None:
    reservations = getattr(adapter, "_pending_dispatch_reservations", None)
    if not isinstance(reservations, dict):
        return
    reserved = reservations.get(session_key)
    dispatch = _dispatch.get()
    owning_dispatch = (dispatch is not None and dispatch.adapter is adapter
                       and dispatch.session_key == session_key and dispatch.task is asyncio.current_task())
    record = reserved if reserved is not None and reserved.event is event else None
    if record is None and owning_dispatch:
        record = dispatch.reservation
    if record is None:
        return
    record.claimed = record.claimed or claimed
    if reserved is record:
        reservations.pop(session_key, None)


@contextmanager
def pending_dispatch_scope(adapter: object, session_key: str,
                           event: MessageEvent) -> Iterator[None]:
    reservations = getattr(adapter, "_pending_dispatch_reservations", {})
    reservation = reservations.get(session_key) if isinstance(reservations, dict) else None
    if reservation is not None and reservation.event is not event:
        reservation = None
    token = _dispatch.set(_PendingDispatch(adapter, session_key, event, asyncio.current_task(), reservation))
    try:
        yield
    finally:
        _dispatch.reset(token)


def is_pending_redispatch(adapter: object, session_key: str, event: MessageEvent) -> bool:
    dispatch = _dispatch.get()
    if (dispatch is None or dispatch.adapter is not adapter or dispatch.session_key != session_key
            or dispatch.task is not asyncio.current_task()):
        return False
    original = dispatch.event
    return (
        original.source == event.source
        and original.message_id == event.message_id
        and (bool(event.message_id) or original.timestamp == event.timestamp)
    )
