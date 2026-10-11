"""Resolve native reply destinations before cron continuation bookkeeping."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.config import Platform
    from gateway.delivery import DeliveryTransport, ResolvedDeliveryDestination


def resolve_live_destination(
    transport: DeliveryTransport,
    platform: Platform,
    chat_id: str,
    thread_id: str | None,
    origin: dict,
    loop: asyncio.AbstractEventLoop,
    *,
    refresh: bool = False,
) -> ResolvedDeliveryDestination | None:
    """Resolve the destination on the adapter's event loop under the cron profile's scope.
    None when the transport keeps targets as given."""
    if not transport.resolves_destinations:
        return None
    from agent.async_utils import safe_schedule_threadsafe
    from gateway.session import SessionSource
    from hermes_cli.profiles import get_active_profile_name

    source = SessionSource(
        platform=platform,
        chat_id=chat_id,
        thread_id=thread_id,
        chat_type=origin.get("chat_type") or "group",
        user_id=origin.get("user_id"),
        scope_id=origin.get("scope_id"),
        profile=get_active_profile_name(),
    )
    future = safe_schedule_threadsafe(
        transport.resolve_destination(source, refresh=refresh), loop
    )
    if future is None:
        raise RuntimeError("destination resolution could not be scheduled")
    try:
        return future.result(timeout=90)
    except TimeoutError:
        future.cancel()
        raise
