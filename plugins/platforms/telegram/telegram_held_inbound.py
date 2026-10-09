"""Telegram held-inbound queue: events PTB already acked that cannot dispatch yet are held,
redispatched on reconnect, and handed to the adapter that replaces this one (#132829)."""

import asyncio
import logging
import weakref
from typing import Optional

from gateway.platforms.event import MessageEvent
from gateway.platforms.helpers import cancel_task

logger = logging.getLogger("plugins.platforms.telegram.adapter")


class TelegramHeldInboundMixin:
    """Hold / redispatch / hand-over of inbound events for ``TelegramAdapter``."""

    def _schedule_held_inbound_redispatch(self) -> None:
        """Ensure a tracked drain runs when held events exist and delivery is live (no-op while
        down or after permanent fatal; an in-flight drain schedules its own follow-up)."""
        if self._is_permanent_fatal() or self._should_drop_delayed_delivery():
            return
        if not getattr(self, "_held_inbound_events", None):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        prior = getattr(self, "_held_inbound_redispatch_task", None)
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if prior is not None and not prior.done() and prior is not current:
            return
        self._held_inbound_redispatch_task = loop.create_task(self._redispatch_held_inbound(prior=None if prior is current else prior))

    def _hold_inbound_event(self, event: MessageEvent, *, where: str, schedule: bool = True) -> None:
        """Preserve an inbound event that cannot be dispatched now (PTB already acked the update, so dropping is silent loss).
        Capped, identity-deduped; permanent fatal discards. ``schedule=False`` inside a drain avoids poison-event loops.

        The disconnect drop-guard (#55971) correctly prevents dispatch into a torn-down session. Destroying
        the event is wrong: by the time we reach enqueue/flush, python-telegram-bot has already acked the
        update and advanced the offset — silent permanent loss, no log, no error.
        """
        if self._is_permanent_fatal():
            logger.warning(
                "[Telegram] Discarding inbound under non-retryable fatal (%s, %d chars)", where, len(getattr(event, "text", None) or ""))
            return
        successor = getattr(self, "_held_inbound_successor", None)
        target = successor() if successor is not None else None
        if target is not None:  # retired by a runner rebuild (#132829)
            self._accept_update()  # the claim being dispatched is ours, not the successor's
            target._adopt_held_event(event, where=f"{where}-forwarded", schedule=schedule)
            return
        held = getattr(self, "_held_inbound_events", None)
        if held is None:
            self._held_inbound_events = held = []
        if any(existing is event for existing in held):
            return
        max_n = int(getattr(self, "HELD_INBOUND_MAX", 64) or 64)
        while len(held) >= max_n:
            dropped = held.pop(0)
            logger.warning(
                "[Telegram] Held-inbound queue full (%d); dropping oldest (%d chars)", max_n, len(getattr(dropped, "text", None) or ""))
        held.append(event)
        self._accept_update()
        logger.warning(
            "[Telegram] Holding inbound (%s, %d chars, queue=%d)%s", where, len(getattr(event, "text", None) or ""), len(held),
            " - will redispatch on reconnect" if self._should_drop_delayed_delivery() else (" - scheduling redispatch" if schedule else ""))
        # A live-path hold must not orphan the event waiting for a reconnect that never comes.
        if schedule and not self._should_drop_delayed_delivery():
            self._schedule_held_inbound_redispatch()

    def adopt_held_inbound(self, predecessor: TelegramHeldInboundMixin) -> None:
        """Take over the hold queue of the instance the runner just replaced with us (#132829): it only
        drains on its own ``_mark_connected``, which never comes; later holds there forward here."""
        self._held_inbound_successor = None  # we own the queue again: a reverse link would forward in a cycle
        predecessor._held_inbound_successor = weakref.ref(self)
        held = getattr(predecessor, "_held_inbound_events", None) or []
        events = list(held)
        held.clear()
        for event in events:
            self._adopt_held_event(event, where="adopted", schedule=False)
        self._schedule_held_inbound_redispatch()

    def _adopt_held_event(self, event: MessageEvent, *, where: str, schedule: bool) -> None:
        """Hold a predecessor's event here; only the transport ref moves (profile/authz stay as resolved)."""
        if getattr(event, "source", None) is not None:
            event.source._transport_adapter_ref = weakref.ref(self)
        self._hold_inbound_event(event, where=where, schedule=schedule)

    def _rehold_from(self, events: list, idx: int, where: str) -> None:
        """Re-hold ``events[idx:]`` without rescheduling (drain interrupted / failed / cancelled)."""
        for rest in events[idx:]:
            self._hold_inbound_event(rest, where=where, schedule=False)

    async def _redispatch_held_inbound(self, prior: Optional[asyncio.Task] = None) -> None:
        """Drain the hold queue after reconnect or a connected-path hold; ``prior`` (previous
        redispatch task) is cancelled+awaited here so ``_mark_connected`` stays synchronous."""
        if prior is not asyncio.current_task():  # a self-redispatch must not cancel itself
            await cancel_task(prior)
        held = getattr(self, "_held_inbound_events", None)
        if self._is_permanent_fatal():
            if held:
                n = len(held)
                held.clear()
                logger.warning("[Telegram] Redispatch aborted; discarded %d held inbound under non-retryable fatal", n)
            return
        if not held:
            return
        # Take ownership atomically; concurrent holds append to the fresh list for a follow-up.
        events = list(held)
        held.clear()
        logger.warning("[Telegram] Redispatching %d held inbound message(s)", len(events))
        allow_followup_schedule = True
        try:
            for idx, event in enumerate(events):
                if self._is_permanent_fatal() or self._should_drop_delayed_delivery():
                    self._rehold_from(events, idx, "redispatch-interrupted")
                    return
                try:
                    await self.handle_message(event)
                except asyncio.CancelledError:
                    self._rehold_from(events, idx, "redispatch-cancelled")
                    raise
                except Exception:
                    # Retryable failure: re-hold but do NOT reschedule now (a poison event would
                    # tight-loop); the next mark_connected/live hold drains.
                    logger.exception(
                        "[Telegram] Failed to redispatch held inbound (%d chars); re-holding", len(getattr(event, "text", None) or ""))
                    self._rehold_from(events, idx, "redispatch-failed")
                    allow_followup_schedule = False
                    return
        finally:
            # Events that arrived mid-drain while still connected need another pass.
            if (
                allow_followup_schedule
                and getattr(self, "_held_inbound_events", None)
                and not self._should_drop_delayed_delivery()
                and not self._is_permanent_fatal()):
                self._schedule_held_inbound_redispatch()
