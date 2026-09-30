"""Busy-text debounce and tracked delayed dispatch."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base_pending import _can_join_pending_event, ingress_order, merge_recorded
from gateway.platforms import base_pending_merge

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.platforms.base")


@dataclass
class TextDebounceState:
    event: MessageEvent
    task: asyncio.Task | None
    first_ts: float
    last_ts: float
    earlier_events: list[MessageEvent] = field(default_factory=list)

    def cancel_timer(self, *, unless: asyncio.Task | None = None) -> None:
        """Cancel the pending flush timer (if live and not ``unless``)."""
        if self.task is not None and self.task is not unless and not self.task.done():
            self.task.cancel()


class BaseTextDebounceMixin:
    def _text_debounce_store(self: BasePlatformAdapter) -> dict[str, TextDebounceState]:
        from gateway.platforms.base import _lazy_attr

        return _lazy_attr(self, "_text_debounce", dict)


    def _is_queue_text_debounce_candidate(self: BasePlatformAdapter, event: MessageEvent) -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT and not getattr(event, "internal", False)
            and not event.is_command() and bool((event.text or "").strip()))
        if result:
            logger.debug("[%s] Queue-text debounce candidate accepted: session=%s text_len=%d",
                         self.name, getattr(event, "session_key", "?"), len(event.text or ""))
        return result


    def _can_merge_text_debounce_events(self: BasePlatformAdapter, existing: MessageEvent, event: MessageEvent) -> bool:
        """Whether one debounce burst can preserve both events' attribution and reply context."""
        return (
            {existing.message_type, event.message_type} <= {MessageType.TEXT, MessageType.PHOTO}
            and _can_join_pending_event(existing, event)
        )


    def _text_debounce_delay(self: BasePlatformAdapter, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        deadline = min(state.last_ts + self._busy_text_debounce_seconds,
                       state.first_ts + self._busy_text_hard_cap_seconds)
        return max(0.0, deadline - time.monotonic())


    async def _queue_text_debounce(self: BasePlatformAdapter, session_key: str, event: MessageEvent) -> bool:
        """Buffer a busy burst and schedule a bounded flush."""
        ingress_order(event)
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None or not self._can_merge_text_debounce_events(state.event, event):
            queue_depth = getattr(self.gateway_runner, "_queue_depth", None)
            depth = (queue_depth(session_key, adapter=self) if callable(queue_depth) else
                     int(session_key in self._pending_messages)
                     + (len(state.earlier_events) + 1 if state else 0))
            limit = getattr(self.gateway_runner, "_BUSY_QUEUE_MAX_PENDING", 32)
            if depth >= limit:
                return False
        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                state.earlier_events.append(state.event)
                state.event = event
                state.first_ts = state.last_ts = time.monotonic()
                state.cancel_timer()
                state.task = asyncio.create_task(
                    self._flush_text_debounce(session_key, self._text_debounce_delay(session_key)))
                return True
        now = time.monotonic()
        if state is None:
            state = TextDebounceState(event=event, task=None, first_ts=now, last_ts=now)
            store[session_key] = state
        else:
            merger = (base_pending_merge._absorb_pending_media
                      if MessageType.PHOTO in {state.event.message_type, event.message_type}
                      else base_pending_merge._append_debounced_text)
            merge_recorded(state.event, event, merger)
            state.last_ts = now
        state.cancel_timer()
        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))
        return True


    async def _flush_text_debounce(self: BasePlatformAdapter, session_key: str, delay: float) -> None:
        """Timer task that flushes the debounced text buffer."""
        try:
            await asyncio.sleep(delay)
            await self._flush_text_debounce_now(session_key)
        except asyncio.CancelledError:
            return
        finally:
            current = asyncio.current_task()
            state = self._text_debounce_store().get(session_key)
            if state is not None and state.task is current:
                state.task = None


    async def _flush_text_debounce_now(self: BasePlatformAdapter, session_key: str) -> bool:
        """Submit one debounced burst through FIFO admission when a runner is available."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False
        state.cancel_timer(unless=asyncio.current_task())
        state.task = None
        enqueue = getattr(self.gateway_runner, "_queue_or_replace_pending_event", None)
        if callable(enqueue):
            store.pop(session_key, None)
            for event in (*state.earlier_events, state.event):
                enqueue(session_key, event)
            return True
        event = state.earlier_events[0] if state.earlier_events else state.event
        pending = self._pending_messages.get(session_key)
        if pending is not None and not self._can_merge_text_debounce_events(pending, event):
            return False
        if state.earlier_events:
            state.earlier_events.pop(0)
        else:
            store.pop(session_key, None)
        self._merge_into_pending_slot(session_key, event, merge_text=True)
        return True


    def _discard_text_debounce(self: BasePlatformAdapter, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()
