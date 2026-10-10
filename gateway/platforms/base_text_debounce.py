"""Busy-text debounce state and processing for BasePlatformAdapter."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
import time
from dataclasses import dataclass, field

from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base_pending_merge import (
    _append_text,
    merge_pending_message_event,
)
from gateway.platforms.base_pending import can_join_pending_event

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

    def _is_queue_text_debounce_candidate(
        self: BasePlatformAdapter, event: MessageEvent
    ) -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT
            and not getattr(event, "internal", False)
            and not event.is_command()
            and bool((event.text or "").strip())
        )
        if result:
            logger.debug(
                "[%s] Queue-text debounce candidate accepted: session=%s text_len=%d",
                self.name,
                getattr(event, "session_key", "?"),
                len(event.text or ""),
            )
        return result

    def _can_merge_text_debounce_events(
        self: BasePlatformAdapter, existing: MessageEvent, event: MessageEvent
    ) -> bool:
        """Whether one debounce burst can preserve both events' attribution and reply context."""
        return can_join_pending_event(existing, event)

    def _text_debounce_delay(self: BasePlatformAdapter, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        deadline = min(
            state.last_ts + self._busy_text_debounce_seconds,
            state.first_ts + self._busy_text_hard_cap_seconds,
        )
        return max(0.0, deadline - time.monotonic())

    async def _queue_text_debounce(
        self: BasePlatformAdapter, session_key: str, event: MessageEvent
    ) -> bool:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None or not self._can_merge_text_debounce_events(
            state.event, event
        ):
            queue_depth = getattr(self.gateway_runner, "_queue_depth", None)
            depth = (
                queue_depth(session_key, adapter=self)
                if callable(queue_depth)
                else int(session_key in self._pending_messages)
                + (len(state.earlier_events) + 1 if state else 0)
            )
            limit = getattr(self.gateway_runner, "_BUSY_QUEUE_MAX_PENDING", 32)
            if depth >= limit:
                return False
        if state is not None and not self._can_merge_text_debounce_events(
            state.event, event
        ):
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(
                state.event, event
            ):
                state.earlier_events.append(state.event)
                state.event = event
                state.first_ts = state.last_ts = time.monotonic()
                state.cancel_timer()
                state.task = asyncio.create_task(
                    self._flush_text_debounce(
                        session_key, self._text_debounce_delay(session_key)
                    )
                )
                return True
        now = time.monotonic()
        if state is None:
            state = TextDebounceState(event=event, task=None, first_ts=now, last_ts=now)
            store[session_key] = state
        else:
            state.event.absorb_context_dependencies(event)
            if event.text:
                state.event.text = _append_text(state.event.text, event.text)
            if event.media_urls:
                state.event.absorb_media(event)
            state.event.absorb_reply_context(event)
            state.event.absorb_reply_expected(event)
            latest_message_id = getattr(event, "message_id", None)
            if latest_message_id is not None:
                state.event.merged_message_ids.extend(
                    message_id for message_id in (state.event.message_id, *event.merged_message_ids)
                    if message_id
                )
                state.event.message_id = str(latest_message_id)
            else:
                state.event.absorb_message_ids(event)
            state.last_ts = now
        state.cancel_timer()
        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))
        return True

    async def _flush_text_debounce(
        self: BasePlatformAdapter, session_key: str, delay: float
    ) -> None:
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

    async def _flush_text_debounce_now(
        self: BasePlatformAdapter, session_key: str
    ) -> bool:
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
        if pending is not None and not self._can_merge_text_debounce_events(
            pending, event
        ):
            return False
        if state.earlier_events:
            state.earlier_events.pop(0)
        else:
            store.pop(session_key, None)
        merge_pending_message_event(
            self._pending_messages, session_key, event, merge_text=True
        )
        return True

    def _discard_text_debounce(self: BasePlatformAdapter, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()
