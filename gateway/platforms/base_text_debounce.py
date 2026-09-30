"""Busy-text debounce state and processing for BasePlatformAdapter."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
import time
from dataclasses import dataclass

from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.base_pending_merge import (
    _append_text,
    merge_pending_message_event,
)

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.platforms.base")


@dataclass
class TextDebounceState:
    event: MessageEvent
    task: asyncio.Task | None
    first_ts: float
    last_ts: float

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
        """Return True when two text debounce events came from the same sender and do not reply
        to different messages."""
        return self._same_text_debounce_sender(
            existing, event
        ) and not existing.reply_context_conflicts(event)

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
    ) -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is not None and not self._can_merge_text_debounce_events(
            state.event, event
        ):
            # Preserve sender attribution: flush the buffer as the next turn, new sender starts
            # fresh.
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(
                state.event, event
            ):
                if not self._same_text_debounce_sender(state.event, event):
                    merge_pending_message_event(
                        self._pending_messages, session_key, event, merge_text=True
                    )
                    return
                logger.debug(
                    "[%s] Busy text for %s replies to a third message; merging it into the "
                    "debounce buffer, which keeps its own reply context",
                    self.name,
                    session_key,
                )
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
        """Force-flush one debounced busy-text burst into the pending slot, or into the runner's
        queue behind the slot when the slot's event cannot absorb it."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False
        state.cancel_timer(unless=asyncio.current_task())
        state.task = None
        pending = self._pending_messages.get(session_key)
        if pending is not None and not self._can_merge_text_debounce_events(
            pending, state.event
        ):
            enqueue = getattr(
                self.gateway_runner, "_queue_or_replace_pending_event", None
            )
            if not callable(enqueue):
                return False
            store.pop(session_key, None)
            enqueue(session_key, state.event)
            return True
        store.pop(session_key, None)
        merge_pending_message_event(
            self._pending_messages, session_key, state.event, merge_text=True
        )
        return True

    def _discard_text_debounce(self: BasePlatformAdapter, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()

    @staticmethod
    def _same_text_debounce_sender(existing: MessageEvent, event: MessageEvent) -> bool:
        """Return True when two text debounce events came from the same sender."""

        from gateway.platforms.base import _platform_name

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                return None
            platform = _platform_name(getattr(source, "platform", None))
            sender = getattr(source, "user_id_alt", None) or getattr(
                source, "user_id", None
            )
            if sender:
                return (platform, str(sender))
            if getattr(source, "chat_type", None) in {"dm", "private"} and getattr(
                source, "chat_id", None
            ):
                return (platform, "dm", str(source.chat_id))
            return None

        existing_sender = _identity(existing)
        return existing_sender is not None and existing_sender == _identity(event)
