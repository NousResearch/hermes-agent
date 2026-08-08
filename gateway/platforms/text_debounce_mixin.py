"""Busy-text debounce helpers for ``BasePlatformAdapter``.

Extracted from ``gateway/platforms/base.py`` (god-file decomposition, wave 1,
shard s4, cluster c9) following the mechanical mixin lift that produced
``plugins/platforms/telegram/authz_mixin.py`` (#75742).  This mixin holds the
queue-mode busy-text debounce cluster: candidate classification, sender
attribution merging, the bounded flush timer, and discard-on-control-command.

Behavior-neutral: every method is lifted verbatim from ``BasePlatformAdapter``.
``self.*`` calls resolve unchanged via the MRO, and ``TextDebounceMixin``
precedes ``BasePlatformAdapter`` in the bases so resolution order is what it
was when these methods sat on the class.

* ``logger`` is bound by explicit name rather than ``__name__``, so records
  emitted from these methods keep the logger name
  ``"gateway.platforms.base"``.
* Class attributes referenced via ``self.`` (``_text_debounce``,
  ``_pending_messages``, ``_busy_text_mode``, ``_busy_text_debounce_seconds``,
  ``_busy_text_hard_cap_seconds``) stay on ``BasePlatformAdapter`` and resolve
  through the MRO.
"""

import asyncio
import logging
import time

from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    TextDebounceState,
    _platform_name,
    merge_pending_message_event,
)

# Bind the adapter's logger by name so log records lifted with these methods
# are emitted under exactly the name they were before.
logger = logging.getLogger("gateway.platforms.base")


class TextDebounceMixin:
    """Busy-text debounce cluster lifted verbatim from ``BasePlatformAdapter``."""

    def _text_debounce_store(self) -> dict[str, TextDebounceState]:
        store = getattr(self, "_text_debounce", None)
        if store is None:
            store = {}
            self._text_debounce = store
        return store

    def _is_queue_text_debounce_candidate(self, event: MessageEvent) -> bool:
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

    def _can_merge_text_debounce_events(self, existing: MessageEvent, event: MessageEvent) -> bool:
        """Return True when two text debounce events came from the same sender."""

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                return None
            platform = _platform_name(getattr(source, "platform", None))
            sender = getattr(source, "user_id_alt", None) or getattr(source, "user_id", None)
            if sender:
                return (platform, str(sender))
            if getattr(source, "chat_type", None) in {"dm", "private"} and getattr(source, "chat_id", None):
                return (platform, "dm", str(source.chat_id))
            return None

        existing_sender = _identity(existing)
        incoming_sender = _identity(event)
        return existing_sender is not None and existing_sender == incoming_sender

    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        now = time.monotonic()
        window_deadline = state.last_ts + self._busy_text_debounce_seconds
        hard_cap_deadline = state.first_ts + self._busy_text_hard_cap_seconds
        return max(0.0, min(window_deadline, hard_cap_deadline) - now)

    async def _queue_text_debounce(self, session_key: str, event: MessageEvent) -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        store = self._text_debounce_store()
        state = store.get(session_key)

        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            # Preserve sender attribution in shared sessions. The current
            # buffer becomes the next pending turn; the new sender starts a
            # fresh debounce burst when the pending slot allows it.
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                existing_pending = self._pending_messages.get(session_key)
                if existing_pending is not None and self._can_merge_text_debounce_events(existing_pending, event):
                    merge_pending_message_event(
                        self._pending_messages,
                        session_key,
                        event,
                        merge_text=True,
                    )
                return

        now = time.monotonic()
        if state is None:
            state = TextDebounceState(
                event=event,
                task=None,
                first_ts=now,
                last_ts=now,
            )
            store[session_key] = state
        else:
            if event.text:
                state.event.text = (
                    f"{state.event.text}\n{event.text}"
                    if state.event.text
                    else event.text
                )
            latest_message_id = getattr(event, "message_id", None)
            latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
            if latest_message_id is not None:
                state.event.message_id = str(latest_message_id)
            if latest_anchor is not None and hasattr(state.event, "reply_to_message_id"):
                state.event.reply_to_message_id = str(latest_anchor)
            state.last_ts = now

        if state.task is not None and not state.task.done():
            state.task.cancel()

        delay = self._text_debounce_delay(session_key)
        state.task = asyncio.create_task(self._flush_text_debounce(session_key, delay))

    async def _flush_text_debounce(self, session_key: str, delay: float) -> None:
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

    async def _flush_text_debounce_now(self, session_key: str) -> bool:
        """Force-flush one debounced busy-text burst into the pending slot."""
        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False

        current = asyncio.current_task()
        if state.task is not None and state.task is not current and not state.task.done():
            state.task.cancel()
        state.task = None

        existing_pending = self._pending_messages.get(session_key)
        if (
            existing_pending is not None
            and not self._can_merge_text_debounce_events(existing_pending, state.event)
        ):
            return False

        state = store.pop(session_key, None)
        if state is None:
            return False
        merge_pending_message_event(
            self._pending_messages,
            session_key,
            state.event,
            merge_text=True,
        )
        return True

    def _discard_text_debounce(self, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None and state.task is not None and not state.task.done():
            state.task.cancel()
