"""Text debounce mixin for BasePlatformAdapter.

Extracted from base.py to keep the base adapter module focused. The debounce
state machine buffers follow-up text messages that arrive while a session is
busy (in queue mode), coalescing them into a single turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from gateway.platforms.event import MessageEvent, MessageType

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter, TextDebounceState

logger = logging.getLogger(__name__)


class TextDebounceMixin:
    """Mixin providing text debounce state and methods for BasePlatformAdapter.

    Requires the host class to provide:
    - ``self._pending_messages``: dict[str, MessageEvent]
    - ``self._busy_text_mode``: str
    - ``self._busy_text_debounce_seconds``: float
    - ``self._busy_text_hard_cap_seconds``: float
    - ``self.name``: str
    - ``self.gateway_runner``: optional runner ref
    """

    def _text_debounce_store(self) -> dict[str, "TextDebounceState"]:
        from gateway.platforms.base import _lazy_attr
        return _lazy_attr(self, "_text_debounce", dict)

    def _is_queue_text_debounce_candidate(self, event: "MessageEvent") -> bool:
        """Return True for normal text eligible for queue-mode debounce."""
        result = (
            getattr(self, "_busy_text_mode", "interrupt") == "queue"
            and event.message_type == MessageType.TEXT and not getattr(event, "internal", False)
            and not event.is_command() and bool((event.text or "").strip()))
        if result:
            logger.debug("[%s] Queue-text debounce candidate accepted: session=%s text_len=%d",
                         self.name, getattr(event, "session_key", "?"), len(event.text or ""))
        return result

    def _can_merge_text_debounce_events(self, existing: "MessageEvent", event: "MessageEvent") -> bool:
        """Return True when two text debounce events came from the same sender AND carry the same
        prompt identity (see :func:`prompt_identity_conflict`)."""
        from gateway.platforms.base import _platform_name

        def _identity(candidate: "MessageEvent"):
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
        if existing_sender is None or existing_sender != _identity(event):
            return False
        return bool(getattr(existing, "preserve_prompt_pins", False)) == bool(
            getattr(event, "preserve_prompt_pins", False))

    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        deadline = min(state.last_ts + self._busy_text_debounce_seconds,
                       state.first_ts + self._busy_text_hard_cap_seconds)
        return max(0.0, deadline - time.monotonic())

    def _route_preserved_debounce_event(self, session_key: str, event: "MessageEvent") -> bool:
        """Hand a buffered turn whose PROMPT IDENTITY differs from the turn it would coalesce with
        to the runner's bounded FIFO admission, so it runs as its own turn.

        Returns True only when that admission ACCEPTED the event (it now owns a queued slot);
        False means nothing was claimed, so the caller keeps its own buffer intact for a later
        flush. Acceptance is read from the admission helper's return value, never from
        ``_gateway_accepted``, which may be left over from an earlier merge onto a different
        event. Without a runner FIFO the merge is refused outright: there is no admission owner.
        """
        runner = getattr(self, "gateway_runner", None)
        admit = getattr(runner, "_queue_or_replace_pending_event", None)
        if not callable(admit):
            logger.debug(
                "[%s] Refusing to coalesce a differing-identity follow-up for session %s — "
                "no runner FIFO to admit it", self.name, session_key,
            )
            return False
        return bool(admit(session_key, event))

    async def _queue_text_debounce(self, session_key: str, event: "MessageEvent") -> None:
        """Buffer normal queue-mode busy text and schedule a bounded flush."""
        from gateway.platforms.base import TextDebounceState, _append_text, merge_pending_message_event, prompt_identity_conflict

        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is not None and not self._can_merge_text_debounce_events(state.event, event):
            # Preserve sender attribution: flush the buffer as the next turn, new sender starts
            # fresh.
            await self._flush_text_debounce_now(session_key)
            state = store.get(session_key)
            if state is not None and not self._can_merge_text_debounce_events(state.event, event):
                existing_pending = self._pending_messages.get(session_key)
                if existing_pending is not None and self._can_merge_text_debounce_events(existing_pending, event):
                    merge_pending_message_event(self._pending_messages, session_key, event, merge_text=True)
                elif prompt_identity_conflict(state.event, event):
                    # Buffering onto, or dropping, a differing-identity turn would run this
                    # human turn under the other turn's pinned prompt identity. Give it its own.
                    if not self._route_preserved_debounce_event(session_key, event):
                        # Refused admission (queue at cap, no runner FIFO). This is a FRESH
                        # arrival that owns no slot yet, so the fresh-input cap may legitimately
                        # leave it unaccepted; the buffered head keeps its place and is admitted
                        # at the next flush. Say so at WARNING rather than dropping a human turn
                        # on a debug line.
                        logger.warning(
                            "[%s] Dropped a differing-identity follow-up for session %s — "
                            "admission refused (queue at cap or no runner FIFO); it was never "
                            "queued, so nothing will retry it",
                            self.name, session_key,
                        )
                return
        now = time.monotonic()
        if state is None:
            state = TextDebounceState(event=event, task=None, first_ts=now, last_ts=now)
            store[session_key] = state
        else:
            if event.text:
                state.event.text = _append_text(state.event.text, event.text)
            state.event.absorb_reply_expected(event)
            latest_message_id = getattr(event, "message_id", None)
            latest_anchor = latest_message_id or getattr(event, "reply_to_message_id", None)
            if latest_message_id is not None:
                state.event.message_id = str(latest_message_id)
            if latest_anchor is not None and hasattr(state.event, "reply_to_message_id"):
                state.event.reply_to_message_id = str(latest_anchor)
            state.last_ts = now
        state.cancel_timer()
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
        from gateway.platforms.base import merge_pending_message_event, prompt_identity_conflict

        store = self._text_debounce_store()
        state = store.get(session_key)
        if state is None:
            return False
        state.cancel_timer(unless=asyncio.current_task())
        state.task = None
        pending = self._pending_messages.get(session_key)
        if pending is not None and not self._can_merge_text_debounce_events(pending, state.event):
            if not prompt_identity_conflict(pending, state.event):
                return False
            # The occupant keeps the slot and holds a DIFFERENT prompt identity, so this
            # buffered turn may not run under its pins. Admit it as its own turn; if that
            # admission refuses (queue at cap, no runner FIFO) the buffer is RETAINED for a
            # later flush — never dropped, never merged into the occupant.
            if not self._route_preserved_debounce_event(session_key, state.event):
                return False
            store.pop(session_key, None)
            return True
        store.pop(session_key, None)
        merge_pending_message_event(self._pending_messages, session_key, state.event, merge_text=True)
        return True

    def _discard_text_debounce(self, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()
