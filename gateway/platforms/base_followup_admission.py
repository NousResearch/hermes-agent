"""The adapter side of follow-up admission (``gateway/platforms/base.py`` sibling).

While an agent turn runs, every follow-up the adapter accepts for that session — a slot merge
in ``_handle_message_while_active``, a debounced busy-text burst, a busy follow-up committed
to the durable FIFO — is published to review admission under one per-session
``FollowupAdmissionState``: its lock orders the slot write against the turn's automatic
review (``_record_and_fence_followup_locked`` bumps the epoch and cancels that review), so
the review never spawns beside, or runs beside, the follower. The same mixin completes a
turn's gateway review ownership with its delivery outcome (``_take_review_delivery_callback``
/ ``_complete_review_delivery``: the completion ``gateway/run_turn_review.py`` parked on the
live session Event) and answers the review's own pending-follow-up probe
(``has_pending_message``). Bound into ``BasePlatformAdapter`` via the MRO; adapters override
nothing here.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from gateway.platforms.event import MessageType

if TYPE_CHECKING:  # annotations only; the facade imports this module
    from gateway.platforms.base import TextDebounceState
    from gateway.platforms.event import MessageEvent

# Log-record parity with the facade.
logger = logging.getLogger("gateway.platforms.base")


def _base():
    """The facade, read at call time: its helpers (``merge_pending_message_event``, the lazy
    attribute guard, the debounce state) are patched there by tests, so they are looked up
    there, never bound at import."""
    from gateway.platforms import base

    return base


@dataclass
class FollowupAdmissionState:
    """Cross-thread queue/review handshake for one active gateway session."""

    lock: Any = field(default_factory=threading.RLock)
    epoch: int = 0
    cancel_review: Optional[Callable[[], None]] = None


class FollowupAdmissionMixin:
    """Follow-up fence, busy-text debounce and delivery-gated review completion."""

    def _text_debounce_store(self) -> dict[str, TextDebounceState]:
        return _base()._lazy_attr(self, "_text_debounce", dict)

    def followup_admission_state(self, session_key: str) -> FollowupAdmissionState:
        """Stable state shared by gateway queue mutation and review request admission."""
        store = self._followup_admission_store()
        state = store.get(session_key)
        if state is None:
            state = store.setdefault(session_key, FollowupAdmissionState())
        return state

    def _followup_admission_store(self) -> dict[str, FollowupAdmissionState]:
        return _base()._lazy_attr(self, "_followup_admission", dict)

    def _existing_followup_admission_state(
        self, session_key: str
    ) -> Optional[FollowupAdmissionState]:
        """Return an existing state without making a read-only probe retain the session."""
        store = getattr(self, "_followup_admission", None)
        return store.get(session_key) if isinstance(store, dict) else None

    def register_followup_review_cancel(
        self, session_key: str, callback: Callable[[], None]
    ) -> FollowupAdmissionState:
        """Wire the active turn's non-blocking background-review fence."""
        state = self.followup_admission_state(session_key)
        with state.lock:
            state.cancel_review = callback if callable(callback) else None
        return state

    @staticmethod
    def _record_and_fence_followup_locked(state: FollowupAdmissionState) -> None:
        """Publish queue state and fence its automatic review before releasing admission.

        The caller owns ``state.lock``; the callback may take the review-run lock. This matches
        ``_BackgroundReviewRun.begin_request``'s admission-lock-then-run-lock order.
        """
        state.epoch += 1
        callback = state.cancel_review
        if callable(callback):
            try:
                callback()
            except Exception:
                logger.debug("Background-review follow-up fence failed", exc_info=True)

    def apply_followup_queue_mutation(
        self, session_key: str, mutation: Callable[[], bool]
    ) -> bool:
        """Apply one accepted future-turn mutation and publish/cancel it atomically."""
        state = self.followup_admission_state(session_key)
        with state.lock:
            if not mutation():
                return False
            self._record_and_fence_followup_locked(state)
            return True

    def park_pending_message(self, session_key: str, event: MessageEvent) -> None:
        """Make ``event`` the session's pending follow-up through the admission fence: it is the
        next live turn, so review admission learns of it like every other accepted follow-up."""

        def _park() -> bool:
            self._pending_messages[session_key] = event
            return True

        self.apply_followup_queue_mutation(session_key, _park)

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

    def _can_merge_text_debounce_events(
        self, existing: MessageEvent, event: MessageEvent
    ) -> bool:
        """Return True when two text debounce events came from the same sender."""

        def _identity(candidate: MessageEvent) -> tuple[str, ...] | None:
            source = getattr(candidate, "source", None)
            if source is None:
                return None
            platform = _base()._platform_name(getattr(source, "platform", None))
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

    def _text_debounce_delay(self, session_key: str) -> float:
        """Return bounded busy-text debounce delay for ``session_key``."""
        state = self._text_debounce_store().get(session_key)
        if state is None:
            return 0.0
        deadline = min(
            state.last_ts + self._busy_text_debounce_seconds,
            state.first_ts + self._busy_text_hard_cap_seconds,
        )
        return max(0.0, deadline - time.monotonic())

    async def _queue_text_debounce(self, session_key: str, event: MessageEvent) -> None:
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
                existing_pending = self._pending_messages.get(session_key)
                if (
                    existing_pending is not None
                    and self._can_merge_text_debounce_events(existing_pending, event)
                ):
                    admission = self.followup_admission_state(session_key)
                    with admission.lock:
                        _base().merge_pending_message_event(
                            self._pending_messages, session_key, event, merge_text=True
                        )
                        self._record_and_fence_followup_locked(admission)
                return
        now = time.monotonic()
        admission = self.followup_admission_state(session_key)
        with admission.lock:
            if state is None:
                state = _base().TextDebounceState(
                    event=event, task=None, first_ts=now, last_ts=now
                )
                store[session_key] = state
            else:
                if event.text:
                    state.event.text = _base()._append_text(
                        state.event.text, event.text
                    )
                state.event.absorb_reply_expected(event)
                latest_message_id = getattr(event, "message_id", None)
                latest_anchor = latest_message_id or getattr(
                    event, "reply_to_message_id", None
                )
                if latest_message_id is not None:
                    state.event.message_id = str(latest_message_id)
                if latest_anchor is not None and hasattr(
                    state.event, "reply_to_message_id"
                ):
                    state.event.reply_to_message_id = str(latest_anchor)
                state.last_ts = now
            self._record_and_fence_followup_locked(admission)
            state.cancel_timer()
            delay = self._text_debounce_delay(session_key)
            state.task = asyncio.create_task(
                self._flush_text_debounce(session_key, delay)
            )

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
        admission = self.followup_admission_state(session_key)
        with admission.lock:
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
                return False
            store.pop(session_key, None)
            _base().merge_pending_message_event(
                self._pending_messages, session_key, state.event, merge_text=True
            )
            return True

    def _discard_text_debounce(self, session_key: str) -> None:
        """Cancel and drop pending text debounce state for control commands."""
        state = self._text_debounce_store().pop(session_key, None)
        if state is not None:
            state.cancel_timer()
        from gateway.session_ingress_burst import discard_busy_burst

        discard_busy_burst(self, session_key)

    async def _complete_stranded_review_delivery(
        self, command_guard: asyncio.Event
    ) -> None:
        """Complete a review-ownership completion left parked on a reset-like command's guard.

        A turn whose task started under the previous guard but reaches the runner's carrier
        lookup after /stop, /new or /reset swapped it parks its completion on the COMMAND
        guard: the command's inline dispatch reads that guard only while its own handler runs,
        and the turn's task reads the guard it started under. Nobody else reads the command
        guard before it is released, so the turn's live-turn token would leak for the process
        lifetime and every later automatic review on the session would be refused as
        ``live_turn_active``. Completed unconfirmed (the turn was interrupted or its reply
        never went out), with the usual owner-tagged skip line. Never done from the generic
        guard release: an outer task legitimately drops a guard that still carries a nested
        turn's completion which the inline dispatch holds and will take."""
        stranded = self._take_review_delivery_callback(None, command_guard)
        if callable(stranded):
            await self._complete_review_delivery(stranded, delivery_succeeded=False)

    @staticmethod
    def _take_review_delivery_callback(
        event: Optional[MessageEvent],
        interrupt_event: Optional[asyncio.Event],
    ) -> Optional[Callable[..., Any]]:
        """Detach THIS turn's review-ownership completion from both carriers (Event first). Taken
        the moment the handler returns: a drain handoff re-uses the session Event, and the
        follow-up turn attaches its own callback there while this turn is still unwinding. An
        inline dispatch passes the Event only when a completion appeared during its own call."""
        callback = None
        for carrier in (interrupt_event, event):
            if carrier is None:
                continue
            found = getattr(carrier, "_gateway_review_delivery_complete", None)
            with contextlib.suppress(Exception):
                delattr(carrier, "_gateway_review_delivery_complete")
            if callback is None and callable(found):
                callback = found
        return callback

    async def _complete_review_delivery(
        self,
        callback: Callable[..., Any],
        *,
        delivery_succeeded: bool,
        cause: Optional[str] = None,
        outer: Any = None,
    ) -> None:
        """Complete one turn's gateway review ownership with its delivery outcome; ``outer`` is
        the still-open owner of the turn this one was nested in (inline dispatch).

        ``finish()`` released ownership and dropped the candidate before the spawn hop, so a
        failure here silently disables the review: one owner-tagged line, never fatal."""
        from agent.review_admission import REASON_COMPLETION_ERROR

        try:
            kwargs = {"delivery_succeeded": delivery_succeeded, "cause": cause}
            if outer is not None:
                kwargs["outer"] = outer
            result = callback(**kwargs)
            if inspect.isawaitable(result):
                await asyncio.wait_for(
                    result, timeout=_base()._POST_DELIVERY_CALLBACK_TIMEOUT_SECONDS
                )
        except Exception:
            logger.warning(
                "[%s] Background review delivery completion failed (owner=%s, reason=%s)",
                self.name,
                getattr(getattr(callback, "__self__", None), "owner_tag", "unbound"),
                REASON_COMPLETION_ERROR,
                exc_info=True,
            )

    def has_pending_message(self, session_key: str) -> bool:
        """Return whether a session has a queued message without consuming it.

        Lock-free on purpose. The per-turn review probe already runs under the
        ``FollowupAdmissionState`` lock its turn captured, and that lock — not this read — is what
        orders a slot write against the review's publication (``_record_and_fence_followup_locked``
        bumps the epoch under it). The store entry may since have been replaced by a later
        activation of the same key, whose lock the gateway loop holds while it cancels through the
        review registry; a stale probe taking that lock is a lock-order inversion.
        """
        pending = getattr(self, "_pending_messages", None) or {}
        debounce = getattr(self, "_text_debounce", None) or {}
        return (
            session_key in pending
            or session_key in debounce
            or self._shared_followup_pending(session_key)
        )

    def _shared_followup_pending(self, session_key: str) -> bool:
        """A busy follow-up the shared runtime holds for ``session_key`` instead of the slot: a
        same-sender burst still merging, or an admission already committed to the durable FIFO
        behind the running turn. Body-free (burst keys and admission statuses only)."""
        from gateway.session_ingress_burst import holds_busy_burst

        if holds_busy_burst(self, session_key):
            return True
        from gateway.session_authorities import all_authorities

        runner = getattr(getattr(self, "_message_handler", None), "__self__", None)
        return runner is not None and any(
            authority.queued_followup_on_route(session_key)
            for authority in all_authorities(runner)
        )
