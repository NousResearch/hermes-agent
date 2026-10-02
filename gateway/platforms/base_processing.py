"""Adapter processing tasks, pending dispatch ownership and cancellation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from gateway.config import Platform
from gateway.platforms.base_pending import (
    _PendingDispatchReservation, pending_dispatch_scope, release_pending_dispatch,
    pending_dispatch_records, reserve_pending_dispatch, release_pending_dispatch_record, ingress_order,
)
from gateway.platforms.event import MessageEvent, MessageType, ProcessingOutcome
from gateway.warning_notifications import diagnostic_wake_muted

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter, MessageHandler
    from gateway.platforms.base_text_debounce import TextDebounceState

logger = logging.getLogger("gateway.platforms.base")


_PLAINTEXT_GATEWAY_RESTART_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+(?:the\s+)?hermes\s+gateway[.!?\s]*$", re.IGNORECASE),
    re.compile(r"^(?:please\s+)?restart\s+hermes[.!?\s]*$", re.IGNORECASE))


def coerce_plaintext_gateway_command(event: "MessageEvent") -> None:
    """Rewrite a tiny set of DM plaintext admin phrases (exact matches only) into slash commands so
    ``restart gateway`` never reaches the LLM/tool path (a self-restart from inside the running
    agent leaves the gateway stuck in ``draining`` waiting on that agent)."""
    with contextlib.suppress(Exception):
        if event is None or event.message_type != MessageType.TEXT:
            return
        text = (event.text or "").strip()
        if not text or text.startswith("/"):
            return
        if getattr(getattr(event, "source", None), "chat_type", None) != "dm":
            return
        if any(pattern.match(text) for pattern in _PLAINTEXT_GATEWAY_RESTART_PATTERNS):
            event.text = "/restart"



class BaseProcessingMixin:
    name: str
    gateway_runner: Any
    _active_sessions: dict[str, asyncio.Event]
    _session_tasks: dict[str, asyncio.Task]
    _background_tasks: set[asyncio.Task]
    _expected_cancelled_tasks: set[asyncio.Task]
    _requeue_counts: dict[str, int]
    _pending_messages: dict[str, MessageEvent]
    _pending_dispatch_reservations: dict[str, _PendingDispatchReservation]
    _message_handler: MessageHandler | None
    _streaming_tts_completed_turns: set[str]
    _text_debounce_store: Callable[[], dict[str, TextDebounceState]]

    if TYPE_CHECKING:
        _cleanup_finished_session_task = BasePlatformAdapter._cleanup_finished_session_task
        _clear_session_guard = BasePlatformAdapter._clear_session_guard
        _deliver_attachments = BasePlatformAdapter._deliver_attachments
        _discard_text_debounce = BasePlatformAdapter._discard_text_debounce
        _extract_response_content = BasePlatformAdapter._extract_response_content
        _finish_session_task = BasePlatformAdapter._finish_session_task
        _fire_post_delivery_callback = BasePlatformAdapter._fire_post_delivery_callback
        _flush_text_debounce_now = BasePlatformAdapter._flush_text_debounce_now
        _media_delivery_scope = BasePlatformAdapter._media_delivery_scope
        _notify_turn_error = BasePlatformAdapter._notify_turn_error
        _play_tts_file = BasePlatformAdapter._play_tts_file
        _release_turn_marker = BasePlatformAdapter._release_turn_marker
        _requeue_backoff_delay = BasePlatformAdapter._requeue_backoff_delay
        _run_processing_hook = BasePlatformAdapter._run_processing_hook
        _send_final_text = BasePlatformAdapter._send_final_text
        _stage_next_queued_event = BasePlatformAdapter._stage_next_queued_event
        _start_typing_refresh = BasePlatformAdapter._start_typing_refresh
        _stop_typing_refresh = BasePlatformAdapter._stop_typing_refresh
        _streaming_tts_turn_key = BasePlatformAdapter._streaming_tts_turn_key
        _synthesize_auto_tts = BasePlatformAdapter._synthesize_auto_tts
        _unwrap_ephemeral = BasePlatformAdapter._unwrap_ephemeral
        _wants_auto_tts = BasePlatformAdapter._wants_auto_tts
        pause_typing_for_chat = BasePlatformAdapter.pause_typing_for_chat
        _drop_unresolved = BasePlatformAdapter._drop_unresolved
        _apply_topic_recovery = BasePlatformAdapter._apply_topic_recovery
        _event_session_key = BasePlatformAdapter._event_session_key
        _handle_message_while_active = BasePlatformAdapter._handle_message_while_active

    async def handle_message(self, event: MessageEvent) -> None:
        """Process an incoming message; returns quickly by spawning a background
        task so new messages (and interrupts) can arrive while an agent runs."""
        ingress_order(event)
        event._gateway_accepted = False
        if not self._message_handler:
            # No handler = every inbound silently discarded on an adapter that still polls and sends;
            # say so once per adapter (#102260).
            if not getattr(self, "_no_message_handler_logged", False):
                self._no_message_handler_logged = True
                logger.error(
                    "[%s] Dropping inbound message: no gateway message handler "
                    "is installed on this adapter. The adapter is connected and "
                    "can send, but every inbound message is discarded.",
                    self.name,
                )
            return

        if event.allow_gateway_control:
            coerce_plaintext_gateway_command(event)
        # Identity FIRST: every key below (routing check, guard lookup, batch lane) derives from it.
        if self._drop_unresolved(event):
            return
        expected_session_key = str((event.metadata or {}).get("gateway_session_key") or "").strip()
        # Explicitly routed events already name their destination; recovering a
        # different topic would redirect them and yield before the session claim.
        if (not expected_session_key and getattr(self, "_topic_recovery_fn", None) is not None
                and event.source.platform == Platform.TELEGRAM and event.source.chat_type == "dm"):
            await asyncio.to_thread(self._apply_topic_recovery, event)
        session_key = self._event_session_key(event)
        if expected_session_key and session_key != expected_session_key:
            logger.warning("Dropping internally routed event: expected session=%s derived=%s",
                           expected_session_key, session_key)
            return
        # On-entry self-heal: clear a guard whose owner task already exited.
        if session_key in self._active_sessions:
            self._heal_stale_session_lock(session_key)
        if session_key in self._active_sessions:
            await self._handle_message_while_active(event, session_key)
            return
        # Guard installed synchronously BEFORE the task spawns so a second message can't race in.
        event._gateway_accepted = self._start_session_processing(event, session_key)


    def _release_session_guard(self, session_key: str, *, guard: Optional[asyncio.Event] = None) -> None:
        """Release the session guard; with ``guard`` given, only if the entry is still that exact
        Event (an old task's unwind must not clear the guard a reset-like command swapped in)."""
        current_guard = self._active_sessions.get(session_key)
        if current_guard is None or (guard is not None and current_guard is not guard):
            return
        del self._active_sessions[session_key]

    def _session_task_is_stale(self, session_key: str) -> bool:
        """True if the recorded owner task for ``session_key`` has exited. No owner task at all is
        NOT stale (guards installed outside handle_message, as tests do, must not be healed)."""
        done = getattr(self._session_tasks.get(session_key), "done", None)
        return bool(done and done())

    def _heal_stale_session_lock(self, session_key: str) -> bool:
        """Clear a stale session lock; True if healed. On-entry safety net: without it a split-brain
        (guard held, nothing processing) traps the chat in "Interrupting..." until restart."""
        if session_key not in self._active_sessions or not self._session_task_is_stale(session_key):
            return False
        logger.warning("[%s] Healing stale session lock for %s (owner task is done/absent)",
                       self.name, session_key)
        self._active_sessions.pop(session_key, None)
        self._pending_messages.pop(session_key, None)
        self._requeue_counts.pop(session_key, None)
        task = self._session_tasks.pop(session_key, None)
        self._discard_text_debounce(session_key)
        for record in pending_dispatch_records(self, session_key):
            if record.task is task:
                release_pending_dispatch_record(self, session_key, record)
        return True

    def _start_session_processing(self, event: MessageEvent, session_key: str, *,
                                  interrupt_event: Optional[asyncio.Event] = None) -> bool:
        """Spawn a background processing task under the session guard; True on success. If
        ``create_task`` is stubbed with a non-Task sentinel (tests), the guard is rolled back
        (False)."""
        guard = interrupt_event or asyncio.Event()
        self._active_sessions[session_key] = guard
        reserved = reserve_pending_dispatch(self, session_key, event)
        task = asyncio.create_task(self._process_message_background(event, session_key))
        if not self._track_session_task(session_key, task, event):
            release_pending_dispatch_record(self, session_key, reserved)
            self._session_tasks.pop(session_key, None)
            self._release_session_guard(session_key, guard=guard)
            return False
        return True

    def _track_session_task(self, session_key: str, task: Any, event: MessageEvent | None = None) -> bool:
        """Record ``task`` as the session owner and track it for shutdown; False when
        ``create_task`` was stubbed with an unhashable sentinel (tests) — the owner entry is left
        for the caller."""
        self._session_tasks[session_key] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            return False
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._expected_cancelled_tasks.discard)
            reserved = next((record for record in pending_dispatch_records(self, session_key)
                             if record.event is event), None)
            if reserved is not None:
                reserved.task = task
                task.add_done_callback(lambda _: release_pending_dispatch_record(self, session_key, reserved))
        return True

    async def cancel_session_processing(self, session_key: str, *, release_guard: bool = True,
                                        discard_pending: bool = True) -> None:
        """Cancel in-flight processing for one session. ``release_guard=False`` keeps the guard so
        reset-like commands finish atomically; the await is bounded (5s) so a wedged finally can't
        stall."""
        self._requeue_counts.pop(session_key, None)
        task = self._session_tasks.pop(session_key, None)
        reserved_inputs = [record for record in pending_dispatch_records(self, session_key)
                           if record.task is task or record.task is None]
        if task is not None and not task.done():
            logger.debug("[%s] Cancelling active processing for session %s", self.name, session_key)
            self._expected_cancelled_tasks.add(task)
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("[%s] Cancelled task for %s did not exit within 5s; "
                               "unblocking dispatch and letting the task unwind in the background",
                               self.name, session_key)
            except Exception:
                logger.debug("[%s] Session cancellation raised while unwinding %s", self.name,
                             session_key, exc_info=True)
        if task is None or task.done():
            for reserved in reversed(reserved_inputs):
                if not discard_pending and not reserved.claimed and not reserved.withdrawn:
                    restore = getattr(self.gateway_runner, "_restore_pending_dispatch", None)
                    if callable(restore):
                        restore(session_key, reserved.event, self)
                release_pending_dispatch_record(self, session_key, reserved)
        if discard_pending:
            self._pending_messages.pop(session_key, None)
            self._discard_text_debounce(session_key)
        if release_guard:
            self._release_session_guard(session_key)

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        """Background task that actually processes the message."""
        from gateway.platforms.base import EphemeralReply, _mark_notify_metadata, _thread_metadata_for_event

        delivery_attempted = delivery_succeeded = False  # feeds the processing-complete hook

        def _record_delivery(result):
            nonlocal delivery_attempted, delivery_succeeded
            if result is not None:
                delivery_attempted = True
                delivery_succeeded = delivery_succeeded or bool(getattr(result, "success", False))
        # Reuse the interrupt event handle_message() installed; new Event only if removed externally.
        interrupt_event = self._active_sessions.get(session_key) or asyncio.Event()
        self._active_sessions[session_key] = interrupt_event
        _thread_metadata = _thread_metadata_for_event(event)
        record = reserve_pending_dispatch(self, session_key, event)
        record.task = asyncio.current_task()
        typing_task = self._start_typing_refresh(event, interrupt_event, _thread_metadata)
        try:
            if record.withdrawn:
                return
            await self._run_processing_hook("on_processing_start", event)
            if record.withdrawn:
                await self._run_processing_hook("on_processing_complete", event, ProcessingOutcome.CANCELLED)
                return
            _thread_metadata = _thread_metadata_for_event(event)
            event._turn_marker_handoff = self.gateway_runner is not None  # it can release the marker
            with pending_dispatch_scope(self, session_key, event):
                response = await self._message_handler(event)
            # A muted diagnostic wake ran for the session; its reply is not presented. The
            # policy read binds the routed profile; delivery itself stays in the launch scope.
            with self._media_delivery_scope(event.source):
                if diagnostic_wake_muted(event):
                    response = None
            is_ephemeral_response = isinstance(response, EphemeralReply)
            # Unwrap EphemeralReply for downstream text processing; TTL applies after send.
            response, _ephemeral_ttl = self._unwrap_ephemeral(response)
            # None/empty is normal (streamed/queued). Suppress a stale response after an interrupt.
            if response and interrupt_event.is_set() and session_key in self._pending_messages:
                logger.info("[%s] Suppressing stale response for interrupted session %s", self.name,
                            session_key)
                response = None
            if not response:
                logger.debug("[%s] Handler returned empty/None response for %s", self.name, event.source.chat_id)
            else:
                extracted = await self._extract_response_content(
                    response, event, session_key, is_ephemeral_response=is_ephemeral_response)
                text_content, media_files = extracted.text_content, extracted.media_files
                # Final content gets notify=True; typing metadata stays unmarked (thread-strict).
                _final_thread_metadata = _mark_notify_metadata(_thread_metadata)
                _tts_paths, _tts_requested_path = [], None
                if self._wants_auto_tts(
                        event, session_key, interrupt_event, text_content, media_files):
                    _tts_paths, _tts_requested_path = await self._synthesize_auto_tts(text_content)
                # TTS plays before text; generated files are removed afterwards.
                _tts_caption_delivered = False
                for _tts_index, _tts_path in enumerate(_tts_paths):
                    try:
                        _tts_caption_delivered |= await self._play_tts_file(
                            event, text_content, _tts_path, _tts_index == 0, _final_thread_metadata,
                            _record_delivery)
                    finally:
                        with contextlib.suppress(OSError):
                            os.remove(_tts_path)
                if not _tts_paths and _tts_requested_path is not None:
                    with contextlib.suppress(OSError):
                        os.remove(_tts_requested_path)
                # Suspend the typing refresh before the first delivery attempt, not just in
                # the turn's finally (#117300): if the final send stalls (platform accepted it
                # but the HTTP ack never returns), control never reaches the finally, and
                # _keep_typing keeps refreshing sendChatAction forever while the agent is
                # already idle and the user can read the answer. Reuse the existing
                # _typing_paused mechanism: _keep_typing skips paused chats each tick and
                # _stop_typing_refresh's finally discards it, so it cannot leak into the next
                # turn. No new await on the delivery path (a fire-and-forget stop task was
                # measured to have no effect).
                if text_content or extracted.images or extracted.media_files or extracted.local_files \
                        or _tts_paths or _tts_caption_delivered:
                    self.pause_typing_for_chat(event.source.chat_id)
                if text_content and not _tts_caption_delivered:
                    await self._send_final_text(
                        event, session_key, text_content, _final_thread_metadata,
                        is_ephemeral_response, _ephemeral_ttl, _record_delivery)
                await self._deliver_attachments(
                    event, extracted, _final_thread_metadata,
                    anything_sent=delivery_attempted or _tts_caption_delivered,
                    record_delivery=_record_delivery)
            await self._release_turn_marker(event)
            processing_ok = delivery_succeeded if delivery_attempted else not bool(response)
            # Clean up the per-turn streaming-TTS flag.
            self._streaming_tts_completed_turns.discard(self._streaming_tts_turn_key(
                session_key, getattr(interrupt_event, "_hermes_run_generation", None),
                event=event) or "")
            await self._run_processing_hook(
                "on_processing_complete", event,
                ProcessingOutcome.SUCCESS if processing_ok else ProcessingOutcome.FAILURE)
            if asyncio.current_task() in self._expected_cancelled_tasks:
                return
            # Force-flush an unfired debounce timer so this task hands off to a fresh drain task.
            # Clear the Event BEFORE the stop-typing await so concurrent inbound sees a live guard.
            await self._flush_text_debounce_now(session_key)
            if session_key in self._pending_messages:
                pending_event = self._pending_messages[session_key]
                delay = self._requeue_backoff_delay(session_key, pending_event, event)
                if not delay:  # a backed-off event stays queued until the drain task wakes
                    self._pending_messages.pop(session_key)
                    reserve_pending_dispatch(self, session_key, pending_event).task = asyncio.current_task()
                    self._stage_next_queued_event(session_key, pending_event)
                logger.debug("[%s] Processing queued follow-up message", self.name)
                self._clear_session_guard(session_key)
                await self._stop_typing_refresh(event.source.chat_id, typing_task, metadata=_thread_metadata)
                self._spawn_drain_task(pending_event, session_key, delay=delay)
                return  # Drain task owns the session now.
        except asyncio.CancelledError:
            expected = asyncio.current_task() in self._expected_cancelled_tasks
            await self._run_processing_hook(
                "on_processing_complete", event,
                ProcessingOutcome.CANCELLED if expected else ProcessingOutcome.FAILURE)
            raise
        except BaseException as e:
            await self._run_processing_hook("on_processing_complete", event, ProcessingOutcome.FAILURE)
            logger.error("[%s] Error handling message: %s", self.name, e, exc_info=True)
            _thread_metadata = (await self._notify_turn_error(event, e)) or _thread_metadata
            # SystemExit/KeyboardInterrupt propagate; other BaseExceptions are contained.
            if isinstance(e, (SystemExit, KeyboardInterrupt)):
                raise
        finally:
            release_pending_dispatch_record(self, session_key, record)
            await self._release_turn_marker(event)
            event._turn_marker_handoff = False  # a later run of this object clears its own marker
            # Stop typing BEFORE the post-delivery callback: a stuck callback must not keep it
            # alive.
            await self._stop_typing_refresh(event.source.chat_id, typing_task, metadata=_thread_metadata)
            await self._fire_post_delivery_callback(session_key, interrupt_event)
            # Callback work or a late refresh may have recreated typing — one final bounded stop.
            await self._stop_typing_refresh(
                event.source.chat_id, None, metadata=_thread_metadata, stop_attempts=1)
            # Flush any timer that missed the in-band drain, then reconcile ownership.
            await self._flush_text_debounce_now(session_key)
            if asyncio.current_task() in self._expected_cancelled_tasks:
                self._cleanup_finished_session_task(session_key, interrupt_event)
            else:
                self._finish_session_task(session_key, interrupt_event)

    def _spawn_drain_task(self, pending_event: MessageEvent, session_key: str,
                          delay: float = 0.0) -> None:
        """Hand the session to a fresh task for a queued follow-up — never recurse (chained
        follow-ups grew the C stack to SIGSEGV). Clearing (not deleting) the Event keeps the guard
        live for concurrent inbound; ownership moves so stale-lock detection works. With ``delay``
        the event stays in ``_pending_messages`` and the new owner task pops the slot only after
        sleeping, so a cancel/discard during the back-off needs no put-back and can't drop a
        newer message."""
        self._clear_session_guard(session_key)
        # Capture the guard this drain owns now: a /stop//new guard swapped in during the
        # back-off must survive the slot-empty exit (#48300).
        guard = self._active_sessions.get(session_key)
        if not delay:
            reserve_pending_dispatch(self, session_key, pending_event)
        self._track_session_task(
            session_key,
            asyncio.create_task(self._drain_after(pending_event, session_key, delay, guard)), pending_event)

    async def _drain_after(self, pending_event: MessageEvent, session_key: str, delay: float,
                           guard: Optional[asyncio.Event]) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
            await self._flush_text_debounce_now(session_key)  # as every other task exit does
            pending_event = self._pending_messages.pop(session_key, None)
            if pending_event is None:  # consumed elsewhere during the back-off
                self._cleanup_finished_session_task(session_key, guard)
                return
            self._stage_next_queued_event(session_key, pending_event)
        await self._process_message_background(pending_event, session_key)

    async def cancel_background_tasks(self) -> None:
        """Cancel in-flight background tasks (shutdown/replacement); 5s bound each,
        stragglers are untracked and left to unwind."""
        pending_reservations: dict[str, list[_PendingDispatchReservation]] = {}
        # Re-drain (max 5 rounds): a message arriving mid-gather spawns a task clear() would
        # untrack.
        for _ in range(5):
            for key in list(getattr(self, "_pending_dispatch_reservations", {})):
                recorded = pending_reservations.setdefault(key, [])
                for reserved in pending_dispatch_records(self, key):
                    if not any(previous is reserved for previous in recorded):
                        recorded.append(reserved)
            candidates = list(self._background_tasks)
            for attr in ("_pending_text_batch_tasks", "_pending_photo_batch_tasks", "_media_group_tasks"):
                candidates.extend(getattr(self, attr, {}).values())
            held_task = getattr(self, "_held_inbound_redispatch_task", None)
            if held_task is not None:
                candidates.append(held_task)
            tasks = [task for task in set(candidates) if not task.done() and task is not asyncio.current_task()]
            if not tasks:
                break
            for task in tasks:
                self._expected_cancelled_tasks.add(task)
                task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*(asyncio.shield(t) for t in tasks), return_exceptions=True),
                    timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("[%s] %d background task(s) did not exit within 5s; "
                               "releasing tracking and letting them unwind in the background",
                               self.name, sum(not t.done() for t in tasks))
                break
        from gateway.shutdown_pending import discard_persisted_ingress, flush_adapter_pending
        persisted = flush_adapter_pending(self, pending_reservations)
        discard_persisted_ingress(self, persisted)
        for key in persisted:
            self._pending_messages.pop(key, None)
            state = self._text_debounce_store().pop(key, None)
            if state is not None:
                state.cancel_timer()
            getattr(self, "_pending_dispatch_reservations", {}).pop(key, None)
        for key, records in pending_reservations.items():
            for reserved in records:
                if key not in persisted and not reserved.claimed and not reserved.withdrawn:
                    if not any(existing is reserved for existing in pending_dispatch_records(self, key)):
                        reserved.previous = getattr(self, "_pending_dispatch_reservations", {}).get(key)
                        self._pending_dispatch_reservations[key] = reserved
        for state in self._text_debounce_store().values():
            state.cancel_timer()
        for bucket in (self._background_tasks, self._expected_cancelled_tasks, self._session_tasks,
                       self._active_sessions, self._requeue_counts):
            bucket.clear()
