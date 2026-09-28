"""Adapter dispatch while a session is already active."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agent.i18n import t
from gateway.platforms import base_pending_merge
from gateway.platforms.event import MessageEvent, ProcessingOutcome, MessageType
from gateway.platforms.base_pending import _can_join_pending_event, pending_dispatch_scope, reserve_pending_dispatch, release_pending_dispatch_record

if TYPE_CHECKING:
    from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger("gateway.platforms.base")


class BaseBusyMixin:
    _active_sessions: dict[str, asyncio.Event]
    _pending_messages: dict[str, MessageEvent]
    _busy_session_handler: Callable[[MessageEvent, str], Awaitable[bool]] | None
    _busy_text_debounce_seconds: float
    gateway_runner: Any
    name: str

    if TYPE_CHECKING:
        _canonicalize = BasePlatformAdapter._canonicalize
        _discard_text_debounce = BasePlatformAdapter._discard_text_debounce
        _dispatch_active_session_command = BasePlatformAdapter._dispatch_active_session_command
        _dispatch_inline_reply = BasePlatformAdapter._dispatch_inline_reply
        _is_queue_text_debounce_candidate = BasePlatformAdapter._is_queue_text_debounce_candidate
        _queue_text_debounce = BasePlatformAdapter._queue_text_debounce
        _send_with_retry = BasePlatformAdapter._send_with_retry
        _stage_next_queued_event = BasePlatformAdapter._stage_next_queued_event
        _start_session_processing = BasePlatformAdapter._start_session_processing
        _text_debounce_store = BasePlatformAdapter._text_debounce_store

    async def _notify_busy_queue_refusal(self, event: MessageEvent) -> None:
        from gateway.platforms.base import _thread_metadata_for_event

        notify = getattr(self.gateway_runner, "_send_pending_queue_refusal", None)
        if callable(notify):
            await notify(event, self)
            return
        await self._send_with_retry(
            chat_id=event.source.chat_id, content=t("gateway.queue.full"),
            reply_to=event.message_id, metadata=_thread_metadata_for_event(event))

    async def _handle_message_while_active(self: BasePlatformAdapter, event: MessageEvent, session_key: str) -> None:
        """Route a message that arrived while ``session_key`` is busy: bypass
        commands / clarify replies dispatch inline, everything else is queued."""
        # Bypass commands run inline: queued they'd leak as user text (/new) or deadlock
        # (/approve, /deny — the agent is blocked on Event.wait).  Dispatch inline by
        # calling the message handler directly and sending the response.  Do NOT use
        # _process_message_background — it manages session lifecycle and its cleanup
        # races with the running task (split-brain, see PR #4926).
        # Certain commands must bypass the active-session guard and be dispatched directly to the gateway
        # runner. Without this, they are queued as pending messages and either: See #4926.
        self._canonicalize(event.source)  # identity FIRST (direct callers may skip handle_message)
        cmd = event.get_command()
        from hermes_cli.commands import (is_interrupt_then_dispatch, should_bypass_active_session)
        if should_bypass_active_session(cmd):
            try:
                # /stop, /new, /reset: cancel + response + drain; other bypasses don't cancel.
                if cmd and is_interrupt_then_dispatch(cmd):
                    self._discard_text_debounce(session_key)
                    await self._dispatch_active_session_command(event, session_key, cmd)
                else:
                    logger.debug("[%s] Command '/%s' bypassing active-session guard for %s",
                                 self.name, cmd, session_key)
                    await self._dispatch_inline_reply(event)
            except Exception as e:
                await self._run_processing_hook("on_inline_processing_complete", event, ProcessingOutcome.FAILURE)
                logger.error("[%s] Command '/%s' dispatch failed: %s", self.name, cmd, e, exc_info=True)
            return
        # Clarify bypass: while blocked on clarify_tool the next message must reach the
        # text-intercept so numeric/exact/"Other" answers resolve it and unblock the agent.
        # Otherwise it lands in _pending_messages as a follow-up turn and the answer is
        # discarded.  Same shape as the /approve deadlock fix (PR #4926): agent thread
        # blocked on Event.wait, message must reach the resolver before being a new turn.
        # See #4926.
        if not cmd and event.allow_gateway_control:
            try:
                from tools import clarify_gateway as _clarify_mod
                _has_text_clarify = _clarify_mod.get_pending_for_session(
                    session_key, include_choice_prompts=True) is not None
            except Exception:
                _has_text_clarify = False
            if _has_text_clarify:
                logger.debug("[%s] Routing message to clarify text-intercept for %s", self.name, session_key)
                try:
                    await self._dispatch_inline_reply(event)
                except Exception as e:
                    await self._run_processing_hook("on_inline_processing_complete", event, ProcessingOutcome.FAILURE)
                    logger.error("[%s] Clarify text-intercept dispatch failed: %s", self.name, e, exc_info=True)
                return
        reservation = reserve_pending_dispatch(self, session_key, event, accepted=False)
        reservation.task = asyncio.current_task()
        try:
            with pending_dispatch_scope(self, session_key, event):
                if self._busy_session_handler is not None:
                    try:
                        handled = await self._busy_session_handler(event, session_key)
                    except Exception as e:
                        logger.error("[%s] Busy-session handler failed: %s", self.name, e, exc_info=True)
                        # It may have stored the event before raising: queuing or starting it again below
                        # would run it twice.
                        handled = event._gateway_accepted is True
                    if reservation.withdrawn:
                        return
                    # The handler awaits (profile scope load, compression-lock read). If the owner task
                    # finished meanwhile, it found the slot empty and released the guard, so nothing would
                    # drain what the handler queued: start that now. If the handler left this event to the
                    # base path instead (returned False, or raised before storing it) and nothing is
                    # queued, start this event.
                    if session_key not in self._active_sessions:
                        # Busy admission queues through the delivery adapter resolved when it stored the
                        # event; a reconnect during the handler's later awaits can replace that adapter.
                        # Recover from whichever slot actually holds it — this one, or the replacement —
                        # and start it on that slot's owner. A replacement with a live guard drains its
                        # own slot, so leave that one alone.
                        owners = [self]
                        runner = getattr(self, "gateway_runner", None)
                        if runner is not None:
                            try:
                                delivery_adapter = runner._delivery_adapter_for(event.source)
                            except Exception:
                                delivery_adapter = None
                                logger.debug("[%s] Delivery-adapter lookup failed during pending recovery",
                                             self.name, exc_info=True)
                            if (delivery_adapter is not None and delivery_adapter is not self
                                    and session_key not in delivery_adapter._active_sessions):
                                owners.append(delivery_adapter)
                        orphan = None
                        for owner in owners:
                            orphan = owner.get_pending_message(session_key)
                            if orphan is not None:
                                owner._stage_next_queued_event(session_key, orphan)
                                owner._start_session_processing(orphan, session_key)
                                break
                        if orphan is None and not handled:
                            event._gateway_accepted = self._start_session_processing(event, session_key)
                            return
                    if handled:
                        return
                # Without a runner FIFO, do not merge a wake into an occupied human slot
                # (or collapse distinct wakes into one turn). Its caller can retry admission.
                if event.internal and session_key in self._pending_messages:
                    return
                existing = self._pending_messages.get(session_key)
                if self._text_debounce_store().get(session_key) is not None or (
                    existing is not None and not _can_join_pending_event(existing, event)
                ):
                    enqueue = getattr(self.gateway_runner, "_queue_or_replace_pending_event", None)
                    event._gateway_accepted = (
                        await self._queue_text_debounce(session_key, event)
                        if self._text_debounce_store().get(session_key) is not None or not callable(enqueue)
                        else enqueue(session_key, event)
                    )
                    if not event._gateway_accepted:
                        await self._notify_busy_queue_refusal(event)
                    return
                # Photo bursts/albums: queue without interrupting; they run after the current task.
                if event.message_type == MessageType.PHOTO:
                    logger.debug("[%s] Queuing photo follow-up for session %s without interrupt", self.name, session_key)
                    base_pending_merge.merge_pending_message_event(self._pending_messages, session_key, event)
                    event._gateway_accepted = True
                    return
                if self._is_queue_text_debounce_candidate(event):
                    logger.debug("[%s] New text message while session %s is active — "
                                 "debouncing follow-up (busy_text_mode=queue, window=%.2fs)", self.name,
                                 session_key, self._busy_text_debounce_seconds)
                    event._gateway_accepted = await self._queue_text_debounce(session_key, event)
                    if not event._gateway_accepted:
                        await self._notify_busy_queue_refusal(event)
                else:
                    logger.debug("[%s] New message while session %s is active — queuing follow-up "
                                 "(no interrupt, will cascade after current turn)", self.name, session_key)
                    base_pending_merge.merge_pending_message_event(self._pending_messages, session_key, event,
                                                merge_text=event.message_type == MessageType.TEXT)
                    event._gateway_accepted = True
        finally:
            if reservation.task is asyncio.current_task():
                release_pending_dispatch_record(self, session_key, reservation)
