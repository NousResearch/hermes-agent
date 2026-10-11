"""Processing-lifecycle hooks for runner-drained queued follow-ups.

A message that arrives mid-turn is parked in the adapter's pending slot and drained in-band by
``TurnRunner._run_agent_queued_followup``, never by ``BasePlatformAdapter._process_message_background``
— the only other call site for ``on_processing_start`` / ``on_processing_complete``. Without firing
them here every adapter that renders a read-receipt reaction from the hooks silently skips queued,
interrupting and steer-demoted messages (#72502, salvage #72503).
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Optional
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome


def _followup_processing_hooks_apply(adapter, event: MessageEvent | None) -> bool:
    """Both conditions are necessary: a real inbound platform message to acknowledge (adapters key their
    marker off ``message_id`` — Slack/Telegram/Feishu/Matrix/Photon — or off ``raw_message`` — Signal,
    Discord; synthetic drains such as ``/goal`` continuations, wake-ups and startup auto-resume carry
    neither and must stay silent), and an adapter that overrides ``on_processing_start``. We bracket, so
    both halves must belong to us: a complete-only adapter (Google Chat reaps its typing card there,
    webhook ends its per-delivery session) would otherwise be handed a completion for a turn whose reply
    is delivered only after the drain chain unwinds back into ``_process_message_background``."""
    if adapter is None or event is None:
        return False
    if not (getattr(event, "message_id", None) or getattr(event, "raw_message", None)):
        return False
    start_hook = getattr(type(adapter), "on_processing_start", None)
    return start_hook is not None and start_hook is not BasePlatformAdapter.on_processing_start


def _followup_cancel_outcome(adapter) -> ProcessingOutcome:
    """Classify a cancelled follow-up exactly as ``_process_message_background`` does: only cancels the
    adapter itself routed (``/stop``, ``/new``, ``/reset``, cleanup) are CANCELLED, anything else is a
    failure. Signal and Matrix leave the in-progress marker in place on CANCELLED, so reporting an
    unexpected cancellation as CANCELLED would strand it."""
    expected = getattr(adapter, "_expected_cancelled_tasks", None)
    if expected is None:
        return ProcessingOutcome.FAILURE
    try:
        current = asyncio.current_task()
    except RuntimeError:
        current = None
    if current is None:
        return ProcessingOutcome.FAILURE
    try:
        return ProcessingOutcome.CANCELLED if current in expected else ProcessingOutcome.FAILURE
    except TypeError:
        return ProcessingOutcome.FAILURE


async def _run_followup_processing_hook(adapter, event: MessageEvent | None, hook_name: str, *args) -> None:
    """Fire one lifecycle hook for a runner-drained follow-up; no-op per ``_followup_processing_hooks_apply``."""
    if not _followup_processing_hooks_apply(adapter, event):
        return
    run_hook = getattr(adapter, "_run_processing_hook", None)
    if not callable(run_hook):
        return
    await run_hook(hook_name, event, *args)


class GatewayPendingFollowupMixin:

    async def _run_agent_drain_pending(
        self, result: Any, adapter: Any, source: SessionSource, session_key: Optional[str]
    ) -> tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up as ``(pending_event, pending)``.

        Keyed by session_key (not source.chat_id) to match the adapter's storage keys."""
        from gateway.run import (
            _build_media_placeholder, _dequeue_pending_event, _is_control_interrupt_message
        )
        pending_event = None
        pending = None
        if result and adapter and session_key:
            pending_event = _dequeue_pending_event(adapter, session_key)
            # /queue overflow: promote the next queued event into the consumed "next-up" slot so the
            # recursive drain sees it (keeps FIFO order; a mid-chain /queue can't jump the queue).
            pending_event = self._promote_queued_event(session_key, adapter, pending_event)
            # A consumed wake must not strand the FIFO sibling behind it.
            pending_event = self._next_unconsumed_pending_completion(pending_event, adapter, session_key)
            if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
                interrupt_message = result.get("interrupt_message")
                if _is_control_interrupt_message(interrupt_message):
                    logger.info(
                        "Ignoring control interrupt message for session %s: %s",
                        session_key or "?", interrupt_message,
                    )
                else:
                    pending = interrupt_message
            elif pending_event:
                # Transcribe audio BEFORE it becomes the next user turn (real transcript, not a path).
                _pending_text = pending_event.text or ""
                if self._pending_event_audio_paths(pending_event):
                    pending, _ = await self._transcribe_and_echo_pending_voice(
                        pending_event, adapter, source, _pending_text, log_context="Voice-drain",
                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                    )
                    pending = pending or _build_media_placeholder(pending_event)
                else:
                    pending = _pending_text or _build_media_placeholder(pending_event)
                if pending:
                    logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

        # Leftover /steer (arrived after the last tool batch): deliver as the next user turn.
        leftover_steer = (result.get("pending_steer") or "").strip() if result else None
        if leftover_steer:
            if not pending and not pending_event:
                pending = leftover_steer
                logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])
            elif pending_event and adapter and session_key:
                # User steered during the turn, but a background event or queued message arrived.
                # Deliver the user's steer first and restore the pending event to the head of the queue.
                overflow = self._overflow_queue(session_key)
                if hasattr(adapter, "_pending_messages") and isinstance(adapter._pending_messages, dict):
                    promoted = adapter._pending_messages.get(session_key)
                    if promoted is not None:
                        if overflow is not None:
                            overflow.insert(0, promoted)
                        else:
                            self._session_state(session_key).conversation.queued_events.insert(0, promoted)
                    adapter._pending_messages[session_key] = pending_event
                else:
                    if overflow is not None:
                        overflow.insert(0, pending_event)
                    else:
                        self._session_state(session_key).conversation.queued_events.insert(0, pending_event)
                pending_event = None
                pending = leftover_steer
                logger.debug(
                    "Delivering leftover /steer before queued event for session %s: '%s...'",
                    session_key, pending[:40],
                )
            elif pending and not pending_event:
                from gateway.platforms.base import MessageEvent
                # A deferred steer continues the active channel context, just like an
                # eventless follow-up. Reuse its pins without marking the user as internal.
                steer_prompt, steer_source = self._pinned_channel_inputs(
                    session_key, None, source, internal=True,
                )
                steer_event = MessageEvent(
                    text=leftover_steer, source=steer_source, channel_prompt=steer_prompt,
                )
                if session_key:
                    self._enqueue_fifo(session_key, steer_event, adapter)
                logger.debug(
                    "Enqueued leftover /steer behind pending message for session %s: '%s...'",
                    session_key or "?", leftover_steer[:40],
                )

        # Safety net: a pending slash command is never passed to the agent as user input.
        if pending and pending.strip().startswith("/"):
            _pending_cmd_word = pending.strip().split(None, 1)[0][1:].lower()
            if _pending_cmd_word:
                with suppress(Exception):
                    from hermes_cli.commands import resolve_command as _rc_pending
                    if _rc_pending(_pending_cmd_word):
                        logger.info(
                            "Discarding command '/%s' from pending queue — "
                            "commands must not be passed as agent input", _pending_cmd_word,
                        )
                        pending_event = None
                        pending = None

        if self._draining and (pending_event or pending):
            logger.info(
                "Discarding pending follow-up for session %s during gateway %s",
                session_key or "?", self._status_action_label(),
            )
            pending_event = None
            pending = None
        return pending_event, pending
