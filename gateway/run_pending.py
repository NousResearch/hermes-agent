"""Generation-owned draining and recursive follow-ups for GatewayRunner.

Mixed into gateway.run_turn; facade helpers are imported at their production read seam.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, Optional

from gateway.platforms.base import ProcessingOutcome
from gateway.platforms.event import MessageEvent, event_preserves_prompt_pins
from gateway.response_filters import display_kind_for_event, reply_expected_metadata
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from gateway.warning_notifications import diagnostic_metadata

logger = logging.getLogger("gateway.run")


class GatewayPendingTurnMixin:
    """Keep accepted follow-ups attached to the turn that dequeued them."""

    def _run_agent_restage_pending_steer(
        self, steer: Optional[str], adapter: Any, source: SessionSource, session_key: Optional[str],
        run_generation: Optional[int],
    ) -> None:
        """Retain an accepted late correction with its priority over the queued next turn."""
        if not steer or not session_key or not self._run_still_current_fn(session_key, run_generation)():
            return
        from gateway.shutdown_flush import mark_pending_execution_front

        event = self._synthetic_prompt_event(source, steer)
        slot = getattr(adapter, "_pending_messages", None) if adapter is not None else None
        before = slot.get(session_key) if isinstance(slot, dict) else None
        before = before or next(iter(self._overflow_queue(session_key) or ()), None)
        mark_pending_execution_front(event, before if isinstance(before, MessageEvent) else None)
        event._gateway_accepted = True
        self._restage_accepted_front(session_key, adapter, event, run_generation=run_generation)

    def _run_agent_restore_queued_followup(
        self, turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
    ) -> None:
        """Before model handoff, the dequeuing turn still owns this accepted work."""
        if pending_event is not None:
            self._restage_accepted_front(
                turn_ctx.session_key, adapter, pending_event, run_generation=turn_ctx.run_generation)
        else:
            self._run_agent_restage_pending_steer(
                pending, adapter, turn_ctx.source, turn_ctx.session_key, turn_ctx.run_generation)

    def _run_agent_hold_queued_followup(
        self, turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
    ) -> bool:
        """A stopped owner discards its work; a draining owner retains it for restart."""
        if not self._run_still_current_fn(turn_ctx.session_key, turn_ctx.run_generation)():
            return True
        if not self._draining:
            return False
        self._run_agent_restore_queued_followup(turn_ctx, adapter, pending, pending_event)
        return True

    async def _run_agent_dequeue_followup(
        self, result: Any, adapter: Any, source: SessionSource, session_key: Optional[str],
        run_generation: Optional[int], leftover_steer: Optional[str],
    ) -> tuple[Any, Optional[str]]:
        """Claim the FIFO head and transcribe it while its finishing generation owns it."""
        from gateway.run import (
            _build_media_placeholder,
            _dequeue_pending_event,
            _is_control_interrupt_message,
        )

        if not result or not adapter or not session_key:
            return None, None
        pending_event = _dequeue_pending_event(adapter, session_key)
        pending_event = self._promote_queued_event(session_key, adapter, pending_event)
        pending = None
        if result.get("interrupted") and not pending_event and result.get("interrupt_message"):
            interrupt_message = result.get("interrupt_message")
            if _is_control_interrupt_message(interrupt_message):
                logger.info("Ignoring control interrupt message for session %s: %s", session_key, interrupt_message)
            else:
                pending = interrupt_message
        elif pending_event:
            _pending_text = pending_event.text or ""
            try:
                if self._pending_event_audio_paths(pending_event):
                    pending, _ = await self._transcribe_and_echo_pending_voice(
                        pending_event, adapter, source, _pending_text, log_context="Voice-drain",
                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                    )
                    pending = pending or _build_media_placeholder(pending_event)
                else:
                    pending = _pending_text or _build_media_placeholder(pending_event)
            except BaseException:
                self._restage_accepted_front(session_key, adapter, pending_event, run_generation=run_generation)
                self._run_agent_restage_pending_steer(leftover_steer, adapter, source, session_key, run_generation)
                raise
            if pending:
                logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])
        return pending_event, pending

    @staticmethod
    def _run_agent_pending_is_command(pending: Optional[str]) -> bool:
        """Recognized slash commands never become queued model input."""
        if pending and pending.strip().startswith("/"):
            command = pending.strip().split(None, 1)[0][1:].lower()
            if command:
                with suppress(Exception):
                    from hermes_cli.commands import resolve_command
                    if resolve_command(command):
                        logger.info(
                            "Discarding command '/%s' from pending queue — commands must not be passed as agent input",
                            command,
                        )
                        return True
        return False

    async def _run_agent_drain_pending(
        self, result: Any, adapter: Any, source: SessionSource, session_key: Optional[str], *,
        run_generation: Optional[int] = None,
    ) -> tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up as ``(pending_event, pending)``.

        Keyed by session_key (not source.chat_id) to match the adapter's storage keys."""
        pending_event = None
        pending = None
        _run_still_current = self._run_still_current_fn(session_key, run_generation)
        if not _run_still_current():
            return None, None
        leftover_steer = (result.get("pending_steer") or "").strip() if result else None
        if self._draining:
            # Shutdown owns the queue snapshot; do not consume its accepted work (#126167).
            logger.info(
                "Leaving queued follow-ups intact for session %s during gateway %s",
                session_key or "?", self._status_action_label(),
            )
            self._run_agent_restage_pending_steer(leftover_steer, adapter, source, session_key, run_generation)
            return pending_event, pending
        pending_event, pending = await self._run_agent_dequeue_followup(
            result, adapter, source, session_key, run_generation, leftover_steer)

        # Post-transcription fence: a /new or /stop invalidated the run generation while the
        # drain was suspended, so the stale A (and any steer) must not be returned or restaged.
        if not _run_still_current():
            return None, None

        # Leftover /steer (arrived after the last tool batch): deliver as the next user turn.
        if leftover_steer:
            if not pending and not pending_event:
                pending = leftover_steer
                logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])
            elif pending_event and adapter and session_key:
                # User steered during the turn, but a background event or queued message arrived.
                # Deliver the user's steer first and restore the pending event to the head of the queue.
                self._restage_accepted_front(session_key, adapter, pending_event, run_generation=run_generation)
                pending_event = None
                pending = leftover_steer
                logger.debug(
                    "Delivering leftover /steer before queued event for session %s: '%s...'",
                    session_key, pending[:40],
                )
            elif pending and not pending_event:
                # A deferred steer continues the active channel context, just like an eventless
                # follow-up: mark it preserving so the drain reuses the session's pins, without
                # pre-resolving them here or marking the user as internal.
                steer_event = MessageEvent(
                    text=leftover_steer, source=source, preserve_prompt_pins=True,
                )
                if session_key:
                    self._enqueue_fifo(session_key, steer_event, adapter)
                logger.debug(
                    "Enqueued leftover /steer behind pending message for session %s: '%s...'",
                    session_key or "?", leftover_steer[:40],
                )

        # Safety net: a pending slash command is never passed to the agent as user input.
        if self._run_agent_pending_is_command(pending):
            pending_event = None
            pending = None

        if self._draining and (pending_event or pending):
            logger.info(
                "Stopping follow-up dispatch for session %s during gateway %s",
                session_key or "?", self._status_action_label(),
            )
            self._restage_accepted_front(
                session_key, adapter, pending_event, run_generation=run_generation)
            self._run_agent_restage_pending_steer(leftover_steer, adapter, source, session_key, run_generation)
            pending_event = None
            pending = None
        return pending_event, pending

    async def _run_agent_restart_followup_typing(self, turn_ctx: TurnContext) -> None:
        """The next voice turn owns its own TTS marker and typing indicator."""
        adapter = self._delivery_adapter_for(turn_ctx.source)
        completed = getattr(adapter, "_streaming_tts_completed_turns", None)
        turn_key = getattr(adapter, "_streaming_tts_turn_key", None)
        if completed is not None and callable(turn_key) and turn_ctx.session_key and turn_ctx.run_generation is not None:
            prior_key = turn_key(turn_ctx.session_key, turn_ctx.run_generation)
            if prior_key:
                completed.discard(prior_key)
        if adapter:
            with suppress(Exception):
                await adapter.send_typing(turn_ctx.source.chat_id, metadata=turn_ctx._status_thread_metadata)

    async def _run_agent_prepare_queued_followup(
        self, turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
        response: Any, result: Any, stream_task: Any,
    ) -> Optional[dict]:
        """Prepare the next turn only while the dequeuing turn still owns the session."""
        def hold() -> bool:
            return self._run_agent_hold_queued_followup(turn_ctx, adapter, pending, pending_event)
        if not result.get("interrupted"):
            await self._run_agent_deliver_first_response(turn_ctx, adapter, response, result, stream_task)
            if hold():
                return None
        next_reply_expected = pending_event.reply_expected if pending_event is not None else None
        next_turn = {
            "message": pending, "context_prompt": turn_ctx.context_prompt,
            "history": result.get("messages", turn_ctx.history), "source": turn_ctx.source,
            "session_id": turn_ctx.session_id, "session_key": turn_ctx.session_key,
            "run_generation": turn_ctx.run_generation, "_interrupt_depth": turn_ctx._interrupt_depth + 1,
            "event_message_id": None, "inbound_message_id": None, "channel_prompt": None,
            "message_type": None, "persist_user_message": None,
            "persist_user_display_kind": display_kind_for_event(pending_event),
            "reply_expected": next_reply_expected,
            "persist_user_display_metadata": {
                **reply_expected_metadata(next_reply_expected), **diagnostic_metadata(pending_event)} or None,
        }
        if pending_event is not None:
            next_source = getattr(pending_event, "source", None) or turn_ctx.source
            if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(turn_ctx.session_id):
                logger.info("Discarding stale goal continuation for session %s: goal is no longer active", turn_ctx.session_key or "?")
                return None
            # Native-image preparation and consumption must use the follow-up's own routing key.
            try:
                next_turn["session_key"] = self._session_key_for_source(next_source)
            except Exception:
                logger.debug("Queued follow-up session-key resolution failed; reusing %s", turn_ctx.session_key or "?", exc_info=True)
            next_message = await self._prepare_profile_scoped_inbound_message_text(
                event=pending_event, source=next_source, history=next_turn["history"], session_key=next_turn["session_key"],
            )
            if hold() or next_message is None:
                return None
            from gateway.run_inbound import strip_discord_triggering_note
            next_turn["message"] = next_message
            next_turn["persist_user_message"] = strip_discord_triggering_note(pending_event, next_message)
            next_turn["event_message_id"] = self._reply_anchor_for_event(pending_event)
            # A topic's reply anchor can be None; its raw inbound id still owns a final obligation.
            next_turn["inbound_message_id"] = str(pending_event.message_id) if getattr(pending_event, "message_id", None) else None
            next_turn["channel_prompt"], next_turn["source"] = self._pinned_channel_inputs(
                next_turn["session_key"], pending_event.channel_prompt, next_source,
                preserve_pin=event_preserves_prompt_pins(pending_event),
            )
            # Every turn persists; the store no-ops an unchanged snapshot, so this only writes
            # when a preserving turn re-rendered (a /sethome config refresh) — stale durable
            # bytes would resurrect the old home after a crash/restart.
            await self._persist_prompt_pins(next_turn["session_key"], turn_ctx.session_id)
            if not event_preserves_prompt_pins(pending_event):
                if hold():
                    return None
            next_turn["message_type"] = getattr(pending_event, "message_type", None)
        else:
            next_turn["channel_prompt"] = turn_ctx.channel_prompt
        await self._run_agent_restart_followup_typing(turn_ctx)
        return None if hold() else next_turn

    async def _run_agent_queued_followup(
        self, turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
        response: Any, result: Any, stream_task: Any,
    ) -> Any:
        """Run the queued / interrupting follow-up as the next turn (recursive ``_run_agent``)."""
        from gateway.run import _preserve_queued_followup_history_offset
        from gateway.run_turn_followup_ack import (
            _followup_cancel_outcome,
            _run_followup_processing_hook,
        )

        def hold() -> bool:
            return self._run_agent_hold_queued_followup(turn_ctx, adapter, pending, pending_event)
        if hold():
            return result
        logger.debug("Processing pending message: '%s...'", pending[:40])
        active = getattr(adapter, "_active_sessions", None) if adapter else None
        if active and turn_ctx.session_key and turn_ctx.session_key in active:
            active[turn_ctx.session_key].clear()
        if turn_ctx._interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
            logger.warning(
                "Interrupt recursion depth %d reached for session %s: queueing message instead of recursing.",
                turn_ctx._interrupt_depth, turn_ctx.session_key,
            )
            adapter = self._delivery_adapter_for(turn_ctx.source)
            if adapter and pending_event:
                self._restage_accepted_front(
                    turn_ctx.session_key, adapter, pending_event, run_generation=turn_ctx.run_generation)
            elif adapter and hasattr(adapter, 'queue_message'):
                adapter.queue_message(turn_ctx.session_key, pending)
            return turn_ctx.result_holder[0] or {"final_response": response, "messages": turn_ctx.history}

        try:
            next_turn = await self._run_agent_prepare_queued_followup(
                turn_ctx, adapter, pending, pending_event, response, result, stream_task)
        except BaseException:
            self._run_agent_restore_queued_followup(turn_ctx, adapter, pending, pending_event)
            raise
        if next_turn is None:
            return result
        hook_adapter = self._intake_adapter_for(next_turn["source"]) if pending_event is not None else None
        turn_started = False
        try:
            await _run_followup_processing_hook(hook_adapter, pending_event, "on_processing_start")
            if hold():
                await _run_followup_processing_hook(hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.CANCELLED)
                return result
            # Keep the cached prefix: this in-band turn must acknowledge the rows its predecessor flushed.
            await self._refresh_agent_cache_message_count(turn_ctx.session_key, turn_ctx.session_id)
            if hold():
                await _run_followup_processing_hook(hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.CANCELLED)
                return result
            # The recursive model turn owns the event from here; replaying its failure would duplicate it.
            turn_started = True
            followup_result = await self._run_agent(**next_turn)
        except BaseException as exc:
            if not turn_started:
                self._run_agent_restore_queued_followup(turn_ctx, adapter, pending, pending_event)
            outcome = (
                _followup_cancel_outcome(hook_adapter)
                if isinstance(exc, asyncio.CancelledError) else ProcessingOutcome.FAILURE)
            await _run_followup_processing_hook(hook_adapter, pending_event, "on_processing_complete", outcome)
            raise
        await _run_followup_processing_hook(hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.SUCCESS)
        merged = _preserve_queued_followup_history_offset(result, followup_result)
        # The terminal queued turn owns the final ledger identity; an inner recursion wins.
        if isinstance(merged, dict) and "queued_terminal_inbound_id" not in merged:
            merged = {
                **merged, "queued_terminal_inbound_id": next_turn["inbound_message_id"],
                "queued_terminal_display_kind": next_turn["persist_user_display_kind"],
                "queued_terminal_reply_expected": next_turn["reply_expected"],
                "queued_terminal_notification_category": (
                    (pending_event.metadata or {}).get("notification_category", "result")
                    if pending_event is not None and pending_event.internal else "result"),
            }
        return merged
