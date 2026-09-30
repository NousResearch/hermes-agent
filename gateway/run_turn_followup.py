"""Recursive execution of queued gateway follow-up turns."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Optional

from gateway.input_owner import gateway_input_owner
from gateway.run_inbound_logging import log_inbound_reply_context
from gateway.platforms.event import MessageEvent, ProcessingOutcome, _ProcessingCompletion
from gateway.platforms.base_pending import reserve_pending_dispatch, release_pending_dispatch_record
from gateway.response_filters import display_kind_for_event, reply_expected_metadata
from gateway.run_inbound_turn_context import channel_state_metadata
from gateway.session import SessionContext
from gateway.turn_context import TurnContext
from gateway.warning_notifications import diagnostic_metadata

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")


class GatewayQueuedFollowupMixin:
    _MAX_INTERRUPT_DEPTH: int
    _reply_anchor_for_event: Callable[[MessageEvent], str | None]

    if TYPE_CHECKING:
        _delivery_adapter_for = GatewayRunner._delivery_adapter_for
        _goal_still_active_for_session = GatewayRunner._goal_still_active_for_session
        _intake_adapter_for = GatewayRunner._intake_adapter_for
        _is_goal_continuation_event = GatewayRunner._is_goal_continuation_event
        _persist_prompt_pins = GatewayRunner._persist_prompt_pins
        _pinned_channel_inputs = GatewayRunner._pinned_channel_inputs
        _prepare_profile_scoped_inbound_message_text = GatewayRunner._prepare_profile_scoped_inbound_message_text
        _refresh_agent_cache_message_count = GatewayRunner._refresh_agent_cache_message_count
        _run_agent = GatewayRunner._run_agent
        _run_agent_deliver_first_response = GatewayRunner._run_agent_deliver_first_response
        _session_env_scope = GatewayRunner._session_env_scope
        _session_key_for_source = GatewayRunner._session_key_for_source
        _restore_pending_dispatch = GatewayRunner._restore_pending_dispatch
        _strict_session_current = GatewayRunner._strict_session_current

    async def _run_agent_queued_followup(
        self: "GatewayRunner", turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
        response: Any, result: Any, stream_task: Any,
    ) -> Any:
        """Run the queued / interrupting follow-up as the next turn (recursive ``_run_agent``).

        A started message that waits for this follow-up (a queued message, or a leftover steer that
        the drain handed to it) has no other completion: its own handler skipped it. Every exit
        before the follow-up starts therefore completes it here, except at the recursion cap, where
        the follow-up returns to the pending slot."""
        from gateway.run_turn_followup_ack import _followup_cancel_outcome

        try:
            merged = await self._run_agent_queued_followup_turn(
                turn_ctx, adapter, pending, pending_event, response, result, stream_task)
        except asyncio.CancelledError:
            await self._complete_unstarted_followup(
                pending_event, _followup_cancel_outcome(self._followup_hook_adapter(pending_event)))
            raise
        except BaseException:
            await self._complete_unstarted_followup(pending_event, ProcessingOutcome.FAILURE)
            raise
        # At the recursion cap the follow-up returns to the pending slot and runs later.
        if turn_ctx._interrupt_depth < self._MAX_INTERRUPT_DEPTH:
            await self._complete_unstarted_followup(pending_event, ProcessingOutcome.FAILURE)
        return merged

    def _followup_hook_adapter(self: "GatewayRunner", pending_event: Any) -> Any:
        source = getattr(pending_event, "source", None)
        return self._intake_adapter_for(source) if source is not None else None

    async def _complete_unstarted_followup(self: "GatewayRunner", pending_event: Any, outcome: ProcessingOutcome) -> None:
        """Complete a started follow-up, and the messages attached to it, that this lane took from
        the queue but never started."""
        await self._complete_discarded_event(pending_event, outcome)

    async def _run_agent_queued_followup_turn(
        self: "GatewayRunner", turn_ctx: TurnContext, adapter: Any, pending: Optional[str], pending_event: Any,
        response: Any, result: Any, stream_task: Any,
    ) -> Any:
        """Run the queued / interrupting follow-up as the next turn (recursive ``_run_agent``)."""
        reservation = None
        if pending_event is not None and adapter is not None and turn_ctx.session_key:
            reservation = reserve_pending_dispatch(adapter, turn_ctx.session_key, pending_event)
            reservation.task = asyncio.current_task()
        try:
            from gateway.run import _preserve_queued_followup_history_offset
            source, session_id, session_key, run_generation = (
                turn_ctx.source, turn_ctx.session_id, turn_ctx.session_key, turn_ctx.run_generation,
            )
            _interrupt_depth, history, _status_thread_metadata = (
                turn_ctx._interrupt_depth, turn_ctx.history, turn_ctx._status_thread_metadata,
            )
            logger.debug("Processing pending message: '%s...'", pending[:40])

            # Clear the interrupt event so the recursive _run_agent isn't re-interrupted (infinite loop).
            _active = getattr(adapter, "_active_sessions", None) if adapter else None
            if _active and session_key and session_key in _active:
                _active[session_key].clear()

            # Cap recursion depth (user keeps sending while the agent keeps failing).
            # (#816)
            if _interrupt_depth >= self._MAX_INTERRUPT_DEPTH:
                logger.warning(
                    "Interrupt recursion depth %d reached for session %s — "
                    "queueing message instead of recursing.", _interrupt_depth, session_key,
                )
                adapter = self._delivery_adapter_for(source)
                if adapter and pending_event:
                    await self._park_followup_at_recursion_cap(adapter, session_key, pending_event)
                elif adapter and hasattr(adapter, 'queue_message'):
                    adapter.queue_message(session_key, pending)
                return turn_ctx.result_holder[0] or {"final_response": response, "messages": history}

            from gateway.run_turn_followup_ack import (
                _followup_cancel_outcome, _followup_processing_hooks_apply,
                _run_followup_processing_hook, _turn_result_outcome,
            )
            completed_event = turn_ctx.processing_event
            completed_adapter = self._intake_adapter_for(completed_event.source) if completed_event is not None else None
            outcome = _turn_result_outcome(result)
            if not result.get("interrupted"):
                delivered = await self._run_agent_deliver_first_response(turn_ctx, adapter, response, result, stream_task)
                if not delivered:
                    outcome = ProcessingOutcome.FAILURE
            if not result.get("interrupted") or outcome == ProcessingOutcome.CANCELLED:
                await _run_followup_processing_hook(completed_adapter, completed_event, "on_processing_complete", outcome)

            if pending_event is not None and not await self._strict_session_current(
                pending_event, session_key, session_id=session_id,
            ):
                return result

            if reservation is not None and reservation.withdrawn:
                return result
            updated_history = result.get("messages", history)
            next_source, next_message, next_session_key = source, pending, session_key
            # message_type is carried into the recursive call so queued voice turns can stream TTS.
            next_message_id = next_channel_prompt = next_message_type = None
            # The raw inbound id keys the delivery-ledger obligation for the follow-up's own final send,
            # distinct from the reply anchor above (None in forum topics). Carry it or two chained
            # topic turns with the same text would collide on one obligation id (queued-final-ledger).
            next_inbound_id = None
            # Queued Discord turns carry the same routing note as first turns; persist the authored text.
            next_persist_message = None
            next_display_kind = display_kind_for_event(pending_event)
            next_reply_expected = pending_event.reply_expected if pending_event is not None else None
            # See #60671.
            if pending_event is not None:
                next_source = getattr(pending_event, "source", None) or source
                if self._is_goal_continuation_event(pending_event) and not self._goal_still_active_for_session(session_id):
                    logger.info(
                        "Discarding stale goal continuation for session %s — goal is no longer active",
                        session_key or "?",
                    )
                    return result
                # Resolve the follow-up's session key BEFORE preparing the inbound text: native image
                # paths are buffered under the key given and consumed under next_session_key.
                try:
                    next_session_key = self._session_key_for_source(next_source)
                except Exception:
                    logger.debug(
                        "Queued follow-up session-key resolution failed; reusing %s",
                        session_key or "?", exc_info=True,
                    )
                prepared_revision = reservation.revision if reservation is not None else 0
                next_message = await self._prepare_profile_scoped_inbound_message_text(
                    event=pending_event, source=next_source, history=updated_history, session_key=next_session_key,
                )
                if next_message is None:
                    return result
                from gateway.run_inbound_context import strip_discord_triggering_note
                next_persist_message = strip_discord_triggering_note(pending_event, next_message)
                next_message_id = self._reply_anchor_for_event(pending_event)
                next_inbound_id = str(pending_event.message_id) if getattr(pending_event, "message_id", None) else None
                next_channel_prompt, next_source = self._pinned_channel_inputs(
                    next_session_key, pending_event.channel_prompt, next_source, internal=pending_event.internal,
                )
                if not pending_event.internal:
                    # A drained human turn re-pins its channel inputs; make them durable like a first turn.
                    await self._persist_prompt_pins(next_session_key, session_id)
                next_message_type = getattr(pending_event, "message_type", None)
            else:
                # Event-less interrupt/steer follow-ups continue the effective prompt
                # of the turn they are recursively following.
                next_channel_prompt = turn_ctx.channel_prompt

            log_inbound_reply_context(next_source, next_message, pending_event, queued=True)

            # Clear the prior turn's streaming-TTS completion marker so the recursive turn isn't suppressed.
            # See #60671.
            _clear_adapter = self._delivery_adapter_for(source)
            _completed_turns = getattr(_clear_adapter, "_streaming_tts_completed_turns", None)
            _prior_key = getattr(_clear_adapter, "_streaming_tts_turn_key", None)
            if _completed_turns is not None and callable(_prior_key) and session_key and run_generation is not None:
                _pk = _prior_key(session_key, run_generation)
                if _pk:
                    _completed_turns.discard(_pk)

            # Restart the typing indicator; the outer typing task may be stale.
            if _clear_adapter:
                with suppress(Exception):
                    await _clear_adapter.send_typing(source.chat_id, metadata=_status_thread_metadata)

            # Re-baseline the cached agent's message_count before recursing, else the coherence guard
            # rebuilds on OUR OWN flushed rows (the outer handler re-baselines only after the chain).
            # Re-baseline the cached agent's message_count snapshot before recursing into the in-band queued
            # (/queue) follow-up turn. The first turn has completed and flushed its own user + assistant rows to
            # the SessionDB, so the cross-process coherence guard (#45966) — which this recursive _run_agent
            # call re-enters — would otherwise see the grown on-disk count against the stale build-time snapshot
            # and rebuild the agent on THIS process's OWN writes, destroying the prompt-cache prefix #46237 was
            # merged to preserve. The existing re-baseline in _handle_message_with_agent only runs after the
            # whole _run_agent chain unwinds — too late for the in-band follow-up. Use the same (session_key,
            # session_id) the recursive call runs under so the snapshot matches exactly what the follow-up's
            # guard will consult. Fail-safe in helper.
            # Acknowledge the follow-up the way an idle-session message is: this in-band drain is the only
            # place a queued/interrupting message ever runs, so base.py's hook site is never entered for it.
            # Resolve the adapter from the follow-up's OWN source — a multiplexed gateway can route it to a
            # different profile's adapter, and only that instance holds the per-message reaction state.
            _hook_adapter = self._intake_adapter_for(next_source) if pending_event is not None else None
            await _run_followup_processing_hook(_hook_adapter, pending_event, "on_processing_start")
            # The re-baseline sits inside the try: a /stop landing on its DB await must still close the marker
            # (the helper's own ``except Exception`` does not catch cancellation).
            try:
                await self._refresh_agent_cache_message_count(session_key, session_id)

                while pending_event is not None:
                    if reservation is not None and reservation.withdrawn:
                        await _run_followup_processing_hook(
                            _hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.CANCELLED,
                        )
                        return result
                    if reservation is not None and reservation.revision != prepared_revision:
                        prepared_revision = reservation.revision
                        next_source = pending_event.source
                        next_session_key = self._session_key_for_source(next_source)
                        next_message = await self._prepare_profile_scoped_inbound_message_text(
                            event=pending_event, source=next_source, history=updated_history,
                            session_key=next_session_key,
                        )
                        if next_message is None:
                            await _run_followup_processing_hook(
                                _hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.CANCELLED,
                            )
                            return result
                        next_persist_message = strip_discord_triggering_note(pending_event, next_message)
                        next_message_id = self._reply_anchor_for_event(pending_event)
                        next_inbound_id = str(pending_event.message_id) if pending_event.message_id else None
                        next_channel_prompt, next_source = self._pinned_channel_inputs(
                            next_session_key, pending_event.channel_prompt, next_source, internal=pending_event.internal,
                        )
                        if not pending_event.internal:
                            await self._persist_prompt_pins(next_session_key, session_id)
                        next_message_type = pending_event.message_type
                        next_display_kind = display_kind_for_event(pending_event)
                        next_reply_expected = pending_event.reply_expected
                    if not await self._strict_session_current(
                        pending_event, next_session_key, session_id=session_id,
                    ):
                        await _run_followup_processing_hook(
                            _hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.CANCELLED,
                        )
                        return result
                    if reservation is None or (not reservation.withdrawn and reservation.revision == prepared_revision):
                        break
                if reservation is not None and session_key:
                    release_pending_dispatch_record(adapter, session_key, reservation, claimed=True)

                from gateway.session_identity import replace_source
                tool_context = SessionContext(
                    source=replace_source(next_source, message_id=next_inbound_id),
                    connected_platforms=[], home_channels={},
                    session_key=next_session_key or "", session_id=session_id or "",
                )
                with self._session_env_scope(tool_context):
                    followup_result = await self._run_agent(
                        message=next_message, context_prompt=turn_ctx.context_prompt, history=updated_history,
                        source=next_source, session_id=session_id, session_key=next_session_key,
                        run_generation=run_generation, _interrupt_depth=_interrupt_depth + 1,
                        event_message_id=next_message_id, inbound_message_id=next_inbound_id,
                        processing_event=pending_event,
                        channel_prompt=next_channel_prompt, message_type=next_message_type,
                        persist_user_message=next_persist_message,
                        persist_user_display_kind=next_display_kind,
                        reply_expected=next_reply_expected,
                        input_snapshot=getattr(pending_event, "_prepared_inbound", None),
                        persist_user_display_metadata={
                            "gateway_input_owner": gateway_input_owner(pending_event, next_source),
                            **channel_state_metadata(pending_event),
                            **reply_expected_metadata(next_reply_expected), **diagnostic_metadata(pending_event)} or None,
                    )
            except asyncio.CancelledError:
                await _run_followup_processing_hook(
                    _hook_adapter, pending_event, "on_processing_complete", _followup_cancel_outcome(_hook_adapter))
                raise
            except BaseException:
                await _run_followup_processing_hook(
                    _hook_adapter, pending_event, "on_processing_complete", ProcessingOutcome.FAILURE)
                raise
            followup_outcome = _turn_result_outcome(followup_result)
            if (completed_event is not None and _hook_adapter is not None
                    and (completed_event.message_id or completed_event.raw_message)
                    and _followup_processing_hooks_apply(_hook_adapter, pending_event)
                    and followup_outcome == ProcessingOutcome.SUCCESS and not followup_result.get("already_sent")):
                completed_event._processing_state.pending_completion = _ProcessingCompletion(_hook_adapter, pending_event)
            else:
                await _run_followup_processing_hook(
                    _hook_adapter, pending_event, "on_processing_complete", followup_outcome)
            merged = _preserve_queued_followup_history_offset(result, followup_result)
            # The TERMINAL turn of the chain owns the ledger identity for the outer final send, which
            # the adapter brackets against the event that OPENED the chain. Without this the terminal
            # reply is recorded under the first message's id, so a first reply that was refused (flood
            # control) has its outstanding row replaced and marked delivered by an identical-text
            # terminal reply, and is never redelivered. A deeper recursion has already set its own id,
            # so only fill the key while it is still absent: the innermost turn wins.
            if isinstance(merged, dict) and "queued_terminal_inbound_id" not in merged:
                merged = {
                    **merged,
                    "queued_terminal_inbound_id": next_inbound_id,
                    "queued_terminal_display_kind": next_display_kind,
                    "queued_terminal_reply_expected": next_reply_expected,
                    "queued_terminal_notification_category": (
                        (pending_event.metadata or {}).get("notification_category", "result")
                        if pending_event is not None and pending_event.internal else "result"),
                }
            return merged
        finally:
            if reservation is not None and turn_ctx.session_key:
                release_pending_dispatch_record(adapter, turn_ctx.session_key, reservation)


    async def _park_followup_at_recursion_cap(self: "GatewayRunner", adapter: Any, session_key: str, pending_event: Any) -> None:
        """Put the follow-up back in the pending slot. A started message keeps one lifecycle: an event
        that the follow-up replaces in the slot is discarded, and a follow-up merged into the event
        already there completes with that event."""
        from gateway.platforms.base_pending_merge import merge_pending_message_event

        existing = adapter._pending_messages.get(session_key)
        merge_pending_message_event(adapter._pending_messages, session_key, pending_event)
        held = adapter._pending_messages.get(session_key)
        state = getattr(pending_event, "_processing_state", None)
        if existing is None or existing is pending_event or state is None:
            return
        if held is pending_event:
            await self._complete_discarded_event(existing)
        elif state.start_notified and state.awaiting_start:
            existing._processing_state.attach(_ProcessingCompletion(self._followup_hook_adapter(pending_event), pending_event))
