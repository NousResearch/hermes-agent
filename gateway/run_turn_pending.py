"""Post-turn pending input selection for GatewayRunner."""

import asyncio
import dataclasses
import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Callable, Optional

from gateway.session import SessionSource
from gateway.platforms.event import MessageEvent, MessageType, _ProcessingPhase
from gateway.platforms.base_pending import reserve_pending_dispatch, release_pending_dispatch, release_pending_dispatch_record, pending_dispatch_withdrawn

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

logger = logging.getLogger("gateway.run")


class GatewayPendingDrainMixin:
    """Select pending input after a completed or interrupted turn."""

    if TYPE_CHECKING:
        _restore_pending_dispatch = GatewayRunner._restore_pending_dispatch
        _peek_session_state = GatewayRunner._peek_session_state


    async def _run_agent_drain_pending(
        self: "GatewayRunner", result: Any, adapter: Any, source: SessionSource, session_key: Optional[str],
        processing_event: Optional[MessageEvent] = None,
    ) -> tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up as ``(pending_event, pending)``.

        Keyed by session_key (not source.chat_id) to match the adapter's storage keys."""
        from gateway.run_inbound_media import _build_media_placeholder
        from gateway.run import _dequeue_pending_event, _is_control_interrupt_message
        pending_event = None
        pending = None
        pending_steer = result.get("pending_steer") if result else None
        pending_input = None
        if result and processing_event is not None:
            pending_inputs = processing_event._processing_state.take_pending_inputs(pending_steer or "")
        # The steer runs as a copy of its latest message. An input whose lifecycle already completed
        # runs as plain text, as its hooks have fired.
        pending_input = pending_inputs[-1] if pending_inputs else None
        if pending_input is not None and pending_input._processing_state.phase is _ProcessingPhase.COMPLETED:
            pending_inputs = []
        steer_event = None
        steer_enqueued = False
        if result and adapter and session_key:
            live_adapter = self._delivery_adapter_for(source)
            if (live_adapter is not None and live_adapter is not adapter
                    and isinstance(getattr(live_adapter, "_pending_messages", None), dict)):
                earlier = _dequeue_pending_event(adapter, session_key)
                if earlier is not None:
                    newer = _dequeue_pending_event(live_adapter, session_key)
                    if newer is not None:
                        self._session_state(session_key).conversation.queued_events.insert(0, newer)
                    pending_event = earlier
                adapter = live_adapter
            if pending_event is None:
                pending_event = _dequeue_pending_event(adapter, session_key)
            # /queue overflow: promote the next queued event into the consumed "next-up" slot so the
            # recursive drain sees it (keeps FIFO order; a mid-chain /queue can't jump the queue).
            pending_event = self._promote_queued_event(session_key, adapter, pending_event)
            if pending_event is not None:
                reserve_pending_dispatch(adapter, session_key, pending_event).task = asyncio.current_task()
            try:
                while pending_event is not None and not await self._strict_session_current(
                    pending_event, session_key,
                ):
                    release_pending_dispatch(adapter, session_key, pending_event)
                    pending_event = _dequeue_pending_event(adapter, session_key)
                    pending_event = self._promote_queued_event(session_key, adapter, pending_event)
                    if pending_event is not None:
                        reserve_pending_dispatch(adapter, session_key, pending_event).task = asyncio.current_task()
                if pending_event is not None and pending_dispatch_withdrawn(adapter, session_key, pending_event):
                    release_pending_dispatch(adapter, session_key, pending_event)
                    return None, None
                _state = self._peek_session_state(session_key)
                followup_withdrawn = bool(_state and _state.turn.followup_withdrawn)
                if _state:
                    _state.turn.followup_withdrawn = False
                if (result.get("interrupted") and not pending_event and result.get("interrupt_message")
                        and not followup_withdrawn):
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
                        if pending_dispatch_withdrawn(adapter, session_key, pending_event):
                            release_pending_dispatch(adapter, session_key, pending_event)
                            return None, None
                        pending = pending or _build_media_placeholder(pending_event)
                    else:
                        pending = _pending_text or _build_media_placeholder(pending_event)
                    if pending:
                        logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])
            except BaseException:
                if pending_event is not None and not pending_dispatch_withdrawn(adapter, session_key, pending_event):
                    self._restore_pending_dispatch(session_key, pending_event, adapter)
                raise

        # Leftover /steer (arrived after the last tool batch): deliver as the next user turn.
        if pending_steer:
            if pending_input is not None:
                from copy import copy
                steer_event = copy(pending_input)
                steer_event.text = pending_steer
                if not pending_input.media_urls:
                    steer_event.message_type = MessageType.TEXT
                if hasattr(steer_event, "_gateway_pending_stt_text"):
                    steer_event._gateway_pending_stt_text = pending_steer
            if pending or pending_event:
                if adapter and session_key:
                    if steer_event is None:
                        steer_prompt, steer_source = self._pinned_channel_inputs(
                            session_key, None, source, internal=True,
                        )
                        steer_event = MessageEvent(
                            text=pending_steer, source=steer_source, channel_prompt=steer_prompt,
                        )
                    self._enqueue_fifo(session_key, steer_event, adapter)
                    steer_enqueued = True
            else:
                pending_event, pending = steer_event, pending_steer
                logger.debug("Delivering leftover /steer as next turn: '%s...'", pending[:40])

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
                        if pending_event is not None and session_key:
                            release_pending_dispatch(adapter, session_key, pending_event)
                        pending_event = None
                        pending = None

        if self._draining and (pending_event or pending):
            logger.info(
                "Discarding pending follow-up for session %s during gateway %s",
                session_key or "?", self._status_action_label(),
            )
            if pending_event is not None and session_key:
                self._restore_pending_dispatch(session_key, pending_event, adapter)
            pending_event = None
            pending = None
        if steer_event is not None and processing_event is not None and (
                steer_enqueued or pending_event is steer_event):
            self._hand_leftover_steer_to_its_turn(processing_event, steer_event, pending_inputs)
        return pending_event, pending

    async def _run_agent_fire_pending_interrupt(
        self, adapter: Any, agent: Any, source: SessionSource, session_key: str,
        _interrupt_detected: asyncio.Event, streaming_tts_consumer_holder: list, *,
        log_context: str, log: Callable[[], None],
    ) -> None:
        """Peek the adapter's pending event, transcribe voice, then signal the agent + abort streaming TTS.

        Peek WITHOUT consuming: the event must stay for the post-run ``_dequeue_pending_event()``
        (popping races the agent finishing). Transcribe BEFORE signaling so voice interrupts carry
        the real transcript."""
        from gateway.run_inbound_media import _build_media_placeholder
        _peek_event = adapter._pending_messages.get(session_key)
        if _peek_event is None:
            state = self._peek_session_state(session_key)
            if state is not None and state.turn.followup_withdrawn:
                return
        reservation = (reserve_pending_dispatch(adapter, session_key, _peek_event)
                       if _peek_event is not None else None)
        if reservation is not None and reservation.task is None:
            reservation.task = asyncio.current_task()
        try:
            pending_text = None
            if _peek_event is not None:
                pending_text = _peek_event.text or ""
                if self._pending_event_audio_paths(_peek_event):
                    pending_text, _ = await self._transcribe_and_echo_pending_voice(
                        _peek_event, adapter, source, pending_text, log_context=log_context,
                        metadata={"thread_id": source.thread_id} if source.thread_id else None,
                    )
                elif not pending_text and (getattr(_peek_event, "media_urls", None) or []):
                    pending_text = _build_media_placeholder(_peek_event)
            if reservation is not None and (reservation.withdrawn or reservation.claimed):
                return
            log()
            agent.interrupt(pending_text)
            _interrupt_detected.set()
            # Abort streaming TTS on barge-in.
            # See #60671.
            # See #60671.
            # See #60671.
            # Finalize the streaming-TTS consumer (#60671). finish() is called from the outer event-loop thread
            # (not the executor worker) so early returns from run_sync are also finalised.  wait_complete()
            # drains queued audio; on timeout the consumer is aborted unconditionally — if audio was audible,
            # suppression is preserved so the gateway does not replay from the beginning; if no audio was
            # audible, the whole-file fallback path is permitted.
            _stts = streaming_tts_consumer_holder[0]
            if _stts is not None:
                _stts.abort("barge-in")
        finally:
            if reservation is not None and reservation.task is asyncio.current_task():
                release_pending_dispatch_record(adapter, session_key, reservation)


    def _hand_leftover_steer_to_its_turn(
        self: "GatewayRunner", processing_event: MessageEvent, steer_event: MessageEvent,
        pending_inputs: list[MessageEvent],
    ) -> None:
        """The steer's own turn completes every started message whose text it carries, not the turn
        that returned the steer unconsumed. The earlier messages complete with the steer copy."""
        from gateway.run_turn_followup_ack import _followup_processing_hooks_apply

        if not _followup_processing_hooks_apply(self._intake_adapter_for(steer_event.source), steer_event):
            return
        running_state, steer_state = processing_event._processing_state, steer_event._processing_state
        for incoming in pending_inputs:
            completion = running_state.release(incoming)
            if completion is not None and incoming._processing_state is not steer_state:
                steer_state.absorbed.append(completion)
        # The message's own handler may still be running, for example while it sends a /steer
        # acknowledgement. Its completion must not end the lifecycle that the copy now runs.
        if steer_state.phase in {_ProcessingPhase.ABSORBED, _ProcessingPhase.RUNNING}:
            steer_state.owner = steer_event
