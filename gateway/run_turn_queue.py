"""Native follow-up draining, including the compulsory rollover bootstrap fence."""
from contextlib import suppress
import logging
from typing import Any, Optional

from gateway.run_rollover import lane_rollover
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayTurnQueueMixin:
    def _rollover_holds_followups(self, result, adapter, source, key):
        lane = lane_rollover(self, key)
        closing = lane is not None and lane.phase in {"authoring", "committing", "bootstrapping", "admitting"}
        if not (closing or (isinstance(result, dict) and result.get("rollover_requested"))):
            return False
        # A late /steer was accepted by the old agent but not yet put in its transcript.
        # Preserve it ahead of ordinary arrivals, just as the normal drain does.
        steer = (result.get("pending_steer") or "").strip() if result else ""
        if steer:
            from gateway.platforms.event import MessageEvent
            if adapter is None:
                raise RuntimeError("Cannot retain accepted steering without its adapter")
            previous = adapter._pending_messages.get(key)
            if previous is not None:
                self._session_state(key).conversation.queued_events.insert(0, previous)
            adapter._pending_messages[key] = MessageEvent(text=steer, source=source)
            adapter._pending_messages[key]._gateway_accepted = True
        if adapter is not None and key not in adapter._pending_messages:
            pending = self._promote_queued_event(key, adapter, None)
            if pending is not None:
                adapter._pending_messages[key] = pending
        if adapter is not None and key in adapter._pending_messages:
            adapter._pending_messages[key]._rollover_fifo_input = True
        return True

    async def _run_agent_drain_pending(
        self, result: Any, adapter: Any, source: SessionSource, session_key: Optional[str]
    ) -> tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up as ``(pending_event, pending)``.

        Keyed by session_key (not source.chat_id) to match the adapter's storage keys."""
        if self._rollover_holds_followups(result, adapter, source, session_key):
            return None, None
        from gateway.run import (
            _build_media_placeholder, _dequeue_pending_event, _is_control_interrupt_message
        )
        pending_event = None
        pending = None
        if result and adapter and session_key:
            pending_event = _dequeue_pending_event(adapter, session_key)
            if result.get("compression_exhausted") and pending_event is not None:
                self._defer_past_exhaustion_reset(result, adapter, source, session_key, pending_event)
                return None, None
            # /queue overflow: promote the next queued event into the consumed "next-up" slot so the
            # recursive drain sees it (keeps FIFO order; a mid-chain /queue can't jump the queue).
            pending_event = self._promote_queued_event(session_key, adapter, pending_event)
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
        pending_event, pending = self._merge_leftover_steer(result, adapter, source, session_key, pending_event, pending)

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
