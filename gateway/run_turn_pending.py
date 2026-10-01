"""Post-turn pending input selection for GatewayRunner."""

import logging
from contextlib import suppress
from typing import Any, Optional, Tuple

from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayPendingDrainMixin:
    """Select pending input after a completed or interrupted turn."""

    async def _run_agent_drain_pending(
        self, result: Any, adapter: Any, source: SessionSource, session_key: Optional[str]
    ) -> Tuple[Any, Optional[str]]:
        """Dequeue the adapter's pending / interrupt / leftover-steer follow-up as ``(pending_event, pending)``.

        Keyed by session_key (not source.chat_id) to match the adapter's storage keys."""
        from gateway.run import (
            _build_media_placeholder, _dequeue_pending_event, _is_control_interrupt_message
        )
        pending_event = None
        pending = None
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
            while pending_event is not None and not await self._strict_session_current(
                pending_event, session_key,
            ):
                pending_event = _dequeue_pending_event(adapter, session_key)
                pending_event = self._promote_queued_event(session_key, adapter, pending_event)
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
                    pending = pending or _build_media_placeholder(pending_event)
                else:
                    pending = _pending_text or _build_media_placeholder(pending_event)
                if pending:
                    logger.debug("Processing queued message after agent completion: '%s...'", pending[:40])

        # Leftover /steer (arrived after the last tool batch): deliver as the next user turn.
        if result and not pending and not pending_event and result.get("pending_steer"):
            pending = result.get("pending_steer")
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
