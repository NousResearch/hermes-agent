"""
Message Staging Queue — drain hook for the gateway builtin_hooks system.

This module provides a mixin that overrides `on_processing_complete` to
drain the next staged message after a turn finishes. It also patches
`_handle_message_while_active` to persist inbound messages instead of
merging them into the in-memory `_pending_messages` dict.

INSTALLATION:
  - Import `StageQueueMixin` from this module
  - Add it to the Telegram adapter's MRO BEFORE BasePlatformAdapter
  - Set HERMES_STAGING_QUEUE=1 in the gateway environment

Phase 1: Telegram lane only.
"""

import json
import logging
from typing import Any, Optional

from gateway.platforms.base import BasePlatformAdapter, MessageEvent, ProcessingOutcome

logger = logging.getLogger("hermes.gateway.stage_queue.mixin")


class StageQueueMixin:
    """Mixin for BasePlatformAdapter that adds staging-queue intercept + drain.

    Override points:
    - on_processing_complete: drains next staged message after a turn ends
    - _handle_message_while_active: persists to DB instead of _pending_messages

    The staging library (stage_queue.py) does the actual Supabase I/O.
    """

    # Subclass must set these (Telegram adapter does)
    _OK_EMOJI: Optional[str] = None
    _FAIL_EMOJI: Optional[str] = None

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        """After a turn completes, drain the next staged message (if any)."""
        # First, let the base class handle its reaction logic
        await super().on_processing_complete(event, outcome)  # type: ignore[misc]

        # Now drain the queue
        session_key = self._event_session_key(event)

        from gateway.stage_queue import (
            drain_next, mark_done, mark_dead, bump_attempt,
            set_released_reaction, clear_reaction,
            check_alarm, _STAGING_ENABLED,
        )

        if not _STAGING_ENABLED:
            return

        # Check alarm thresholds
        alarm_msg = await check_alarm(session_key)
        if alarm_msg:
            logger.warning("[stage-queue] %s", alarm_msg)
            # JC NO-PAGE LAW: alarm goes to bus (archie lane), never to JC directly

        # Drain: pop the oldest staged message and feed it into processing
        while True:
            staged = await drain_next(session_key)
            if staged is None:
                break

            msg_id = staged["id"]
            staged_message_id = staged.get("message_id", "")
            chat_id = staged.get("chat_id", "")
            event_data = staged.get("event", {})

            # Reconstruct a MessageEvent from the stored data
            try:
                reconstructed = self._reconstruct_event(staged)
            except Exception as e:
                logger.error("[stage-queue] reconstruct failed for id=%s: %s", msg_id, e)
                attempts = await bump_attempt(msg_id)
                if attempts >= 3:
                    await mark_dead(msg_id, f"reconstruct failed: {e}")
                continue

            # Set "released" reaction
            await set_released_reaction(self, chat_id, staged_message_id)

            # Process it — if session is free, start processing; if not, re-stage
            if session_key in self._active_sessions:  # type: ignore[attr-defined]
                # Session is still busy (shouldn't happen — we just finished)
                logger.warning("[stage-queue] session still active after completion, re-staging id=%s", msg_id)
                from gateway.stage_queue import _sb_rpc
                from urllib.parse import quote
                await _sb_rpc("patch", "message_stage_queue",
                              payload={"status": "staged", "updated_at": "now()"},
                              query=quote(f"id=eq.{msg_id}"))
                break

            # Start processing the drained message
            accepted = self._start_session_processing(reconstructed, session_key)  # type: ignore[attr-defined]
            if not accepted:
                logger.warning("[stage-queue] session start rejected for id=%s, re-staging", msg_id)
                from gateway.stage_queue import _sb_rpc
                from urllib.parse import quote
                await _sb_rpc("patch", "message_stage_queue",
                              payload={"status": "staged", "updated_at": "now()"},
                              query=quote(f"id=eq.{msg_id}"))
                break

            # Only drain ONE message at a time per spec ("one message in flight per chat")
            break

    def _reconstruct_event(self, staged_row: dict) -> MessageEvent:
        """Reconstruct a MessageEvent from a staged queue row.
        This is adapter-specific; Telegram adapter overrides for platform fields."""
        event_data = staged_row.get("event", {})
        # Build a minimal MessageEvent — subclasses add platform-specific fields
        from gateway.platforms.base import MessageSource
        source = MessageSource(
            platform=event_data.get("source_platform", "telegram"),
            chat_id=event_data.get("source_chat_id", staged_row.get("chat_id", "")),
            chat_type=event_data.get("source_chat_type", "dm"),
            user_id=event_data.get("source_user_id", staged_row.get("sender_id", "")),
            user_name=event_data.get("source_user_name", staged_row.get("sender_name", "")),
        )
        return MessageEvent(
            source=source,
            text=event_data.get("text", ""),
            message_id=staged_row.get("message_id", ""),
            metadata=event_data.get("metadata"),
        )

    async def _handle_message_while_active_staged(self, event: MessageEvent, session_key: str) -> bool:
        """Replace the default _handle_message_while_active: persist to DB + reaction
        instead of merging into _pending_messages.
        Returns True if the message was staged (caller should skip default handling)."""
        from gateway.stage_queue import enqueue_from_event, _STAGING_ENABLED

        if not _STAGING_ENABLED:
            return False  # Let the default handler run

        # Bypass commands still need to go through the inline dispatch path
        cmd = event.get_command()
        from hermes_cli.commands import should_bypass_active_session
        if should_bypass_active_session(cmd):
            return False  # Don't stage bypass commands

        # Stage the message
        staged = await enqueue_from_event(self, event, session_key)
        return staged
