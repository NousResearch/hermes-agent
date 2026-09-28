"""Matrix processing feedback and read receipts."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from gateway.platforms.event import MessageEvent, ProcessingOutcome

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixFeedbackMixin:
    async def on_processing_start(self: MatrixAdapter, event: MessageEvent) -> None:
        if actions := getattr(self, "_reaction_followup_actions", None):
            self._discard_followup_action(self._event_session_key(event))
        msg_id, room_id = event.message_id, event.source.chat_id
        if self._reactions_enabled and msg_id and room_id:
            reaction_event_id = await self._send_reaction(room_id, msg_id, "\U0001f440")
            if reaction_event_id:
                self._pending_reactions[(room_id, msg_id)] = reaction_event_id


    async def on_processing_complete(self: MatrixAdapter, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        if outcome != ProcessingOutcome.SUCCESS and (actions := getattr(self, "_reaction_followup_actions", None)):
            self._discard_followup_action(self._event_session_key(event))
        msg_id, room_id = event.message_id, event.source.chat_id
        if self._read_receipts_mode.should_send_on_completion(outcome) and msg_id and room_id:
            receipt_id = event.metadata.get("matrix_read_receipt_event_id") or msg_id
            self._background_read_receipt(room_id, receipt_id)
        if not self._reactions_enabled or not msg_id or not room_id or outcome == ProcessingOutcome.CANCELLED:
            return
        eyes_event_id = self._pending_reactions.pop((room_id, msg_id), None)
        if eyes_event_id:
            self._schedule_reaction_redaction(room_id, eyes_event_id, "processing complete")
        await self._send_reaction(room_id, msg_id, "\u2705" if outcome == ProcessingOutcome.SUCCESS else "\u274c")


    def _background_read_receipt(self: MatrixAdapter, room_id: str, event_id: str) -> None:

        async def _send() -> None:
            try:
                await self.send_read_receipt(room_id, event_id)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Matrix: background read receipt failed: %s", exc)
        asyncio.ensure_future(_send())


    async def send_read_receipt(self: MatrixAdapter, room_id: str, event_id: str) -> bool:
        from plugins.platforms.matrix.adapter import RoomID, EventID
        if not self._client:
            return False
        try:
            room, event = RoomID(room_id), EventID(event_id)
            if hasattr(self._client, "set_fully_read_marker"):
                await self._client.set_fully_read_marker(room, event, event)
            elif hasattr(self._client, "send_receipt"):
                await self._client.send_receipt(room, event)
            elif hasattr(self._client, "set_read_markers"):
                await self._client.set_read_markers(room, fully_read_event=event, read_receipt=event)
            else:
                logger.debug("Matrix: client has no read receipt method")
                return False
            logger.debug("Matrix: sent read receipt for %s in %s", event_id, room_id)
            return True
        except Exception as exc:
            logger.debug("Matrix: read receipt failed: %s", exc)
            return False
