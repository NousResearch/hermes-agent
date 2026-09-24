"""Matrix processing feedback and read receipts."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

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
        if (
            self._read_receipts_mode == "after_processing"
            and msg_id
            and room_id
            and outcome != ProcessingOutcome.CANCELLED
        ):
            self._background_read_receipt(room_id, msg_id)
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



    @staticmethod
    def _parse_read_receipts_mode(config) -> str:
        """Resolve read-receipt mode from config.yaml ``matrix.read_receipts``.

        Falls back to the legacy ``MATRIX_READ_RECEIPTS`` env var; the YAML
        value always wins when set. Returns one of ``"immediate"``,
        ``"after_processing"``, or ``"disabled"``.

        Back-compat: earlier releases exposed only a boolean env var, so
        boolean / ``"true"`` / ``"false"`` / ``"on"`` / ``"off"`` / ``"1"`` /
        ``"0"`` values are still accepted — truthy maps to ``"immediate"`` and
        falsy to ``"disabled"``. Unrecognized values fall back to the default.
        """
        default = "immediate"
        valid = {"immediate", "after_processing", "disabled"}

        def _coerce(raw) -> Optional[str]:
            if raw is None:
                return None
            if isinstance(raw, bool):
                return "immediate" if raw else "disabled"
            token = str(raw).strip().lower()
            if not token:
                return None
            if token in valid:
                return token
            # Legacy boolean spellings (the pre-mode env flag).
            if token in {"true", "1", "yes", "on", "enabled"}:
                return "immediate"
            if token in {"false", "0", "no", "off"}:
                return "disabled"
            return default

        configured = _coerce(config.extra.get("read_receipts"))
        if configured is not None:
            return configured
        env_mode = _coerce(os.getenv("MATRIX_READ_RECEIPTS"))
        if env_mode is not None:
            return env_mode
        return default


    @staticmethod
    def _parse_reactions_enabled(config) -> bool:
        """Resolve lifecycle-reaction toggle from ``matrix.reactions``.

        Falls back to the legacy ``MATRIX_REACTIONS`` env var; the YAML value
        always wins when set. Defaults to enabled. A value is falsy when it is
        the boolean ``False`` or one of ``"false"`` / ``"0"`` / ``"no"`` /
        ``"off"`` (case-insensitive); everything else is truthy.
        """

        def _coerce(raw) -> Optional[bool]:
            if raw is None:
                return None
            if isinstance(raw, bool):
                return raw
            token = str(raw).strip().lower()
            if not token:
                return None
            return token not in {"false", "0", "no", "off"}

        configured = _coerce(config.extra.get("reactions"))
        if configured is not None:
            return configured
        env_val = _coerce(os.getenv("MATRIX_REACTIONS"))
        if env_val is not None:
            return env_val
        return True

