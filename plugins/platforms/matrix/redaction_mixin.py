"""Effective event invalidation for Matrix redactions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _redacted_event_id(event: Any) -> str:
    """The event ID that an ``m.room.redaction`` event redacts, or an empty string."""
    content = getattr(event, "content", None)
    # Room version 11 moved ``redacts`` into the content.
    return str(getattr(event, "redacts", None) or (content.get("redacts") if content else None) or "")


class MatrixRedactionMixin:
    """Redaction handling for the Matrix adapter."""

    async def _on_redaction(self, event: Any) -> None:
        room_id = str(getattr(event, "room_id", "") or "")
        target = _redacted_event_id(event)
        if room_id and target:
            self._event_context_cache.redact(room_id, target)
            for action in self._reaction_followup_actions.values():
                if action.room_id == room_id:
                    action.pending.discard(target)

        if room_id and target:
            self._withdraw_redacted_message(room_id, str(getattr(event, "sender", "") or ""), target)

    def _withdraw_redacted_message(self, room_id: str, sender: str, target: str) -> None:
        """Drop ``target`` if it is still waiting for its turn and ``sender`` wrote it. A
        redaction by anyone else, such as a moderator, leaves the message queued."""
        withdrawn = self._parked_voices.discard(room_id, sender, target)
        withdrawn = self.withdraw_pending_message(target, chat_id=room_id, sender_id=sender) or withdrawn
        if withdrawn:
            logger.info("Matrix: dropped queued message %s in %s; its sender redacted it", target, room_id)
