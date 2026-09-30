"""Effective event invalidation for Matrix redactions."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MatrixRedactionMixin:
    """Redaction handling for the Matrix adapter."""

    async def _on_redaction(self, event: Any) -> None:
        room_id = str(getattr(event, "room_id", "") or "")
        target = str(getattr(event, "redacts", "") or "")
        if not target:
            content = getattr(event, "content", None)
            target = str(content.get("redacts") or "") if isinstance(content, dict) else ""
        if room_id and target:
            self._event_context_cache.redact(room_id, target)
            for action in self._reaction_followup_actions.values():
                if action.room_id == room_id:
                    action.pending.discard(target)

        sender = str(getattr(event, "sender", "") or "")
        if not (room_id and target and sender):
            return
        withdrawn = self._parked_voices.discard(room_id, sender, target)
        withdrawn = self.withdraw_pending_message(target, chat_id=room_id, sender_id=sender) or withdrawn
        if withdrawn:
            logger.info("Matrix: dropped queued message %s in %s; its sender redacted it", target, room_id)
