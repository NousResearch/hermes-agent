"""Effective event invalidation for Matrix redactions."""

from typing import Any


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
