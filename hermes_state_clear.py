"""Same-session conversation boundaries for :class:`hermes_state.SessionDB`."""

from __future__ import annotations

import time


class SessionClearMixin:
    """Durably retire the active transcript while keeping its session identity."""

    def clear_conversation(self, session_id: str) -> int:
        """Start a new empty conversation epoch in *session_id*."""
        if not session_id:
            raise ValueError("session_id is required")

        def _do(conn):
            row = conn.execute(
                "SELECT conversation_epoch FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown session {session_id!r}")
            conn.execute(
                "UPDATE messages SET active = 0, compacted = 0 "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            )
            conn.execute(
                "UPDATE sessions SET conversation_epoch = conversation_epoch + 1, "
                "conversation_cleared_at = ?, message_count = 0, tool_call_count = 0 "
                "WHERE id = ?",
                (time.time(), session_id),
            )
            return int(row["conversation_epoch"] or 0) + 1

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
