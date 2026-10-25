"""Inactive cron-delivery references, acknowledged with a durable admitted user row."""
from __future__ import annotations

import time
from typing import Any

from hermes_state_common import _placeholders


class SessionPendingDeliveriesMixin:
    def append_pending_delivery(self, session_id: str, content: str, *, source: str) -> int:
        """Durable reference, not an active user turn. Admission coalesces it with a real input.

        Inactive, non-compacted rows survive active flushes without entering compression or
        changing the model's cached prefix. The source is producer metadata, never inferred
        from a user-supplied text label.
        """
        from hermes_state_messages import _INSERT_MESSAGE_SQL

        def _do(conn):
            self._check_transcript_write_guards(conn, session_id, None)
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            params = self._message_row_params(conversation_id, "user", {
                "content": content, "display_kind": "pending_delivery",
                "display_metadata": {"source": source, "delivery_session_id": session_id},
            }, None, time.time(), keep_reasoning=False)
            row_id = conn.execute(_INSERT_MESSAGE_SQL, params).lastrowid
            conn.execute("UPDATE messages SET active = 0 WHERE id = ?", (row_id,))
            return row_id
        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def pending_deliveries(self, session_id: str, *, limit: int = 16) -> list[dict[str, Any]]:
        """Bounded references under the existing compression-aware conversation lease key.

        An explicit fork/reset has its own key; compression keeps the old one. Inactive
        delivery rows need neither copying nor transcript repair when the active tip rotates.
        """
        with self._read_ctx() as conn:
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            return [dict(row) for row in conn.execute(
                """SELECT id, substr(content, 1, 32000) AS content, length(content) AS size
                FROM messages WHERE session_id = ? AND display_kind = 'pending_delivery'
                AND active = 0 ORDER BY id LIMIT ?""", (conversation_id, limit)).fetchall()]

    def _consume_pending_deliveries(self, conn, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Acknowledge only with the user carrier's successful commit; failed flushes retry."""
        for message in messages:
            metadata = self._decode_display_metadata(message.get("display_metadata")) or {}
            ids = metadata.get("pending_delivery_ids") if message.get("role") == "user" else None
            if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
                continue
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            conn.execute(
                f"""UPDATE messages SET display_kind = 'consumed_delivery'
                WHERE session_id = ? AND display_kind = 'pending_delivery'
                AND active = 0 AND id IN ({_placeholders(ids)})""", (conversation_id, *ids))
