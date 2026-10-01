"""External message identity lookups for SessionDB (platform message ids, gateway input owners, client
message ids). Mixin bound via the MRO, built on SessionDB's _read_one / _resume_lineage_ids /
_decode_display_metadata primitives."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hermes_state_common import _placeholders, _sql_json_extract


class SessionMessageIdentityMixin:
    def has_gateway_input_owner(self, session_id: str, owner: str) -> bool:
        """Probe the accepted-input marker without allocating message bodies or archives."""
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND role = 'user' "
            "AND observed = 0 AND (active = 1 OR compacted = 1) "
            "AND CASE WHEN json_valid(display_metadata) "
            "THEN json_extract(display_metadata, '$.gateway_input_owner') END = ? LIMIT 1",
            (session_id, owner)) is not None

    def has_platform_message_id(self, session_id: str, platform_message_id: str) -> bool:
        """True when *platform_message_id* exists (partial-index probe; the gateway's transient-failure dedupe).

        Uses the idx_messages_platform_msg_id partial index for efficient lookup. Used by the gateway's
        transient-failure dedupe guard (#47237) to skip re-persisting a user message that was already saved
        on a prior retry of the same inbound platform message.
        """
        return self._read_one(
            "SELECT 1 FROM messages WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
            (session_id, platform_message_id)) is not None

    def find_user_message_by_client_message_id(self, session_id: str, client_message_id: str) -> Optional[Dict[str, Any]]:
        """Return the first durable user row carrying ``client_message_id`` across the resume lineage.

        Inactive rows are intentional: queued prompts are written at acceptance and later re-placed at
        the transcript tail, while the original acknowledgment remains bound to the first row.
        """
        session_ids = self._resume_lineage_ids(session_id)
        if not session_ids or not client_message_id:
            return None
        row = self._read_one(
            f"SELECT id, timestamp, display_metadata FROM messages "
            f"WHERE session_id IN ({_placeholders(session_ids)}) AND role = 'user' "
            f"AND {_sql_json_extract('display_metadata', '$.client_message_id')} = ? "
            "ORDER BY id LIMIT 1",
            (*session_ids, client_message_id),
        )
        if row is None:
            return None
        return {
            "_row_id": row["id"],
            "timestamp": row["timestamp"],
            "display_metadata": self._decode_display_metadata(row["display_metadata"]) or {},
        }
