"""External message identity lookups for SessionDB (platform message ids, gateway input owners, client
message ids). Mixin bound via the MRO, built on SessionDB's _read_one / _resume_lineage_ids
primitives."""

from __future__ import annotations

from typing import Any, Dict, Optional

from hermes_state_common import QUEUED_PROMPT_METADATA_KEY, _placeholders, _sql_json_extract


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
        """Return the newest visible (active or compaction-archived) user row for ``client_message_id``
        across the resume lineage that does not carry the never-drained marker, or None.

        The row is the acknowledgement a retry receives when the in-memory queue cannot answer (a
        restart, or the prompt already drained). A row still carrying the marker (#125577) is a prompt
        nothing will run — ``reopen_session`` retires the tip's, but a compression rotation leaves the
        parent's original active and an in-place compaction without coverage archives it — so it is
        never an acknowledgement; the drain's unmarked replacement row is.
        """
        session_ids = self._resume_lineage_ids(session_id)
        if not session_ids or not client_message_id:
            return None
        row = self._read_one(
            f"SELECT id, timestamp FROM messages "
            f"WHERE session_id IN ({_placeholders(session_ids)}) AND role = 'user' "
            "AND (active = 1 OR compacted = 1) "
            f"AND COALESCE({_sql_json_extract('display_metadata', '$.' + QUEUED_PROMPT_METADATA_KEY)}, 0) != 1 "
            f"AND {_sql_json_extract('display_metadata', '$.client_message_id')} = ? "
            "ORDER BY id DESC LIMIT 1",
            (*session_ids, client_message_id),
        )
        if row is None:
            return None
        return {"_row_id": row["id"], "timestamp": row["timestamp"]}
