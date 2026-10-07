"""Tool-retry mixin for SessionDB: a card's Retry (the start_chat handoff card) records its outcome on the
original tool row's ``display_metadata``. The row the model replays stays byte-identical, so the cached prefix
survives; a reload shows the outcome, and ``announced`` tells the model once, on the next user turn (the
reactions pattern)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from hermes_state_common import _placeholders, _sql_json_extract
from hermes_state_messages import _DISPLAY_ACTIVE_CLAUSE, _SET_DISPLAY_META_SQL


class SessionToolRetriesMixin:
    """Retry outcomes stamped on tool rows, and their one-time announcement to the model."""

    TOOL_RETRY_METADATA_KEY = "retried"

    def tool_row_retry(self, session_id: str, tool_call_id: str) -> Optional[Tuple[int, Optional[Dict[str, Any]]]]:
        """``(row id, recorded retry outcome or None)`` of the newest visible tool row for *tool_call_id* in the
        session's resume lineage, or ``None`` when no such row is saved."""
        if not session_id or not tool_call_id:
            return None
        lineage = self._resume_lineage_ids(session_id)
        row = self._read_one(
            "SELECT id, display_metadata FROM messages WHERE role = 'tool' AND tool_call_id = ? "
            f"AND session_id IN ({_placeholders(lineage)}){_DISPLAY_ACTIVE_CLAUSE} ORDER BY id DESC LIMIT 1",
            (tool_call_id, *lineage))
        if row is None:
            return None
        retried = (self._decode_display_metadata(row["display_metadata"]) or {}).get(self.TOOL_RETRY_METADATA_KEY)
        result = retried.get("result") if isinstance(retried, dict) else None
        return row["id"], result if isinstance(result, dict) else None

    def set_tool_row_retry(self, row_id: int, outcome: Dict[str, Any]) -> None:
        """Record *outcome* as the retry of tool row *row_id*, not yet announced to the model."""
        def _do(conn):
            row = conn.execute("SELECT display_metadata FROM messages WHERE id = ?", (row_id,)).fetchone()
            if row is None:
                return
            meta = self._decode_display_metadata(row[0]) or {}
            meta[self.TOOL_RETRY_METADATA_KEY] = {"result": outcome, "announced": False}
            conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta), row_id))
        self._execute_write(_do)

    def take_unannounced_tool_retries(self, session_id: str) -> List[Dict[str, Any]]:
        """``{result, tool_name}`` for each retry the model has not heard about yet, marked announced."""
        if not session_id:
            return []
        lineage = self._resume_lineage_ids(session_id)
        key = self.TOOL_RETRY_METADATA_KEY
        # Every turn of every desktop chat asks. The LIKE pre-filter skips the JSON parse on rows that cannot hold
        # the key (nearly all of them); json_extract still decides.
        select = ("SELECT id, tool_name, display_metadata FROM messages "
                  f"WHERE session_id IN ({_placeholders(lineage)}){_DISPLAY_ACTIVE_CLAUSE} "
                  f"AND display_metadata LIKE '%\"{key}\"%' "
                  f"AND {_sql_json_extract('display_metadata', '$.' + key)} IS NOT NULL ORDER BY id")

        def _unannounced(row) -> Optional[Dict[str, Any]]:
            meta = self._decode_display_metadata(row["display_metadata"]) or {}
            retried = meta.get(key)
            if not isinstance(retried, dict) or retried.get("announced") or not isinstance(retried.get("result"), dict):
                return None
            return meta

        # Every turn asks; almost none have a retry to announce. Read first so those turns never take the write lock.
        if not any(_unannounced(row) for row in self._read_all(select, tuple(lineage))):
            return []

        def _do(conn):
            pending = []
            for row in conn.execute(select, tuple(lineage)).fetchall():
                meta = _unannounced(row)
                if meta is None:
                    continue
                retried = meta[key]
                retried["announced"] = True
                conn.execute(_SET_DISPLAY_META_SQL, (self._encode_display_metadata(meta), row["id"]))
                pending.append({"result": retried["result"], "tool_name": row["tool_name"] or "tool"})
            return pending
        return self._execute_write(_do)
