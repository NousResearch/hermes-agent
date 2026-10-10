"""Atomic session-topic selection and exact current-turn retraction."""
from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Optional
from hermes_state_common import _placeholders

def _normalized_topic_title(value: Any) -> str:
    """Database comparison key for model/CLI supplied topic titles."""
    text = " ".join(str(value or "").strip().lower().split())
    return "-".join(re.findall(r"[a-z0-9]+", text))[:96]

class TopicMessagesMixin:
    def _topic_conversation_id_on_conn(self, conn, session_id: str) -> str:
        """Compression-lineage root used by topics and the session turn lease."""
        return self._session_turn_lease_key_on_conn(conn, session_id)

    def _topic_lineage_ids_on_conn(self, conn, session_id: str) -> List[str]:
        """Current compression lineage, newest to oldest, on one connection."""
        result: List[str] = []
        current = session_id
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            result.append(current)
            row = conn.execute(
                "SELECT id, parent_session_id, source, model_config, end_reason "
                "FROM sessions WHERE id = ?", (current,),
            ).fetchone()
            if row is None:
                break
            current_row = dict(row)
            parent_id = current_row.get("parent_session_id")
            if not parent_id or self._is_explicit_fork_child_row(current_row, include_reset=True):
                break
            parent = conn.execute(
                "SELECT end_reason FROM sessions WHERE id = ?", (parent_id,),
            ).fetchone()
            if parent is None or parent["end_reason"] != "compression":
                break
            current = str(parent_id)
        return result

    def _topics_for_conversation_on_conn(self, conn, conversation_id: str) -> List[Dict[str, Any]]:
        rows = conn.execute(
            """SELECT t.id, t.title, t.normalized_title, t.summary, t.state,
                      t.created_at, t.last_active_at,
                      (SELECT COUNT(*) FROM messages m
                       WHERE m.topic_id = t.id AND m.active = 1) AS message_count
               FROM session_topics t WHERE t.session_id = ?
               ORDER BY t.last_active_at DESC, t.id DESC""",
            (conversation_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_topics(self, session_id: str) -> List[Dict[str, Any]]:
        """Topics for a session's compression lineage, most recently active first."""
        if not session_id:
            return []
        with self._read_ctx() as conn:
            conversation_id = self._topic_conversation_id_on_conn(conn, session_id)
            return self._topics_for_conversation_on_conn(conn, conversation_id)

    def get_active_topic(self, session_id: str) -> Optional[Dict[str, Any]]:
        return next((topic for topic in self.get_topics(session_id) if topic["state"] == "active"), None)

    def ensure_session_topic(self, session_id: str, title: str) -> Dict[str, Any]:
        """Return/create one active topic and adopt unlabelled legacy rows atomically."""
        clean_title = " ".join(str(title or "session").strip().split())[:64] or "session"
        normalized = _normalized_topic_title(clean_title) or "session"
        now = time.time()

        def _do(conn):
            conversation_id = self._topic_conversation_id_on_conn(conn, session_id)
            topics = self._topics_for_conversation_on_conn(conn, conversation_id)
            active = next((topic for topic in topics if topic["state"] == "active"), None)
            if active is None and topics:
                active = topics[0]
                conn.execute(
                    "UPDATE session_topics SET state = 'active', last_active_at = ? WHERE id = ?",
                    (now, active["id"]),
                )
                active = {**active, "state": "active", "last_active_at": now}
            if active is None:
                topic_id = conn.execute(
                    """INSERT INTO session_topics
                       (session_id, title, normalized_title, summary, state, created_at, last_active_at)
                       VALUES (?, ?, ?, NULL, 'active', ?, ?)""",
                    (conversation_id, clean_title, normalized, now, now),
                ).lastrowid
                active = {
                    "id": topic_id, "title": clean_title, "normalized_title": normalized,
                    "summary": None, "state": "active", "created_at": now,
                    "last_active_at": now, "message_count": 0,
                }
            lineage_ids = self._topic_lineage_ids_on_conn(conn, session_id)
            conn.execute(
                f"UPDATE messages SET topic_id = ? WHERE topic_id IS NULL "
                f"AND session_id IN ({_placeholders(lineage_ids)})",
                (active["id"], *lineage_ids),
            )
            return active

        return self._execute_write(_do)

    def create_topic(self, session_id: str, title: str, summary: Optional[str] = None) -> int:
        """Archive the prior active topic and create a new active topic atomically."""
        selected = self.activate_topic_for_messages(
            session_id, title=title, summary=summary, message_ids=[]
        )
        return int(selected["id"])

    def set_active_topic(self, session_id: str, topic_id: int) -> bool:
        """Activate an existing topic; an invalid id leaves the prior topic unchanged."""
        try:
            self.activate_topic_for_messages(session_id, topic_id=topic_id, message_ids=[])
            return True
        except (LookupError, ValueError):
            return False

    def activate_topic_for_messages(
        self, session_id: str, *, topic_id: Optional[int] = None,
        title: Optional[str] = None, summary: Optional[str] = None,
        message_ids: Optional[List[int]] = None,
        turn_lease_holder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Activate/create a topic and retag exact current-turn rows in one transaction."""
        clean_title = " ".join(str(title or "").strip().split())[:64]
        normalized = _normalized_topic_title(clean_title)
        ids = list(dict.fromkeys(
            int(row_id) for row_id in (message_ids or [])
            if isinstance(row_id, int) and not isinstance(row_id, bool) and row_id > 0
        ))
        now = time.time()

        def _do(conn):
            self._check_transcript_write_guards(
                conn, session_id, None, turn_lease_holder=turn_lease_holder
            )
            conversation_id = self._topic_conversation_id_on_conn(conn, session_id)
            target = None
            if topic_id is not None:
                target = conn.execute(
                    "SELECT * FROM session_topics WHERE id = ? AND session_id = ?",
                    (int(topic_id), conversation_id),
                ).fetchone()
                if target is None:
                    raise LookupError(f"Topic {topic_id} does not belong to session {session_id}")
            else:
                if not clean_title or not normalized:
                    raise ValueError("topic title must not be empty")
                target = conn.execute(
                    """SELECT * FROM session_topics
                       WHERE session_id = ? AND normalized_title = ?
                       ORDER BY last_active_at DESC, id DESC LIMIT 1""",
                    (conversation_id, normalized),
                ).fetchone()
                if target is None:
                    new_id = conn.execute(
                        """INSERT INTO session_topics
                           (session_id, title, normalized_title, summary, state, created_at, last_active_at)
                           VALUES (?, ?, ?, ?, 'warm', ?, ?)""",
                        (conversation_id, clean_title, normalized, summary, now, now),
                    ).lastrowid
                    target = conn.execute(
                        "SELECT * FROM session_topics WHERE id = ?", (new_id,),
                    ).fetchone()
            target_id = int(target["id"])
            conn.execute(
                "UPDATE session_topics SET state = 'warm' "
                "WHERE session_id = ? AND state = 'active' AND id != ?",
                (conversation_id, target_id),
            )
            conn.execute(
                "UPDATE session_topics SET state = 'active', last_active_at = ? WHERE id = ?",
                (now, target_id),
            )
            if ids:
                rows = conn.execute(
                    f"SELECT id, session_id FROM messages WHERE id IN ({_placeholders(ids)})",
                    ids,
                ).fetchall()
                if len(rows) != len(ids) or any(
                    self._topic_conversation_id_on_conn(conn, row["session_id"]) != conversation_id
                    for row in rows
                ):
                    raise LookupError("message ids do not all belong to this conversation")
                conn.execute(
                    f"UPDATE messages SET topic_id = ? WHERE id IN ({_placeholders(ids)})",
                    (target_id, *ids),
                )
            updated = conn.execute(
                """SELECT t.id, t.title, t.normalized_title, t.summary, t.state,
                          t.created_at, t.last_active_at,
                          (SELECT COUNT(*) FROM messages m
                           WHERE m.topic_id = t.id AND m.active = 1) AS message_count
                   FROM session_topics t WHERE t.id = ?""",
                (target_id,),
            ).fetchone()
            return dict(updated)

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def retract_topic_turn_messages(
        self, session_id: str, message_ids: List[int],
        *, turn_lease_holder: Optional[str] = None,
    ) -> int:
        """Atomically retire exact current-session rows after a failed topic transition.

        Reject missing/foreign ids before changing anything. This is an internal
        cleanup operation, never a suffix or cross-lineage transcript rewind.
        """
        ids = list(dict.fromkeys(message_ids))
        if not ids or any(type(row_id) is not int or row_id <= 0 for row_id in ids):
            raise ValueError("expected positive current-turn row ids")

        def _do(conn):
            self._check_transcript_write_guards(
                conn, session_id, None, turn_lease_holder=turn_lease_holder,
                reject_active_turn_lease=not bool(turn_lease_holder),
                reject_active_compression_lock=True,
            )
            rows = conn.execute(
                f"SELECT id, tool_calls FROM messages WHERE session_id = ? "
                f"AND id IN ({_placeholders(ids)})",
                (session_id, *ids),
            ).fetchall()
            if len(rows) != len(ids):
                raise LookupError("current-turn message ids are not all in this session")
            from hermes_state_messages import _tool_calls_len
            tool_calls = sum(_tool_calls_len(row["tool_calls"], scalar=1) for row in rows)
            conn.execute(
                f"DELETE FROM messages WHERE session_id = ? AND id IN ({_placeholders(ids)})",
                (session_id, *ids),
            )
            conn.execute(
                "UPDATE sessions SET message_count = MAX(0, message_count - ?), "
                "tool_call_count = MAX(0, tool_call_count - ?) WHERE id = ?",
                (len(ids), tool_calls, session_id),
            )
            return len(ids)

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)
