"""Results that arrive between turns, recorded once as user rows: a detached API delegation's delivery and
an unviewed local completion's note (#134889 LC-7). Split out of ``hermes_state_messages`` and bound onto
SessionDB through ``SessionMessagesMixin``; built on its _message_row_params/_execute_write primitives."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from agent.message_metadata import stamp_message_uid
from hermes_state_common import _placeholders

# Rows of a session and of the parents it was compressed from; a detached result's match condition follows.
_LINEAGE_MATCH_SQL = """WITH RECURSIVE lineage(id) AS (SELECT ? UNION
    SELECT s.parent_session_id FROM sessions s JOIN lineage l ON s.id = l.id
    JOIN sessions p ON p.id = s.parent_session_id WHERE p.end_reason = 'compression'
) SELECT m.id FROM messages m JOIN lineage l ON m.session_id = l.id WHERE """
_DELEGATION_DELIVERY_MATCH = ("m.display_kind IN ('async_delegation_complete', 'hidden') AND json_extract(m.display_metadata, "
                              "'$.delegation_id') = ? AND coalesce(json_extract(m.display_metadata, '$.delivery_notice'), '') = ?")


class SessionDetachedResultsMixin:
    """Detached API delegation deliveries and unviewed completion notes."""

    def append_delegation_delivery(self, session_id: str, content: str, metadata: dict[str, Any]) -> int:
        """Record a detached API result once, keyed by the event's unit id (not its text or active flag)."""
        delegation_id = metadata.get("delegation_id")
        if not delegation_id:
            raise ValueError("Delegation delivery requires a stable delegation_id")
        kind = "hidden" if metadata.get("presentation_suppressed") else "async_delegation_complete"
        return self._append_detached_result(session_id, content, kind, metadata, _DELEGATION_DELIVERY_MATCH,
                                            (delegation_id, metadata.get("delivery_notice", "")))

    def append_completion_note(self, session_id: str, admission_id: str, content: str, display_kind: str,
                               metadata: dict[str, Any]) -> int:
        """Record an unviewed completion's note once, keyed by its admission (stored as ``completion_admission``)."""
        return self._append_detached_result(
            session_id, content, display_kind, {**metadata, "completion_admission": admission_id},
            "json_extract(m.display_metadata, '$.completion_admission') = ?", (admission_id,))

    def _append_detached_result(self, session_id: str, content: str, display_kind: str, metadata: dict[str, Any],
                                match_sql: str, params: tuple) -> int:
        """Record a result that arrived between turns as a user row once, including replay after rotation: a row
        ``match_sql`` (a condition on ``m``) finds in the compression lineage is returned instead. Check and insert
        share the writer transaction, so independent gateway processes cannot duplicate it; a live turn refuses it."""
        from hermes_state_messages import _INSERT_MESSAGE_SQL  # that module binds this mixin
        msg = {"content": content, "display_kind": display_kind, "display_metadata": metadata}
        stamp_message_uid(msg)  # a fresh occurrence: mint its durable id
        row = self._message_row_params(session_id, "user", msg, None, time.time(), keep_reasoning=True)

        def _do(conn):
            existing = conn.execute(_LINEAGE_MATCH_SQL + match_sql + " LIMIT 1", (session_id, *params)).fetchone()
            if existing is not None:
                return existing[0]
            self._check_transcript_write_guards(conn, session_id, None, reject_active_turn_lease=True)
            msg_id = conn.execute(_INSERT_MESSAGE_SQL, row).lastrowid
            self._bump_session_counters(conn, session_id, 1, 0, unit=True)
            return msg_id

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    # Caller-history delivery state lives in display_metadata: ``caller_history_consumed`` is final;
    # ``caller_history_reservation`` = {token, owner, until} is a lease that expires on its own.
    _CH_SELECT = (
        " AND role = 'user' AND display_kind IN ('async_delegation_complete', 'hidden')"
        " AND COALESCE(active, 1) = 1"
        " AND json_extract(display_metadata, '$.delegation_id') IS NOT NULL"
        " AND json_extract(display_metadata, '$.caller_history_consumed') IS NULL"
        " AND COALESCE(json_extract(display_metadata, '$.caller_history_reservation.until'), 0) <= ?")

    def _pending_caller_history_rows(self, conn, lineage, now, limit=None):
        sql = (f"SELECT id, session_id, content, display_metadata FROM messages"
               f" WHERE session_id IN ({_placeholders(lineage)})" + self._CH_SELECT + " ORDER BY id")
        params: list = [*lineage, now]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, int(limit)))
        return conn.execute(sql, tuple(params)).fetchall()

    def _caller_history_row(self, row, token=None):
        content = self._decode_content(row["content"])
        out = {"id": row["id"], "delivery_id": row["id"], "session_id": row["session_id"],
               "content": content, "result": content,
               "display_metadata": self._decode_display_metadata(row["display_metadata"]) or {}}
        if token is not None:
            out["reservation_token"] = token
        return out

    def claim_caller_history_deliveries(self, session_id: str) -> list[dict[str, Any]]:
        """Finally consume the lineage's pending delegation rows (skipping live reservations).

        Check and stamp share one writer txn, so each row has exactly one consumer.
        """
        lineage = self._resume_lineage_ids(session_id)

        def _do(conn):
            rows = self._pending_caller_history_rows(conn, lineage, time.time())
            for row in rows:
                conn.execute("UPDATE messages SET display_metadata = json_remove(json_set(display_metadata,"
                             " '$.caller_history_consumed', 1), '$.caller_history_reservation') WHERE id = ?",
                             (row["id"],))
            return [self._caller_history_row(row) for row in rows]

        return self._execute_write(_do)

    def reserve_caller_history_deliveries(self, session_id: str, owner: str, ttl_seconds: float,
                                          limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Lease pending rows to ``owner`` for ``ttl_seconds``; commit or release them under the
        returned ``reservation_token``. Expired leases are reclaimable by any caller."""
        import uuid
        lineage = self._resume_lineage_ids(session_id)
        token = uuid.uuid4().hex

        def _do(conn):
            now = time.time()
            rows = self._pending_caller_history_rows(conn, lineage, now, limit)
            lease = json.dumps({"token": token, "owner": str(owner), "until": now + float(ttl_seconds)})
            for row in rows:
                conn.execute("UPDATE messages SET display_metadata = json_set(display_metadata,"
                             " '$.caller_history_reservation', json(?)) WHERE id = ?", (lease, row["id"]))
            return [self._caller_history_row(row, token) for row in rows]

        return self._execute_write(_do)

    def _held_reservation_where(self, token_or_ids, owner):
        clause = (" AND json_extract(display_metadata, '$.caller_history_consumed') IS NULL"
                  " AND json_extract(display_metadata, '$.caller_history_reservation.owner') = ?")
        if isinstance(token_or_ids, str):
            # A re-reserve replaces the token, so a matching token proves nobody took the rows over:
            # its holder may settle them even after the lease ran out (e.g. after a consumer restart).
            return ("json_extract(display_metadata, '$.caller_history_reservation.token') = ?" + clause,
                    [token_or_ids, str(owner)])
        clause += " AND json_extract(display_metadata, '$.caller_history_reservation.until') > ?"
        ids = [int(i) for i in token_or_ids or ()]
        if not ids:
            return None, None
        return (f"id IN ({_placeholders(ids)})" + clause, [*ids, str(owner), time.time()])

    def commit_caller_history_deliveries(self, reservation_token_or_ids, owner: str) -> int:
        """Mark rows still leased to ``owner`` delivered for good; returns rows committed."""
        where, params = self._held_reservation_where(reservation_token_or_ids, owner)
        if where is None:
            return 0
        return self._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET display_metadata = json_remove(json_set(display_metadata,"
            " '$.caller_history_consumed', 1), '$.caller_history_reservation') WHERE " + where,
            tuple(params)).rowcount)

    def release_caller_history_deliveries(self, session_id: Optional[str] = None,
                                          row_ids: Optional[list[int]] = None, *,
                                          reservation_token: Optional[str] = None,
                                          owner: Optional[str] = None) -> int:
        """Return rows to pending: ``(session_id, row_ids)`` undoes a claim; ``reservation_token=``
        + ``owner=`` (or ``row_ids`` + ``owner``) drops a lease held by ``owner``."""
        if owner is not None:
            where, params = self._held_reservation_where(
                reservation_token if reservation_token is not None else (row_ids or []), owner)
            if where is None:
                return 0
            return self._execute_write(lambda conn: conn.execute(
                "UPDATE messages SET display_metadata = json_remove(display_metadata,"
                " '$.caller_history_reservation') WHERE " + where, tuple(params)).rowcount)
        ids = [int(i) for i in row_ids or ()]
        if not ids or not session_id:
            return 0
        lineage = self._resume_lineage_ids(session_id)

        def _do(conn):
            return conn.execute(
                "UPDATE messages SET display_metadata = json_remove(display_metadata, '$.caller_history_consumed')"
                f" WHERE id IN ({_placeholders(ids)}) AND session_id IN ({_placeholders(lineage)})"
                " AND json_extract(display_metadata, '$.caller_history_consumed') IS NOT NULL",
                (*ids, *lineage)).rowcount

        return self._execute_write(_do)
