"""Caller-history delivery state of detached delegation rows: claim, reserve, commit and release.

The state lives in ``display_metadata``: ``caller_history_consumed`` is final; ``caller_history_reservation`` =
``{token, owner, until}`` is a lease that expires on its own.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Optional, Sequence

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import _decode_content, _decode_display_metadata, _resume_lineage_ids
from persistence.sqlite.conversation._store.sql_common import _placeholders

_CONSUME_META_SQL = "json_remove(json_set(display_metadata, '$.caller_history_consumed', 1), '$.caller_history_reservation')"
_RELEASE_META_SQL = "json_remove(display_metadata, '$.caller_history_reservation')"
_PENDING_SQL = (
    " AND role = 'user' AND display_kind IN ('async_delegation_complete', 'hidden')"
    " AND COALESCE(active, 1) = 1"
    " AND json_extract(display_metadata, '$.delegation_id') IS NOT NULL"
    " AND json_extract(display_metadata, '$.caller_history_consumed') IS NULL"
    " AND COALESCE(json_extract(display_metadata, '$.caller_history_reservation.until'), 0) <= ?")


def _pending_rows(conn, lineage: Sequence[str], now: float, limit: Optional[int] = None) -> list:
    sql = (f"SELECT id, session_id, content, display_metadata FROM messages"
           f" WHERE session_id IN ({_placeholders(lineage)}){_PENDING_SQL} ORDER BY id")
    if limit is None:
        return conn.execute(sql, (*lineage, now)).fetchall()
    return conn.execute(sql + " LIMIT ?", (*lineage, now, max(0, int(limit)))).fetchall()


def _delivery(row, token: Optional[str] = None) -> dict[str, Any]:
    content = _decode_content(row["content"])
    out = {"id": row["id"], "delivery_id": row["id"], "session_id": row["session_id"],
           "content": content, "result": content,
           "display_metadata": _decode_display_metadata(row["display_metadata"]) or {}}
    if token is not None:
        out["reservation_token"] = token
    return out


def claim_caller_history_deliveries(holder: StoreHolder, session_id: str) -> list[dict[str, Any]]:
    """Consume the lineage's pending rows for good in one writer txn (skips live leases)."""
    lineage = _resume_lineage_ids(holder, session_id)

    def _do(conn):
        rows = _pending_rows(conn, lineage, time.time())
        conn.executemany(f"UPDATE messages SET display_metadata = {_CONSUME_META_SQL} WHERE id = ?",
                         [(row["id"],) for row in rows])
        return [_delivery(row) for row in rows]

    return holder._execute_write(_do)


def reserve_caller_history_deliveries(holder: StoreHolder, session_id: str, owner: str, ttl_seconds: float,
                                      limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Lease pending rows to ``owner`` for ``ttl_seconds`` under one ``reservation_token``."""
    lineage = _resume_lineage_ids(holder, session_id)
    token = uuid.uuid4().hex

    def _do(conn):
        now = time.time()
        rows = _pending_rows(conn, lineage, now, limit)
        lease = json.dumps({"token": token, "owner": str(owner), "until": now + float(ttl_seconds)})
        conn.executemany("UPDATE messages SET display_metadata = json_set(display_metadata,"
                         " '$.caller_history_reservation', json(?)) WHERE id = ?",
                         [(lease, row["id"]) for row in rows])
        return [_delivery(row, token) for row in rows]

    return holder._execute_write(_do)


def _settle_reservation(holder: StoreHolder, new_metadata_sql: str, token_or_ids: str | Sequence[int] | None,
                        owner: str) -> int:
    """Rewrite metadata of unconsumed rows leased to ``owner``; returns rows changed."""
    where = ("json_extract(display_metadata, '$.caller_history_consumed') IS NULL"
             " AND json_extract(display_metadata, '$.caller_history_reservation.owner') = ?")
    if isinstance(token_or_ids, str):
        # A re-reserve replaces the token, so its holder may settle even after the lease expired.
        where += " AND json_extract(display_metadata, '$.caller_history_reservation.token') = ?"
        params: tuple = (str(owner), token_or_ids)
    else:
        ids = [int(i) for i in token_or_ids or ()]
        if not ids:
            return 0
        where += (f" AND json_extract(display_metadata, '$.caller_history_reservation.until') > ?"
                  f" AND id IN ({_placeholders(ids)})")
        params = (str(owner), time.time(), *ids)
    return holder._execute_write(lambda conn: conn.execute(
        f"UPDATE messages SET display_metadata = {new_metadata_sql} WHERE {where}", params).rowcount)


def commit_caller_history_deliveries(holder: StoreHolder, reservation_token_or_ids: str | Sequence[int],
                                     owner: str) -> int:
    """Mark rows still leased to ``owner`` delivered for good; returns rows committed."""
    return _settle_reservation(holder, _CONSUME_META_SQL, reservation_token_or_ids, owner)


def release_caller_history_deliveries(holder: StoreHolder, session_id: Optional[str] = None,
                                      row_ids: Optional[Sequence[int]] = None, *,
                                      reservation_token: Optional[str] = None, owner: Optional[str] = None) -> int:
    """Return rows to pending: with ``owner`` drop its lease, else undo a ``(session_id, row_ids)`` claim."""
    if owner is not None:
        return _settle_reservation(holder, _RELEASE_META_SQL,
                                   reservation_token if reservation_token is not None else row_ids, owner)
    ids = [int(i) for i in row_ids or ()]
    if not ids or not session_id:
        return 0
    lineage = _resume_lineage_ids(holder, session_id)
    return holder._execute_write(lambda conn: conn.execute(
        "UPDATE messages SET display_metadata = json_remove(display_metadata, '$.caller_history_consumed')"
        f" WHERE id IN ({_placeholders(ids)}) AND session_id IN ({_placeholders(lineage)})"
        " AND json_extract(display_metadata, '$.caller_history_consumed') IS NOT NULL",
        (*ids, *lineage)).rowcount)
