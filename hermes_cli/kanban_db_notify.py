"""Notification subscriptions consumed by the gateway kanban-notifier: per-(task, platform, chat, thread) rows with delivery metadata, unseen-event cursors and purge of stale done-task subs.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Event


# Notifier reaction to a terminal event: "notify" = passive adapter.send only
# (default); "notify+wake" = send AND wake the destination agent; "wake" = wake only.
_NOTIFY_DELIVERY_MODES = ("notify", "notify+wake", "wake")

_SCALAR_TYPES = (str, int, float, bool)

# Subscription primary key predicate; every per-row statement below binds
# ``(task_id, platform, chat_id, thread_id or "")`` against it.
_SUB_KEY_WHERE = "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?"
_OUTBOX_OPEN_STATES = ("pending", "sending", "retry_wait", "delivery_unknown", "dead_letter")
_MAX_DELIVERY_ATTEMPTS = 12


def _sub_key(task_id: str, platform: str, chat_id: str, thread_id: Optional[str]) -> tuple:
    return (task_id, platform, chat_id, thread_id or "")


def _safe_delivery_error(error: Any, limit: int = 240) -> str:
    text = " ".join(str(error or "delivery failed").split())
    if any(prefix in text for prefix in ("/home/", "/Users/", "/private/", "/tmp/", "C:\\")):
        text = "delivery failed (local detail redacted)"
    return text[:limit]


def _delivery_payload(event: Event, sub: Mapping[str, Any]) -> tuple[str, str, str]:
    payload = {
        "event_id": int(event.id), "kind": event.kind, "payload": event.payload or {},
        "delivery_mode": sub.get("delivery_mode") or "notify",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    origin = json.dumps([
        sub["task_id"], sub["platform"], sub["chat_id"], sub.get("thread_id") or "",
        sub.get("notifier_profile") or "",
    ], separators=(",", ":"))
    key = hashlib.sha256(f"{event.id}:{origin}:{digest}".encode()).hexdigest()
    return key, digest, encoded


def enqueue_delivery(conn: sqlite3.Connection, *, event: Event, sub: Mapping[str, Any]) -> dict:
    """Idempotently persist one exact task-event-origin delivery obligation."""
    key, digest, encoded = _delivery_payload(event, sub)
    now = int(time.time())
    with _kb.write_txn(conn):
        conn.execute(
            """INSERT OR IGNORE INTO kanban_delivery_outbox
               (delivery_key,task_id,event_id,platform,chat_id,thread_id,notifier_profile,
                payload_digest,payload_json,state,created_at,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?)""",
            (key, sub["task_id"], int(event.id), sub["platform"], sub["chat_id"],
             sub.get("thread_id") or "", sub.get("notifier_profile"), digest, encoded, now, now),
        )
        row = conn.execute("SELECT * FROM kanban_delivery_outbox WHERE delivery_key=?", (key,)).fetchone()
    return dict(row)


def release_expired_delivery_leases(conn: sqlite3.Connection, *, now: Optional[int] = None) -> int:
    """Quarantine possible transport acceptance; never blindly replay it."""
    stamp = int(time.time()) if now is None else int(now)
    with _kb.write_txn(conn):
        cur = conn.execute(
            """UPDATE kanban_delivery_outbox SET state='delivery_unknown', lease_token=NULL,
               lease_expires_at=NULL, last_error='delivery lease expired after send may have started', updated_at=?
               WHERE state='sending' AND lease_expires_at IS NOT NULL AND lease_expires_at<=?""",
            (stamp, stamp),
        )
    return int(cur.rowcount or 0)


def claim_delivery(conn: sqlite3.Connection, *, delivery_key: str, now: Optional[int] = None,
                   lease_seconds: int = 60) -> Optional[dict]:
    """CAS-claim one due obligation; concurrent claimers cannot share it."""
    stamp = int(time.time()) if now is None else int(now)
    token = secrets.token_hex(16)
    with _kb.write_txn(conn):
        conn.execute(
            """UPDATE kanban_delivery_outbox SET state='delivery_unknown', lease_token=NULL,
               lease_expires_at=NULL, last_error='delivery lease expired after send may have started', updated_at=?
               WHERE delivery_key=? AND state='sending' AND lease_expires_at<=?""",
            (stamp, delivery_key, stamp),
        )
        cur = conn.execute(
            """UPDATE kanban_delivery_outbox SET state='sending',lease_token=?,lease_expires_at=?,updated_at=?
               WHERE delivery_key=? AND state IN ('pending','retry_wait') AND next_attempt_at<=?""",
            (token, stamp + max(1, int(lease_seconds)), stamp, delivery_key, stamp),
        )
        if cur.rowcount != 1:
            return None
        row = conn.execute("SELECT * FROM kanban_delivery_outbox WHERE delivery_key=?", (delivery_key,)).fetchone()
    return dict(row)


def mark_delivery_ping_delivered(
    conn: sqlite3.Connection, *, delivery_key: str, lease_token: str,
    transport_receipt: str, now: Optional[int] = None,
) -> bool:
    """Checkpoint this obligation's accepted passive ping under its live lease."""
    receipt = " ".join(str(transport_receipt or "").split())[:240]
    if not receipt:
        raise ValueError("transport receipt is required")
    stamp = int(time.time()) if now is None else int(now)
    with _kb.write_txn(conn):
        cur = conn.execute(
            """UPDATE kanban_delivery_outbox
               SET ping_delivered_at=?,ping_receipt=?,updated_at=?
               WHERE delivery_key=? AND state='sending' AND lease_token=?
                 AND ping_delivered_at IS NULL""",
            (stamp, receipt, stamp, delivery_key, lease_token),
        )
    return cur.rowcount == 1


def acknowledge_delivery(conn: sqlite3.Connection, *, delivery_key: str, lease_token: str,
                         transport_receipt: str) -> bool:
    """Mark delivered only with the current lease and receipt evidence."""
    receipt = " ".join(str(transport_receipt or "").split())[:240]
    if not receipt:
        raise ValueError("transport receipt is required")
    with _kb.write_txn(conn):
        cur = conn.execute(
            """UPDATE kanban_delivery_outbox SET state='delivered',transport_receipt=?,
               lease_token=NULL,lease_expires_at=NULL,last_error=NULL,updated_at=?
               WHERE delivery_key=? AND state='sending' AND lease_token=?""",
            (receipt, int(time.time()), delivery_key, lease_token),
        )
    return cur.rowcount == 1


def fail_delivery(conn: sqlite3.Connection, *, delivery_key: str, lease_token: str, error: Any,
                  retry_base_seconds: int = 5, max_attempts: int = _MAX_DELIVERY_ATTEMPTS,
                  now: Optional[int] = None) -> Optional[str]:
    """Persist a confirmed failure, bounded retry budget, and durable backoff."""
    stamp = int(time.time()) if now is None else int(now)
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT attempts FROM kanban_delivery_outbox WHERE delivery_key=? AND state='sending' AND lease_token=?",
            (delivery_key, lease_token),
        ).fetchone()
        if row is None:
            return None
        attempts = int(row["attempts"]) + 1
        state = "dead_letter" if attempts >= max(1, int(max_attempts)) else "retry_wait"
        delay = 0 if state == "dead_letter" else min(300, max(0, int(retry_base_seconds)) * (2 ** min(attempts - 1, 6)))
        conn.execute(
            """UPDATE kanban_delivery_outbox SET state=?,attempts=?,next_attempt_at=?,lease_token=NULL,
               lease_expires_at=NULL,last_error=?,updated_at=?
               WHERE delivery_key=? AND state='sending' AND lease_token=?""",
            (state, attempts, stamp + delay, _safe_delivery_error(error), stamp, delivery_key, lease_token),
        )
    return state


def mark_delivery_ambiguous(conn: sqlite3.Connection, *, delivery_key: str, lease_token: str,
                            error: Any, transport_receipt: Optional[str] = None,
                            now: Optional[int] = None) -> bool:
    """Hold uncertain sends for reconciliation; never return them to the retry queue.

    ``delivery_unknown`` is the existing schema state for a transport that may
    already have accepted a non-idempotent side effect.  Keep that single state
    name rather than introducing an incompatible synonym.
    """
    stamp = int(time.time()) if now is None else int(now)
    receipt = " ".join(str(transport_receipt or "").split())[:240] or None
    with _kb.write_txn(conn):
        c = conn.execute(
            "UPDATE kanban_delivery_outbox SET state='delivery_unknown', updated_at=?, last_error=?, "
            "transport_receipt=COALESCE(?, transport_receipt), lease_token=NULL, lease_expires_at=NULL "
            "WHERE delivery_key=? AND state='sending' AND lease_token=?",
            (stamp, _safe_delivery_error(error), receipt, delivery_key, lease_token),
        )
    return c.rowcount == 1


def mark_delivery_exception_recorded(conn: sqlite3.Connection, *, delivery_key: str) -> bool:
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_delivery_outbox SET exception_recorded=1,updated_at=? "
            "WHERE delivery_key=? AND exception_recorded=0 AND state IN ('dead_letter','delivery_unknown')",
            (int(time.time()), delivery_key),
        )
    return cur.rowcount == 1


def claim_delivery_unknown_warnings(
    conn: sqlite3.Connection, *, task_id: Optional[str] = None,
    platform: Optional[str] = None, chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> list[dict]:
    """Atomically claim once-only operator warnings for quarantined deliveries."""
    clauses = ["state='delivery_unknown'", "exception_recorded=0"]
    params: list[Any] = []
    for column, value in (
        ("task_id", task_id), ("platform", platform), ("chat_id", chat_id),
        ("thread_id", thread_id),
    ):
        if value is not None:
            clauses.append(f"{column}=?")
            params.append(value)
    where = " AND ".join(clauses)
    with _kb.write_txn(conn):
        rows = conn.execute(
            f"SELECT * FROM kanban_delivery_outbox WHERE {where} ORDER BY event_id,id",
            params,
        ).fetchall()
        if rows:
            keys = [row["delivery_key"] for row in rows]
            placeholders = ",".join("?" for _ in keys)
            conn.execute(
                f"UPDATE kanban_delivery_outbox SET exception_recorded=1,updated_at=? "
                f"WHERE exception_recorded=0 AND state='delivery_unknown' "
                f"AND delivery_key IN ({placeholders})",
                (int(time.time()), *keys),
            )
    return [dict(row) for row in rows]


def list_delivery_outbox(
    conn: sqlite3.Connection, *, state: Optional[str] = "delivery_unknown",
) -> list[dict]:
    """Return local operator status without exposing stored message payloads."""
    allowed = {
        "pending", "sending", "retry_wait", "delivered", "delivery_unknown", "dead_letter",
    }
    if state is not None and state not in allowed:
        raise ValueError(f"invalid delivery state: {state}")
    fields = (
        "delivery_key,task_id,event_id,platform,chat_id,thread_id,notifier_profile,state,attempts,"
        "next_attempt_at,lease_expires_at,last_error,transport_receipt,exception_recorded,created_at,updated_at"
    )
    sql = f"SELECT {fields} FROM kanban_delivery_outbox"
    params: tuple[Any, ...] = ()
    if state is not None:
        sql += " WHERE state=?"
        params = (state,)
    sql += " ORDER BY event_id,id"
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def reconcile_delivery_unknown(
    conn: sqlite3.Connection, *, delivery_key: str, action: str, reason: str,
    operator: str, accept_duplicate_risk: bool = False,
    now: Optional[int] = None,
) -> dict:
    """Guardedly resolve one uncertain delivery and append a durable task audit.

    ``retry`` can duplicate a delivery whose transport side effect actually
    succeeded, so it requires an explicit risk acknowledgement. ``mark-delivered``
    suppresses replay and should only be used after the operator verifies the
    external effect.
    """
    if action not in {"retry", "mark-delivered"}:
        raise ValueError("action must be retry or mark-delivered")
    if action == "retry" and not accept_duplicate_risk:
        raise ValueError("retry requires --accept-duplicate-risk")
    clean_reason = " ".join(str(reason or "").split())[:500]
    if not clean_reason:
        raise ValueError("a reconciliation reason is required")
    clean_operator = " ".join(str(operator or "unknown").split())[:120] or "unknown"
    stamp = int(time.time()) if now is None else int(now)
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT * FROM kanban_delivery_outbox WHERE delivery_key=?", (delivery_key,),
        ).fetchone()
        if row is None:
            return {"ok": False, "delivery_key": delivery_key, "state": None}
        if row["state"] != "delivery_unknown":
            return {"ok": False, "delivery_key": delivery_key, "state": row["state"]}
        if action == "retry":
            new_state = "retry_wait"
            receipt = row["transport_receipt"]
        else:
            new_state = "delivered"
            receipt = f"operator-verified:{clean_operator}"[:240]
        cur = conn.execute(
            """UPDATE kanban_delivery_outbox
               SET state=?,next_attempt_at=?,lease_token=NULL,lease_expires_at=NULL,
                   last_error=NULL,transport_receipt=?,exception_recorded=0,updated_at=?
               WHERE delivery_key=? AND state='delivery_unknown'""",
            (new_state, stamp, receipt, stamp, delivery_key),
        )
        if cur.rowcount != 1:
            current = conn.execute(
                "SELECT state FROM kanban_delivery_outbox WHERE delivery_key=?", (delivery_key,),
            ).fetchone()
            return {"ok": False, "delivery_key": delivery_key,
                    "state": current["state"] if current else None}
        _kb._append_event(conn, row["task_id"], "delivery_reconciled", {
            "delivery_key": delivery_key,
            "action": action,
            "operator": clean_operator,
            "reason": clean_reason,
            "duplicate_risk_accepted": bool(accept_duplicate_risk),
            "prior_error": row["last_error"],
            "prior_transport_receipt": row["transport_receipt"],
        })
    return {"ok": True, "delivery_key": delivery_key, "state": new_state, "action": action}


def open_delivery_obligations(conn: sqlite3.Connection, *, task_id: Optional[str] = None) -> int:
    params: list[Any] = list(_OUTBOX_OPEN_STATES)
    sql = "SELECT COUNT(*) FROM kanban_delivery_outbox WHERE state IN (" + ",".join("?" for _ in params) + ")"
    if task_id is not None:
        sql += " AND task_id=?"
        params.append(task_id)
    row = conn.execute(sql, params).fetchone()
    return int(row[0]) if row else 0


def _encode_notify_delivery_metadata(metadata: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Serialize platform send metadata stored on notification subscriptions."""
    if not isinstance(metadata, Mapping):
        return None
    clean = {
        str(key): value
        for key, value in metadata.items()
        if value is not None and isinstance(value, _SCALAR_TYPES)
    }
    if not clean:
        return None
    return json.dumps(clean, sort_keys=True, separators=(",", ":"))


def _decode_notify_delivery_metadata(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        data = json.loads(str(raw))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(key): value for key, value in data.items() if isinstance(value, _SCALAR_TYPES)}


def add_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    user_id_alt: Optional[str] = None,
    chat_type: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    delivery_mode: Optional[str] = None,
    delivery_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register a gateway source wanting terminal-state notifications for
    ``task_id``; idempotent on (task, platform, chat, thread).

    ``user_id_alt`` (Signal UUID, Feishu union_id, ...) and ``chat_type`` are
    replayed on active wake: ``build_session_key`` prefers the alt id, so
    omitting it would key the wake into a different session. ``None`` keeps an
    existing row's value. ``delivery_mode``: ``None`` leaves an existing row
    untouched, an explicit valid value is last-write-wins, unknown falls back
    to ``"notify"``. New subs start caught up (``last_event_id`` =
    ``MAX(task_events.id)``) so the notifier never replays history at boot.
    """
    valid_mode = delivery_mode if delivery_mode in _NOTIFY_DELIVERY_MODES else None
    # api_server is stateless: the adapter has no send(), the wake self-post IS
    # the delivery. A plain 'notify' default would leave those subs with no
    # delivery mechanism at all. Explicit modes still win.
    insert_mode = valid_mode or ("notify+wake" if platform == "api_server" else "notify")
    metadata_json = _encode_notify_delivery_metadata(delivery_metadata)
    key = _sub_key(task_id, platform, chat_id, thread_id)
    with _kb.write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO kanban_notify_subs
                (task_id, platform, chat_id, thread_id, user_id, user_id_alt,
                 chat_type, notifier_profile, delivery_mode, delivery_metadata,
                 created_at, last_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT MAX(id) FROM task_events WHERE task_id = ?), 0))
            """,
            (
                *key, user_id, user_id_alt, chat_type or "dm", notifier_profile,
                insert_mode, metadata_json, int(time.time()), task_id,
            ),
        )
        # chat_type / delivery_mode / delivery_metadata are last-write-wins;
        # user_id_alt and notifier_profile only self-heal legacy rows lacking one.
        for column, value, fill_only in (
            ("chat_type", chat_type, False),
            ("user_id_alt", user_id_alt, True),
            ("notifier_profile", notifier_profile, True),
            ("delivery_mode", valid_mode, False),
            ("delivery_metadata", metadata_json, False),
        ):
            if not value:
                continue
            guard = f" AND ({column} IS NULL OR {column} = '')" if fill_only else ""
            conn.execute(
                f"UPDATE kanban_notify_subs SET {column} = ? " + _SUB_KEY_WHERE + guard,
                (value, *key),
            )


def _notify_profile_filter(
    notifier_profiles: Optional[Iterable[str]],
    *,
    include_unowned: bool,
) -> tuple[str, list[str]]:
    """Build an optional SQL predicate for notification profile ownership."""
    if notifier_profiles is None:
        return "", []

    profiles = sorted({str(p).strip() for p in notifier_profiles if str(p).strip()})
    clauses: list[str] = []
    params: list[str] = []
    if profiles:
        clauses.append("notifier_profile IN (" + ",".join("?" for _ in profiles) + ")")
        params.extend(profiles)
    if include_unowned:
        clauses.append("notifier_profile IS NULL OR notifier_profile = ''")
    if not clauses:
        return "0", []
    return "(" + ") OR (".join(clauses) + ")", params


def list_notify_subs(
    conn: sqlite3.Connection,
    task_id: Optional[str] = None,
    *,
    notifier_profiles: Optional[Iterable[str]] = None,
    include_unowned: bool = False,
) -> list[dict]:
    """List subscriptions, optionally restricted to notifier profile owners.

    No ``notifier_profiles`` -> all subscriptions. Gateway notifiers pass the
    profiles they own so they cannot claim another gateway's events;
    ``include_unowned`` (dispatch owner) covers legacy rows without a stamp.
    """
    owner_where, owner_params = _notify_profile_filter(
        notifier_profiles, include_unowned=include_unowned,
    )
    where: list[str] = []
    params: list[Any] = []
    if task_id is not None:
        where.append("task_id = ?")
        params.append(task_id)
    if owner_where:
        where.append(owner_where)
        params.extend(owner_params)
    sql = "SELECT * FROM kanban_notify_subs"
    if where:
        sql += " WHERE " + " AND ".join(f"({clause})" for clause in where)
    out: list[dict] = []
    for row in conn.execute(sql, params).fetchall():
        item = dict(row)
        if "delivery_metadata" in item:
            item["delivery_metadata"] = _decode_notify_delivery_metadata(item.get("delivery_metadata"))
        out.append(item)
    return out


def count_notify_subs(
    db_path: Optional[Path] = None,
    *,
    board: Optional[str] = None,
    notifier_profiles: Optional[Iterable[str]] = None,
    include_unowned: bool = False,
    platform: Optional[str] = None,
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> int:
    """Count ``kanban_notify_subs`` rows via a read-only connection — the
    notifier's cheap zero-subscription early exit. Unlike :func:`connect` it
    never creates the file, runs init/migration or opens writable; WAL rows are
    still visible so a fresh sub is never missed. Missing DB / missing table
    counts as zero; platform matches case-insensitively (as notifier routing),
    chat/thread exactly. Raises :class:`sqlite3.Error` if the DB exists but is
    unreadable — callers pick their own fallback.
    """
    path = db_path if db_path is not None else _kb.kanban_db_path(board=board)
    if not path.exists():
        return 0
    owner_where, owner_params = _notify_profile_filter(
        notifier_profiles, include_unowned=include_unowned,
    )
    clauses: list[str] = []
    params: list[Any] = []
    if owner_where:
        clauses.append(f"({owner_where})")
        params.extend(owner_params)
    for clause, value in (
        ("LOWER(platform) = LOWER(?)", platform),
        ("chat_id = ?", chat_id),
        ("thread_id = ?", thread_id),
    ):
        if value is not None:
            clauses.append(clause)
            params.append(value)
    query = "SELECT COUNT(*) FROM kanban_notify_subs"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            row = conn.execute(query, params).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc).lower():
                return 0
            raise
        return int(row[0]) if row else 0
    finally:
        conn.close()


def remove_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> bool:
    with _kb.write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs " + _SUB_KEY_WHERE,
            _sub_key(task_id, platform, chat_id, thread_id),
        )
    return cur.rowcount > 0


def purge_stale_done_notify_subs(conn: sqlite3.Connection, *, max_age_days: int = 30) -> int:
    """Delete notify subs whose task sat in ``done``/``blocked`` untouched for
    longer than ``max_age_days`` (``<= 0`` disables); returns rows deleted.

    Subs survive ``done`` because a reopened task must still notify its origin,
    which accumulates forever on never-archiving boards. ``blocked`` is
    abandoned (unlike ``backlog``/``ready``) so it reaps on the same clock. Age
    = latest event, else ``completed_at``, else ``created_at`` — any activity,
    including a reopen, exempts the sub.

    The notifier keeps subscriptions alive through ``done`` because a completed task can be reopened (review
    corrections, continuation) and the reopened cycle must still notify its origin session. On boards that
    never archive, that retention would otherwise accumulate subscription rows forever — each one scanned
    every notifier tick. This GC bounds that: a task that has been ``done`` with no new events for the
    retention window is treated as settled and its subscriptions are purged. ``blocked`` tasks
    (circuit-breaker trips, dead workers) are reaped on the same clock — they are abandoned, not idle,
    unlike a ``backlog``/``ready`` card that is merely waiting for pickup (#100955).
    """
    try:
        days = int(max_age_days)
    except (TypeError, ValueError):
        days = 30
    if days <= 0:
        return 0
    cutoff = int(time.time()) - days * 86400
    with _kb.write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id IN ("
            " SELECT t.id FROM tasks t"
            " WHERE t.status IN ('done', 'blocked')"
            " AND COALESCE("
            "  (SELECT MAX(e.created_at) FROM task_events e"
            "   WHERE e.task_id = t.id),"
            "  t.completed_at, t.created_at, 0"
            " ) < ?)"
            " AND NOT EXISTS ("
            "  SELECT 1 FROM kanban_delivery_outbox o"
            "  WHERE o.task_id = kanban_notify_subs.task_id"
            "  AND o.platform = kanban_notify_subs.platform"
            "  AND o.chat_id = kanban_notify_subs.chat_id"
            "  AND o.thread_id = kanban_notify_subs.thread_id"
            "  AND o.state IN ('pending','retry_wait','sending','delivery_unknown','dead_letter')"
            " )",
            (cutoff,),
        )
    return int(cur.rowcount or 0)


def _notify_cursor(
    conn: sqlite3.Connection, task_id: str, platform: str, chat_id: str, thread_id: Optional[str],
) -> Optional[int]:
    """``last_event_id`` of one subscription row, or ``None`` when unsubscribed."""
    row = conn.execute(
        "SELECT last_event_id FROM kanban_notify_subs " + _SUB_KEY_WHERE,
        _sub_key(task_id, platform, chat_id, thread_id),
    ).fetchone()
    return None if row is None else int(row["last_event_id"])


def unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, list[Event]]:
    """Return ``(new_cursor, events)`` with ``id > last_event_id``. The cursor
    is NOT advanced here; call :func:`advance_notify_cursor` after delivery.
    """
    cursor = _notify_cursor(conn, task_id, platform, chat_id, thread_id)
    if cursor is None:
        return 0, []
    kind_list = list(kinds) if kinds else None
    q = (
        "SELECT * FROM task_events WHERE task_id = ? AND id > ? "
        + ("AND kind IN (" + ",".join("?" * len(kind_list)) + ") " if kind_list else "")
        + "ORDER BY id ASC"
    )
    params: list[Any] = [task_id, cursor]
    if kind_list:
        params.extend(kind_list)
    rows = conn.execute(q, params).fetchall()
    out = [_kb.Event.from_row(r) for r in rows]
    max_id = max([cursor, *(int(r["id"]) for r in rows)])
    return max_id, out


def claim_unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, int, list[Event]]:
    """Atomically claim unseen events for one subscription.

    Returns ``(old_cursor, new_cursor, events)``; when events are returned the
    row's ``last_event_id`` has already been advanced inside ``BEGIN IMMEDIATE``,
    so concurrent gateway watchers on the same board DB serialize on SQLite's
    writer lock and only the first claims a given event range. Callers send the
    events, then leave the cursor or call :func:`rewind_notify_cursor` on
    delivery failure.
    """
    with _kb.write_txn(conn):
        old_cursor = _notify_cursor(conn, task_id, platform, chat_id, thread_id)
        if old_cursor is None:
            return 0, 0, []
        new_cursor, events = unseen_events_for_sub(
            conn, task_id=task_id, platform=platform, chat_id=chat_id,
            thread_id=thread_id, kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        # Persist obligations before advancing the subscription cursor. A crash
        # after this transaction can lose neither the event nor its exact origin.
        now = int(time.time())
        sub = {
            "task_id": task_id, "platform": platform, "chat_id": chat_id,
            "thread_id": thread_id or "",
        }
        stored = conn.execute(
            "SELECT notifier_profile,delivery_mode FROM kanban_notify_subs " + _SUB_KEY_WHERE,
            _sub_key(task_id, platform, chat_id, thread_id),
        ).fetchone()
        if stored:
            sub.update(dict(stored))
        for event in events:
            key, digest, encoded = _delivery_payload(event, sub)
            conn.execute(
                """INSERT OR IGNORE INTO kanban_delivery_outbox
                   (delivery_key,task_id,event_id,platform,chat_id,thread_id,notifier_profile,
                    payload_digest,payload_json,state,created_at,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,'pending',?,?)""",
                (key, task_id, int(event.id), platform, chat_id, thread_id or "",
                 sub.get("notifier_profile"), digest, encoded, now, now),
            )
        _cas_cursor(conn, _sub_key(task_id, platform, chat_id, thread_id), new_cursor, old_cursor)
        return old_cursor, new_cursor, events


def list_due_deliveries_for_sub(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None, now: Optional[int] = None,
) -> list[tuple[dict, Event]]:
    """List due rows in cursor order without leasing rows that are still waiting.

    Callers must CAS-claim each row immediately before its transport attempt.
    """
    stamp = int(time.time()) if now is None else int(now)
    release_expired_delivery_leases(conn, now=stamp)
    rows = conn.execute(
        """SELECT * FROM kanban_delivery_outbox
           WHERE task_id=? AND platform=? AND chat_id=? AND thread_id=?
             AND state IN ('pending','retry_wait') AND next_attempt_at<=?
           ORDER BY event_id,id""",
        (*_sub_key(task_id, platform, chat_id, thread_id), stamp),
    ).fetchall()
    due: list[tuple[dict, Event]] = []
    for row in rows:
        event_row = conn.execute("SELECT * FROM task_events WHERE id=?", (row["event_id"],)).fetchone()
        if event_row is not None:
            due.append((dict(row), _kb.Event.from_row(event_row)))
    return due


def _cas_cursor(conn: sqlite3.Connection, key: tuple, new_cursor: int, expected: int) -> sqlite3.Cursor:
    """Move ``last_event_id`` only if it still equals ``expected``."""
    return conn.execute(
        "UPDATE kanban_notify_subs SET last_event_id = ? " + _SUB_KEY_WHERE + " AND last_event_id = ?",
        (int(new_cursor), *key, int(expected)),
    )


def advance_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    new_cursor: int,
) -> None:
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? " + _SUB_KEY_WHERE,
            (int(new_cursor), *_sub_key(task_id, platform, chat_id, thread_id)),
        )


def record_notify_ping(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None, event_id: int,
) -> None:
    """Checkpoint a sent ping independently of the retryable wake cursor."""
    with _kb.write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_ping_event_id = MAX(last_ping_event_id, ?) "
            + _SUB_KEY_WHERE,
            (int(event_id), *_sub_key(task_id, platform, chat_id, thread_id)),
        )


def rewind_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    claimed_cursor: int,
    old_cursor: int,
) -> bool:
    """Undo a claim when delivery fails. The CAS guard only rewinds if no later
    notifier advanced the row, so retries never clobber newer progress.
    """
    with _kb.write_txn(conn):
        cur = _cas_cursor(conn, _sub_key(task_id, platform, chat_id, thread_id), old_cursor, claimed_cursor)
    return cur.rowcount > 0


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb  # noqa: E402
