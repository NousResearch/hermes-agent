"""Notification subscriptions consumed by the gateway kanban-notifier: per-(task, platform, chat, thread) rows with delivery metadata, unseen-event cursors and purge of stale done-task subs.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
import secrets
import hashlib
import os
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
_NOTIFY_DELIVERY_PROCESS_TOKEN = secrets.token_urlsafe(8)

# Subscription primary key predicate; every per-row statement below binds
# ``(task_id, platform, chat_id, thread_id or "")`` against it.
_SUB_KEY_WHERE = "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?"


def _sub_key(task_id: str, platform: str, chat_id: str, thread_id: Optional[str]) -> tuple:
    return (task_id, platform, chat_id, thread_id or "")


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
        conn.execute("DELETE FROM kanban_notify_deliveries " + _SUB_KEY_WHERE, _sub_key(task_id, platform, chat_id, thread_id))
    return cur.rowcount > 0


def stage_unseen_notify_deliveries_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> list[_kb.Event]:
    """Durably stage unseen events without treating staging as delivery.

    The legacy claim API advances ``last_event_id`` before transport and has to
    rewind a whole batch on failure.  Gateway push delivery uses this ledger
    instead: each event/destination gets an idempotent pending row, while the
    cursor remains at the last fully acknowledged contiguous delivery.  A
    process that dies after this transaction but before ``adapter.send`` leaves
    replayable work, not a falsely acknowledged notification.
    """
    thread = thread_id or ""
    kind_list = list(kinds) if kinds else None
    now = int(time.time())
    staged: list[_kb.Event] = []
    with _kb.write_txn(conn):
        sub = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread),
        ).fetchone()
        if sub is None:
            return []
        cursor = int(sub["last_event_id"])
        query = (
            "SELECT * FROM task_events WHERE task_id = ? AND id > ? "
            + (
                "AND kind IN (" + ",".join("?" for _ in kind_list) + ") "
                if kind_list else ""
            )
            + "ORDER BY id ASC"
        )
        params: list[Any] = [task_id, cursor]
        if kind_list:
            params.extend(kind_list)
        for row in conn.execute(query, params).fetchall():
            event_id = int(row["id"])
            inserted = conn.execute(
                "INSERT OR IGNORE INTO kanban_notify_deliveries "
                "(task_id, platform, chat_id, thread_id, event_id, "
                " delivery_token, delivery_identity, state, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'content_marker_v1', 'pending', ?, ?)",
                (
                    task_id, platform, chat_id, thread, event_id,
                    secrets.token_urlsafe(12), now, now,
                ),
            )
            if inserted.rowcount:
                try:
                    payload = json.loads(row["payload"]) if row["payload"] else None
                except Exception:
                    payload = None
                staged.append(_kb.Event(
                    id=event_id,
                    task_id=row["task_id"],
                    kind=row["kind"],
                    payload=payload,
                    created_at=int(row["created_at"]),
                    run_id=(
                        int(row["run_id"])
                        if "run_id" in row.keys() and row["run_id"] is not None
                        else None
                    ),
                ))
    return staged


def list_notify_deliveries(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    states: Iterable[str] = ("pending", "ambiguous", "parked"),
) -> list[dict]:
    """Return unsettled per-event deliveries with their source events."""
    state_list = list(states)
    if not state_list:
        return []
    rows = conn.execute(
        "SELECT d.*, e.kind, e.payload, e.created_at AS event_created_at, "
        "e.run_id FROM kanban_notify_deliveries d "
        "JOIN task_events e ON e.id = d.event_id AND e.task_id = d.task_id "
        "WHERE d.task_id = ? AND d.platform = ? AND d.chat_id = ? "
        "AND d.thread_id = ? AND d.state IN ("
        + ",".join("?" for _ in state_list)
        + ") ORDER BY d.event_id ASC",
        (task_id, platform, chat_id, thread_id or "", *state_list),
    ).fetchall()
    deliveries: list[dict] = []
    for row in rows:
        item = dict(row)
        try:
            payload = json.loads(row["payload"]) if row["payload"] else None
        except Exception:
            payload = None
        item["event"] = _kb.Event(
            id=int(row["event_id"]),
            task_id=row["task_id"],
            kind=row["kind"],
            payload=payload,
            created_at=int(row["event_created_at"]),
            run_id=(int(row["run_id"]) if row["run_id"] is not None else None),
        )
        deliveries.append(item)
    return deliveries


def begin_notify_delivery(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
) -> Optional[str]:
    """Exclusively claim one pending row for an in-flight transport call.

    ``sending`` is deliberately distinct from ``ambiguous``: another watcher
    must not query remote history while the owning adapter call is still in
    flight. The random claim token fences every later transition so a stale
    sender cannot acknowledge or retry work owned by a successor.
    """
    claim_token = secrets.token_urlsafe(18)
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'sending', "
            "delivery_identity = 'content_marker_v1', "
            "claim_token = ?, claim_owner = ?, "
            "attempt_count = attempt_count + 1, updated_at = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND event_id = ? AND state = 'pending'",
            (
                claim_token, _notify_delivery_owner_id(), int(time.time()), task_id, platform,
                chat_id, thread_id or "", int(event_id),
            ),
        )
    return claim_token if cur.rowcount == 1 else None


def mark_notify_delivery_ambiguous(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
    claim_token: str,
    error: str,
) -> bool:
    """End an in-flight send without asserting acceptance or rejection."""
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'ambiguous', "
            "claim_token = NULL, claim_owner = NULL, last_error = ?, updated_at = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND event_id = ? AND state = 'sending' AND claim_token = ?",
            (
                str(error)[:500], int(time.time()), task_id, platform, chat_id,
                thread_id or "", int(event_id), claim_token,
            ),
        )
    return cur.rowcount == 1


def _notify_delivery_owner_id() -> str:
    """Identify this process without confusing a later PID reuse for it."""
    return f"{_kb._claimer_id()}:{_NOTIFY_DELIVERY_PROCESS_TOKEN}"


def _notify_delivery_owner_alive(owner: Optional[str]) -> Optional[bool]:
    """Return same-host owner liveness, or ``None`` when it cannot be proved."""
    if not owner or ":" not in owner:
        return None
    parts = owner.rsplit(":", 2)
    if len(parts) == 3:
        owner_host, raw_pid, process_token = parts
    else:
        owner_host, raw_pid = parts
        process_token = None
    local_host = _kb._claimer_id().rsplit(":", 1)[0]
    if owner_host != local_host:
        return None
    try:
        pid = int(raw_pid)
    except (TypeError, ValueError):
        return None
    if (
        pid == os.getpid()
        and process_token is not None
        and process_token != _NOTIFY_DELIVERY_PROCESS_TOKEN
    ):
        return False
    return _kb._pid_alive(pid)


def claim_notify_reconciliation(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
) -> Optional[str]:
    """Exclusively claim an ambiguous or provably abandoned delivery.

    A live ``sending``/``reconciling`` owner is never raced. If a same-host
    process died at either boundary, its row is safe to reconcile by its
    persisted transport identity.
    Unknown or cross-host ownership stays held instead of turning an absence
    scan into a concurrent duplicate send.
    """
    thread = thread_id or ""
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT state, claim_owner FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND event_id = ?",
            (task_id, platform, chat_id, thread, int(event_id)),
        ).fetchone()
        if row is None:
            return None
        state = row["state"]
        reclaim_abandoned = (
            state in ("sending", "reconciling")
            and _notify_delivery_owner_alive(row["claim_owner"]) is False
        )
        if state != "ambiguous" and not reclaim_abandoned:
            return None
        claim_token = secrets.token_urlsafe(18)
        cur = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'reconciling', "
            "claim_token = ?, claim_owner = ?, updated_at = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND event_id = ? AND state = ?",
            (
                claim_token, _notify_delivery_owner_id(), int(time.time()), task_id, platform,
                chat_id, thread, int(event_id), state,
            ),
        )
    return claim_token if cur.rowcount == 1 else None


def retry_notify_delivery(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
    claim_token: str,
    error: str,
) -> bool:
    """Record a confirmed rejection and make only that event retryable."""
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'pending', "
            "claim_token = NULL, claim_owner = NULL, last_error = ?, updated_at = ? "
            "WHERE task_id = ? AND platform = ? "
            "AND chat_id = ? AND thread_id = ? AND event_id = ? "
            "AND state IN ('sending', 'reconciling') AND claim_token = ?",
            (
                str(error)[:500], int(time.time()), task_id, platform, chat_id,
                thread_id or "", int(event_id), claim_token,
            ),
        )
    return cur.rowcount == 1


def release_notify_reconciliation(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
    claim_token: str,
    error: str,
) -> bool:
    """Return an abandoned reconciliation claim to ambiguous, never pending.

    Remote reconciliation already established acceptance. If persisting that
    acknowledgement fails, clearing only the matching fenced owner lets the
    next tick verify the durable identity again in the same process without
    ever making the accepted notice sendable.
    """
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'ambiguous', "
            "claim_token = NULL, claim_owner = NULL, last_error = ?, updated_at = ? "
            "WHERE task_id = ? AND platform = ? "
            "AND chat_id = ? AND thread_id = ? AND event_id = ? "
            "AND state = 'reconciling' AND claim_token = ?",
            (
                str(error)[:500], int(time.time()), task_id, platform, chat_id,
                thread_id or "", int(event_id), claim_token,
            ),
        )
    return cur.rowcount == 1


def park_notify_delivery(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
    claim_token: str,
    error: str,
) -> bool:
    """Hold an unreconcilable ambiguous send for operator disposition."""
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'parked', "
            "claim_token = NULL, claim_owner = NULL, last_error = ?, updated_at = ? "
            "WHERE task_id = ? AND platform = ? "
            "AND chat_id = ? AND thread_id = ? AND event_id = ? "
            "AND state = 'reconciling' AND claim_token = ?",
            (
                str(error)[:500], int(time.time()), task_id, platform, chat_id,
                thread_id or "", int(event_id), claim_token,
            ),
        )
    return cur.rowcount == 1


def acknowledge_notify_delivery(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    event_id: int,
    claim_token: str,
    message_id: Optional[str] = None,
) -> bool:
    """Acknowledge one event and advance only through contiguous acknowledgements."""
    thread = thread_id or ""
    with _kb.write_txn(conn):
        delivered = conn.execute(
            "UPDATE kanban_notify_deliveries SET state = 'delivered', message_id = ?, "
            "claim_token = NULL, claim_owner = NULL, last_error = NULL, updated_at = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND event_id = ? AND state IN ('sending', 'reconciling') "
            "AND claim_token = ?",
            (
                str(message_id) if message_id else None, int(time.time()), task_id,
                platform, chat_id, thread, int(event_id), claim_token,
            ),
        )
        if delivered.rowcount != 1:
            return False
        first_unsettled = conn.execute(
            "SELECT MIN(event_id) AS event_id FROM kanban_notify_deliveries "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND state != 'delivered'",
            (task_id, platform, chat_id, thread),
        ).fetchone()
        boundary = first_unsettled["event_id"] if first_unsettled else None
        if boundary is None:
            row = conn.execute(
                "SELECT MAX(event_id) AS event_id FROM kanban_notify_deliveries "
                "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
                "AND state = 'delivered'",
                (task_id, platform, chat_id, thread),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(event_id) AS event_id FROM kanban_notify_deliveries "
                "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
                "AND state = 'delivered' AND event_id < ?",
                (task_id, platform, chat_id, thread, int(boundary)),
            ).fetchone()
        acknowledged_through = row["event_id"] if row else None
        if acknowledged_through is not None:
            conn.execute(
                "UPDATE kanban_notify_subs SET last_event_id = MAX(last_event_id, ?) "
                "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
                (int(acknowledged_through), task_id, platform, chat_id, thread),
            )
            conn.execute(
                "DELETE FROM kanban_notify_deliveries WHERE task_id = ? "
                "AND platform = ? AND chat_id = ? AND thread_id = ? "
                "AND state = 'delivered' AND event_id <= ?",
                (
                    task_id, platform, chat_id, thread,
                    int(acknowledged_through),
                ),
            )
    return True




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
            "DELETE FROM kanban_notify_subs AS s WHERE NOT EXISTS ("
            " SELECT 1 FROM kanban_notify_deliveries d"
            " WHERE d.task_id = s.task_id AND d.platform = s.platform"
            " AND d.chat_id = s.chat_id AND d.thread_id = s.thread_id"
            ") AND task_id IN ("
            " SELECT t.id FROM tasks t"
            " WHERE t.status IN ('done', 'blocked')"
            " AND COALESCE("
            "  (SELECT MAX(e.created_at) FROM task_events e"
            "   WHERE e.task_id = t.id),"
            "  t.completed_at, t.created_at, 0"
            " ) < ?)",
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
        _cas_cursor(conn, _sub_key(task_id, platform, chat_id, thread_id), new_cursor, old_cursor)
        return old_cursor, new_cursor, events


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
