"""Notification-subscription helpers extracted from ``hermes_cli/kanban_db.py``.

Owned by the gateway kanban-notifier (``gateway/kanban_watchers.py`` and the
notifier wake path): per-task notification subscriptions, unseen-event claims
with cursor CAS, and notifier-profile ownership filters.

Extracted verbatim from ``hermes_cli/kanban_db.py`` (godfile decomposition,
wave 1, shard s5, cluster c9, agreement move=44). The functions are
re-exported from ``kanban_db`` so the ``kanban_db.<name>`` public API surface
is byte-for-byte unchanged. The deferred import at the bottom of this module
resolves the shared helpers it binds from ``kanban_db`` after that module has
finished loading, which keeps both import directions cycle-free.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Notification subscriptions (used by the gateway kanban-notifier)
# ---------------------------------------------------------------------------

def _encode_notify_delivery_metadata(
    metadata: Optional[Mapping[str, Any]],
) -> Optional[str]:
    """Serialize platform send metadata stored on notification subscriptions."""
    if not isinstance(metadata, Mapping):
        return None
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[str(key)] = value
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
    return {
        str(key): value
        for key, value in data.items()
        if isinstance(value, (str, int, float, bool))
    }


def add_notify_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    chat_type: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    notifier_profile: Optional[str] = None,
    delivery_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Register a gateway source that wants terminal-state notifications
    for ``task_id``. Idempotent on (task, platform, chat, thread).

    New subscriptions start "caught up": ``last_event_id`` snaps to the
    task's current ``MAX(task_events.id)`` at creation instead of the
    schema default 0. A cursor of 0 on an already-active task made the
    gateway notifier replay every historical terminal event on its next
    tick — and with many stale subs, a single boot-time burst of 100+
    messages (issue #29905). Subscribers only want events that occur
    AFTER they subscribe; the gateway/tool auto-subscribe paths run at
    task creation, where the snapshot is 0 anyway.
    """
    now = int(time.time())
    metadata_json = _encode_notify_delivery_metadata(delivery_metadata)
    with write_txn(conn):
        conn.execute(
            """
            INSERT OR IGNORE INTO kanban_notify_subs
                (task_id, platform, chat_id, chat_type, thread_id, user_id,
                 notifier_profile, delivery_metadata, created_at, last_event_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT MAX(id) FROM task_events WHERE task_id = ?), 0))
            """,
            (
                task_id,
                platform,
                chat_id,
                chat_type,
                thread_id or "",
                user_id,
                notifier_profile,
                metadata_json,
                now,
                task_id,
            ),
        )
        if chat_type:
            # Self-heal rows created before chat_type was persisted.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET chat_type = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                   AND (chat_type IS NULL OR chat_type = '')
                """,
                (chat_type, task_id, platform, chat_id, thread_id or ""),
            )
        if notifier_profile:
            # Self-heal legacy rows that predate notifier ownership by
            # backfilling only when the existing value is unset.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET notifier_profile = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                   AND (notifier_profile IS NULL OR notifier_profile = '')
                """,
                (notifier_profile, task_id, platform, chat_id, thread_id or ""),
            )
        if metadata_json:
            # A duplicate subscribe from the same chat/thread should refresh
            # the routing anchor. Telegram DM-topic notifications need the
            # latest reply anchor to stay inside the visible topic lane.
            conn.execute(
                """
                UPDATE kanban_notify_subs
                   SET delivery_metadata = ?
                 WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?
                """,
                (metadata_json, task_id, platform, chat_id, thread_id or ""),
            )


def _notify_profile_filter(
    notifier_profiles: Optional[Iterable[str]],
    *,
    include_unowned: bool,
) -> tuple[str, list[str]]:
    """Build an optional SQL predicate for notification profile ownership."""
    if notifier_profiles is None:
        return "", []

    profiles = sorted(
        {
            str(profile).strip()
            for profile in notifier_profiles
            if str(profile).strip()
        }
    )
    clauses: list[str] = []
    params: list[str] = []
    if profiles:
        clauses.append(
            "notifier_profile IN (" + ",".join("?" for _ in profiles) + ")"
        )
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

    Passing no ``notifier_profiles`` preserves the historical all-subscriptions
    result. Gateway notifier processes pass the profiles whose adapters they
    own so they cannot claim another gateway's events. ``include_unowned`` is
    used by the dispatch owner for legacy rows created before profile stamping.
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
    rows = conn.execute(sql, params).fetchall()
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        if "delivery_metadata" in item:
            item["delivery_metadata"] = _decode_notify_delivery_metadata(
                item.get("delivery_metadata")
            )
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
    """Count ``kanban_notify_subs`` rows via a read-only connection.

    Cheap probe for the gateway notifier's zero-subscription early exit:
    unlike :func:`connect`, this never creates the DB file, never runs
    schema init/migration, and never opens the database writable (no
    write locks, no checkpoints — though a read-only open of a WAL
    database may still create the ``-shm``/``-wal`` sidecars, it cannot
    write table content). Rows in a not-yet-checkpointed WAL are
    visible, so a freshly added subscription is never missed. A missing
    DB, or a legacy DB that predates the subscriptions table, counts as
    zero. When ``notifier_profiles`` is supplied, only subscriptions owned
    by those profiles are counted; ``include_unowned`` also includes legacy
    rows without an owner stamp. Optional platform/chat/thread filters narrow
    the probe to one notification owner without changing the unfiltered count.
    Platform matching is case-insensitive, matching notifier routing; chat and
    thread identifiers are exact. Path resolution matches :func:`connect`
    (explicit ``db_path``, else ``board`` via :func:`kanban_db_path`). Raises
    :class:`sqlite3.Error` when the DB exists but cannot be read
    (locked, corrupt); callers choose their own fallback.
    """
    path = db_path if db_path is not None else kanban_db_path(board=board)
    if not path.exists():
        return 0
    conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        try:
            owner_where, owner_params = _notify_profile_filter(
                notifier_profiles, include_unowned=include_unowned,
            )
            clauses: list[str] = []
            params: list[Any] = []
            if owner_where:
                clauses.append(f"({owner_where})")
                params.extend(owner_params)
            if platform is not None:
                clauses.append("LOWER(platform) = LOWER(?)")
                params.append(platform)
            if chat_id is not None:
                clauses.append("chat_id = ?")
                params.append(chat_id)
            if thread_id is not None:
                clauses.append("thread_id = ?")
                params.append(thread_id)
            query = "SELECT COUNT(*) FROM kanban_notify_subs"
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
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
    with write_txn(conn):
        cur = conn.execute(
            "DELETE FROM kanban_notify_subs WHERE task_id = ? "
            "AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        )
    return cur.rowcount > 0


def unseen_events_for_sub(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    kinds: Optional[Iterable[str]] = None,
) -> tuple[int, list[Event]]:
    """Return ``(new_cursor, events)`` for a given subscription.

    Only events with ``id > last_event_id`` are returned. The subscription's
    cursor is NOT advanced here; call :func:`advance_notify_cursor` after
    the gateway has successfully delivered the notifications.
    """
    row = conn.execute(
        "SELECT last_event_id FROM kanban_notify_subs "
        "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
        (task_id, platform, chat_id, thread_id or ""),
    ).fetchone()
    if row is None:
        return 0, []
    cursor = int(row["last_event_id"])
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
    out: list[Event] = []
    max_id = cursor
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if r["payload"] else None
        except Exception:
            payload = None
        out.append(Event(
            id=r["id"], task_id=r["task_id"], kind=r["kind"],
            payload=payload, created_at=r["created_at"],
            run_id=(int(r["run_id"]) if "run_id" in r.keys() and r["run_id"] is not None else None),
        ))
        max_id = max(max_id, int(r["id"]))
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
    """Atomically claim unseen notification events for one subscription.

    Returns ``(old_cursor, new_cursor, events)``. When events are returned,
    ``kanban_notify_subs.last_event_id`` has already been advanced to
    ``new_cursor`` inside a ``BEGIN IMMEDIATE`` transaction. That makes the
    notifier's read/claim step single-owner across multiple gateway watcher
    processes pointed at the same board DB: concurrent watchers serialize on
    SQLite's writer lock, and only the first process sees and claims a given
    event range.

    Callers should send the claimed events, then either leave the cursor at
    ``new_cursor`` on success or call :func:`rewind_notify_cursor` if delivery
    failed before any terminal unsubscribe removed the row.
    """
    with write_txn(conn):
        row = conn.execute(
            "SELECT last_event_id FROM kanban_notify_subs "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (task_id, platform, chat_id, thread_id or ""),
        ).fetchone()
        if row is None:
            return 0, 0, []
        old_cursor = int(row["last_event_id"])
        new_cursor, events = unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform=platform,
            chat_id=chat_id,
            thread_id=thread_id,
            kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or "", int(old_cursor)),
        )
        return old_cursor, new_cursor, events


def advance_notify_cursor(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str] = None,
    new_cursor: int,
) -> None:
    with write_txn(conn):
        conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ?",
            (int(new_cursor), task_id, platform, chat_id, thread_id or ""),
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
    """Undo a notification claim when delivery fails.

    The CAS guard only rewinds if no later notifier advanced the row after our
    claim. This keeps retry behavior for transient send failures without
    clobbering newer progress.
    """
    with write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_subs SET last_event_id = ? "
            "WHERE task_id = ? AND platform = ? AND chat_id = ? AND thread_id = ? "
            "AND last_event_id = ?",
            (
                int(old_cursor), task_id, platform, chat_id, thread_id or "",
                int(claimed_cursor),
            ),
        )
    return cur.rowcount > 0

# Deferred import: kanban_db re-imports this module at the bottom of the file
# (after its own definitions); resolving the shared helpers here avoids the
# circular-import deadlock that a top-level ``from hermes_cli import kanban_db``
# would create in both import directions.
from hermes_cli import kanban_db as _kanban_db  # noqa: E402

Event = _kanban_db.Event
kanban_db_path = _kanban_db.kanban_db_path
write_txn = _kanban_db.write_txn
