"""Notification subscriptions consumed by the gateway kanban-notifier: per-(task, platform, chat, thread) rows with delivery metadata, unseen-event cursors and purge of stale done-task subs.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import json
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

# Delivery-failure handling for a notify sub: 'default' = stock (rewind/drop on failure);
# 'durable' = GOV-F25 Option B never-drop (the single pending_event_id batch fence, see the
# kanban_notify_subs schema in kanban_db.py).
_NOTIFY_RETRY_POLICIES = ("default", "durable")
# The only transport whose delivery can confirm persistence — its wake self-post carries the
# X-Hermes-Turn-Persisted ack (GOV-F25 PR-B) — so the only platform a 'durable' retry_policy is
# deliverable on. A string, not gateway.config.Platform.API_SERVER: the adapter is not in scope
# here and hermes_cli must not import the gateway layer.
_DURABLE_RETRY_PLATFORM = "api_server"
# Only a woken turn produces the persist-ack a durable fence waits for, so a durable sub MUST be
# wake-capable; a plain 'notify' sub could never confirm and would strand the fence forever.
_WAKE_CAPABLE_DELIVERY_MODES = ("notify+wake", "wake")

_SCALAR_TYPES = (str, int, float, bool)

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
    retry_policy: Optional[str] = None,
) -> None:
    """Register a gateway source wanting terminal-state notifications for
    ``task_id``; idempotent on (task, platform, chat, thread).

    ``user_id_alt`` (Signal UUID, Feishu union_id, ...) and ``chat_type`` are
    replayed on active wake: ``build_session_key`` prefers the alt id, so
    omitting it would key the wake into a different session. ``None`` keeps an
    existing row's value. ``delivery_mode``: ``None`` leaves an existing row
    untouched, an explicit valid value is last-write-wins, unknown falls back
    to ``"notify"``. ``delivery_metadata`` merges supplied routing anchors
    into an existing row so re-subscribing never discards them. New subs start
    caught up (``last_event_id`` =
    ``MAX(task_events.id)``) so the notifier never replays history at boot.
    """
    valid_mode = delivery_mode if delivery_mode in _NOTIFY_DELIVERY_MODES else None
    valid_retry_policy = retry_policy if retry_policy in _NOTIFY_RETRY_POLICIES else None
    if retry_policy is not None and valid_retry_policy is None:
        raise ValueError(
            f"unknown retry_policy {retry_policy!r} (expected one of {_NOTIFY_RETRY_POLICIES})")
    # api_server is stateless: the adapter has no send(), the wake self-post IS
    # the delivery. A plain 'notify' default would leave those subs with no
    # delivery mechanism at all. Explicit modes still win.
    insert_mode = valid_mode or ("notify+wake" if platform == "api_server" else "notify")
    key = _sub_key(task_id, platform, chat_id, thread_id)
    with _kb.write_txn(conn):
        existing = conn.execute(
            "SELECT delivery_metadata, delivery_mode, retry_policy, pending_event_id "
            "FROM kanban_notify_subs " + _SUB_KEY_WHERE,
            key,
        ).fetchone()
        existing_metadata = _decode_notify_delivery_metadata(existing["delivery_metadata"]) if existing else {}
        # (M4) A 'durable' sub must always have a confirmable, wake-capable delivery — otherwise its
        # persist-ack never arrives and the pending_event_id fence never clears (a silent
        # never-deliver). Compute the retry_policy / delivery_mode that WILL apply after this call
        # (last-write-wins: an explicit value wins, an omitted one keeps the existing row's) and
        # re-check on EVERY subscribe / policy update, so an omitted-policy re-subscribe can never
        # silently downgrade a durable sub into a non-wake / non-api_server state. Raise BEFORE any
        # write so a rejected update leaves the existing row untouched.
        effective_retry_policy = valid_retry_policy or (existing["retry_policy"] if existing else "default")
        if effective_retry_policy == "durable":
            effective_mode = valid_mode or (existing["delivery_mode"] if existing else insert_mode)
            if platform != _DURABLE_RETRY_PLATFORM:
                raise ValueError(
                    f"retry_policy='durable' requires the {_DURABLE_RETRY_PLATFORM!r} platform "
                    f"(only it can confirm persistence); got platform={platform!r}")
            if effective_mode not in _WAKE_CAPABLE_DELIVERY_MODES:
                raise ValueError(
                    f"retry_policy='durable' requires a wake-capable delivery_mode "
                    f"(one of {_WAKE_CAPABLE_DELIVERY_MODES}); got delivery_mode={effective_mode!r}")
        elif existing is not None and existing["retry_policy"] == "durable" \
                and existing["pending_event_id"] is not None:
            # Refuse to downgrade a durable sub OUT of 'durable' while it still holds an un-acked
            # pending_event_id fence: the next (non-durable) claim would ignore the fence and a crash
            # could lose the in-flight batch. Drain/settle the fenced batch before downgrading.
            raise ValueError(
                "cannot downgrade retry_policy from 'durable' while an un-acked pending_event_id "
                f"fence is set (task {task_id!r}, chat {chat_id!r}); drain the in-flight batch first")
        merged_metadata = dict(existing_metadata)
        if delivery_metadata:
            merged_metadata.update(delivery_metadata)
        metadata_json = _encode_notify_delivery_metadata(merged_metadata) if merged_metadata else None
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
        # chat_type / delivery_mode are last-write-wins; delivery metadata
        # preserves existing routing fields while supplied fields overwrite them.
        # user_id, user_id_alt and notifier_profile only self-heal legacy rows lacking one.
        for column, value, fill_only in (
            ("chat_type", chat_type, False),
            ("user_id", user_id, True),
            ("user_id_alt", user_id_alt, True),
            ("notifier_profile", notifier_profile, True),
            ("delivery_mode", valid_mode, False),
            ("delivery_metadata", metadata_json, False),
            ("retry_policy", valid_retry_policy, False),
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
            # GOV-F25 (M4): NEVER GC a durable sub — an aged durable sub may still hold an un-acked
            # pending_event_id fence, and reaping it would drop that event (defeating never-drop).
            # Only 'default' subs are stale-swept.
            "DELETE FROM kanban_notify_subs WHERE retry_policy = 'default' AND task_id IN ("
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

    GOV-F25 Option B — a ``retry_policy='durable'`` sub carries a single
    ``pending_event_id`` fence. On a fresh durable claim the fence is set to the
    FIRST claimed event id in the SAME transaction that advances the cursor (one
    column, no peek); while the fence is set a fresh claim returns NO newer events
    — a newer claim can never bypass an un-acked event, and recovery replays the
    fenced range via :func:`pending_events_for_sub`. The fence is CAS'd forward per
    event by :func:`settle_notify_pending` after each persist-ack (NULL after the
    final one). A non-durable ('default') sub is unchanged — it has no fence.
    """
    key = _sub_key(task_id, platform, chat_id, thread_id)
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT last_event_id, retry_policy, pending_event_id "
            "FROM kanban_notify_subs " + _SUB_KEY_WHERE,
            key,
        ).fetchone()
        if row is None:
            return 0, 0, []
        old_cursor = int(row["last_event_id"])
        durable = row["retry_policy"] == "durable"
        if durable and row["pending_event_id"] is not None:
            # Fence held: an un-acked batch is still in flight. A fresh claim must NOT bypass it
            # (recovery replays [pending_event_id, last_event_id]); return empty.
            return old_cursor, old_cursor, []
        new_cursor, events = unseen_events_for_sub(
            conn, task_id=task_id, platform=platform, chat_id=chat_id,
            thread_id=thread_id, kinds=kinds,
        )
        if not events:
            return old_cursor, old_cursor, []
        _cas_cursor(conn, key, new_cursor, old_cursor)
        if durable:
            # Set the fence to the oldest (first) claimed event, atomically with the cursor
            # advance. CAS-guarded on NULL so a racing claim can never overwrite a live fence.
            conn.execute(
                "UPDATE kanban_notify_subs SET pending_event_id = ? " + _SUB_KEY_WHERE
                + " AND pending_event_id IS NULL",
                (int(events[0].id), *key),
            )
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


# --- GOV-F25 Option B durable fence (pending_event_id) accessors ---


def notify_retry_policy(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None,
) -> Optional[str]:
    """The sub's ``retry_policy`` ('default' | 'durable'), or ``None`` when unsubscribed."""
    row = conn.execute(
        "SELECT retry_policy FROM kanban_notify_subs " + _SUB_KEY_WHERE,
        _sub_key(task_id, platform, chat_id, thread_id),
    ).fetchone()
    return None if row is None else str(row["retry_policy"])


def notify_pending_event_id(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None,
) -> Optional[int]:
    """The durable fence — the oldest un-acked event id in flight — or ``None`` when the sub is
    unsubscribed or the fence is clear (nothing in flight)."""
    row = conn.execute(
        "SELECT pending_event_id FROM kanban_notify_subs " + _SUB_KEY_WHERE,
        _sub_key(task_id, platform, chat_id, thread_id),
    ).fetchone()
    if row is None or row["pending_event_id"] is None:
        return None
    return int(row["pending_event_id"])


def pending_events_for_sub(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None, kinds: Optional[Iterable[str]] = None,
) -> list[Event]:
    """The un-acked events of a durable sub's in-flight batch: those with
    ``pending_event_id <= id <= last_event_id`` — the claimed-but-not-yet-settled range the fence
    guards. Empty when the fence is clear. Recovery / redelivery reads ONLY this range (never beyond
    ``last_event_id``), so a restart replays exactly the un-acked tail and never the already-acked
    head (GOV-F25 Option B, AC-10).
    """
    row = conn.execute(
        "SELECT last_event_id, pending_event_id FROM kanban_notify_subs " + _SUB_KEY_WHERE,
        _sub_key(task_id, platform, chat_id, thread_id),
    ).fetchone()
    if row is None or row["pending_event_id"] is None:
        return []
    pending, last = int(row["pending_event_id"]), int(row["last_event_id"])
    kind_list = list(kinds) if kinds else None
    q = (
        "SELECT * FROM task_events WHERE task_id = ? AND id >= ? AND id <= ? "
        + ("AND kind IN (" + ",".join("?" * len(kind_list)) + ") " if kind_list else "")
        + "ORDER BY id ASC"
    )
    params: list[Any] = [task_id, pending, last]
    if kind_list:
        params.extend(kind_list)
    return [_kb.Event.from_row(r) for r in conn.execute(q, params).fetchall()]


def settle_notify_pending(
    conn: sqlite3.Connection, *, task_id: str, platform: str, chat_id: str,
    thread_id: Optional[str] = None, settled_event_id: int, next_pending_id: Optional[int],
) -> bool:
    """CAS the durable fence forward after ONE event settles — a TRUE persist-ack, OR a non-waking
    kind (archived/unblocked) that can never produce an ack and so is treated as immediately
    settled. ``next_pending_id`` is the next un-acked event id, or ``None`` after the final event
    (fence cleared -> the sub may claim again). CAS-guarded on ``settled_event_id`` so a concurrent
    settle cannot double-advance. Returns True when it moved. On a durable delivery FAILURE the
    caller does NOT call this — the fence is RETAINED and recovery replays the range.
    """
    with _kb.write_txn(conn):
        cur = conn.execute(
            "UPDATE kanban_notify_subs SET pending_event_id = ? " + _SUB_KEY_WHERE
            + " AND pending_event_id = ?",
            (None if next_pending_id is None else int(next_pending_id),
             *_sub_key(task_id, platform, chat_id, thread_id), int(settled_event_id)),
        )
    return cur.rowcount > 0


# Late-bound origin namespace (see module docstring); imported LAST so this
# module is fully populated before ``kanban_db`` imports from it.
from hermes_cli import kanban_db as _kb
