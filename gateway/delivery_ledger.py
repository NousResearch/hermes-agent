"""Durable delivery-obligation ledger for gateway final responses.

A final agent response that was generated but not yet confirmed-delivered
to the messaging platform is the one artifact the gateway can lose without
a trace: the turn already burned its tokens, the text exists only in a
Python local, and a crash / planned restart between finalize and platform
ACK drops it silently (#58818, #41696, #63695).

This module records a small durable row per outbound final response in the
shared ``state.db`` (same file and conventions as
``tools.async_delegation`` — WAL, owner pid + process-start-time liveness,
bounded retention). The gateway writes three checkpoints around the send:

    record_obligation()   state='pending'     before any send attempt
    mark_attempting()     state='attempting'  immediately before the await
    mark_delivered() /    state='delivered'   only on SendResult.success
    mark_failed()         state='failed'      on a definitive rejection

On startup, ``sweep_recoverable()`` claims rows whose owning process is
dead and hands them to the gateway for redelivery. Crash semantics are
explicit about ambiguity (the contract review of the earlier
delivery-outbox attempt, #61790, closed it for silently resending
ambiguous sends):

- ``pending``     — the send never started: redeliver plainly, no dup risk.
- ``attempting``  — crashed mid-await: the platform MAY already have the
  message. Redelivered WITH a visible recovered-reply marker so the
  contract is honest at-least-once, never a silent duplicate.
- ``failed``      — definitively rejected once; the restart is a natural
  retry boundary. Also carries the marker.
- ``delivered``   — nothing to do; retention prunes.

Poison rows cannot spin: attempts are capped, stale rows expire, and both
transition to ``abandoned`` (kept briefly for inspection, then pruned).

Everything here is best-effort by design: ledger failures must never block
or delay an actual send. Callers wrap every call in try/except.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional

from hermes_constants import get_hermes_home
from hermes_state_common import state_db_begin_immediate

logger = logging.getLogger(__name__)

_DB_LOCK = threading.Lock()

# Redelivery policy knobs (module constants; deliberately not config — the
# ledger itself is gated by ``gateway.delivery_ledger`` and these bounds
# only matter in the rare recovery path).
MAX_ATTEMPTS = 3
STALE_AFTER_SECONDS = 24 * 60 * 60
_RETENTION_SECONDS = 7 * 24 * 60 * 60
_MAX_ROWS = 500

# Visible prefix for redeliveries that might duplicate an already-received
# message (crash mid-send / post-rejection retry). Honest at-least-once.
RECOVERED_MARKER = (
    "♻️ Recovered reply — the gateway restarted during delivery, "
    "so this may be a duplicate:\n\n"
)


def _db_path():
    return get_hermes_home() / "state.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``isolation_level=None`` opts out of sqlite3's implicit transaction
    # wrapper so ``_transaction`` can issue ``BEGIN IMMEDIATE`` explicitly
    # (the WAL write lock is acquired at transaction start, not lazily on
    # the first INSERT/UPDATE).  ``timeout=10`` keeps sqlite3's built-in
    # busy handler engaged for mid-body contention; the BEGIN itself rides
    # out longer holds via the shared ``state_db_begin_immediate`` primitive.
    conn = sqlite3.connect(path, timeout=10, isolation_level=None)
    try:
        _initialize_schema(conn)
    except Exception:
        # A PRAGMA/DDL failure after a successful connect() must not leak the
        # just-opened connection back to the caller.
        conn.close()
        raise
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    from hermes_state import apply_wal_with_fallback

    apply_wal_with_fallback(conn, db_label="state.db (delivery_ledger)")
    conn.execute(
        """CREATE TABLE IF NOT EXISTS delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT
        )"""
    )


@contextmanager
def _transaction() -> Iterator[Any]:
    """Open a short-lived connection and run the body under a
    ``BEGIN IMMEDIATE`` write transaction with bounded jitter retry.

    ``sqlite3.Connection.__enter__``/``__exit__`` only commits or rolls
    back but never closes, which historically leaked ``db/-wal/-shm``
    file descriptors (see #69567, #69594).  This helper guarantees the
    deterministic close (in ``finally``) AND adds the reliability
    discipline via the shared ``hermes_state_common.state_db_begin_immediate``
    context manager:

    * ``BEGIN IMMEDIATE`` is issued before the body runs and retries with
      the same fast-then-slow jitter schedule ``SessionDB._execute_write``
      uses.
    * Mid-body contention rides on SQLite's own ``timeout=10`` busy handler.
    * On body success the transaction commits; on body exception it
      rolls back and the exception propagates.
    * The connection is ALWAYS closed in ``finally`` so the FD leaks the
      original ``with conn:`` had cannot return.
    """
    conn = _connect()
    try:
        with state_db_begin_immediate(conn):
            yield conn
    finally:
        conn.close()


def _owner_stamp() -> tuple[int, Optional[int]]:
    pid = os.getpid()
    try:
        from gateway.status import get_process_start_time

        return pid, get_process_start_time(pid)
    except Exception:
        return pid, None


def _owner_alive(pid: Any, started_at: Any) -> bool:
    """True when the recorded owning process still exists (pid + start time)."""
    if not pid:
        return False
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        from gateway.status import get_process_start_time

        current_start = get_process_start_time(pid)
    except Exception:
        current_start = None
    if current_start is None:
        # No such process (or unreadable) — treat unreadable-but-extant
        # processes as alive only if the pid exists. Route through the
        # cross-platform probe: ``os.kill(pid, 0)`` on Windows is NOT a
        # no-op (bpo-14484 — CPython maps sig=0 to
        # ``GenerateConsoleCtrlEvent(0, pid)``), so a raw probe here could
        # Ctrl+C the gateway's own console group whenever psutil failed to
        # read the start time of a live pid. ``_pid_exists`` keeps the
        # EPERM-means-alive semantics (exists but owned by another user).
        try:
            from gateway.status import _pid_exists
        except Exception:
            if os.name == "nt":
                # Never fall back to a raw sig-0 probe on Windows.
                return False
            try:
                os.kill(pid, 0)  # windows-footgun: ok — POSIX-only fallback branch
            except ProcessLookupError:
                return False
            except PermissionError:
                return True
            except OSError:
                return False
            return True
        try:
            return bool(_pid_exists(pid))
        except Exception:
            return False
    if started_at is None:
        return True
    try:
        return int(current_start) == int(started_at)
    except (TypeError, ValueError):
        return True


def compute_obligation_id(session_key: str, message_ref: str, content: str) -> str:
    """Stable id: same turn + same content re-records idempotently, while
    distinct threads/topics on the same chat can never collide (the
    session_key carries platform, chat and thread; ``message_ref`` is the
    triggering inbound message id, distinguishing turns in one session)."""
    payload = f"{session_key}|{message_ref}|{content}"
    return hashlib.sha256(payload.encode("utf-8", "replace")).hexdigest()[:24]


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
) -> None:
    """Record a final response as owed to the platform (state='pending')."""
    now = time.time()
    pid, started = _owner_stamp()
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?)""",
            (obligation_id, session_key, platform, str(chat_id),
             str(thread_id) if thread_id else None, content, now, now,
             pid, started),
        )
    _prune()


def mark_attempting(obligation_id: str) -> None:
    _update_state(obligation_id, "attempting")


def mark_delivered(obligation_id: str) -> None:
    _update_state(obligation_id, "delivered")


def mark_failed(obligation_id: str, error: str = "") -> None:
    _update_state(obligation_id, "failed", error=error)


def _update_state(obligation_id: str, state: str, error: str = "") -> None:
    with _DB_LOCK, _transaction() as conn:
        conn.execute(
            """UPDATE delivery_obligations
               SET state=?, updated_at=?, last_error=?
               WHERE obligation_id=?""",
            (state, time.time(), error[:500] if error else None, obligation_id),
        )


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Claim undelivered rows owned by dead processes; return them for
    redelivery.

    Claiming atomically re-stamps the owner to THIS process and increments
    ``attempts``, so a second gateway racing the same sweep cannot
    double-claim (the UPDATE is guarded on the previous owner stamp).
    Rows over the attempts cap or older than the stale cutoff transition to
    'abandoned' instead of being returned.

    ``deliverable_platforms`` (platform value strings) restricts claiming to
    platforms the caller can actually send on this boot.  ``attempts`` is the
    redelivery budget, so it must only be spent on a real send: a platform
    that failed to connect would otherwise burn one attempt per boot and hit
    the cap having never been sent once.  Rows for absent platforms are left
    untouched for a later boot; the stale cutoff still bounds them.
    """
    now = now if now is not None else time.time()
    pid, started = _owner_stamp()
    claimed: List[Dict[str, Any]] = []
    with _DB_LOCK, _transaction() as conn:
        claimed = _sweep_recoverable_body(
            conn, now, pid, started, deliverable_platforms,
        )
    return claimed


def _sweep_recoverable_body(
    conn: sqlite3.Connection,
    now: float,
    pid: int,
    started: Optional[int],
    deliverable_platforms: Optional[set],
) -> List[Dict[str, Any]]:
    claimed: List[Dict[str, Any]] = []
    rows = conn.execute(
        """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                  content, state, attempts, created_at,
                  owner_pid, owner_started_at
           FROM delivery_obligations
           WHERE state IN ('pending', 'attempting', 'failed')"""
    ).fetchall()
    for (oid, session_key, platform, chat_id, thread_id, content, state,
         attempts, created_at, owner_pid, owner_started_at) in rows:
        if _owner_alive(owner_pid, owner_started_at):
            continue  # a live gateway still owns this row
        if attempts >= MAX_ATTEMPTS or (now - created_at) > STALE_AFTER_SECONDS:
            conn.execute(
                """UPDATE delivery_obligations
                   SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                (now, oid),
            )
            continue
        if (
            deliverable_platforms is not None
            and platform not in deliverable_platforms
        ):
            # No adapter for this platform this boot — the caller cannot
            # send, so claiming would spend an attempt on a no-op.
            continue
        cursor = conn.execute(
            """UPDATE delivery_obligations
               SET owner_pid=?, owner_started_at=?, attempts=attempts+1,
                   updated_at=?
               WHERE obligation_id=? AND (owner_pid IS ? OR owner_pid=?)""",
            (pid, started, now, oid, owner_pid, owner_pid),
        )
        if cursor.rowcount:
            claimed.append({
                "obligation_id": oid,
                "session_key": session_key,
                "platform": platform,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "content": content,
                # pending = send never started, redeliver plainly;
                # attempting/failed = ambiguous or rejected, carry marker.
                "needs_marker": state != "pending",
                "attempts": attempts + 1,
            })
    return claimed


def _prune(now: Optional[float] = None) -> None:
    now = now if now is not None else time.time()
    cutoff = now - _RETENTION_SECONDS
    try:
        with _transaction() as conn:
            _prune_body(conn, cutoff)
    except Exception:
        logger.debug("delivery ledger prune failed", exc_info=True)


def _prune_body(conn: sqlite3.Connection, cutoff: float) -> None:
    conn.execute(
        """DELETE FROM delivery_obligations
           WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
        (cutoff,),
    )
    total = conn.execute(
        "SELECT COUNT(*) FROM delivery_obligations"
    ).fetchone()[0]
    excess = max(0, total - _MAX_ROWS)
    if excess:
        conn.execute(
            """DELETE FROM delivery_obligations WHERE obligation_id IN (
                 SELECT obligation_id FROM delivery_obligations
                 ORDER BY CASE state
                            WHEN 'delivered' THEN 0
                            WHEN 'abandoned' THEN 1
                            ELSE 2
                          END, updated_at ASC
                 LIMIT ?)""",
            (excess,),
        )


def ledger_enabled(config: Optional[Dict[str, Any]] = None) -> bool:
    """Read the ``gateway.delivery_ledger`` config gate (default on)."""
    try:
        if config is None:
            from hermes_cli.config import load_config

            config = load_config()
        gw = config.get("gateway") or {}
        value = gw.get("delivery_ledger", True)
        if isinstance(value, str):
            return value.strip().lower() not in {"false", "0", "no", "off"}
        return bool(value)
    except Exception:
        return True


def debug_rows(limit: int = 20) -> str:
    """Human-readable dump for ad-hoc inspection (sqlite3-free path)."""
    with _DB_LOCK, _transaction() as conn:
        rows = conn.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return json.dumps(
        [
            {
                "id": r[0], "session": r[1], "state": r[2], "attempts": r[3],
                "created_at": r[4], "updated_at": r[5], "last_error": r[6],
            }
            for r in rows
        ],
        indent=2,
    )
