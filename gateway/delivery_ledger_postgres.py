"""PostgreSQL storage for the gateway delivery-obligation ledger."""

from __future__ import annotations

import os
import socket
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, TypeVar

from hermes_state import SessionDB
from hermes_state_registry import acquire, release_or_close


_OWNER_TOKEN = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
_OWNER_LEASE_SECONDS = 5 * 60

T = TypeVar("T")


@contextmanager
def _database() -> Iterator[SessionDB]:
    from gateway.delivery_ledger import _db_path

    database = acquire(_db_path())
    try:
        if database.backend != "postgres":
            raise RuntimeError(
                "PostgreSQL delivery storage requires database.backend: postgres"
            )
        yield database
    finally:
        release_or_close(database)


def _write(operation: Callable[[Any], T]) -> T:
    with _database() as database:
        return database._execute_write(operation)


def _read(operation: Callable[[Any], T]) -> T:
    with _database() as database, database._read_ctx() as connection:
        return operation(connection)


def _lease_deadline(now: float) -> float:
    return now + _OWNER_LEASE_SECONDS


def _prune_rows(connection, now: float) -> None:
    from gateway import delivery_ledger as ledger

    connection.execute(
        """DELETE FROM delivery_obligations
           WHERE state IN ('delivered', 'abandoned') AND updated_at < ?""",
        (now - ledger._RETENTION_SECONDS,),
    )
    total = connection.execute("SELECT COUNT(*) FROM delivery_obligations").fetchone()[
        0
    ]
    if total > ledger._MAX_ROWS:
        connection.execute(
            """DELETE FROM delivery_obligations WHERE obligation_id IN (
                 SELECT obligation_id FROM delivery_obligations
                 ORDER BY CASE state
                            WHEN 'delivered' THEN 0
                            WHEN 'abandoned' THEN 1
                            ELSE 2
                          END, updated_at ASC
                 LIMIT ?)""",
            (total - ledger._MAX_ROWS,),
        )


def record_obligation(
    *,
    obligation_id: str,
    session_key: str,
    platform: str,
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    adapter_profile: Optional[str] = None,
) -> None:
    from gateway.delivery_ledger import _owner_stamp

    now = time.time()
    owner_pid, owner_started_at = _owner_stamp()
    profile = str(adapter_profile).strip() if adapter_profile else "default"

    def store(connection):
        connection.execute(
            """INSERT INTO delivery_obligations
               (obligation_id, session_key, platform, chat_id, thread_id,
                content, state, attempts, created_at, updated_at,
                owner_pid, owner_started_at, adapter_profile, owner_token, lease_expires_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(obligation_id) DO UPDATE SET
                   session_key = excluded.session_key,
                   platform = excluded.platform,
                   chat_id = excluded.chat_id,
                   thread_id = excluded.thread_id,
                   content = excluded.content,
                   state = 'pending',
                   attempts = 0,
                   created_at = excluded.created_at,
                   updated_at = excluded.updated_at,
                   owner_pid = excluded.owner_pid,
                   owner_started_at = excluded.owner_started_at,
                   last_error = NULL,
                   adapter_profile = excluded.adapter_profile,
                   owner_token = excluded.owner_token,
                   lease_expires_at = excluded.lease_expires_at""",
            (
                obligation_id,
                session_key,
                platform,
                str(chat_id),
                str(thread_id) if thread_id else None,
                content,
                now,
                now,
                owner_pid,
                owner_started_at,
                profile,
                _OWNER_TOKEN,
                _lease_deadline(now),
            ),
        )
        _prune_rows(connection, now)

    _write(store)


def update_state(obligation_id: str, state: str, error: str = "") -> None:
    now = time.time()
    lease_expires_at = _lease_deadline(now) if state == "attempting" else None

    def update(connection):
        if lease_expires_at is None:
            connection.execute(
                """UPDATE delivery_obligations
                   SET state=?, updated_at=?, last_error=?
                   WHERE obligation_id=?""",
                (state, now, error[:500] if error else None, obligation_id),
            )
        else:
            connection.execute(
                """UPDATE delivery_obligations
                   SET state=?, updated_at=?, last_error=?, owner_token=?, lease_expires_at=?
                   WHERE obligation_id=?""",
                (
                    state,
                    now,
                    error[:500] if error else None,
                    _OWNER_TOKEN,
                    lease_expires_at,
                    obligation_id,
                ),
            )

    _write(update)


def release_runtime_claim(obligation_id: str, error: str = "") -> bool:
    def release(connection):
        cursor = connection.execute(
            """UPDATE delivery_obligations
               SET state='failed', attempts=CASE
                       WHEN attempts > 0 THEN attempts - 1 ELSE 0 END,
                   updated_at=?, last_error=?
               WHERE obligation_id=? AND state='attempting' AND owner_token=?""",
            (time.time(), error[:500] if error else None, obligation_id, _OWNER_TOKEN),
        )
        return bool(cursor.rowcount)

    return _write(release)


def sweep_recoverable(
    now: Optional[float] = None,
    *,
    deliverable_platforms: Optional[set] = None,
    deliverable_targets: Optional[set] = None,
) -> List[Dict[str, Any]]:
    from gateway import delivery_ledger as ledger

    current_time = time.time() if now is None else now

    def claim(connection):
        rows = connection.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, state, attempts, created_at, adapter_profile,
                      last_error, updated_at
               FROM delivery_obligations
               WHERE state IN ('pending', 'attempting', 'failed')
                 AND (owner_token IS NULL OR
                      (owner_token <> ? AND COALESCE(lease_expires_at, 0) <= ?))
               FOR UPDATE SKIP LOCKED""",
            (_OWNER_TOKEN, current_time),
        ).fetchall()
        claimed: List[Dict[str, Any]] = []
        for (
            obligation_id,
            session_key,
            platform,
            chat_id,
            thread_id,
            content,
            state,
            attempts,
            created_at,
            adapter_profile,
            last_error,
            updated_at,
        ) in rows:
            if (
                attempts >= ledger.MAX_ATTEMPTS
                or (current_time - created_at) > ledger.STALE_AFTER_SECONDS
            ):
                connection.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=? WHERE obligation_id=?""",
                    (current_time, obligation_id),
                )
                continue
            if (
                deliverable_platforms is not None
                and platform not in deliverable_platforms
            ) or (
                deliverable_targets is not None
                and (platform, adapter_profile) not in deliverable_targets
            ):
                continue
            flood_row = state == "failed" and ledger.is_flood_error(last_error)
            if flood_row and current_time < ledger.flood_not_before(
                updated_at, last_error
            ):
                connection.execute(
                    """UPDATE delivery_obligations
                       SET owner_token=?, lease_expires_at=?, owner_pid=?, owner_started_at=?,
                           adapter_profile=COALESCE(adapter_profile, 'default')
                       WHERE obligation_id=?""",
                    (
                        _OWNER_TOKEN,
                        _lease_deadline(current_time),
                        os.getpid(),
                        ledger._start_time(os.getpid()),
                        obligation_id,
                    ),
                )
                claimed.append({
                    "obligation_id": obligation_id,
                    "session_key": session_key,
                    "platform": platform,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "content": content,
                    "profile": adapter_profile or "default",
                    "attempts": attempts,
                    "adopted": True,
                    "not_before": ledger.flood_not_before(updated_at, last_error),
                })
                continue
            connection.execute(
                """UPDATE delivery_obligations
                   SET owner_token=?, lease_expires_at=?, owner_pid=?, owner_started_at=?,
                       attempts=attempts+1, updated_at=?,
                       adapter_profile=COALESCE(adapter_profile, 'default'),
                       state=CASE WHEN ? THEN 'attempting' ELSE state END,
                       last_error=CASE WHEN ? THEN NULL ELSE last_error END
                   WHERE obligation_id=?""",
                (
                    _OWNER_TOKEN,
                    _lease_deadline(current_time),
                    os.getpid(),
                    ledger._start_time(os.getpid()),
                    current_time,
                    flood_row,
                    flood_row,
                    obligation_id,
                ),
            )
            claimed.append(
                ledger._claimed_row(
                    obligation_id,
                    session_key,
                    platform,
                    chat_id,
                    thread_id,
                    content,
                    attempts,
                    adapter_profile or "default",
                    needs_marker=state != "pending",
                    flood=flood_row,
                )
            )
        return claimed

    return _write(claim)


def sweep_failed_for_runtime(
    platform: str, now: Optional[float] = None, *, profile: Optional[str] = None
) -> List[Dict[str, Any]]:
    from gateway import delivery_ledger as ledger

    current_time = time.time() if now is None else now
    expected_profile = (
        "default" if not profile or profile == "default" else str(profile)
    )

    def claim(connection):
        rows = connection.execute(
            """SELECT obligation_id, session_key, platform, chat_id, thread_id,
                      content, attempts, created_at, last_error, adapter_profile, updated_at
               FROM delivery_obligations
               WHERE state='failed' AND platform=? AND owner_token=?
               FOR UPDATE SKIP LOCKED""",
            (platform, _OWNER_TOKEN),
        ).fetchall()
        claimed: List[Dict[str, Any]] = []
        for (
            obligation_id,
            session_key,
            row_platform,
            chat_id,
            thread_id,
            content,
            attempts,
            created_at,
            last_error,
            adapter_profile,
            updated_at,
        ) in rows:
            if adapter_profile != expected_profile:
                continue
            due = ledger.retry_not_before(updated_at, last_error, attempts)
            if due is None:
                continue
            if (
                attempts >= ledger.MAX_ATTEMPTS
                or (current_time - created_at) > ledger.STALE_AFTER_SECONDS
            ):
                connection.execute(
                    """UPDATE delivery_obligations
                       SET state='abandoned', updated_at=?
                       WHERE obligation_id=? AND state='failed' AND owner_token=?""",
                    (current_time, obligation_id, _OWNER_TOKEN),
                )
                continue
            if current_time < due:
                continue
            cursor = connection.execute(
                """UPDATE delivery_obligations
                   SET state='attempting', attempts=attempts+1, updated_at=?, last_error=NULL,
                       lease_expires_at=?
                   WHERE obligation_id=? AND state='failed' AND owner_token=?""",
                (
                    current_time,
                    _lease_deadline(current_time),
                    obligation_id,
                    _OWNER_TOKEN,
                ),
            )
            if cursor.rowcount:
                claimed.append(
                    ledger._claimed_row(
                        obligation_id,
                        session_key,
                        row_platform,
                        chat_id,
                        thread_id,
                        content,
                        attempts,
                        adapter_profile,
                        needs_marker=True,
                        runtime=True,
                        flood=ledger.is_flood_error(last_error),
                        last_error=last_error,
                    )
                )
        return claimed

    return _write(claim)


def pending_retries(now: Optional[float] = None) -> List[Dict[str, Any]]:
    from gateway import delivery_ledger as ledger

    current_time = time.time() if now is None else now

    def load(connection):
        return connection.execute(
            """SELECT platform, adapter_profile, updated_at, last_error, attempts, created_at
               FROM delivery_obligations
               WHERE state='failed' AND owner_token=?""",
            (_OWNER_TOKEN,),
        ).fetchall()

    earliest: Dict[tuple, float] = {}
    for (
        platform,
        adapter_profile,
        updated_at,
        last_error,
        attempts,
        created_at,
    ) in _read(load):
        if ledger.is_reconnect_only(last_error):
            continue
        due = ledger.retry_not_before(updated_at, last_error, attempts)
        if (
            due is None
            or attempts >= ledger.MAX_ATTEMPTS
            or (current_time - created_at) > ledger.STALE_AFTER_SECONDS
        ):
            continue
        key = (platform, adapter_profile or "default")
        if key not in earliest or due < earliest[key]:
            earliest[key] = due
    return [
        {"platform": platform, "profile": profile, "not_before": due}
        for (platform, profile), due in sorted(earliest.items())
    ]


def prune(now: Optional[float] = None) -> None:
    current_time = time.time() if now is None else now
    _write(lambda connection: _prune_rows(connection, current_time))


def debug_rows(limit: int = 20) -> List[Dict[str, Any]]:
    def load(connection):
        return connection.execute(
            """SELECT obligation_id, session_key, state, attempts,
                      created_at, updated_at, last_error
               FROM delivery_obligations
               ORDER BY updated_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    return [
        {
            "id": row[0],
            "session": row[1],
            "state": row[2],
            "attempts": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "last_error": row[6],
        }
        for row in _read(load)
    ]
