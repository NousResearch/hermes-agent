"""SQLite storage helpers for gateway-hosted Group Chats.

This module owns schema initialization, root-database transactions, capacity,
peer-link receipts, and row serialization. Public room operations remain in
gateway.hosted_rooms.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, NoReturn

from gateway.hosted_room_contract import (
    AuthorityConflictError,
    HostedRoomError,
    RoomHistoryExpiredError,
    RoomNotFoundError,
    _EVENT_SCHEMA_COLUMNS,
    _LINK_SCHEMA_COLUMNS,
    _PEER_RESERVATION_SCHEMA_COLUMNS,
    _REMOTE_RUN_IDENTITY_COLUMNS,
    _REMOTE_RUN_SCHEMA_COLUMNS,
    _RETIRED_ROOM_SCHEMA_COLUMNS,
    _REVOKED_GRANT_SCHEMA_COLUMNS,
    _ROOM_SCHEMA_COLUMNS,
    _canonical_json,
    _validate_identifier,
)


def _public_api():
    """Resolve facade-owned patch points only when an operation runs."""
    from gateway import hosted_rooms

    return hosted_rooms


def _primary_key_columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    return tuple(
        str(row[1])
        for row in sorted(
            (row for row in conn.execute(f"PRAGMA table_info({table})") if row[5]),
            key=lambda row: int(row[5]),
        )
    )


def _migrate_remote_run_schema(conn: sqlite3.Connection) -> None:
    """Fence legacy receipts behind a complete authority-lineage key."""

    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(hosted_room_remote_runs)")
    }
    if (
        _REMOTE_RUN_SCHEMA_COLUMNS.issubset(columns)
        and _primary_key_columns(conn, "hosted_room_remote_runs")
        == _REMOTE_RUN_IDENTITY_COLUMNS
    ):
        return

    conn.execute("DROP TABLE IF EXISTS hosted_room_remote_runs_migrating")
    conn.execute(
        """CREATE TABLE hosted_room_remote_runs_migrating (
            room_id TEXT NOT NULL,
            home_install_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            member_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL CHECK (execution_generation >= 1),
            target_install_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (
                room_id, home_install_id, authority_gateway_id, authority_epoch,
                member_id, target_install_id, target_profile, task_id,
                execution_generation
            )
        )"""
    )
    if columns:
        home = "home_install_id" if "home_install_id" in columns else "'legacy'"
        gateway = (
            "authority_gateway_id" if "authority_gateway_id" in columns else "'legacy'"
        )
        epoch = "authority_epoch" if "authority_epoch" in columns else "1"
        conn.execute(
            f"""INSERT OR IGNORE INTO hosted_room_remote_runs_migrating(
                    room_id, home_install_id, authority_gateway_id,
                    authority_epoch, member_id, task_id,
                    execution_generation, target_install_id, target_profile,
                    run_id, session_id, created_at, updated_at
                )
                SELECT room_id, {home}, {gateway}, {epoch}, member_id, task_id,
                       execution_generation, target_install_id, target_profile,
                       run_id, session_id, created_at, updated_at
                  FROM hosted_room_remote_runs"""
        )
    conn.execute("DROP TABLE hosted_room_remote_runs")
    conn.execute(
        "ALTER TABLE hosted_room_remote_runs_migrating "
        "RENAME TO hosted_room_remote_runs"
    )


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_rooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL DEFAULT 1 CHECK (authority_epoch >= 1),
            next_seq INTEGER NOT NULL DEFAULT 1 CHECK (next_seq >= 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0),
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            disbanded_at REAL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_events (
            room_id TEXT NOT NULL,
            seq INTEGER NOT NULL CHECK (seq >= 1),
            event_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor_json TEXT NOT NULL,
            authority_epoch INTEGER CHECK (authority_epoch IS NULL OR authority_epoch >= 1),
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq),
            UNIQUE (room_id, event_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_retired_ids (
            room_id TEXT PRIMARY KEY,
            retired_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_links (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_url TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            grant TEXT NOT NULL,
            catalog_json TEXT NOT NULL,
            cancellation_scope_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            transport_security TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_remote_runs (
            room_id TEXT NOT NULL,
            home_install_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            member_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL CHECK (execution_generation >= 1),
            target_install_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (
                room_id, home_install_id, authority_gateway_id, authority_epoch,
                member_id, target_install_id, target_profile, task_id,
                execution_generation
            )
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_revoked_grants (
            scope_key TEXT PRIMARY KEY,
            expires_at REAL NOT NULL,
            revoked_before REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_peer_reservations (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            expires_at REAL NOT NULL,
            revoked_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id, target_profile)
        )"""
    )
    room_columns = {row[1] for row in conn.execute("PRAGMA table_info(hosted_rooms)")}
    if "authority_gateway_id" not in room_columns:
        conn.execute(
            "ALTER TABLE hosted_rooms "
            "ADD COLUMN authority_gateway_id TEXT NOT NULL DEFAULT 'legacy'"
        )
    if "authority_epoch" not in room_columns:
        conn.execute(
            "ALTER TABLE hosted_rooms "
            "ADD COLUMN authority_epoch INTEGER NOT NULL DEFAULT 1"
        )
    backfill_event_bytes = "event_bytes" not in room_columns
    if backfill_event_bytes:
        conn.execute(
            "ALTER TABLE hosted_rooms ADD COLUMN event_bytes INTEGER NOT NULL DEFAULT 0"
        )

    event_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_events)")
    }
    if "actor_json" not in event_columns:
        # Draft builds before the actor contract carried no identity. Preserve
        # their inert replay rows explicitly as legacy system events rather
        # than guessing a user or Bot author.
        legacy_actor = _canonical_json(
            {"kind": "system", "id": "legacy"},
            label="actor",
            max_bytes=4 * 1024,
        )
        escaped_actor = legacy_actor.replace("'", "''")
        conn.execute(
            "ALTER TABLE hosted_room_events "
            f"ADD COLUMN actor_json TEXT NOT NULL DEFAULT '{escaped_actor}'"
        )
    if "authority_epoch" not in event_columns:
        conn.execute(
            "ALTER TABLE hosted_room_events ADD COLUMN authority_epoch INTEGER"
        )
    if backfill_event_bytes:
        conn.execute(
            """UPDATE hosted_rooms
                  SET event_bytes=COALESCE((
                      SELECT SUM(
                          length(CAST(event_id AS BLOB)) +
                          length(CAST(kind AS BLOB)) +
                          length(CAST(actor_json AS BLOB)) +
                          length(CAST(payload_json AS BLOB))
                      )
                      FROM hosted_room_events
                      WHERE hosted_room_events.room_id=hosted_rooms.room_id
                  ), 0)"""
        )
    # Old schemas kept the final identity tombstone in hosted_rooms itself.
    # Copy those identities before bounded history pruning can remove their
    # heavier room/event payloads. This compact registry is intentionally
    # permanent: a stale coordinate must never name a different Group Chat.
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at)
           SELECT room_id, disbanded_at FROM hosted_rooms
            WHERE disbanded_at IS NOT NULL"""
    )
    _migrate_remote_run_schema(conn)
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_hosted_room_events_cursor
           ON hosted_room_events(room_id, seq)"""
    )
    if not _schema_is_current(conn):
        raise HostedRoomError("hosted room schema migration did not complete")


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    room_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_rooms)")
    )
    event_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_events)")
    )
    retired_room_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_retired_ids)")
    )
    link_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_links)")
    )
    remote_run_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_remote_runs)")
    )
    revoked_grant_columns = frozenset(
        row[1] for row in conn.execute("PRAGMA table_info(hosted_room_revoked_grants)")
    )
    peer_reservation_columns = frozenset(
        row[1]
        for row in conn.execute("PRAGMA table_info(hosted_room_peer_reservations)")
    )
    if not _ROOM_SCHEMA_COLUMNS.issubset(room_columns):
        return False
    if not _EVENT_SCHEMA_COLUMNS.issubset(event_columns):
        return False
    if not _RETIRED_ROOM_SCHEMA_COLUMNS.issubset(retired_room_columns):
        return False
    if not _LINK_SCHEMA_COLUMNS.issubset(link_columns):
        return False
    if not _REMOTE_RUN_SCHEMA_COLUMNS.issubset(remote_run_columns):
        return False
    if (
        _primary_key_columns(conn, "hosted_room_remote_runs")
        != _REMOTE_RUN_IDENTITY_COLUMNS
    ):
        return False
    if not _REVOKED_GRANT_SCHEMA_COLUMNS.issubset(revoked_grant_columns):
        return False
    if not _PEER_RESERVATION_SCHEMA_COLUMNS.issubset(peer_reservation_columns):
        return False
    index = conn.execute(
        """SELECT 1 FROM sqlite_master
           WHERE type='index' AND name='idx_hosted_room_events_cursor'"""
    ).fetchone()
    return index is not None


def list_room_link_records(db_path: Path | str) -> list[dict[str, Any]]:
    """Return private RoomLink records without logging or formatting grants."""
    with _transaction(db_path) as conn:
        rows = conn.execute(
            """SELECT room_id, member_id, target_url, target_profile, grant,
                      catalog_json, cancellation_scope_id, trace_id,
                      transport_security, status, updated_at
                 FROM hosted_room_links
             ORDER BY room_id, member_id"""
        ).fetchall()
    return [dict(row) for row in rows]


def upsert_room_link_record(
    db_path: Path | str,
    *,
    record: Mapping[str, Any],
    max_links: int,
) -> None:
    """Atomically insert or replace one private RoomLink record."""
    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            "SELECT 1 FROM hosted_room_links WHERE room_id=? AND member_id=?",
            (record["room_id"], record["member_id"]),
        ).fetchone()
        if existing is None:
            count = int(
                conn.execute("SELECT COUNT(*) FROM hosted_room_links").fetchone()[0]
            )
            if count >= max_links:
                raise HostedRoomError("too many stored room links")
        conn.execute(
            """INSERT INTO hosted_room_links(
                   room_id, member_id, target_url, target_profile, grant,
                   catalog_json, cancellation_scope_id, trace_id,
                   transport_security, status, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(room_id, member_id) DO UPDATE SET
                   target_url=excluded.target_url,
                   target_profile=excluded.target_profile,
                   grant=excluded.grant,
                   catalog_json=excluded.catalog_json,
                   cancellation_scope_id=excluded.cancellation_scope_id,
                   trace_id=excluded.trace_id,
                   transport_security=excluded.transport_security,
                   status=excluded.status,
                   updated_at=excluded.updated_at""",
            (
                record["room_id"],
                record["member_id"],
                record["target_url"],
                record["target_profile"],
                record["grant"],
                record["catalog_json"],
                record["cancellation_scope_id"],
                record["trace_id"],
                record["transport_security"],
                record["status"],
                record["updated_at"],
            ),
        )


def update_room_link_status(
    db_path: Path | str,
    *,
    room_id: str,
    member_id: str,
    status: str,
    now: float | None = None,
) -> bool:
    """Persist a non-secret route health classification."""
    with _transaction(db_path, immediate=True) as conn:
        cursor = conn.execute(
            """UPDATE hosted_room_links SET status=?, updated_at=?
                 WHERE room_id=? AND member_id=?""",
            (
                status,
                float(now if now is not None else time.time()),
                room_id,
                member_id,
            ),
        )
        return cursor.rowcount == 1


def delete_room_link_records(db_path: Path | str, *, room_id: str) -> int:
    """Delete persisted peer routes after their target grants are revoked."""
    with _transaction(db_path, immediate=True) as conn:
        cursor = conn.execute(
            "DELETE FROM hosted_room_links WHERE room_id=?",
            (room_id,),
        )
        return cursor.rowcount


def _room_grant_scope_key(claims: Mapping[str, Any]) -> str:
    """Return a stable non-secret key for one room/home/target/profile scope."""
    import hashlib

    fields = {
        key: str(claims.get(key) or "")
        for key in (
            "room_id",
            "home_install_id",
            "authority_gateway_id",
            "authority_epoch",
            "member_id",
            "target_install_id",
            "target_profile",
        )
    }
    if not all(fields.values()):
        raise HostedRoomError("room grant scope is incomplete")
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def revoke_room_grant_scope(
    db_path: Path | str,
    *,
    claims: Mapping[str, Any],
    expires_at: float,
    now: float | None = None,
) -> None:
    """Revoke every grant issued at or before now for one exact room scope."""
    scope_key = _room_grant_scope_key(claims)
    timestamp = float(now if now is not None else time.time())
    expiry = float(expires_at)
    if expiry <= timestamp:
        return
    with _transaction(db_path, immediate=True) as conn:
        conn.execute(
            "DELETE FROM hosted_room_revoked_grants WHERE expires_at<=?",
            (timestamp,),
        )
        conn.execute(
            """INSERT INTO hosted_room_revoked_grants(
                   scope_key, expires_at, revoked_before
               ) VALUES (?, ?, ?)
               ON CONFLICT(scope_key) DO UPDATE SET
                   expires_at=MAX(hosted_room_revoked_grants.expires_at,
                                  excluded.expires_at),
                   revoked_before=MAX(hosted_room_revoked_grants.revoked_before,
                                      excluded.revoked_before)""",
            (scope_key, expiry, timestamp),
        )
        conn.execute(
            """UPDATE hosted_room_peer_reservations
                  SET revoked_at=?, updated_at=?
                WHERE room_id=? AND member_id=? AND target_profile=?
                  AND authority_gateway_id=? AND authority_epoch=?""",
            (
                timestamp,
                timestamp,
                str(claims.get("room_id") or ""),
                str(claims.get("member_id") or ""),
                str(claims.get("target_profile") or ""),
                str(claims.get("authority_gateway_id") or ""),
                int(claims.get("authority_epoch") or 0),
            ),
        )


def reserve_peer_room(
    db_path: Path | str,
    *,
    claims: Mapping[str, Any],
    expires_at: float,
    now: float | None = None,
) -> None:
    """Fence direct Desktop prompts before the first peer run is admitted."""

    timestamp = float(now if now is not None else time.time())
    expiry = float(expires_at)
    if expiry <= timestamp:
        raise HostedRoomError("peer room reservation must expire in the future")
    limits = _public_api()
    values = (
        _validate_identifier(
            claims.get("room_id"),
            label="room_id",
            max_chars=limits.MAX_ROOM_ID_CHARS,
        ),
        _validate_identifier(
            claims.get("member_id"),
            label="member_id",
            max_chars=limits.MAX_ACTOR_ID_CHARS,
        ),
        _validate_identifier(
            claims.get("target_profile"),
            label="target_profile",
            max_chars=limits.MAX_ACTOR_ID_CHARS,
        ),
        _validate_identifier(
            claims.get("authority_gateway_id"),
            label="authority_gateway_id",
            max_chars=limits.MAX_ACTOR_ID_CHARS,
        ),
        int(claims.get("authority_epoch") or 0),
    )
    if values[4] < 1:
        raise HostedRoomError("authority_epoch must be positive")
    with _transaction(db_path, immediate=True) as conn:
        conn.execute(
            "DELETE FROM hosted_room_peer_reservations WHERE expires_at<=?",
            (timestamp,),
        )
        authority_rows = conn.execute(
            """SELECT authority_gateway_id, authority_epoch
                 FROM hosted_room_peer_reservations
                WHERE room_id=? AND target_profile=?
                  AND expires_at>? AND revoked_at IS NULL""",
            (values[0], values[2], timestamp),
        ).fetchall()
        if any(
            int(row["authority_epoch"]) > values[4]
            or (
                int(row["authority_epoch"]) == values[4]
                and str(row["authority_gateway_id"]) != values[3]
            )
            for row in authority_rows
        ):
            raise AuthorityConflictError("peer room reservation authority changed")
        conn.execute(
            """UPDATE hosted_room_peer_reservations
                  SET revoked_at=?, updated_at=?
                WHERE room_id=? AND target_profile=?
                  AND authority_epoch<? AND revoked_at IS NULL""",
            (timestamp, timestamp, values[0], values[2], values[4]),
        )
        existing = conn.execute(
            """SELECT authority_gateway_id, authority_epoch
                 FROM hosted_room_peer_reservations
                WHERE room_id=? AND member_id=? AND target_profile=?""",
            values[:3],
        ).fetchone()
        if existing is not None and (
            int(existing["authority_epoch"]) > values[4]
            or (
                int(existing["authority_epoch"]) == values[4]
                and str(existing["authority_gateway_id"]) != values[3]
            )
        ):
            raise AuthorityConflictError("peer room reservation authority changed")
        conn.execute(
            """INSERT INTO hosted_room_peer_reservations(
                   room_id, member_id, target_profile, authority_gateway_id,
                   authority_epoch, expires_at, revoked_at, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
               ON CONFLICT(room_id, member_id, target_profile) DO UPDATE SET
                   authority_gateway_id=excluded.authority_gateway_id,
                   authority_epoch=excluded.authority_epoch,
                   expires_at=MAX(hosted_room_peer_reservations.expires_at,
                                  excluded.expires_at),
                   revoked_at=NULL,
                   updated_at=excluded.updated_at""",
            (*values, expiry, timestamp, timestamp),
        )


def peer_room_is_reserved(
    db_path: Path | str,
    *,
    room_id: str,
    target_profile: str,
    now: float | None = None,
) -> bool:
    """Return whether a live target-side RoomLink reservation fences Desktop."""

    timestamp = float(now if now is not None else time.time())
    limits = _public_api()
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT 1 FROM hosted_room_peer_reservations
                WHERE room_id=? AND target_profile=?
                  AND expires_at>? AND revoked_at IS NULL
                LIMIT 1""",
            (
                _validate_identifier(
                    room_id,
                    label="room_id",
                    max_chars=limits.MAX_ROOM_ID_CHARS,
                ),
                _validate_identifier(
                    target_profile,
                    label="target_profile",
                    max_chars=limits.MAX_ACTOR_ID_CHARS,
                ),
                timestamp,
            ),
        ).fetchone()
    return row is not None


def peer_room_grant_is_current(
    db_path: Path | str,
    *,
    claims: Mapping[str, Any],
    now: float | None = None,
) -> bool:
    """Require a grant to match the target's current live reservation."""

    timestamp = float(now if now is not None else time.time())
    limits = _public_api()
    room_id = _validate_identifier(
        claims.get("room_id"),
        label="room_id",
        max_chars=limits.MAX_ROOM_ID_CHARS,
    )
    member_id = _validate_identifier(
        claims.get("member_id"),
        label="member_id",
        max_chars=limits.MAX_ACTOR_ID_CHARS,
    )
    target_profile = _validate_identifier(
        claims.get("target_profile"),
        label="target_profile",
        max_chars=limits.MAX_ACTOR_ID_CHARS,
    )
    authority_gateway_id = _validate_identifier(
        claims.get("authority_gateway_id"),
        label="authority_gateway_id",
        max_chars=limits.MAX_ACTOR_ID_CHARS,
    )
    authority_epoch = int(claims.get("authority_epoch") or 0)
    if authority_epoch < 1:
        raise HostedRoomError("authority_epoch must be positive")
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT 1 FROM hosted_room_peer_reservations
                WHERE room_id=? AND member_id=? AND target_profile=?
                  AND authority_gateway_id=? AND authority_epoch=?
                  AND expires_at>? AND revoked_at IS NULL
                LIMIT 1""",
            (
                room_id,
                member_id,
                target_profile,
                authority_gateway_id,
                authority_epoch,
                timestamp,
            ),
        ).fetchone()
    return row is not None


def room_grant_is_revoked(
    db_path: Path | str,
    *,
    claims: Mapping[str, Any],
    now: float | None = None,
) -> bool:
    """Return whether a grant predates its exact scope's revocation fence."""
    timestamp = float(now if now is not None else time.time())
    scope_key = _room_grant_scope_key(claims)
    issued_at = float(claims.get("issued_at") or 0)
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT revoked_before FROM hosted_room_revoked_grants
                 WHERE scope_key=? AND expires_at>?""",
            (scope_key, timestamp),
        ).fetchone()
    return row is not None and issued_at <= float(row["revoked_before"])


def _remote_run_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[column] for column in _REMOTE_RUN_IDENTITY_COLUMNS)


def upsert_remote_run_receipt(
    db_path: Path | str,
    *,
    record: Mapping[str, Any],
    now: float | None = None,
) -> None:
    """Durably bind one logical peer task attempt to its remote run handle."""
    timestamp = float(now if now is not None else time.time())
    identity = _remote_run_identity(record)
    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            """SELECT * FROM hosted_room_remote_runs
                 WHERE room_id=? AND home_install_id=?
                   AND authority_gateway_id=? AND authority_epoch=?
                   AND member_id=? AND target_install_id=?
                   AND target_profile=? AND task_id=?
                   AND execution_generation=?""",
            identity,
        ).fetchone()
        immutable = (*identity, record["run_id"], record["session_id"])
        if existing is not None:
            stored = (
                *_remote_run_identity(existing),
                existing["run_id"],
                existing["session_id"],
            )
            if stored != immutable:
                raise HostedRoomError(
                    "remote run receipt conflicts with its logical task"
                )
            conn.execute(
                """UPDATE hosted_room_remote_runs SET updated_at=?
                     WHERE room_id=? AND home_install_id=?
                       AND authority_gateway_id=? AND authority_epoch=?
                       AND member_id=? AND target_install_id=?
                       AND target_profile=? AND task_id=?
                       AND execution_generation=?""",
                (timestamp, *identity),
            )
            return
        conn.execute(
            """INSERT INTO hosted_room_remote_runs(
                   room_id, home_install_id, authority_gateway_id,
                   authority_epoch, member_id, target_install_id,
                   target_profile, task_id, execution_generation, run_id,
                   session_id, created_at, updated_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                *immutable,
                timestamp,
                timestamp,
            ),
        )


def list_remote_run_receipts(
    db_path: Path | str,
    *,
    room_id: str | None = None,
    target_profile: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return remote run handles in durable task order."""
    conditions: list[str] = []
    values: list[Any] = []
    for column, value in (
        ("room_id", room_id),
        ("target_profile", target_profile),
        ("session_id", session_id),
    ):
        if value is not None:
            conditions.append(f"{column}=?")
            values.append(value)
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    with _transaction(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hosted_room_remote_runs"
            + where
            + " ORDER BY created_at, task_id, execution_generation",
            values,
        ).fetchall()
    return [dict(row) for row in rows]


def remote_run_receipt(
    db_path: Path | str,
    *,
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return the exact durable remote run handle for one task attempt."""
    identity = _remote_run_identity(record)
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT * FROM hosted_room_remote_runs
                 WHERE room_id=? AND home_install_id=?
                   AND authority_gateway_id=? AND authority_epoch=?
                   AND member_id=? AND target_install_id=?
                   AND target_profile=? AND task_id=?
                   AND execution_generation=?""",
            identity,
        ).fetchone()
    return dict(row) if row is not None else None


def _connect(db_path: Path | str) -> sqlite3.Connection:
    from hermes_state import apply_wal_with_fallback

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        journal_retries = _public_api()._JOURNAL_MODE_LOCK_RETRIES
        for attempt in range(journal_retries):
            try:
                apply_wal_with_fallback(conn, db_label="state.db (hosted_rooms)")
                break
            except sqlite3.OperationalError as exc:
                if (
                    str(exc).lower() != "database is locked"
                    or attempt + 1 == journal_retries
                ):
                    raise
                # SQLite's journal-mode pragma may not honor the connection's
                # busy timeout while another first opener initializes the DB,
                # especially on Windows. Retry only that transient lock class.
                time.sleep(0.01 * (2**attempt))
        conn.execute("PRAGMA foreign_keys=ON")
        if _schema_is_current(conn):
            return conn
        # Multiple profile gateways share this root database. Serialize every
        # draft-schema transition in SQLite itself so a crash rolls back the
        # whole DDL/data migration and another process can safely retry it.
        conn.execute("BEGIN IMMEDIATE")
        _public_api()._initialize_schema(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


def _read_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open the room store without steady-state journal or migration writes."""

    path = Path(db_path)
    if not path.is_file():
        initialized = _connect(path)
        initialized.close()
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    if _schema_is_current(conn):
        return conn
    conn.close()
    migrated = _connect(path)
    migrated.close()
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def _transaction(
    db_path: Path | str, *, immediate: bool = False
) -> Iterator[sqlite3.Connection]:
    conn = _connect(db_path)
    try:
        if immediate:
            conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _raise_room_not_found(conn: sqlite3.Connection, room_id: str) -> NoReturn:
    retained = conn.execute(
        "SELECT 1 FROM hosted_rooms WHERE room_id=?",
        (room_id,),
    ).fetchone()
    if retained is not None:
        # A retained disband tombstone still has replayable history. The
        # caller simply did not opt into reading disbanded rooms.
        raise RoomNotFoundError("hosted room not found")
    retired = conn.execute(
        "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?",
        (room_id,),
    ).fetchone()
    if retired is not None:
        raise RoomHistoryExpiredError(
            "Group Chat history expired; room_id remains permanently retired"
        )
    raise RoomNotFoundError("hosted room not found")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def _room_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    room = {
        "room_id": row["room_id"],
        "name": row["name"],
        "members": json.loads(row["members_json"]),
        "authority_gateway_id": row["authority_gateway_id"],
        "authority_epoch": int(row["authority_epoch"]),
        "revision": int(row["revision"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "idempotent": idempotent,
    }
    if "disbanded_at" in row.keys() and row["disbanded_at"] is not None:
        room["disbanded_at"] = float(row["disbanded_at"])
    if "next_seq" in row.keys():
        room["latest_seq"] = int(row["next_seq"]) - 1
    return room


def _event_storage_bytes(
    *, event_id: str, kind: str, actor_json: str, payload_json: str
) -> int:
    return len((event_id + kind + actor_json + payload_json).encode("utf-8"))


def _assert_event_capacity(
    conn: sqlite3.Connection,
    *,
    room: sqlite3.Row,
    additional_bytes: int,
    allow_control: bool = False,
) -> None:
    limits = _public_api()
    event_limit = limits.MAX_EVENTS_PER_ROOM + (
        limits.CONTROL_EVENT_COUNT_RESERVE if allow_control else 0
    )
    room_byte_limit = limits.MAX_ROOM_EVENT_BYTES + (
        limits.CONTROL_EVENT_BYTE_RESERVE if allow_control else 0
    )
    gateway_byte_limit = limits.MAX_GATEWAY_EVENT_BYTES + (
        limits.CONTROL_EVENT_BYTE_RESERVE if allow_control else 0
    )
    if int(room["next_seq"]) - 1 >= event_limit:
        raise HostedRoomError(
            "This Group Chat reached its history limit. Start a new Group Chat to continue."
        )
    room_bytes = int(room["event_bytes"])
    if room_bytes + additional_bytes > room_byte_limit:
        raise HostedRoomError(
            "This Group Chat reached its storage limit. Start a new Group Chat to continue."
        )
    gateway_bytes = int(
        conn.execute(
            "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
        ).fetchone()[0]
    )
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        _prune_disbanded_rooms_locked(
            conn,
            now=None,
            max_gateway_event_bytes=max(0, gateway_byte_limit - additional_bytes),
        )
        gateway_bytes = int(
            conn.execute(
                "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
            ).fetchone()[0]
        )
    if gateway_bytes + additional_bytes > gateway_byte_limit:
        raise HostedRoomError(
            "Group Chat storage is full on this host. Delete an old Group Chat and try again."
        )


def _prune_disbanded_rooms_locked(
    conn: sqlite3.Connection,
    *,
    now: float | None,
    max_gateway_event_bytes: int | None = None,
) -> int:
    limits = _public_api()
    candidates: set[str] = set()
    if now is not None:
        cutoff = now - limits.DISBANDED_ROOM_RETENTION_SECONDS
        candidates.update(
            str(row["room_id"])
            for row in conn.execute(
                """SELECT room_id FROM hosted_rooms
                     WHERE disbanded_at IS NOT NULL AND disbanded_at<=?""",
                (cutoff,),
            ).fetchall()
        )
    candidates.update(
        str(row["room_id"])
        for row in conn.execute(
            """SELECT room_id FROM hosted_rooms
                 WHERE disbanded_at IS NOT NULL
                 ORDER BY disbanded_at DESC, room_id ASC
                 LIMIT -1 OFFSET ?""",
            (limits.MAX_DISBANDED_ROOM_TOMBSTONES,),
        ).fetchall()
    )
    if max_gateway_event_bytes is not None:
        retained_bytes = int(
            conn.execute(
                "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_rooms"
            ).fetchone()[0]
        )
        if retained_bytes > max_gateway_event_bytes:
            for row in conn.execute(
                """SELECT room_id, event_bytes FROM hosted_rooms
                     WHERE disbanded_at IS NOT NULL
                     ORDER BY disbanded_at ASC, room_id ASC"""
            ).fetchall():
                room_id = str(row["room_id"])
                if room_id not in candidates:
                    candidates.add(room_id)
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_gateway_event_bytes:
                    break
    if not candidates:
        return 0

    placeholders = ",".join("?" for _ in candidates)
    room_ids = tuple(sorted(candidates))
    conn.execute(
        f"""INSERT OR IGNORE INTO hosted_room_retired_ids (room_id, retired_at)
            SELECT room_id, disbanded_at FROM hosted_rooms
             WHERE room_id IN ({placeholders}) AND disbanded_at IS NOT NULL""",
        room_ids,
    )
    dependent_tables = (
        "hosted_room_policy_transcript_state",
        "hosted_room_policy_transcript",
        "hosted_room_policy_publications",
        "hosted_room_policy_watermarks",
        "hosted_room_policy_events",
        "hosted_room_policy_threads",
        "hosted_room_policy_cursors",
        "hosted_room_driver_tasks",
        "hosted_room_driver_leases",
        "hosted_room_remote_runs",
        "hosted_room_links",
        "hosted_room_peer_reservations",
        "hosted_room_events",
    )
    for table in dependent_tables:
        if _table_exists(conn, table):
            conn.execute(
                f"DELETE FROM {table} WHERE room_id IN ({placeholders})",
                room_ids,
            )
    conn.execute(
        f"DELETE FROM hosted_rooms WHERE room_id IN ({placeholders})",
        room_ids,
    )
    return len(room_ids)


def prune_disbanded_rooms(
    db_path: Path | str,
    *,
    now: float | None = None,
) -> int:
    """Purge deleted Group Chat payloads while reserving their identities."""

    timestamp = time.time() if now is None else float(now)
    with _transaction(db_path, immediate=True) as conn:
        return _prune_disbanded_rooms_locked(conn, now=timestamp)


def _event_from_row(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    return {
        "room_id": row["room_id"],
        "seq": int(row["seq"]),
        "event_id": row["event_id"],
        "kind": row["kind"],
        "actor": json.loads(row["actor_json"]),
        "authority_epoch": (
            int(row["authority_epoch"]) if row["authority_epoch"] is not None else None
        ),
        "payload": json.loads(row["payload_json"]),
        "created_at": float(row["created_at"]),
        "idempotent": idempotent,
    }
