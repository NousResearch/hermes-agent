"""Persisted RoomLink routes and their retirement fences."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping

from gateway.hosted_rooms import (
    _transaction,
    AuthorityConflictError,
    HostedRoomError,
    _validate_identifier,
)






def room_link_record(
    db_path: Path | str,
    *,
    room_id: str,
    member_id: str,
) -> dict[str, Any] | None:
    """Return one private RoomLink record by its exact identity."""

    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT room_id, member_id, target_url, target_profile, grant,
                      catalog_json, cancellation_scope_id, trace_id,
                      transport_security, status, updated_at
                 FROM hosted_room_links
                WHERE room_id=? AND member_id=?""",
            (room_id, member_id),
        ).fetchone()
    return dict(row) if row is not None else None


def begin_room_link_retirement(
    db_path: Path | str,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence new route writes before remote revocation and room disband."""

    from gateway import hosted_rooms as limits
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=limits.MAX_ROOM_ID_CHARS,
    )
    authority_gateway_id = _validate_identifier(
        authority_gateway_id,
        label="authority_gateway_id",
        max_chars=limits.MAX_ACTOR_ID_CHARS,
    )
    if (
        isinstance(authority_epoch, bool)
        or not isinstance(authority_epoch, int)
        or authority_epoch < 1
    ):
        raise HostedRoomError("authority_epoch must be a positive integer")
    timestamp = float(time.time() if now is None else now)
    with _transaction(db_path, immediate=True) as conn:
        room = conn.execute(
            """SELECT authority_gateway_id, authority_epoch
                 FROM hosted_rooms
                WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        lineage = (authority_gateway_id, authority_epoch)
        if (
            room is not None
            and (str(room["authority_gateway_id"]), int(room["authority_epoch"]))
            != lineage
        ):
            raise AuthorityConflictError("Group Chat route fence authority changed")
        existing = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, started_at,
                      revocation_complete_at
                 FROM hosted_room_disband_fences WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if existing is not None:
            if (
                str(existing["authority_gateway_id"]),
                int(existing["authority_epoch"]),
            ) != lineage:
                raise AuthorityConflictError("Group Chat route fence authority changed")
            return dict(existing)
        conn.execute(
            """INSERT INTO hosted_room_disband_fences(
                   room_id, authority_gateway_id, authority_epoch, started_at
               ) VALUES (?, ?, ?, ?)""",
            (room_id, *lineage, timestamp),
        )
        return {
            "room_id": room_id,
            "authority_gateway_id": lineage[0],
            "authority_epoch": lineage[1],
            "started_at": timestamp,
            "revocation_complete_at": None,
        }


def room_link_retirement_started(db_path: Path | str, *, room_id: str) -> bool:
    """Return whether this room has crossed the no-new-route boundary."""

    with _transaction(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM hosted_room_disband_fences WHERE room_id=?",
            (room_id,),
        ).fetchone()
    return row is not None


def complete_room_link_retirement(
    db_path: Path | str,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    now: float | None = None,
) -> None:
    """Allow route deletion only after every remote grant was revoked."""

    from gateway import hosted_rooms as limits
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=limits.MAX_ROOM_ID_CHARS,
    )
    authority_gateway_id = _validate_identifier(
        authority_gateway_id,
        label="authority_gateway_id",
        max_chars=limits.MAX_ACTOR_ID_CHARS,
    )
    if (
        isinstance(authority_epoch, bool)
        or not isinstance(authority_epoch, int)
        or authority_epoch < 1
    ):
        raise HostedRoomError("authority_epoch must be a positive integer")
    timestamp = float(time.time() if now is None else now)
    with _transaction(db_path, immediate=True) as conn:
        cursor = conn.execute(
            """UPDATE hosted_room_disband_fences
                  SET revocation_complete_at=COALESCE(revocation_complete_at, ?)
                WHERE room_id=? AND authority_gateway_id=? AND authority_epoch=?""",
            (timestamp, room_id, authority_gateway_id, authority_epoch),
        )
        if cursor.rowcount != 1:
            raise AuthorityConflictError("Group Chat route fence authority changed")


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
    expected_grant_sha256: str | None = None,
) -> None:
    """Atomically insert or replace one private RoomLink record."""
    with _transaction(db_path, immediate=True) as conn:
        fenced = conn.execute(
            "SELECT 1 FROM hosted_room_disband_fences WHERE room_id=?",
            (record["room_id"],),
        ).fetchone()
        if fenced is not None:
            raise HostedRoomError("Group Chat route registration is fenced")
        existing = conn.execute(
            "SELECT grant FROM hosted_room_links WHERE room_id=? AND member_id=?",
            (record["room_id"], record["member_id"]),
        ).fetchone()
        if expected_grant_sha256 is not None:
            current_hash = (
                hashlib.sha256(str(existing["grant"]).encode("utf-8")).hexdigest()
                if existing is not None
                else ""
            )
            incoming_hash = hashlib.sha256(record["grant"].encode("utf-8")).hexdigest()
            if current_hash not in {incoming_hash, expected_grant_sha256}:
                raise HostedRoomError("peer route changed during reconnect")
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
    expected_grant_sha256: str | None = None,
) -> bool:
    """Persist a non-secret route health classification."""
    with _transaction(db_path, immediate=True) as conn:
        if expected_grant_sha256 is not None:
            row = conn.execute(
                "SELECT grant FROM hosted_room_links WHERE room_id=? AND member_id=?",
                (room_id, member_id),
            ).fetchone()
            if (
                row is None
                or hashlib.sha256(row["grant"].encode()).hexdigest()
                != expected_grant_sha256
            ):
                return False
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
