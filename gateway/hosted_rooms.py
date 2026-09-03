"""Durable operations for gateway-hosted Bot Mode rooms.

The public API in this module owns room identity and its append-only event log.
Validation lives in hosted_room_contract and root-DB mechanics live in
hosted_room_storage.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from gateway.hosted_room_contract import (
    AuthorityConflictError,
    AuthoritySupersededError,
    CONTROL_EVENT_BYTE_RESERVE,
    CONTROL_EVENT_COUNT_RESERVE,
    DISBANDED_ROOM_RETENTION_SECONDS,
    EventConflictError,
    HostedRoomError,
    MAX_ACTIVE_ROOMS,
    MAX_ACTOR_ID_CHARS,
    MAX_ACTOR_LABEL_CHARS,
    MAX_DISBANDED_ROOM_TOMBSTONES,
    MAX_EVENT_ID_CHARS,
    MAX_EVENT_JSON_BYTES,
    MAX_EVENT_KIND_CHARS,
    MAX_EVENTS_PER_ROOM,
    MAX_GATEWAY_EVENT_BYTES,
    MAX_LOG_LIMIT,
    MAX_LOG_PAGE_BYTES,
    MAX_MEMBERS,
    MAX_MEMBERS_JSON_BYTES,
    MAX_ROOM_EVENT_BYTES,
    MAX_ROOM_ID_CHARS,
    MAX_ROOM_LIST_LIMIT,
    MAX_ROOM_NAME_CHARS,
    PROTOCOL_VERSION,
    RoomConflictError,
    RoomHistoryExpiredError,
    RoomNotFoundError,
    RoomProbeUnavailableError,
    _ACTOR_FIELDS,
    _EVENT_KIND_RE,
    _EVENT_KINDS_BY_ACTOR,
    _EVENT_SCHEMA_COLUMNS,
    _IDENTIFIER_RE,
    _JOURNAL_MODE_LOCK_RETRIES,
    _LINK_SCHEMA_COLUMNS,
    _PEER_RESERVATION_SCHEMA_COLUMNS,
    _REMOTE_RUN_IDENTITY_COLUMNS,
    _REMOTE_RUN_SCHEMA_COLUMNS,
    _RETIRED_ROOM_SCHEMA_COLUMNS,
    _REVOKED_GRANT_SCHEMA_COLUMNS,
    _ROOM_SCHEMA_COLUMNS,
    _canonical_json,
    _legacy_members_match,
    _optional_actor_field,
    _validate_actor,
    _validate_event_kind,
    _validate_identifier,
    _validate_members,
    _validate_room_name,
    default_db_path,
    local_authority_gateway_id,
    user_event_id,
)
from gateway.hosted_room_storage import (
    _assert_event_capacity,
    _connect,
    _event_from_row,
    _event_storage_bytes,
    _initialize_schema,
    _migrate_remote_run_schema,
    _primary_key_columns,
    _prune_disbanded_rooms_locked,
    _raise_room_not_found,
    _read_connection,
    _remote_run_identity,
    _room_from_row,
    _room_grant_scope_key,
    _schema_is_current,
    _table_exists,
    _transaction,
    delete_room_link_records,
    list_remote_run_receipts,
    list_room_link_records,
    peer_room_grant_is_current,
    peer_room_is_reserved,
    prune_disbanded_rooms,
    remote_run_receipt,
    reserve_peer_room,
    revoke_room_grant_scope,
    room_grant_is_revoked,
    update_room_link_status,
    upsert_remote_run_receipt,
    upsert_room_link_record,
)


def create_room(
    db_path: Path | str,
    *,
    room_id: Any,
    name: Any,
    members: Any,
    authority_gateway_id: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Create a room, or return the identical existing room idempotently."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    name = _validate_room_name(name)
    normalized_members, members_json = _validate_members(members)
    authority_gateway_id = _validate_identifier(
        authority_gateway_id,
        label="authority_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    now = time.time() if now is None else float(now)

    with _transaction(db_path, immediate=True) as conn:
        if conn.execute(
            "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?",
            (room_id,),
        ).fetchone():
            raise RoomConflictError("room_id belongs to a disbanded room")
        existing = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, event_bytes, revision,
                      created_at, updated_at, disbanded_at
               FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if existing is not None:
            if existing["disbanded_at"] is not None:
                raise RoomConflictError("room_id belongs to a disbanded room")
            legacy_adoption = (
                existing["authority_gateway_id"] == "legacy"
                and authority_gateway_id != "legacy"
            )
            members_match = existing["members_json"] == members_json or (
                legacy_adoption
                and _legacy_members_match(existing["members_json"], normalized_members)
            )
            if existing["name"] != name or not members_match:
                raise RoomConflictError("room_id already exists with different state")
            if legacy_adoption:
                target_epoch = int(existing["authority_epoch"]) + 1
                seq = int(existing["next_seq"])
                claim_actor_json = _canonical_json(
                    {"kind": "system", "id": "authority-control"},
                    label="actor",
                    max_bytes=4 * 1024,
                )
                claim_payload_json = _canonical_json(
                    {
                        "previous_gateway_id": "legacy",
                        "authority_gateway_id": authority_gateway_id,
                        "authority_epoch": target_epoch,
                    },
                    label="payload",
                    max_bytes=MAX_EVENT_JSON_BYTES,
                )
                claim_bytes = _event_storage_bytes(
                    event_id="system:authority-adopted",
                    kind="authority.claimed",
                    actor_json=claim_actor_json,
                    payload_json=claim_payload_json,
                )
                _assert_event_capacity(
                    conn,
                    room=existing,
                    additional_bytes=claim_bytes,
                    allow_control=True,
                )
                conn.execute(
                    """INSERT INTO hosted_room_events
                       (room_id, seq, event_id, kind, actor_json,
                        authority_epoch, payload_json, created_at)
                       VALUES (?, ?, 'system:authority-adopted',
                               'authority.claimed', ?, ?, ?, ?)""",
                    (
                        room_id,
                        seq,
                        claim_actor_json,
                        target_epoch,
                        claim_payload_json,
                        now,
                    ),
                )
                adopted = conn.execute(
                    """UPDATE hosted_rooms
                          SET members_json=?, authority_gateway_id=?, authority_epoch=?,
                              next_seq=next_seq+1, revision=revision+1,
                              event_bytes=event_bytes+?, updated_at=?
                        WHERE room_id=? AND authority_gateway_id='legacy'
                          AND authority_epoch=? AND next_seq=?
                          AND disbanded_at IS NULL""",
                    (
                        members_json,
                        authority_gateway_id,
                        target_epoch,
                        claim_bytes,
                        now,
                        room_id,
                        int(existing["authority_epoch"]),
                        seq,
                    ),
                )
                if adopted.rowcount != 1:
                    raise AuthorityConflictError("legacy room adoption lost its fence")
                existing = conn.execute(
                    """SELECT room_id, name, members_json, authority_gateway_id,
                              authority_epoch, next_seq, revision, created_at,
                              updated_at, disbanded_at
                         FROM hosted_rooms WHERE room_id=?""",
                    (room_id,),
                ).fetchone()
                if existing is None:  # pragma: no cover - row updated above
                    raise RuntimeError("adopted room could not be reloaded")
                result = _room_from_row(existing, idempotent=True)
                result["adopted"] = True
                claim_event = conn.execute(
                    """SELECT room_id, seq, event_id, kind, actor_json,
                              authority_epoch, payload_json, created_at
                         FROM hosted_room_events
                        WHERE room_id=? AND event_id='system:authority-adopted'""",
                    (room_id,),
                ).fetchone()
                if claim_event is None:  # pragma: no cover - inserted above
                    raise RuntimeError("legacy adoption event could not be reloaded")
                result["claim_event"] = _event_from_row(claim_event)
                return result
            if existing["authority_gateway_id"] != authority_gateway_id:
                raise RoomConflictError(
                    "room_id already belongs to a different authority"
                )
            return _room_from_row(existing, idempotent=True)

        active_rooms = int(
            conn.execute(
                "SELECT COUNT(*) FROM hosted_rooms WHERE disbanded_at IS NULL"
            ).fetchone()[0]
        )
        if active_rooms >= MAX_ACTIVE_ROOMS:
            raise HostedRoomError(
                "This host has too many active Group Chats. Delete one and try again."
            )

        conn.execute(
            """INSERT INTO hosted_rooms
               (room_id, name, members_json, authority_gateway_id,
                authority_epoch, next_seq, event_bytes, revision,
                created_at, updated_at, disbanded_at)
               VALUES (?, ?, ?, ?, 1, 1, 0, 1, ?, ?, NULL)""",
            (room_id, name, members_json, authority_gateway_id, now, now),
        )
        row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, revision, created_at, updated_at
               FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the insert above
            raise RuntimeError("created room could not be reloaded")
    result = _room_from_row(row)
    result["members"] = normalized_members
    return result


def list_rooms(
    db_path: Path | str,
    *,
    include_disbanded: bool = False,
    limit: int = MAX_ROOM_LIST_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return one bounded read-only page ordered by most recent change."""
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_ROOM_LIST_LIMIT
    ):
        raise HostedRoomError(f"limit must be between 1 and {MAX_ROOM_LIST_LIMIT}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise HostedRoomError("offset must be a non-negative integer")
    conn = _read_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at, updated_at,
                      disbanded_at
               FROM hosted_rooms
               WHERE disbanded_at IS NULL OR ?
               ORDER BY updated_at DESC, room_id ASC
               LIMIT ? OFFSET ?""",
            (int(include_disbanded), limit, offset),
        ).fetchall()
    finally:
        conn.close()
    return [_room_from_row(row) for row in rows]


def rename_room(
    db_path: Path | str,
    *,
    room_id: Any,
    event_id: Any,
    name: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Rename a live room and append its replay event atomically."""
    room_id = _validate_identifier(
        room_id, label="room_id", max_chars=MAX_ROOM_ID_CHARS
    )
    event_id = _validate_identifier(
        event_id, label="event_id", max_chars=MAX_EVENT_ID_CHARS
    )
    name = _validate_room_name(name)
    now = time.time() if now is None else float(now)
    actor_json = _canonical_json(
        {"kind": "system", "id": "room-control"},
        label="actor",
        max_bytes=4 * 1024,
    )
    payload_json = _canonical_json(
        {"name": name}, label="payload", max_bytes=MAX_EVENT_JSON_BYTES
    )
    with _transaction(db_path, immediate=True) as conn:
        room = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, event_bytes, revision, created_at,
                      updated_at, disbanded_at
                 FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if room is None:
            _raise_room_not_found(conn, room_id)
        if room["disbanded_at"] is not None:
            raise RoomNotFoundError("hosted room not found")
        existing = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM hosted_room_events WHERE room_id=? AND event_id=?""",
            (room_id, event_id),
        ).fetchone()
        if existing is not None:
            if (
                existing["kind"] != "room.renamed"
                or existing["payload_json"] != payload_json
            ):
                raise EventConflictError(
                    "event_id already exists with different immutable content"
                )
            result = _room_from_row(room, idempotent=True)
            result["event"] = _event_from_row(existing, idempotent=True)
            return result
        seq = int(room["next_seq"])
        epoch = int(room["authority_epoch"])
        event_bytes = _event_storage_bytes(
            event_id=event_id,
            kind="room.renamed",
            actor_json=actor_json,
            payload_json=payload_json,
        )
        _assert_event_capacity(conn, room=room, additional_bytes=event_bytes)
        conn.execute(
            """UPDATE hosted_rooms
               SET name=?, next_seq=?, event_bytes=event_bytes+?,
                   revision=revision+1, updated_at=?
               WHERE room_id=?""",
            (name, seq + 1, event_bytes, now, room_id),
        )
        conn.execute(
            """INSERT INTO hosted_room_events(
                   room_id, seq, event_id, kind, actor_json,
                   authority_epoch, payload_json, created_at
               ) VALUES (?, ?, ?, 'room.renamed', ?, ?, ?, ?)""",
            (room_id, seq, event_id, actor_json, epoch, payload_json, now),
        )
        updated = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at,
                      updated_at, disbanded_at
                 FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        event = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM hosted_room_events WHERE room_id=? AND event_id=?""",
            (room_id, event_id),
        ).fetchone()
    result = _room_from_row(updated)
    result["event"] = _event_from_row(event)
    return result


def append_event(
    db_path: Path | str,
    *,
    room_id: Any,
    event_id: Any,
    kind: Any,
    actor: Any,
    payload: Any,
    authority_gateway_id: Any = None,
    authority_epoch: Any = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Append one immutable event and allocate its per-room sequence atomically.

    Repeating the same ``event_id`` and immutable content returns the original
    event. Reusing the id for different content fails closed.
    """
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    event_id = _validate_identifier(
        event_id,
        label="event_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    kind = _validate_event_kind(kind)
    normalized_actor, actor_json = _validate_actor(actor, kind=kind)
    authority_scoped = normalized_actor["kind"] in {
        "user",
        "member",
        "gateway",
        "system",
    }
    normalized_authority_gateway_id: str | None = None
    normalized_authority_epoch: int | None = None
    if authority_scoped:
        normalized_authority_gateway_id = _validate_identifier(
            authority_gateway_id,
            label="authority_gateway_id",
            max_chars=MAX_ACTOR_ID_CHARS,
        )
        if (
            normalized_actor["kind"] == "gateway"
            and normalized_actor["id"] != normalized_authority_gateway_id
        ):
            raise HostedRoomError("gateway actor.id must match authority_gateway_id")
        if (
            isinstance(authority_epoch, bool)
            or not isinstance(authority_epoch, int)
            or authority_epoch < 1
        ):
            raise HostedRoomError("authority_epoch must be a positive integer")
        normalized_authority_epoch = authority_epoch
    elif authority_gateway_id is not None or authority_epoch is not None:
        raise HostedRoomError("authority fields are only valid for room-scoped events")
    if not isinstance(payload, dict):
        raise HostedRoomError("payload must be an object")
    payload_json = _canonical_json(
        payload,
        label="payload",
        max_bytes=MAX_EVENT_JSON_BYTES,
    )
    now = time.time() if now is None else float(now)

    with _transaction(db_path, immediate=True) as conn:
        existing = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
               FROM hosted_room_events WHERE room_id=? AND event_id=?""",
            (room_id, event_id),
        ).fetchone()
        if existing is not None:
            if (
                existing["kind"] != kind
                or existing["actor_json"] != actor_json
                or existing["authority_epoch"] != normalized_authority_epoch
                or existing["payload_json"] != payload_json
            ):
                raise EventConflictError(
                    "event_id already exists with different content"
                )
            return _event_from_row(existing, idempotent=True)

        room = conn.execute(
            """SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch
                  FROM hosted_rooms
               WHERE room_id=? AND disbanded_at IS NULL""",
            (room_id,),
        ).fetchone()
        if room is None:
            _raise_room_not_found(conn, room_id)
        if authority_scoped and (
            room["authority_gateway_id"] != normalized_authority_gateway_id
            or int(room["authority_epoch"]) != normalized_authority_epoch
        ):
            raise AuthorityConflictError("stale hosted room authority")
        seq = int(room["next_seq"])
        event_bytes = _event_storage_bytes(
            event_id=event_id,
            kind=kind,
            actor_json=actor_json,
            payload_json=payload_json,
        )
        _assert_event_capacity(
            conn,
            room=room,
            additional_bytes=event_bytes,
            allow_control=kind
            in {
                "authority.claimed",
                "authority.lost",
                "room.disbanded",
                "room.stop_requested",
            },
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                room_id,
                seq,
                event_id,
                kind,
                actor_json,
                normalized_authority_epoch,
                payload_json,
                now,
            ),
        )
        advanced = conn.execute(
            """UPDATE hosted_rooms
               SET next_seq=?, event_bytes=event_bytes+?, updated_at=?
               WHERE room_id=? AND next_seq=?""",
            (seq + 1, event_bytes, now, room_id, seq),
        )
        if advanced.rowcount != 1:
            raise RuntimeError("hosted room sequence advance lost its write fence")
        row = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
               FROM hosted_room_events WHERE room_id=? AND seq=?""",
            (room_id, seq),
        ).fetchone()
        if row is None:  # pragma: no cover - guarded by the insert above
            raise RuntimeError("appended event could not be reloaded")
    result = _event_from_row(row)
    result["actor"] = normalized_actor
    return result


def probe_hosted_room(db_path: Path | str, *, room_id: Any) -> bool:
    """Check room ownership without creating or migrating the shared store.

    This runs on the synchronous prompt-admission path for older Desktop
    clients, so it fails quickly under contention instead of blocking the
    WebSocket reader for SQLite's normal ten-second timeout.
    """

    checked_room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    path = Path(db_path)
    if not path.is_file():
        return False
    try:
        conn = sqlite3.connect(path, timeout=0.05)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='hosted_rooms' LIMIT 1"
            ).fetchone()
            if table is None:
                return False
            return (
                conn.execute(
                    "SELECT 1 FROM hosted_rooms WHERE room_id=? "
                    "AND disbanded_at IS NULL LIMIT 1",
                    (checked_room_id,),
                ).fetchone()
                is not None
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RoomProbeUnavailableError(
            "hosted room ownership is temporarily unavailable"
        ) from exc


def probe_peer_room_reservation(
    db_path: Path | str,
    *,
    room_id: Any,
    target_profile: Any,
    now: float | None = None,
) -> bool:
    """Check a peer reservation without creating or migrating shared state."""

    checked_room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    checked_profile = _validate_identifier(
        target_profile,
        label="target_profile",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    path = Path(db_path)
    if not path.is_file():
        return False
    checked_now = float(now if now is not None else time.time())
    try:
        conn = sqlite3.connect(path, timeout=0.05)
        try:
            table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='hosted_room_peer_reservations' LIMIT 1"
            ).fetchone()
            if table is None:
                return False
            return (
                conn.execute(
                    """SELECT 1 FROM hosted_room_peer_reservations
                         WHERE room_id=? AND target_profile=?
                           AND expires_at>? AND revoked_at IS NULL
                         LIMIT 1""",
                    (checked_room_id, checked_profile, checked_now),
                ).fetchone()
                is not None
            )
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise RoomProbeUnavailableError(
            "peer room ownership is temporarily unavailable"
        ) from exc


def room_state(
    db_path: Path | str,
    *,
    room_id: Any,
    include_disbanded: bool = False,
) -> dict[str, Any]:
    """Return durable replay and authority state for one room."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at, updated_at,
                      disbanded_at
                 FROM hosted_rooms
                WHERE room_id=? AND (disbanded_at IS NULL OR ?)""",
            (room_id, int(include_disbanded)),
        ).fetchone()
        if row is None:
            _raise_room_not_found(conn, room_id)
        claim_row = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
                 FROM hosted_room_events
                WHERE room_id=? AND kind='authority.claimed'
                  AND authority_epoch=?
                ORDER BY seq DESC LIMIT 1""",
            (room_id, int(row["authority_epoch"])),
        ).fetchone()
    state = _room_from_row(row)
    state["latest_seq"] = int(row["next_seq"]) - 1
    if claim_row is not None:
        state["authority_claim"] = _event_from_row(claim_row)
    return state


def request_room_stop(
    db_path: Path | str,
    *,
    room_id: Any,
    cancel_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
) -> dict[str, Any]:
    """Append an idempotent fence that supersedes earlier user turns."""

    cancel_id = _validate_identifier(
        cancel_id,
        label="cancel_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    digest = hashlib.sha256(cancel_id.encode()).hexdigest()[:32]
    return append_event(
        db_path,
        room_id=room_id,
        event_id=f"room-stop:{digest}",
        kind="room.stop_requested",
        actor={"kind": "gateway", "id": expected_gateway_id},
        payload={"cancel_id": cancel_id},
        authority_gateway_id=expected_gateway_id,
        authority_epoch=expected_epoch,
    )


def claim_authority(
    db_path: Path | str,
    *,
    room_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    new_gateway_id: Any,
    event_id: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Fence a verified authority transfer with a compare-and-swap epoch.

    This storage primitive does not decide *when* takeover is safe. A future
    replicated driver must call it only after its lease/quorum policy has
    established that the previous owner can no longer commit.
    """
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    expected_gateway_id = _validate_identifier(
        expected_gateway_id,
        label="expected_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    new_gateway_id = _validate_identifier(
        new_gateway_id,
        label="new_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    event_id = _validate_identifier(
        event_id,
        label="event_id",
        max_chars=MAX_EVENT_ID_CHARS,
    )
    if (
        isinstance(expected_epoch, bool)
        or not isinstance(expected_epoch, int)
        or expected_epoch < 1
    ):
        raise HostedRoomError("expected_epoch must be a positive integer")
    now = time.time() if now is None else float(now)
    target_epoch = expected_epoch + 1
    claim_actor = {"kind": "system", "id": "authority-control"}
    claim_actor_json = _canonical_json(
        claim_actor,
        label="actor",
        max_bytes=4 * 1024,
    )
    claim_payload = {
        "previous_gateway_id": expected_gateway_id,
        "authority_gateway_id": new_gateway_id,
        "authority_epoch": target_epoch,
    }
    claim_payload_json = _canonical_json(
        claim_payload,
        label="payload",
        max_bytes=MAX_EVENT_JSON_BYTES,
    )

    with _transaction(db_path, immediate=True) as conn:
        row = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, next_seq, event_bytes
                 FROM hosted_rooms
                WHERE room_id=? AND disbanded_at IS NULL""",
            (room_id,),
        ).fetchone()
        if row is None:
            _raise_room_not_found(conn, room_id)
        current_gateway = str(row["authority_gateway_id"])
        current_epoch = int(row["authority_epoch"])
        existing_event = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
                 FROM hosted_room_events WHERE room_id=? AND event_id=?""",
            (room_id, event_id),
        ).fetchone()
        if existing_event is not None:
            if (
                existing_event["kind"] != "authority.claimed"
                or existing_event["actor_json"] != claim_actor_json
                or existing_event["authority_epoch"] != target_epoch
                or existing_event["payload_json"] != claim_payload_json
            ):
                raise EventConflictError(
                    "event_id already exists with different content"
                )
            if current_gateway != new_gateway_id or current_epoch != target_epoch:
                raise AuthoritySupersededError(
                    "authority claim succeeded but was later superseded"
                )
            idempotent = True
        elif current_gateway != expected_gateway_id or current_epoch != expected_epoch:
            raise AuthorityConflictError("hosted room authority changed")
        else:
            seq = int(row["next_seq"])
            claim_bytes = _event_storage_bytes(
                event_id=event_id,
                kind="authority.claimed",
                actor_json=claim_actor_json,
                payload_json=claim_payload_json,
            )
            _assert_event_capacity(
                conn,
                room=row,
                additional_bytes=claim_bytes,
                allow_control=True,
            )
            conn.execute(
                """INSERT INTO hosted_room_events
                   (room_id, seq, event_id, kind, actor_json, authority_epoch,
                    payload_json, created_at)
                   VALUES (?, ?, ?, 'authority.claimed', ?, ?, ?, ?)""",
                (
                    room_id,
                    seq,
                    event_id,
                    claim_actor_json,
                    target_epoch,
                    claim_payload_json,
                    now,
                ),
            )
            updated = conn.execute(
                """UPDATE hosted_rooms
                      SET authority_gateway_id=?, authority_epoch=authority_epoch+1,
                          next_seq=next_seq+1, event_bytes=event_bytes+?,
                          revision=revision+1, updated_at=?
                    WHERE room_id=? AND disbanded_at IS NULL
                      AND authority_gateway_id=? AND authority_epoch=?""",
                (
                    new_gateway_id,
                    claim_bytes,
                    now,
                    room_id,
                    expected_gateway_id,
                    expected_epoch,
                ),
            )
            if updated.rowcount != 1:
                raise AuthorityConflictError("hosted room authority changed")
            idempotent = False
            existing_event = conn.execute(
                """SELECT room_id, seq, event_id, kind, actor_json,
                          authority_epoch, payload_json, created_at
                     FROM hosted_room_events WHERE room_id=? AND event_id=?""",
                (room_id, event_id),
            ).fetchone()
        state_row = conn.execute(
            """SELECT room_id, name, members_json, authority_gateway_id,
                      authority_epoch, next_seq, revision, created_at, updated_at
                 FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if state_row is None:  # pragma: no cover - room exists in this transaction
            raise RuntimeError("claimed room could not be reloaded")
    state = _room_from_row(state_row, idempotent=idempotent)
    state["latest_seq"] = int(state_row["next_seq"]) - 1
    if existing_event is None:  # pragma: no cover - both claim paths set it
        raise RuntimeError("authority claim event could not be reloaded")
    state["claim_event"] = _event_from_row(
        existing_event,
        idempotent=idempotent,
    )
    return state


def disband_room(
    db_path: Path | str,
    *,
    room_id: Any,
    expected_gateway_id: Any,
    expected_epoch: Any,
    now: float | None = None,
) -> dict[str, Any]:
    """Tombstone a room id permanently and idempotently."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    expected_gateway_id = _validate_identifier(
        expected_gateway_id,
        label="expected_gateway_id",
        max_chars=MAX_ACTOR_ID_CHARS,
    )
    if (
        isinstance(expected_epoch, bool)
        or not isinstance(expected_epoch, int)
        or expected_epoch < 1
    ):
        raise HostedRoomError("expected_epoch must be a positive integer")
    now = time.time() if now is None else float(now)

    with _transaction(db_path, immediate=True) as conn:
        room = conn.execute(
            """SELECT authority_gateway_id, authority_epoch, next_seq,
                      event_bytes, disbanded_at
                 FROM hosted_rooms WHERE room_id=?""",
            (room_id,),
        ).fetchone()
        if room is None:
            retired = conn.execute(
                "SELECT retired_at FROM hosted_room_retired_ids WHERE room_id=?",
                (room_id,),
            ).fetchone()
            if retired is None:
                raise RoomNotFoundError("hosted room not found")
            return {
                "room_id": room_id,
                "disbanded_at": float(retired["retired_at"]),
                "idempotent": True,
                "history_expired": True,
            }
        if room["disbanded_at"] is not None:
            conn.execute(
                """INSERT OR IGNORE INTO hosted_room_retired_ids
                   (room_id, retired_at) VALUES (?, ?)""",
                (room_id, float(room["disbanded_at"])),
            )
            event = conn.execute(
                """SELECT room_id, seq, event_id, kind, actor_json,
                          authority_epoch, payload_json, created_at
                     FROM hosted_room_events
                    WHERE room_id=? AND event_id='system:room-disbanded'""",
                (room_id,),
            ).fetchone()
            return {
                "room_id": room_id,
                "disbanded_at": float(room["disbanded_at"]),
                "idempotent": True,
                **(
                    {"event": _event_from_row(event, idempotent=True)}
                    if event is not None
                    else {}
                ),
            }
        if (
            str(room["authority_gateway_id"]) != expected_gateway_id
            or int(room["authority_epoch"]) != expected_epoch
        ):
            raise AuthorityConflictError("stale hosted room authority")
        seq = int(room["next_seq"])
        actor_json = _canonical_json(
            {"kind": "system", "id": "room-control"},
            label="actor",
            max_bytes=4 * 1024,
        )
        payload_json = _canonical_json(
            {"room_id": room_id},
            label="payload",
            max_bytes=MAX_EVENT_JSON_BYTES,
        )
        disband_bytes = _event_storage_bytes(
            event_id="system:room-disbanded",
            kind="room.disbanded",
            actor_json=actor_json,
            payload_json=payload_json,
        )
        _assert_event_capacity(
            conn,
            room=room,
            additional_bytes=disband_bytes,
            allow_control=True,
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES (?, ?, 'system:room-disbanded', 'room.disbanded', ?, ?, ?, ?)""",
            (
                room_id,
                seq,
                actor_json,
                int(room["authority_epoch"]),
                payload_json,
                now,
            ),
        )
        updated = conn.execute(
            """UPDATE hosted_rooms
               SET disbanded_at=?, updated_at=?, revision=revision+1,
                   next_seq=next_seq+1, event_bytes=event_bytes+?
               WHERE room_id=? AND disbanded_at IS NULL
                 AND authority_gateway_id=? AND authority_epoch=?""",
            (
                now,
                now,
                disband_bytes,
                room_id,
                expected_gateway_id,
                expected_epoch,
            ),
        )
        if updated.rowcount != 1:
            raise RoomConflictError("hosted room disband lost its fence")
        conn.execute(
            """INSERT OR IGNORE INTO hosted_room_retired_ids
               (room_id, retired_at) VALUES (?, ?)""",
            (room_id, now),
        )
        event = conn.execute(
            """SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM hosted_room_events
                WHERE room_id=? AND event_id='system:room-disbanded'""",
            (room_id,),
        ).fetchone()
        if event is None:  # pragma: no cover - inserted in this transaction
            raise RuntimeError("room disband event could not be reloaded")
        _prune_disbanded_rooms_locked(
            conn,
            now=now,
            max_gateway_event_bytes=MAX_GATEWAY_EVENT_BYTES,
        )
    return {
        "room_id": room_id,
        "disbanded_at": now,
        "idempotent": False,
        "event": _event_from_row(event),
    }


def read_events(
    db_path: Path | str,
    *,
    room_id: Any,
    since_seq: Any = 0,
    limit: Any = 100,
    include_disbanded: bool = False,
) -> dict[str, Any]:
    """Read a monotonic room-log delta after ``since_seq``."""
    room_id = _validate_identifier(
        room_id,
        label="room_id",
        max_chars=MAX_ROOM_ID_CHARS,
    )
    if isinstance(since_seq, bool) or not isinstance(since_seq, int) or since_seq < 0:
        raise HostedRoomError("since_seq must be a non-negative integer")
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_LOG_LIMIT
    ):
        raise HostedRoomError(f"limit must be between 1 and {MAX_LOG_LIMIT}")

    with _transaction(db_path) as conn:
        room = conn.execute(
            """SELECT next_seq, authority_gateway_id, authority_epoch
               FROM hosted_rooms
               WHERE room_id=? AND (disbanded_at IS NULL OR ?)""",
            (room_id, int(include_disbanded)),
        ).fetchone()
        if room is None:
            _raise_room_not_found(conn, room_id)
        latest_seq = int(room["next_seq"]) - 1
        authority_gateway = str(room["authority_gateway_id"])
        authority_epoch = int(room["authority_epoch"])
        if since_seq > latest_seq:
            raise HostedRoomError("since_seq is ahead of the hosted room log")
        rows = conn.execute(
            """WITH candidates AS (
                   SELECT room_id, seq, event_id, kind, actor_json,
                          authority_epoch, payload_json, created_at,
                          SUM(
                              LENGTH(CAST(event_id AS BLOB)) +
                              LENGTH(CAST(kind AS BLOB)) +
                              LENGTH(CAST(actor_json AS BLOB)) +
                              LENGTH(CAST(payload_json AS BLOB))
                          ) OVER (ORDER BY seq ASC) AS cumulative_bytes
                     FROM hosted_room_events
                    WHERE room_id=? AND seq>?
                    ORDER BY seq ASC LIMIT ?
               )
               SELECT room_id, seq, event_id, kind, actor_json,
                      authority_epoch, payload_json, created_at
                 FROM candidates
                WHERE cumulative_bytes<=?
                ORDER BY seq ASC""",
            (room_id, since_seq, limit, MAX_LOG_PAGE_BYTES),
        ).fetchall()
    events = [_event_from_row(row) for row in rows]

    def build_page(page_events: list[dict[str, Any]]) -> dict[str, Any]:
        cursor = page_events[-1]["seq"] if page_events else since_seq
        return {
            "events": page_events,
            "cursor": cursor,
            "latest_seq": latest_seq,
            "has_more": cursor < latest_seq,
            "authority": {
                "gateway_id": authority_gateway,
                "epoch": authority_epoch,
            },
        }

    def page_bytes(page: dict[str, Any]) -> int:
        return len(
            json.dumps(
                page,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    page = build_page(events)
    if events and page_bytes(page) > MAX_LOG_PAGE_BYTES:
        low, high = 1, len(events)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = build_page(events[:middle])
            if page_bytes(candidate) <= MAX_LOG_PAGE_BYTES:
                low = middle
            else:
                high = middle - 1
        page = build_page(events[:low])
        if page_bytes(page) > MAX_LOG_PAGE_BYTES:
            raise HostedRoomError("hosted room event exceeds replay page limit")
    return page
