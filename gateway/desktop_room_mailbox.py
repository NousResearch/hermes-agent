"""Durable command handoff for Desktop-driven Bot rooms.

Messaging adapters run on the gateway while classic room orchestration lives
in a connected Desktop renderer. This mailbox keeps that compatibility path
idempotent and recoverable without making the gateway a second room runner.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


MAX_ROOM_IDS = 128
MAX_QUERY_ROOM_IDS = 4096
MAX_COMMANDS_PER_CLAIM = 8
MAX_AUTOMATIC_SEND_CLAIMS = 2
MAX_PAYLOAD_BYTES = 64 * 1024
# Desktop refreshes classic-room presence on a 60s retained-socket backstop;
# push events handle command latency. Keep enough overlap for scheduler jitter
# without turning a closed Desktop into a long-lived false-positive.
PRESENCE_TTL_SECONDS = 90.0
CLAIM_TTL_SECONDS = 45.0
PENDING_TTL_SECONDS = 24 * 60 * 60
TERMINAL_RETENTION_SECONDS = 7 * 24 * 60 * 60
MAX_PENDING_COMMANDS_PER_ROOM = 64
MAX_PENDING_COMMANDS_TOTAL = 4096
MAX_RETRY_COMMANDS = 32

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_AUTHORITY_HASH_RE = re.compile(r"^[a-f0-9]{64}$")
_ACTIONS = frozenset({"send", "stop"})
_TERMINAL_STATES = frozenset({"completed", "failed"})


class DesktopRoomMailboxError(ValueError):
    """Raised when a mailbox command is invalid or stale."""


def default_db_path() -> Path:
    """Keep compatibility heartbeats out of the session state database."""

    from gateway.hosted_rooms import default_db_path as hosted_db_path

    return hosted_db_path().with_name("desktop_room_mailbox.db")


def pending_signal_path(db_path: Path | str | None = None) -> Path:
    """Cross-process change signal watched by the gateway WebSocket server."""

    return Path(db_path or default_db_path()).with_name("desktop_room_mailbox.pending")


def _identifier(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > 160 or not _IDENTIFIER_RE.fullmatch(text):
        raise DesktopRoomMailboxError(f"invalid {label}")
    return text


def _room_identifier(value: Any) -> str:
    text = str(value or "").strip()
    if (
        not text
        or len(text) > 200
        or any(char in text for char in ("\x00", "\r", "\n"))
    ):
        raise DesktopRoomMailboxError("invalid room_id")
    return text


def _room_ids(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise DesktopRoomMailboxError("room_ids must be a list")
    if len(value) > MAX_ROOM_IDS:
        raise DesktopRoomMailboxError("too many room_ids")
    return list(dict.fromkeys(_room_identifier(item) for item in value))


def _room_authorities(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, (list, tuple)):
        raise DesktopRoomMailboxError("room_authorities must be a list")
    if len(value) > MAX_ROOM_IDS:
        raise DesktopRoomMailboxError("too many room authorities")
    authorities: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            raise DesktopRoomMailboxError("invalid room authority")
        room_id = _room_identifier(item.get("room_id"))
        token = _identifier(item.get("authority_token"), label="authority_token")
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        prior = authorities.setdefault(room_id, digest)
        if prior != digest:
            raise DesktopRoomMailboxError("conflicting room authority")
    return list(authorities.items())


def _authority_hash(value: Any) -> str:
    authority_hash = str(value or "").strip().casefold()
    if not _AUTHORITY_HASH_RE.fullmatch(authority_hash):
        raise DesktopRoomMailboxError("invalid room authority commitment")
    return authority_hash


def _room_commitments(value: Any) -> list[tuple[str, str]]:
    if not isinstance(value, (list, tuple)):
        raise DesktopRoomMailboxError("room commitments must be a list")
    if len(value) > MAX_ROOM_IDS:
        raise DesktopRoomMailboxError("too many room commitments")
    commitments: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            raise DesktopRoomMailboxError("invalid room commitment")
        room_id = _room_identifier(item.get("room_id"))
        authority_hash = _authority_hash(item.get("authority_hash"))
        prior = commitments.setdefault(room_id, authority_hash)
        if prior != authority_hash:
            raise DesktopRoomMailboxError("conflicting room commitment")
    return list(commitments.items())


def _query_room_ids(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise DesktopRoomMailboxError("room_ids must be a list")
    if len(value) > MAX_QUERY_ROOM_IDS:
        raise DesktopRoomMailboxError("too many room_ids")
    return list(dict.fromkeys(_room_identifier(item) for item in value))


def _payload_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, RecursionError) as exc:
        raise DesktopRoomMailboxError("payload must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise DesktopRoomMailboxError("payload is too large")
    return encoded


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS desktop_room_commands (
            command_id TEXT PRIMARY KEY,
            room_id TEXT NOT NULL,
            action TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            lease_owner TEXT,
            lease_token TEXT,
            lease_expires_at REAL,
            attempts INTEGER NOT NULL DEFAULT 0,
            result_json TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )"""
    )
    command_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(desktop_room_commands)")
    }
    if "lease_token" not in command_columns:
        conn.execute("ALTER TABLE desktop_room_commands ADD COLUMN lease_token TEXT")
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_desktop_room_commands_claim
           ON desktop_room_commands(state, room_id, created_at)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS desktop_room_presence (
            consumer_id TEXT NOT NULL,
            room_id TEXT NOT NULL,
            expires_at REAL NOT NULL,
            PRIMARY KEY (consumer_id, room_id)
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_desktop_room_presence_room
           ON desktop_room_presence(room_id, expires_at)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS desktop_room_owners (
            room_id TEXT PRIMARY KEY,
            consumer_id TEXT NOT NULL,
            expires_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE INDEX IF NOT EXISTS idx_desktop_room_owners_expiry
           ON desktop_room_owners(expires_at)"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS desktop_room_authorities (
            room_id TEXT PRIMARY KEY,
            consumer_id TEXT,
            authority_hash TEXT NOT NULL,
            created_at REAL NOT NULL
        )"""
    )
    authority_columns = {
        row[1] for row in conn.execute("PRAGMA table_info(desktop_room_authorities)")
    }
    if "consumer_id" not in authority_columns:
        conn.execute("ALTER TABLE desktop_room_authorities ADD COLUMN consumer_id TEXT")


def _schema_is_current(conn: sqlite3.Connection) -> bool:
    commands = {
        row[1] for row in conn.execute("PRAGMA table_info(desktop_room_commands)")
    }
    presence = {
        row[1] for row in conn.execute("PRAGMA table_info(desktop_room_presence)")
    }
    owners = {row[1] for row in conn.execute("PRAGMA table_info(desktop_room_owners)")}
    authorities = {
        row[1] for row in conn.execute("PRAGMA table_info(desktop_room_authorities)")
    }
    return (
        {
            "command_id",
            "room_id",
            "action",
            "payload_json",
            "state",
            "lease_owner",
            "lease_token",
            "lease_expires_at",
            "attempts",
            "result_json",
            "created_at",
            "updated_at",
        }.issubset(commands)
        and {"consumer_id", "room_id", "expires_at"}.issubset(presence)
        and {"room_id", "consumer_id", "expires_at"}.issubset(owners)
        and {"room_id", "consumer_id", "authority_hash", "created_at"}.issubset(
            authorities
        )
    )


def _connect(db_path: Path | str) -> sqlite3.Connection:
    from hermes_state import apply_wal_with_fallback

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        apply_wal_with_fallback(conn, db_label="state.db (desktop room mailbox)")
        if _schema_is_current(conn):
            return conn
        conn.execute("BEGIN IMMEDIATE")
        _initialize(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        conn.close()
        raise
    return conn


def _notify_pending(db_path: Path | str) -> None:
    path = pending_signal_path(db_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(time.time_ns()), encoding="ascii")
        os.chmod(path, 0o600)
    except OSError:
        # The command is already durable. A failed best-effort wake signal
        # must never turn success into an error or invite a duplicate send;
        # the retained-socket poll remains the backstop.
        pass


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


def _command(row: sqlite3.Row, *, idempotent: bool = False) -> dict[str, Any]:
    result = {
        "command_id": str(row["command_id"]),
        "room_id": str(row["room_id"]),
        "action": str(row["action"]),
        "payload": json.loads(row["payload_json"]),
        "state": str(row["state"]),
        "attempts": int(row["attempts"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "idempotent": idempotent,
    }
    if row["result_json"]:
        result["result"] = json.loads(row["result_json"])
    if row["lease_token"]:
        result["lease_token"] = str(row["lease_token"])
    return result


def _expire_stale_state(
    conn: sqlite3.Connection,
    *,
    now: float,
    pending_ttl: float = PENDING_TTL_SECONDS,
) -> None:
    """Bound offline work and remove expired ownership before every write path."""

    conn.execute("DELETE FROM desktop_room_presence WHERE expires_at <= ?", (now,))
    conn.execute("DELETE FROM desktop_room_owners WHERE expires_at <= ?", (now,))
    cutoff = now - max(1.0, float(pending_ttl))
    expired_result = _payload_json({
        "code": "command_expired",
        "message": "This Group Chat command expired before Desktop could apply it.",
    })
    conn.execute(
        """UPDATE desktop_room_commands
           SET state = 'failed', result_json = ?, lease_owner = NULL,
               lease_token = NULL, lease_expires_at = NULL, updated_at = ?
           WHERE state = 'pending' AND created_at <= ?""",
        (expired_result, now, cutoff),
    )
    conn.execute(
        """UPDATE desktop_room_commands
           SET state = 'failed', result_json = ?, lease_owner = NULL,
               lease_token = NULL, lease_expires_at = NULL, updated_at = ?
           WHERE state = 'claimed' AND created_at <= ?
             AND COALESCE(lease_expires_at, 0) <= ?""",
        (expired_result, now, cutoff, now),
    )
    exhausted_result = _payload_json({
        "code": "automatic_attempts_exhausted",
        "message": "This Group Chat command needs an explicit Retry.",
    })
    conn.execute(
        """UPDATE desktop_room_commands
           SET state='failed', result_json=?, lease_owner=NULL,
               lease_token=NULL, lease_expires_at=NULL, updated_at=?
           WHERE action='send' AND attempts>=?
             AND (
                 state='pending'
                 OR (state='claimed' AND COALESCE(lease_expires_at, 0)<=?)
             )""",
        (exhausted_result, now, MAX_AUTOMATIC_SEND_CLAIMS, now),
    )
    conn.execute(
        """DELETE FROM desktop_room_commands
           WHERE state IN ('completed', 'failed') AND updated_at <= ?""",
        (now - TERMINAL_RETENTION_SECONDS,),
    )


def _owned_rooms(
    conn: sqlite3.Connection,
    *,
    consumer_id: str,
    authorities: list[tuple[str, str]],
    now: float,
    presence_ttl: float,
) -> list[str]:
    """Bind or safely transfer rooms when the prior Desktop lease is gone."""

    candidates: list[str] = []
    for room_id, authority_hash in authorities:
        authority = conn.execute(
            """SELECT consumer_id, authority_hash
               FROM desktop_room_authorities WHERE room_id = ?""",
            (room_id,),
        ).fetchone()
        if (
            authority is None
            or str(authority["authority_hash"] or "") != authority_hash
        ):
            continue
        bound_consumer = str(authority["consumer_id"] or "")
        if bound_consumer and bound_consumer != consumer_id:
            live_owner = conn.execute(
                """SELECT 1 FROM desktop_room_owners
                   WHERE room_id = ? AND consumer_id = ? AND expires_at > ?""",
                (room_id, bound_consumer, now),
            ).fetchone()
            live_claim = conn.execute(
                """SELECT 1 FROM desktop_room_commands
                   WHERE room_id = ? AND state = 'claimed'
                     AND lease_owner = ? AND lease_expires_at > ? LIMIT 1""",
                (room_id, bound_consumer, now),
            ).fetchone()
            if live_owner is not None or live_claim is not None:
                continue
        conn.execute(
            """UPDATE desktop_room_authorities SET consumer_id = ?
               WHERE room_id = ? AND authority_hash = ?""",
            (consumer_id, room_id, authority_hash),
        )
        candidates.append(room_id)

    if candidates:
        conn.executemany(
            """INSERT INTO desktop_room_owners (
                   room_id, consumer_id, expires_at
               ) VALUES (?, ?, ?)
               ON CONFLICT(room_id) DO UPDATE
               SET consumer_id = excluded.consumer_id,
                   expires_at = excluded.expires_at
               WHERE desktop_room_owners.consumer_id = excluded.consumer_id
                  OR desktop_room_owners.expires_at <= ?""",
            (
                (room_id, consumer_id, now + float(presence_ttl), now)
                for room_id in candidates
            ),
        )
    owned: list[str] = []
    if candidates:
        placeholders = ",".join("?" for _ in candidates)
        owned = [
            str(row["room_id"])
            for row in conn.execute(
                f"""SELECT room_id FROM desktop_room_owners
                    WHERE room_id IN ({placeholders}) AND consumer_id = ?
                      AND expires_at > ?""",
                (*candidates, consumer_id, now),
            )
        ]
    if owned:
        conn.executemany(
            """INSERT INTO desktop_room_presence (
                   consumer_id, room_id, expires_at
               ) VALUES (?, ?, ?)
               ON CONFLICT(consumer_id, room_id) DO UPDATE
               SET expires_at = excluded.expires_at""",
            ((consumer_id, room_id, now + float(presence_ttl)) for room_id in owned),
        )
    return owned


def register_projected_authorities(
    db_path: Path | str,
    commitments: Any,
    *,
    clock: Any = time.time,
) -> list[str]:
    """Record one-way owner proofs read from the trusted room projection.

    Conflicts are isolated per room: an old or corrupted projection cannot
    prevent healthy rooms in the same snapshot from advertising presence.
    """

    parsed = _room_commitments(commitments)
    now = float(clock())
    registered: list[str] = []
    with _transaction(db_path, immediate=True) as conn:
        for room_id, authority_hash in parsed:
            existing = conn.execute(
                """SELECT authority_hash FROM desktop_room_authorities
                   WHERE room_id = ?""",
                (room_id,),
            ).fetchone()
            if existing is None:
                conn.execute(
                    """INSERT INTO desktop_room_authorities (
                           room_id, consumer_id, authority_hash, created_at
                       ) VALUES (?, NULL, ?, ?)""",
                    (room_id, authority_hash, now),
                )
                registered.append(room_id)
            elif str(existing["authority_hash"] or "") == authority_hash:
                registered.append(room_id)
    return registered


def enqueue_command(
    db_path: Path | str,
    *,
    command_id: str,
    room_id: str,
    authority_hash: str,
    action: str,
    payload: Any,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Persist one idempotent command for a compatible Desktop."""

    command_id = _identifier(command_id, label="command_id")
    room_id = _room_identifier(room_id)
    authority_hash = _authority_hash(authority_hash)
    action = str(action or "").strip().casefold()
    if action not in _ACTIONS:
        raise DesktopRoomMailboxError("invalid action")
    encoded = _payload_json(payload)
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        _expire_stale_state(conn, now=now)
        result = _persist_command(
            conn, command_id=command_id, room_id=room_id, authority_hash=authority_hash,
            action=action, encoded=encoded, now=now,
        )
    _notify_pending(db_path)
    return result


def _persist_command(conn, *, command_id, room_id, authority_hash, action, encoded, now):
    existing_authority = conn.execute(
        """SELECT authority_hash FROM desktop_room_authorities
           WHERE room_id = ?""",
        (room_id,),
    ).fetchone()
    if existing_authority is None:
        conn.execute(
            """INSERT INTO desktop_room_authorities (
                   room_id, consumer_id, authority_hash, created_at
               ) VALUES (?, NULL, ?, ?)""",
            (room_id, authority_hash, now),
        )
    elif str(existing_authority["authority_hash"] or "") != authority_hash:
        raise DesktopRoomMailboxError(
            "room authority commitment does not match its existing owner"
        )
    existing = conn.execute(
        "SELECT * FROM desktop_room_commands WHERE command_id = ?",
        (command_id,),
    ).fetchone()
    if existing is not None:
        if (
            str(existing["room_id"]) != room_id
            or str(existing["action"]) != action
            or str(existing["payload_json"]) != encoded
        ):
            raise DesktopRoomMailboxError(
                "command_id was already used for different room work"
            )
        result = _command(existing, idempotent=True)
    else:
        if action == "stop":
            superseded_result = _payload_json({
                "code": "superseded_by_stop",
                "message": "Canceled before Desktop started it.",
            })
            conn.execute(
                """UPDATE desktop_room_commands
                   SET state = 'failed', result_json = ?, lease_owner = NULL,
                       lease_token = NULL, lease_expires_at = NULL, updated_at = ?
                   WHERE room_id = ? AND action IN ('send', 'stop')
                     AND state = 'pending'""",
                (superseded_result, now, room_id),
            )
        else:
            room_count = conn.execute(
                """SELECT COUNT(*) FROM desktop_room_commands
                   WHERE room_id = ? AND state IN ('pending', 'claimed')""",
                (room_id,),
            ).fetchone()[0]
            total_count = conn.execute(
                """SELECT COUNT(*) FROM desktop_room_commands
                   WHERE state IN ('pending', 'claimed')"""
            ).fetchone()[0]
            if int(room_count) >= MAX_PENDING_COMMANDS_PER_ROOM:
                raise DesktopRoomMailboxError(
                    "This Group Chat already has too many commands waiting for Desktop."
                )
            if int(total_count) >= MAX_PENDING_COMMANDS_TOTAL:
                raise DesktopRoomMailboxError(
                    "Too many Group Chat commands are waiting for Desktop."
                )
        conn.execute(
            """INSERT INTO desktop_room_commands (
                   command_id, room_id, action, payload_json, state,
                   created_at, updated_at
               ) VALUES (?, ?, ?, ?, 'pending', ?, ?)""",
            (command_id, room_id, action, encoded, now, now),
        )
        row = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        result = _command(row)
    return result


def enqueue_stop_command(
    db_path: Path | str, *, command_id: str, room_id: str, authority_hash: str,
    target_thread_id: str = "", target_message_id: str = "", clock: Any = time.time,
) -> dict[str, Any]:
    """Freeze the first Stop target and replay duplicate deliveries atomically."""
    command_id = _identifier(command_id, label="command_id")
    room_id = _room_identifier(room_id)
    authority_hash = _authority_hash(authority_hash)
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        _expire_stale_state(conn, now=now)
        existing = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id=?", (command_id,),
        ).fetchone()
        if existing is not None:
            encoded = str(existing["payload_json"])
        else:
            current = conn.execute(
                """SELECT command_id, action, state FROM desktop_room_commands
                   WHERE room_id=? ORDER BY created_at DESC, command_id DESC LIMIT 1""",
                (room_id,),
            ).fetchone()
            target_command_id = (
                str(current["command_id"])
                if current is not None and current["action"] == "send"
                and current["state"] in {"claimed", "pending"} else ""
            )
            if not target_command_id and not target_message_id:
                raise DesktopRoomMailboxError(
                    "Open this Group Chat in Hermes Desktop to Stop it safely."
                )
            encoded = _payload_json({
                **({"target_command_id": target_command_id} if target_command_id else {}),
                **({"target_thread_id": target_thread_id} if target_thread_id else {}),
                **({"target_message_id": target_message_id}
                   if not target_command_id and target_message_id else {}),
            })
        result = _persist_command(
            conn, command_id=command_id, room_id=room_id, authority_hash=authority_hash,
            action="stop", encoded=encoded, now=now,
        )
    _notify_pending(db_path)
    return result


def claim_commands(
    db_path: Path | str,
    *,
    consumer_id: str,
    room_authorities: Any,
    actions: Any = None,
    limit: int = MAX_COMMANDS_PER_CLAIM,
    presence_ttl: float = PRESENCE_TTL_SECONDS,
    claim_ttl: float = CLAIM_TTL_SECONDS,
    clock: Any = time.time,
) -> list[dict[str, Any]]:
    """Refresh room presence and lease pending commands to one Desktop."""

    consumer_id = _identifier(consumer_id, label="consumer_id")
    authorities = _room_authorities(room_authorities)
    requested_actions = (
        {str(action or "").strip().casefold() for action in actions}
        if isinstance(actions, (list, tuple, set, frozenset))
        else set()
    )
    if requested_actions and not requested_actions.issubset(_ACTIONS):
        raise DesktopRoomMailboxError("invalid action filter")
    limit = max(1, min(MAX_COMMANDS_PER_CLAIM, int(limit)))
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        _expire_stale_state(conn, now=now)
        owned = _owned_rooms(
            conn,
            consumer_id=consumer_id,
            authorities=authorities,
            now=now,
            presence_ttl=presence_ttl,
        )
        if not owned:
            return []
        placeholders = ",".join("?" for _ in owned)
        action_sql = ""
        action_params: tuple[str, ...] = ()
        if requested_actions:
            action_placeholders = ",".join("?" for _ in requested_actions)
            action_sql = f" AND command.action IN ({action_placeholders})"
            action_params = tuple(sorted(requested_actions))
        rows = conn.execute(
            f"""SELECT command.* FROM desktop_room_commands AS command
                WHERE command.room_id IN ({placeholders})
                  {action_sql}
                  AND (
                    command.state = 'pending'
                    OR (command.state = 'claimed' AND COALESCE(command.lease_expires_at, 0) <= ?)
                  )
                ORDER BY CASE command.action WHEN 'stop' THEN 0 ELSE 1 END,
                         command.created_at, command.command_id
                LIMIT ?""",
            (*owned, *action_params, now, limit),
        ).fetchall()
        claimed: list[dict[str, Any]] = []
        for row in rows:
            lease_token = secrets.token_hex(16)
            updated = conn.execute(
                """UPDATE desktop_room_commands
                   SET state = 'claimed', lease_owner = ?, lease_token = ?,
                       lease_expires_at = ?, attempts = attempts + 1,
                       updated_at = ?
                   WHERE command_id = ?
                     AND (
                       state = 'pending'
                       OR (state = 'claimed' AND COALESCE(lease_expires_at, 0) <= ?)
                     )""",
                (
                    consumer_id,
                    lease_token,
                    now + float(claim_ttl),
                    now,
                    str(row["command_id"]),
                    now,
                ),
            )
            if updated.rowcount != 1:
                continue
            current = conn.execute(
                "SELECT * FROM desktop_room_commands WHERE command_id = ?",
                (str(row["command_id"]),),
            ).fetchone()
            command = _command(current)
            if str(current["action"]) == "stop":
                target_command_id = str(
                    command.get("payload", {}).get("target_command_id") or ""
                )
                if target_command_id:
                    target = conn.execute(
                        "SELECT state, result_json FROM desktop_room_commands WHERE command_id = ?",
                        (target_command_id,),
                    ).fetchone()
                    if target is not None:
                        command["target_command_state"] = str(target["state"])
                        target_result = (
                            json.loads(target["result_json"])
                            if target["result_json"]
                            else {}
                        )
                        if isinstance(target_result, dict) and target_result.get(
                            "code"
                        ):
                            command["target_result_code"] = str(target_result["code"])
            claimed.append(command)
        return claimed


def refresh_presence(
    db_path: Path | str,
    *,
    consumer_id: str,
    room_authorities: Any,
    presence_ttl: float = PRESENCE_TTL_SECONDS,
    clock: Any = time.time,
) -> list[str]:
    """Renew classic Group Chat ownership without leasing queued commands."""

    consumer_id = _identifier(consumer_id, label="consumer_id")
    authorities = _room_authorities(room_authorities)
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        _expire_stale_state(conn, now=now)
        return _owned_rooms(
            conn,
            consumer_id=consumer_id,
            authorities=authorities,
            now=now,
            presence_ttl=presence_ttl,
        )


def complete_command(
    db_path: Path | str,
    *,
    consumer_id: str,
    command_id: str,
    lease_token: str,
    success: bool,
    result: Any,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Commit one claimed command result, tolerating an ACK retry."""

    consumer_id = _identifier(consumer_id, label="consumer_id")
    command_id = _identifier(command_id, label="command_id")
    lease_token = _identifier(lease_token, label="lease_token")
    encoded = _payload_json(result)
    state = "completed" if success else "failed"
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        row = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        if row is None:
            raise DesktopRoomMailboxError("command not found")
        if str(row["state"]) in _TERMINAL_STATES:
            if str(row["state"]) == state and str(row["result_json"] or "") == encoded:
                return _command(row, idempotent=True)
            raise DesktopRoomMailboxError("command already has a different result")
        if (
            str(row["state"]) != "claimed"
            or str(row["lease_owner"] or "") != consumer_id
            or str(row["lease_token"] or "") != lease_token
            or float(row["lease_expires_at"] or 0) <= now
        ):
            raise DesktopRoomMailboxError(
                "command lease is no longer owned by this Desktop"
            )
        updated = conn.execute(
            """UPDATE desktop_room_commands
               SET state = ?, result_json = ?, lease_owner = NULL,
                   lease_token = NULL, lease_expires_at = NULL, updated_at = ?
               WHERE command_id = ? AND state = 'claimed' AND lease_owner = ?
                 AND lease_token = ? AND lease_expires_at > ?""",
            (state, encoded, now, command_id, consumer_id, lease_token, now),
        )
        if updated.rowcount != 1:
            raise DesktopRoomMailboxError("command completion raced another consumer")
        current = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        return _command(current)


def renew_command(
    db_path: Path | str,
    *,
    consumer_id: str,
    command_id: str,
    lease_token: str,
    claim_ttl: float = CLAIM_TTL_SECONDS,
    presence_ttl: float = PRESENCE_TTL_SECONDS,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Extend one live claim without allowing an expired attempt to revive."""

    consumer_id = _identifier(consumer_id, label="consumer_id")
    command_id = _identifier(command_id, label="command_id")
    lease_token = _identifier(lease_token, label="lease_token")
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        row = conn.execute(
            """SELECT room_id FROM desktop_room_commands
               WHERE command_id = ? AND state = 'claimed'
                 AND lease_owner = ? AND lease_token = ?
                 AND lease_expires_at > ?""",
            (command_id, consumer_id, lease_token, now),
        ).fetchone()
        if row is None:
            raise DesktopRoomMailboxError(
                "command lease is no longer owned by this Desktop"
            )
        room_id = str(row["room_id"])
        owner = conn.execute(
            """UPDATE desktop_room_owners
               SET expires_at = ?
               WHERE room_id = ? AND consumer_id = ? AND expires_at > ?""",
            (now + float(presence_ttl), room_id, consumer_id, now),
        )
        if owner.rowcount != 1:
            raise DesktopRoomMailboxError(
                "room authority is no longer owned by this Desktop"
            )
        updated = conn.execute(
            """UPDATE desktop_room_commands
               SET lease_expires_at = ?, updated_at = ?
               WHERE command_id = ? AND state = 'claimed'
                 AND lease_owner = ? AND lease_token = ?
                 AND lease_expires_at > ?""",
            (
                now + float(claim_ttl),
                now,
                command_id,
                consumer_id,
                lease_token,
                now,
            ),
        )
        if updated.rowcount != 1:
            raise DesktopRoomMailboxError(
                "command lease is no longer owned by this Desktop"
            )
        conn.execute(
            """INSERT INTO desktop_room_presence (
                   consumer_id, room_id, expires_at
               ) VALUES (?, ?, ?)
               ON CONFLICT(consumer_id, room_id) DO UPDATE
               SET expires_at = excluded.expires_at""",
            (consumer_id, room_id, now + float(presence_ttl)),
        )
        current = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
        return _command(current)


def retry_failed_command(
    db_path: Path | str,
    *,
    room_id: Any,
    command_id: Any = None,
    clock: Any = time.time,
) -> dict[str, Any]:
    """Requeue one exact failed Desktop command after explicit owner action."""

    room_id = _room_identifier(room_id)
    requested_id = str(command_id or "").strip()
    now = float(clock())
    with _transaction(db_path, immediate=True) as conn:
        if requested_id:
            row = conn.execute(
                """SELECT * FROM desktop_room_commands
                   WHERE room_id = ? AND command_id = ?""",
                (room_id, _identifier(requested_id, label="command_id")),
            ).fetchone()
        else:
            row = conn.execute(
                """SELECT * FROM desktop_room_commands
                   WHERE room_id = ? AND state IN ('failed', 'pending')
                   ORDER BY updated_at DESC, command_id DESC LIMIT 1""",
                (room_id,),
            ).fetchone()
        if row is None:
            raise DesktopRoomMailboxError("no failed Group Chat command needs retry")
        if str(row["state"]) == "pending":
            return _command(row, idempotent=True)
        if str(row["state"]) != "failed":
            raise DesktopRoomMailboxError("that Group Chat command is not retryable")
        conn.execute(
            """UPDATE desktop_room_commands
                  SET state='pending', attempts=0,
                      lease_token=NULL, lease_owner=NULL,
                      lease_expires_at=NULL, result_json=NULL,
                      created_at=?, updated_at=?
                WHERE command_id=? AND state='failed'""",
            (now, now, str(row["command_id"])),
        )
        retried = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id = ?",
            (str(row["command_id"]),),
        ).fetchone()
        if retried is None:
            raise DesktopRoomMailboxError("Group Chat command disappeared during retry")
        result = _command(retried)
    _notify_pending(db_path)
    return result


def retry_failed_commands(
    db_path: Path | str,
    *,
    room_id: Any,
    command_ids: Any = None,
    clock: Any = time.time,
) -> list[dict[str, Any]]:
    """Requeue a bounded oldest-first batch after explicit owner confirmation."""

    room_id = _room_identifier(room_id)
    now = float(clock())
    retried: list[dict[str, Any]] = []
    requested = tuple(
        _identifier(str(item), label="command_id")
        for item in (command_ids or ())
        if str(item).strip()
    )
    if len(requested) > MAX_RETRY_COMMANDS:
        raise DesktopRoomMailboxError("too many Group Chat commands to retry")
    with _transaction(db_path, immediate=True) as conn:
        if requested:
            placeholders = ",".join("?" for _ in requested)
            rows = conn.execute(
                f"""SELECT * FROM desktop_room_commands
                      WHERE room_id=? AND command_id IN ({placeholders})
                        AND state IN ('failed','pending')
                      ORDER BY created_at, command_id""",
                (room_id, *requested),
            ).fetchall()
            if len(rows) != len(requested):
                raise DesktopRoomMailboxError(
                    "Group Chat retry scope changed before it could be applied"
                )
        else:
            rows = conn.execute(
                """SELECT * FROM desktop_room_commands
                     WHERE room_id=? AND state='failed'
                     ORDER BY created_at, command_id
                     LIMIT ?""",
                (room_id, MAX_RETRY_COMMANDS),
            ).fetchall()
        if not rows:
            pending = conn.execute(
                """SELECT * FROM desktop_room_commands
                     WHERE room_id=? AND state='pending'
                     ORDER BY created_at, command_id LIMIT 1""",
                (room_id,),
            ).fetchone()
            if pending is not None:
                return [_command(pending, idempotent=True)]
            raise DesktopRoomMailboxError("no failed Group Chat command needs retry")
        for index, row in enumerate(rows):
            command_id = str(row["command_id"])
            if str(row["state"]) == "pending":
                retried.append(_command(row, idempotent=True))
                continue
            conn.execute(
                """UPDATE desktop_room_commands
                      SET state='pending', attempts=0,
                          lease_token=NULL, lease_owner=NULL,
                          lease_expires_at=NULL, result_json=NULL,
                          created_at=?, updated_at=?
                    WHERE command_id=? AND state='failed'""",
                (now + index * 0.000001, now, command_id),
            )
            current = conn.execute(
                "SELECT * FROM desktop_room_commands WHERE command_id=?",
                (command_id,),
            ).fetchone()
            if current is None:
                raise DesktopRoomMailboxError(
                    "Group Chat command disappeared during retry"
                )
            retried.append(_command(current))
    _notify_pending(db_path)
    return retried


def retryable_command_ids(
    db_path: Path | str,
    *,
    room_id: Any,
) -> tuple[str, ...]:
    """Freeze the bounded oldest-first retry scope before mutating it."""

    room_id = _room_identifier(room_id)
    with _transaction(db_path) as conn:
        rows = conn.execute(
            """SELECT command_id FROM desktop_room_commands
                 WHERE room_id=? AND state='failed'
                 ORDER BY created_at, command_id
                 LIMIT ?""",
            (room_id, MAX_RETRY_COMMANDS),
        ).fetchall()
        if not rows:
            pending = conn.execute(
                """SELECT command_id FROM desktop_room_commands
                     WHERE room_id=? AND state='pending'
                     ORDER BY created_at, command_id LIMIT 1""",
                (room_id,),
            ).fetchone()
            rows = [pending] if pending is not None else []
    if not rows:
        raise DesktopRoomMailboxError("no failed Group Chat command needs retry")
    return tuple(str(row["command_id"]) for row in rows)


def failed_command_counts(
    db_path: Path | str,
    room_ids: Any,
) -> dict[str, int]:
    """Return bounded failed-command counts for requested Desktop rooms."""

    rooms = _query_room_ids(room_ids)
    if not rooms:
        return {}
    counts: dict[str, int] = {}
    with _transaction(db_path) as conn:
        for index in range(0, len(rooms), MAX_ROOM_IDS):
            batch = rooms[index : index + MAX_ROOM_IDS]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"""SELECT room_id, COUNT(*) AS count
                       FROM desktop_room_commands
                      WHERE room_id IN ({placeholders}) AND state='failed'
                      GROUP BY room_id""",
                tuple(batch),
            ).fetchall()
            counts.update({str(row["room_id"]): int(row["count"]) for row in rows})
    return counts


def room_available(
    db_path: Path | str,
    room_id: str,
    *,
    clock: Any = time.time,
) -> bool:
    """Return whether a connected Desktop currently advertises this room."""

    room_id = _room_identifier(room_id)
    now = float(clock())
    with _transaction(db_path) as conn:
        row = conn.execute(
            """SELECT 1 FROM desktop_room_presence
               WHERE room_id = ? AND expires_at > ? LIMIT 1""",
            (room_id, now),
        ).fetchone()
        return row is not None


def available_room_ids(
    db_path: Path | str,
    room_ids: Any,
    *,
    clock: Any = time.time,
) -> set[str]:
    """Return all advertised room ids using one bounded database read."""

    rooms = _query_room_ids(room_ids)
    if not rooms:
        return set()
    now = float(clock())
    available: set[str] = set()
    with _transaction(db_path) as conn:
        for index in range(0, len(rooms), MAX_ROOM_IDS):
            batch = rooms[index : index + MAX_ROOM_IDS]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"""SELECT DISTINCT room_id FROM desktop_room_presence
                    WHERE room_id IN ({placeholders}) AND expires_at > ?""",
                (*batch, now),
            ).fetchall()
            available.update(str(row["room_id"]) for row in rows)
    return available


def latest_command_states(
    db_path: Path | str,
    room_ids: Any,
) -> dict[str, dict[str, Any]]:
    """Return the newest command state for each requested room."""

    rooms = _query_room_ids(room_ids)
    if not rooms:
        return {}
    states: dict[str, dict[str, Any]] = {}
    with _transaction(db_path) as conn:
        for index in range(0, len(rooms), MAX_ROOM_IDS):
            batch = rooms[index : index + MAX_ROOM_IDS]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"""SELECT c.* FROM desktop_room_commands AS c
                    WHERE c.room_id IN ({placeholders})
                      AND c.command_id = (
                        SELECT newer.command_id
                        FROM desktop_room_commands AS newer
                        WHERE newer.room_id = c.room_id
                        ORDER BY newer.created_at DESC, newer.command_id DESC
                        LIMIT 1
                      )""",
                tuple(batch),
            ).fetchall()
            states.update({str(row["room_id"]): _command(row) for row in rows})
    return states


def command_state(
    db_path: Path | str,
    command_id: str,
) -> dict[str, Any] | None:
    """Return one command by its transport-stable id."""

    command_id = _identifier(command_id, label="command_id")
    with _transaction(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM desktop_room_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()
    return _command(row) if row is not None else None
