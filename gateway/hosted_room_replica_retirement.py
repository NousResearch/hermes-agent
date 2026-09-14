"""Owner-enrolled, one-purpose retirement of passive Group Chat copies.

Only a commitment crosses setup. The home reveals its closing value after a
durable canonical disband, independently of expired execution grants/routes.
Retirement is a local copy fact, never a fabricated canonical log event.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway import hosted_rooms as rooms
from gateway import hosted_room_passive_lineage as lineage
from gateway.hosted_room_peer import validate_room_link_url
from gateway.hosted_rooms_common import compact_json, table_exists

HOME_TABLE = "hosted_room_replica_retirement_home"
ENROLLMENT_TABLE = "hosted_room_replica_retirement_enrollments"
RETIREMENT_TABLE = "hosted_room_replica_retirements"
MAX_PENDING_ENROLLMENTS = 512
_PUBLIC_FIELDS = (
    "enrollment_id",
    "room_id",
    "authority_gateway_id",
    "authority_epoch",
    "target_install_id",
    "roster_sha256",
    "commitment",
)
_SCOPE_FIELDS = _PUBLIC_FIELDS[:-1]
_DOMAIN = b"hermes.group.replica.retirement.v1\0"
_COMMITMENT_DOMAIN = b"hermes.group.replica.retirement.commitment.v1\0"


class RetirementError(rooms.HostedRoomError):
    """Invalid or unavailable copy-retirement operation."""


class RetirementConflictError(RetirementError):
    """An immutable enrollment, room namespace or expected state differs."""


class RetirementAuthorizationError(RetirementError):
    """The closing value does not authorize the exact current enrollment."""


class RetirementCapacityError(RetirementError):
    """Pending cleanup obligations cannot be silently dropped for space."""


class RetirementKeyUnavailable(RetirementError):
    """The original home secret is no longer available."""


@dataclass(frozen=True)
class RetirementNotice:
    enrollment_id: str
    room_id: str
    authority_gateway_id: str
    authority_epoch: int
    target_install_id: str
    endpoint: str
    value: str = field(repr=False)
    version: int | None = None
    lineage_sha256: str | None = None

    def payload(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in _SCOPE_FIELDS if name != "roster_sha256"}
        if self.version == 2:
            result.update(version=2, lineage_sha256=self.lineage_sha256)
        return result


def _identifier(value: Any, name: str) -> str:
    return rooms._validate_identifier(value, label=name, max_chars=128)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RetirementError(f"invalid {name}")
    return value


def roster_digest(members: Any) -> str:
    _, encoded = rooms._validate_members(members)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _public_fields(row):
    return (*_PUBLIC_FIELDS, "version", "lineage_sha256") if lineage.is_v2(row) else _PUBLIC_FIELDS


def _scope_fields(row):
    return (*_SCOPE_FIELDS, "version", "lineage_sha256") if lineage.is_v2(row) else _SCOPE_FIELDS


def _public(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _public_fields(row)}


def _closing_value(secret: bytes, row: Mapping[str, Any]) -> str:
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise RetirementKeyUnavailable("retirement key is unavailable")
    material = {key: row[key] for key in _scope_fields(row)}
    material["nonce"] = row["nonce"]
    raw = hmac.new(
        secret, (_DOMAIN.replace(b".v1\0", b".v2\0") if lineage.is_v2(row) else _DOMAIN)
        + compact_json(material).encode("utf-8"), hashlib.sha256
    ).digest()
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _commitment(value: str, scope: Mapping[str, Any]) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", value) is None:
        raise RetirementAuthorizationError("invalid retirement capability")
    binding = compact_json({key: scope[key] for key in _scope_fields(scope)}).encode("utf-8")
    domain = _COMMITMENT_DOMAIN.replace(b".v1\0", b".v2\0") if lineage.is_v2(scope) else _COMMITMENT_DOMAIN
    return hashlib.sha256(
        domain + binding + b"\0" + value.encode("ascii")
    ).hexdigest()


def _initialize(conn: sqlite3.Connection) -> None:
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {HOME_TABLE} (
        enrollment_id TEXT PRIMARY KEY, room_id TEXT NOT NULL,
        authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
        target_install_id TEXT NOT NULL, roster_sha256 TEXT NOT NULL,
        endpoint TEXT NOT NULL, nonce TEXT NOT NULL, commitment TEXT NOT NULL,
        is_current INTEGER NOT NULL DEFAULT 1,
        state TEXT NOT NULL DEFAULT 'prepared', created_at REAL NOT NULL,
        frozen_at REAL, closed_at REAL, acknowledged_at REAL,
        closing_value TEXT, last_error TEXT, superseded_by TEXT)""")
    conn.execute(f"""CREATE UNIQUE INDEX IF NOT EXISTS idx_replica_retirement_home_current
        ON {HOME_TABLE}(room_id,target_install_id) WHERE is_current=1""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {ENROLLMENT_TABLE} (
        enrollment_id TEXT PRIMARY KEY, room_id TEXT NOT NULL,
        authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
        target_install_id TEXT NOT NULL, roster_sha256 TEXT NOT NULL,
        commitment TEXT NOT NULL, is_current INTEGER NOT NULL DEFAULT 1,
        state TEXT NOT NULL DEFAULT 'active', created_at REAL NOT NULL)""")
    conn.execute(f"""CREATE UNIQUE INDEX IF NOT EXISTS idx_replica_retirement_target_current
        ON {ENROLLMENT_TABLE}(room_id) WHERE is_current=1""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {RETIREMENT_TABLE} (
        room_id TEXT PRIMARY KEY, enrollment_id TEXT NOT NULL,
        authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
        target_install_id TEXT NOT NULL, commitment TEXT NOT NULL,
        retired_at REAL NOT NULL, stored_seq INTEGER NOT NULL, source_latest_seq INTEGER NOT NULL)""")
    lineage.initialize(conn)
    from gateway.hosted_room_replicas import _initialize_replica_schema
    _initialize_replica_schema(conn)
    columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({RETIREMENT_TABLE})")}
    for name, kind in (("version", "INTEGER"), ("lineage_sha256", "TEXT"), ("lineage_status", "TEXT")):
        if name not in columns:
            conn.execute(f"ALTER TABLE {RETIREMENT_TABLE} ADD COLUMN {name} {kind}")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_lineage_update_v2
        BEFORE UPDATE ON hosted_room_replicas
        WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id IN (OLD.room_id,NEW.room_id))
          AND (NEW.replica_version IS NOT OLD.replica_version OR NEW.lineage_sha256 IS NOT OLD.lineage_sha256)
        BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    for table, suffix in (
        ("hosted_room_replicas", "room"),
        ("hosted_room_replica_events", "event"),
    ):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_{suffix}_insert
            BEFORE INSERT ON {table}
            WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id=NEW.room_id)
            BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_event_update_v2
        BEFORE UPDATE ON hosted_room_replica_events
        WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id IN (OLD.room_id,NEW.room_id))
        BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_room_update_v2
        BEFORE UPDATE ON hosted_room_replicas
        WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id IN (OLD.room_id,NEW.room_id))
          AND (NEW.room_id IS NOT OLD.room_id OR NEW.name IS NOT OLD.name
            OR NEW.members_json IS NOT OLD.members_json
            OR NEW.authority_gateway_id IS NOT OLD.authority_gateway_id
            OR NEW.authority_epoch IS NOT OLD.authority_epoch
            OR NEW.last_seq IS NOT OLD.last_seq OR NEW.latest_seq IS NOT OLD.latest_seq
            OR NEW.disbanded_at IS NOT OLD.disbanded_at)
        BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    from gateway.hosted_room_work_records import initialize_retirement_guards
    initialize_retirement_guards(conn)

    # Install both replacement guards before retiring v1, in the caller's writer
    # transaction. The v1 initializer can only add its old guards alongside v2.
    conn.execute("DROP TRIGGER IF EXISTS trg_replica_retired_event_update")
    conn.execute("DROP TRIGGER IF EXISTS trg_replica_retired_room_update")


@contextmanager
def _transaction(db_path: Path | str):
    with rooms._transaction(db_path, immediate=True) as conn:
        _initialize(conn)
        yield conn


def prepare_home_enrollment(
    db_path: Path | str,
    *,
    room_id: str,
    target_install_id: str,
    endpoint: str,
    local_gateway_id: str,
    secret: bytes,
    enrollment_id: str | None = None,
    replace_enrollment_id: str | None = None,
) -> dict[str, Any]:
    """Reserve a cleanup obligation before owner-authorized target enrollment."""
    room_id = _identifier(room_id, "room_id")
    target_install_id = _identifier(target_install_id, "target_install_id")
    local_gateway_id = _identifier(local_gateway_id, "local_gateway_id")
    endpoint, _ = validate_room_link_url(endpoint)
    if len(endpoint) > 2048 or re.search(r"/p/[^/]+$", endpoint):
        raise RetirementError(
            "retirement requires an installation endpoint, not a profile endpoint"
        )
    if enrollment_id is not None:
        enrollment_id = _identifier(enrollment_id, "enrollment_id")
    if replace_enrollment_id is not None and enrollment_id is None:
        raise RetirementError("replacement requires a new idempotent enrollment_id")
    with _transaction(db_path) as conn:
        room = conn.execute(
            "SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)
        ).fetchone()
        if room is None or room["disbanded_at"] is not None:
            raise RetirementConflictError("Group Chat is not active")
        if room["authority_gateway_id"] != local_gateway_id:
            raise RetirementConflictError("retirement setup requires the current authority")
        if not table_exists(conn, "hosted_room_disband_fences"):
            raise RetirementError("irreversible Group Chat disband support is required")
        if conn.execute(
            "SELECT 1 FROM hosted_room_disband_fences WHERE room_id=?", (room_id,)
        ).fetchone():
            raise RetirementConflictError("Group Chat disband has already started")
        if conn.execute(
            "SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)
        ).fetchone():
            raise RetirementConflictError("Group Chat is quarantined")
        members = json.loads(room["members_json"])
        if target_install_id == local_gateway_id or not any(
            isinstance(m.get("target"), dict)
            and m["target"].get("kind") == "peer"
            and m["target"].get("installation_id") == target_install_id
            for m in members
        ):
            raise RetirementConflictError("target is not a participant gateway")
        scope = dict(
            room_id=room_id,
            authority_gateway_id=local_gateway_id,
            authority_epoch=room["authority_epoch"],
            target_install_id=target_install_id,
            roster_sha256=roster_digest(members),
        )
        history_json = None
        if room["authority_epoch"] != 1:
            history, digest = lineage.source_locked(conn, room_id, {
                "gateway_id": local_gateway_id, "epoch": room["authority_epoch"]})
            history_json = lineage.canonical(history)
            scope.update(version=2, lineage_sha256=digest)
        existing = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)
        ).fetchone()
        if existing is not None:
            if (
                not existing["is_current"]
                or any(existing[k] != v for k, v in scope.items())
                or existing["endpoint"] != endpoint
            ):
                raise RetirementConflictError(
                    "enrollment_id already belongs to another setup"
                )
            if not hmac.compare_digest(
                _commitment(_closing_value(secret, existing), existing),
                existing["commitment"],
            ):
                raise RetirementKeyUnavailable(
                    "retirement key no longer matches enrollment"
                )
            return _public(existing)
        current = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE room_id=? AND target_install_id=? AND is_current=1",
            (room_id, target_install_id),
        ).fetchone()
        if current is not None and replace_enrollment_id is None:
            if (
                any(current[k] != v for k, v in scope.items())
                or current["endpoint"] != endpoint
            ):
                raise RetirementConflictError("retirement enrollment changed")
            if not hmac.compare_digest(
                _commitment(_closing_value(secret, current), current),
                current["commitment"],
            ):
                raise RetirementKeyUnavailable(
                    "retirement key no longer matches enrollment"
                )
            return _public(current)
        if replace_enrollment_id is not None and (
            current is None or current["enrollment_id"] != replace_enrollment_id
        ):
            raise RetirementConflictError(
                "retirement enrollment replacement lost its expected state"
            )
        count = conn.execute(
            f"SELECT COUNT(*) FROM {HOME_TABLE} WHERE state NOT IN ('acknowledged','superseded','revoked')"
        ).fetchone()[0]
        if count >= MAX_PENDING_ENROLLMENTS:
            raise RetirementCapacityError(
                "pending retirement delivery capacity is full"
            )
        if history_json is not None:
            lineage.ensure_descriptor_capacity(conn, history_json)
        row = dict(
            scope,
            enrollment_id=enrollment_id or secrets.token_hex(16),
            endpoint=endpoint,
            nonce=secrets.token_hex(16),
        )
        row["commitment"] = _commitment(_closing_value(secret, row), row)
        if current is not None:
            conn.execute(
                f"UPDATE {HOME_TABLE} SET is_current=0 WHERE enrollment_id=?",
                (current["enrollment_id"],),
            )
        row["authority_history_json"] = history_json
        fields = (*_public_fields(row), "endpoint", "nonce", "authority_history_json")
        conn.execute(
            f"INSERT INTO {HOME_TABLE} ({','.join(fields)},created_at) VALUES ({','.join('?' for _ in fields)},?)",
            (*[row[k] for k in fields], time.time()),
        )
        return _public(row)


def _validate_enrollment(value: Any, target_install_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_public_fields(value)):
        raise RetirementError("invalid retirement enrollment fields")
    value = dict(value)
    for key in (
        "enrollment_id",
        "room_id",
        "authority_gateway_id",
        "target_install_id",
    ):
        validated = _identifier(value[key], key)
        if lineage.is_v2(value) and validated != value[key]:
            raise RetirementError("v2 enrollment identifiers must be canonical")
        value[key] = validated
    if (type(value["authority_epoch"]) is not int or not 1 <= value["authority_epoch"] < 2**63
            or (not lineage.is_v2(value) and value["authority_epoch"] != 1)):
        raise RetirementConflictError("verified initial authority or v2 lineage is required")
    if lineage.is_v2(value):
        _digest(value["lineage_sha256"], "lineage_sha256")
    if (
        value["target_install_id"] != target_install_id
        or value["authority_gateway_id"] == target_install_id
    ):
        raise RetirementConflictError(
            "retirement enrollment targets another installation"
        )
    _digest(value["roster_sha256"], "roster_sha256")
    _digest(value["commitment"], "commitment")
    return value


def _check_replica_namespace(
    conn: sqlite3.Connection, value: Mapping[str, Any], *, spans=None
) -> sqlite3.Row | None:
    from gateway.hosted_room_replicas import _audit_existing_replicas_locked

    _audit_existing_replicas_locked(conn)
    room_id = value["room_id"]
    reservation = conn.execute(
        "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?", (room_id,)
    ).fetchone()
    if (
        (reservation is not None and reservation[0] == "authority")
        or conn.execute(
            "SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)
        ).fetchone()
        or conn.execute(
            "SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)
        ).fetchone()
    ):
        raise RetirementConflictError("Group Chat is locally authoritative")
    if conn.execute(
        "SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)
    ).fetchone():
        raise RetirementConflictError(
            "quarantined Group Chat evidence must be preserved"
        )
    replica = conn.execute(
        "SELECT * FROM hosted_room_replicas WHERE room_id=?", (room_id,)
    ).fetchone()
    if replica is not None and (
        replica["quarantine_reason"] is not None
        or (spans is None and (
            replica["authority_gateway_id"] != value["authority_gateway_id"]
            or replica["authority_epoch"] != value["authority_epoch"]))
        or roster_digest(json.loads(replica["members_json"])) != value["roster_sha256"]
    ):
        raise RetirementConflictError(
            "retirement enrollment differs from the retained copy"
        )
    if replica is None and reservation is not None:
        raise RetirementConflictError("retired copy identity must be preserved")
    return replica


def enroll_target(
    db_path: Path | str,
    *,
    enrollment: Mapping[str, Any],
    target_install_id: str,
    expected_enrollment_id: str | None = None,
    expected_state: str = "active",
    authority_history: Any = None,
) -> dict[str, Any]:
    """Owner-only enrollment; ordinary RoomLink bearers must never call this."""
    value = _validate_enrollment(enrollment, target_install_id)
    spans, history_json = None, None
    if lineage.is_v2(value):
        spans, history_json, digest = lineage.descriptor(authority_history,
            gateway_id=value["authority_gateway_id"], epoch=value["authority_epoch"])
        if digest != value["lineage_sha256"]:
            raise RetirementConflictError("enrollment lineage digest differs")
    elif authority_history is not None:
        raise RetirementError("v1 enrollment cannot carry a lineage descriptor")
    if expected_state not in {"active", "revoked"}:
        raise RetirementError("invalid expected enrollment state")
    with _transaction(db_path) as conn:
        if copy_retired_locked(conn, value["room_id"]):
            raise RetirementConflictError(
                "a retired Group Chat copy cannot be reenrolled"
            )
        replica = _check_replica_namespace(conn, value, spans=spans)
        reservations = conn.execute(
            """SELECT authority_gateway_id,authority_epoch
            FROM hosted_room_peer_reservations WHERE room_id=? AND revoked_at IS NULL AND expires_at>?""",
            (value["room_id"], time.time()),
        ).fetchall()
        if any(
            (r[0], r[1]) not in ({(s.gateway_id, s.epoch) for s in spans} if spans else
                {(value["authority_gateway_id"], value["authority_epoch"])})
            for r in reservations
        ):
            raise RetirementConflictError(
                "retirement enrollment differs from a live room reservation"
            )
        existing = conn.execute(
            f"SELECT * FROM {ENROLLMENT_TABLE} WHERE enrollment_id=?",
            (value["enrollment_id"],),
        ).fetchone()
        if existing is not None:
            if (
                _public(existing) != value or existing["authority_history_json"] != history_json
                or not existing["is_current"]
            ):
                raise RetirementConflictError(
                    "enrollment_id already belongs to another setup"
                )
            if existing["state"] != "active":
                raise RetirementAuthorizationError(
                    "retirement capability has been revoked"
                )
            return {**_public(existing), "state": "active"}
        current = conn.execute(
            f"SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND is_current=1",
            (value["room_id"],),
        ).fetchone()
        if current is not None:
            if (
                current["enrollment_id"] != expected_enrollment_id
                or current["state"] != expected_state
            ):
                raise RetirementConflictError(
                    "target enrollment replacement lost its expected state"
                )
            fixed = ("room_id", "target_install_id", "roster_sha256") if spans else tuple(
                k for k in _scope_fields(current) if k != "enrollment_id")
            if any(current[k] != value.get(k) for k in fixed):
                raise RetirementConflictError(
                    "implicit enrollment lineage changes are not allowed"
                )
        elif expected_enrollment_id is not None:
            raise RetirementConflictError("expected target enrollment does not exist")
        if spans is not None:
            lineage.compatible_extension(conn, value["room_id"], spans, previous=current, replica=replica)
            lineage.ensure_descriptor_capacity(conn, history_json)
        active = conn.execute(
            f"SELECT COUNT(*) FROM {ENROLLMENT_TABLE} WHERE is_current=1 AND state='active'"
        ).fetchone()[0]
        if (
            current is None or current["state"] != "active"
        ) and active >= MAX_PENDING_ENROLLMENTS:
            raise RetirementCapacityError("target enrollment capacity is full")
        if current is not None:
            conn.execute(
                f"UPDATE {ENROLLMENT_TABLE} SET is_current=0 WHERE enrollment_id=?",
                (current["enrollment_id"],),
            )
        fields = (*_public_fields(value), "authority_history_json")
        conn.execute(
            f"INSERT INTO {ENROLLMENT_TABLE} ({','.join(fields)},created_at) VALUES ({','.join('?' for _ in fields)},?)",
            (*[value[k] for k in _public_fields(value)], history_json, time.time()),
        )
        if spans is not None and replica is not None:
            conn.execute("UPDATE hosted_room_replicas SET replica_version=2,lineage_sha256=? WHERE room_id=?",
                         (value["lineage_sha256"], value["room_id"]))
        return {**value, "state": "active"}


def revoke_target_enrollment(
    db_path: Path | str, *, room_id: str, enrollment_id: str
) -> dict[str, Any]:
    """Owner action revoking only future copy retirement, not ordinary Bot grants."""
    with _transaction(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND enrollment_id=? AND is_current=1",
            (
                _identifier(room_id, "room_id"),
                _identifier(enrollment_id, "enrollment_id"),
            ),
        ).fetchone()
        if row is None:
            raise RetirementConflictError("retirement enrollment is no longer current")
        conn.execute(
            f"UPDATE {ENROLLMENT_TABLE} SET state='revoked' WHERE enrollment_id=?",
            (enrollment_id,),
        )
        return {"room_id": room_id, "enrollment_id": enrollment_id, "state": "revoked"}


def freeze_home_enrollments_locked(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
) -> None:
    if table_exists(conn, HOME_TABLE):
        conn.execute(
            f"UPDATE {HOME_TABLE} SET state='closing',frozen_at=COALESCE(frozen_at,?) WHERE room_id=? AND authority_gateway_id=? AND authority_epoch=? AND state IN ('prepared','enrolled')",
            (time.time(), room_id, authority_gateway_id, authority_epoch),
        )


def current_home_enrollment(
    db_path: Path | str, *, room_id: str, target_install_id: str
) -> dict[str, Any] | None:
    with rooms._transaction(db_path) as conn:
        if not table_exists(conn, HOME_TABLE):
            return None
        row = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE room_id=? AND target_install_id=? AND is_current=1",
            (room_id, target_install_id),
        ).fetchone()
        return {**_public(row), "state": row["state"]} if row is not None else None


def home_enrollment_history(db_path, *, enrollment_id):
    """The owner setup response carries the descriptor beside its public scope."""
    with _transaction(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)).fetchone()
        if row is None or not lineage.is_v2(row):
            return {}
        return {"authority_history": [s.as_mapping() for s in lineage.enrolled_history(row)]}


def current_target_enrollment(
    db_path: Path | str,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
) -> dict[str, Any] | None:
    with rooms._transaction(db_path) as conn:
        if not table_exists(conn, ENROLLMENT_TABLE):
            return None
        row = conn.execute(
            f"SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND authority_gateway_id=? AND authority_epoch=? AND is_current=1 AND state='active'",
            (room_id, authority_gateway_id, authority_epoch),
        ).fetchone()
        return {**_public(row), "state": "active"} if row is not None else None


def confirm_home_enrollment(
    db_path: Path | str, *, enrollment_id: str, proof: Any
) -> bool:
    with _transaction(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=? AND is_current=1 AND state IN ('prepared','enrolled')",
            (enrollment_id,),
        ).fetchone()
        if row is None:
            return False
        if (
            not isinstance(proof, Mapping)
            or proof.get("state") != "active"
            or type(proof.get("authority_epoch")) is not int
            or any(proof.get(k) != row[k] for k in _public_fields(row))
            or (lineage.is_v2(row) and type(proof.get("version")) is not int)
            or (not lineage.is_v2(row) and "version" in proof)
        ):
            conn.execute(
                f"UPDATE {HOME_TABLE} SET last_error='retirement_enrollment_unconfirmed' WHERE enrollment_id=?",
                (enrollment_id,),
            )
            return False
        conn.execute(
            f"UPDATE {HOME_TABLE} SET state='enrolled',last_error=NULL WHERE enrollment_id=?",
            (enrollment_id,),
        )
        return True


def reconcile_home_close_locked(conn: sqlite3.Connection, room_id: str) -> None:
    """Use canonical close proof, never the route fence alone, to enable reveal."""
    if not table_exists(conn, HOME_TABLE):
        return
    room = conn.execute(
        "SELECT authority_gateway_id,authority_epoch,disbanded_at FROM hosted_rooms WHERE room_id=?",
        (room_id,),
    ).fetchone()
    if room is not None:
        if room["disbanded_at"] is not None:
            conn.execute(
                f"UPDATE {HOME_TABLE} SET state='closed',closed_at=? WHERE room_id=? AND authority_gateway_id=? AND authority_epoch=? AND state IN ('prepared','enrolled','closing')",
                (
                    room["disbanded_at"],
                    room_id,
                    room["authority_gateway_id"],
                    room["authority_epoch"],
                ),
            )
        return
    tombstone = conn.execute(
        "SELECT retired_at FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)
    ).fetchone()
    if tombstone is not None:
        conn.execute(
            f"UPDATE {HOME_TABLE} SET state='closed',closed_at=? WHERE room_id=? AND state='closing' AND frozen_at IS NOT NULL",
            (tombstone[0], room_id),
        )


def _block_stale_home_locked(conn):
    conn.execute(f"""UPDATE {HOME_TABLE} SET state='blocked_authority',last_error='stale_authority'
        WHERE state NOT IN ('acknowledged','superseded','revoked','blocked_authority')
          AND EXISTS (SELECT 1 FROM hosted_rooms r WHERE r.room_id={HOME_TABLE}.room_id
            AND (r.authority_gateway_id!={HOME_TABLE}.authority_gateway_id OR r.authority_epoch!={HOME_TABLE}.authority_epoch))""")


def pending_notice_ids(
    db_path: Path | str, *, local_gateway_id: str, limit: int = MAX_PENDING_ENROLLMENTS
) -> list[str]:
    if type(limit) is not int or not 1 <= limit <= MAX_PENDING_ENROLLMENTS:
        raise RetirementError("invalid retirement notice limit")
    with _transaction(db_path) as conn:
        _block_stale_home_locked(conn)
        for row in conn.execute(
            f"SELECT DISTINCT room_id FROM {HOME_TABLE} WHERE authority_gateway_id=? AND state='closing'",
            (local_gateway_id,),
        ).fetchall():
            reconcile_home_close_locked(conn, row[0])
        return [
            row[0]
            for row in conn.execute(
                f"SELECT enrollment_id FROM {HOME_TABLE} WHERE authority_gateway_id=? AND state IN ('closed','ready') ORDER BY created_at,enrollment_id LIMIT ?",
                (local_gateway_id, limit),
            )
        ]


def notice_route(
    db_path: Path | str, *, enrollment_id: str, local_gateway_id: str
) -> dict[str, str] | None:
    """Read immutable routing without loading or deriving a closing value."""
    with _transaction(db_path) as conn:
        row = conn.execute(
            f"SELECT room_id,target_install_id,endpoint FROM {HOME_TABLE} WHERE enrollment_id=? AND authority_gateway_id=? AND state IN ('closed','ready')",
            (enrollment_id, local_gateway_id),
        ).fetchone()
        return dict(row) if row is not None else None


def materialize_notice(
    db_path: Path | str,
    *,
    enrollment_id: str,
    local_gateway_id: str,
    secret_loader: Callable[[], bytes],
) -> RetirementNotice:
    with _transaction(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)
        ).fetchone()
        if row is None or row["authority_gateway_id"] != local_gateway_id:
            raise RetirementConflictError(
                "retirement notice is not owned by this gateway"
            )
        _block_stale_home_locked(conn)
        reconcile_home_close_locked(conn, row["room_id"])
        row = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)
        ).fetchone()
        if row["state"] not in {"closed", "ready"} or row["closed_at"] is None:
            raise RetirementConflictError(
                "canonical Group Chat disband has not completed"
            )
        value = row["closing_value"]
        if not value:
            try:
                value = _closing_value(secret_loader(), row)
            except (OSError, ValueError) as exc:
                raise RetirementKeyUnavailable("retirement key is unavailable") from exc
        if not hmac.compare_digest(_commitment(value, row), row["commitment"]):
            raise RetirementKeyUnavailable(
                "retirement key no longer matches enrollment"
            )
        conn.execute(
            f"UPDATE {HOME_TABLE} SET state='ready',closing_value=? WHERE enrollment_id=?",
            (value, enrollment_id),
        )
        return RetirementNotice(
            **{
                k: row[k]
                for k in (
                    "enrollment_id",
                    "room_id",
                    "authority_gateway_id",
                    "authority_epoch",
                    "target_install_id",
                    "endpoint",
                )
            },
            value=value, version=row["version"], lineage_sha256=row["lineage_sha256"],
        )


def copy_retired_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    return (
        table_exists(conn, RETIREMENT_TABLE)
        and conn.execute(
            f"SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id=?",
            (room_id,),
        ).fetchone()
        is not None
    )


def copy_scope_matches_locked(
    conn: sqlite3.Connection,
    *,
    room_id: str,
    authority_gateway_id: str,
    authority_epoch: int,
    members_json: str,
    replica_version: int | None = None,
    lineage_sha256: str | None = None,
) -> bool:
    row = lineage.current_locked(conn, room_id)
    if row is None:
        return replica_version is None
    if lineage.is_v2(row):
        if replica_version != 2 or lineage_sha256 != row["lineage_sha256"] or row["state"] != "active":
            return False
    elif replica_version is not None:
        return False
    return (row["authority_gateway_id"], row["authority_epoch"], row["roster_sha256"]) == (
        authority_gateway_id, authority_epoch, hashlib.sha256(members_json.encode("utf-8")).hexdigest())


def _retired_response(row):
    result = dict(row)
    if lineage.is_v2(result):
        _digest(result.get("lineage_sha256"), "lineage_sha256")
        if result.get("lineage_status") not in {"pending", "verified"}:
            raise RetirementConflictError("retired lineage coverage is unavailable")
    else:
        if result["authority_epoch"] != 1 or result.get("version") is not None:
            raise RetirementConflictError("retired lineage format is unavailable")
        for key in ("version", "lineage_sha256", "lineage_status"):
            result.pop(key, None)
    return {"retired": True, **result}


def retire_copy(
    db_path: Path | str,
    *,
    payload: Mapping[str, Any],
    value: str,
    local_gateway_id: str,
) -> dict[str, Any]:
    fields = {
        "enrollment_id",
        "room_id",
        "authority_gateway_id",
        "authority_epoch",
        "target_install_id",
    }
    if isinstance(payload, Mapping) and lineage.is_v2(payload):
        fields |= {"version", "lineage_sha256"}
        _digest(payload.get("lineage_sha256"), "lineage_sha256")
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise RetirementError("invalid retirement notice fields")
    payload = dict(payload)
    for key in fields - {"authority_epoch", "version", "lineage_sha256"}:
        validated = _identifier(payload[key], key)
        if lineage.is_v2(payload) and validated != payload[key]:
            raise RetirementError("v2 notice identifiers must be canonical")
        payload[key] = validated
    if (type(payload["authority_epoch"]) is not int or not 1 <= payload["authority_epoch"] < 2**63
            or (not lineage.is_v2(payload) and payload["authority_epoch"] != 1)):
        raise RetirementError("invalid retirement authority epoch")
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", value) is None:
        raise RetirementAuthorizationError("invalid retirement capability")
    if not Path(db_path).is_file():
        raise RetirementAuthorizationError("retirement enrollment is unavailable")
    # Unknown bearers must not acquire the shared room writer lock or initialize schema.
    with closing(
        sqlite3.connect(
            Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25
        )
    ) as read:
        read.row_factory = sqlite3.Row
        if not table_exists(read, ENROLLMENT_TABLE):
            raise RetirementAuthorizationError("retirement enrollment is unavailable")
        enrolled = read.execute(
            f"SELECT * FROM {ENROLLMENT_TABLE} WHERE enrollment_id=?",
            (payload["enrollment_id"],),
        ).fetchone()
        if enrolled is None or not hmac.compare_digest(
            _commitment(value, enrolled), enrolled["commitment"]
        ):
            raise RetirementAuthorizationError("invalid retirement capability")
        if (
            any(payload[k] != enrolled[k] for k in fields)
            or enrolled["target_install_id"] != local_gateway_id
        ):
            raise RetirementAuthorizationError("retirement capability scope differs")
        if table_exists(read, RETIREMENT_TABLE):
            done = read.execute(
                f"SELECT * FROM {RETIREMENT_TABLE} WHERE room_id=?",
                (payload["room_id"],),
            ).fetchone()
            if done is not None:
                if done["enrollment_id"] != payload["enrollment_id"]:
                    raise RetirementConflictError(
                        "copy was retired under another enrollment"
                    )
                return _retired_response(done)
    with _transaction(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {ENROLLMENT_TABLE} WHERE enrollment_id=?",
            (payload["enrollment_id"],),
        ).fetchone()
        if row is None or not hmac.compare_digest(
            _commitment(value, row), row["commitment"]
        ):
            raise RetirementAuthorizationError("invalid retirement capability")
        if (
            any(payload[k] != row[k] for k in fields)
            or row["target_install_id"] != local_gateway_id
        ):
            raise RetirementAuthorizationError("retirement capability scope differs")
        retired = conn.execute(
            f"SELECT * FROM {RETIREMENT_TABLE} WHERE room_id=?", (row["room_id"],)
        ).fetchone()
        if retired is not None:
            if retired["enrollment_id"] != row["enrollment_id"]:
                raise RetirementConflictError(
                    "copy was retired under another enrollment"
                )
            return _retired_response(retired)
        if not row["is_current"] or row["state"] != "active":
            raise RetirementAuthorizationError(
                "retirement capability has been revoked or replaced"
            )
        spans = lineage.enrolled_history(row) if lineage.is_v2(row) else None
        replica = _check_replica_namespace(conn, row, spans=spans)
        if replica is not None and spans is not None:
            lineage.replica_history_locked(conn, replica)
        now = time.time()
        conn.execute(
            "INSERT OR IGNORE INTO hosted_room_id_reservations(room_id,owner_kind,reserved_at) VALUES (?,'replica',?)",
            (row["room_id"], now),
        )
        conn.execute(
            f"INSERT INTO {RETIREMENT_TABLE} (room_id,enrollment_id,authority_gateway_id,authority_epoch,target_install_id,commitment,retired_at,stored_seq,source_latest_seq) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                row["room_id"],
                row["enrollment_id"],
                row["authority_gateway_id"],
                row["authority_epoch"],
                row["target_install_id"],
                row["commitment"],
                now,
                int(replica["last_seq"]) if replica is not None else 0,
                int(replica["latest_seq"]) if replica is not None else 0,
            ),
        )
        if spans is not None:
            conn.execute(f"UPDATE {RETIREMENT_TABLE} SET version=2,lineage_sha256=?,lineage_status=? WHERE room_id=?",
                (row["lineage_sha256"], lineage.status(spans, replica["last_seq"] if replica is not None else 0), row["room_id"]))
        conn.execute(
            f"UPDATE {ENROLLMENT_TABLE} SET state='retired' WHERE enrollment_id=?",
            (row["enrollment_id"],),
        )
        from gateway.hosted_room_work_records import discard_retired_locked
        discard_retired_locked(conn, row["room_id"])
        return _retired_response(conn.execute(
            f"SELECT * FROM {RETIREMENT_TABLE} WHERE room_id=?", (row["room_id"],),
        ).fetchone())


def acknowledge_notice(
    db_path: Path | str, *, notice: RetirementNotice, response: Mapping[str, Any]
) -> None:
    expected = notice.payload()
    if (
        not isinstance(response, Mapping)
        or response.get("retired") is not True
        or type(response.get("authority_epoch")) is not int
        or any(response.get(k) != v for k, v in expected.items())
    ):
        raise RetirementConflictError(
            "retirement acknowledgment differs from its enrollment"
        )
    with _transaction(db_path) as conn:
        row = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (notice.enrollment_id,)
        ).fetchone()
        if (
            row is None
            or any(row[k] != v for k, v in expected.items())
            or row["state"] not in {"ready", "acknowledged"}
        ):
            raise RetirementConflictError(
                "retirement acknowledgment is no longer current"
            )
        if response.get("commitment") != row["commitment"]:
            raise RetirementConflictError(
                "retirement acknowledgment commitment differs"
            )
        conn.execute(
            f"UPDATE {HOME_TABLE} SET state='acknowledged',closing_value=NULL,acknowledged_at=COALESCE(acknowledged_at,?),last_error=NULL WHERE enrollment_id=?",
            (time.time(), notice.enrollment_id),
        )
        conn.execute(
            f"""UPDATE {HOME_TABLE} SET state='superseded',closing_value=NULL,superseded_by=?
            WHERE room_id=? AND target_install_id=? AND authority_gateway_id=? AND authority_epoch=?
              AND roster_sha256=? AND enrollment_id!=? AND state NOT IN ('acknowledged','superseded','revoked')""",
            (
                notice.enrollment_id,
                notice.room_id,
                notice.target_install_id,
                notice.authority_gateway_id,
                notice.authority_epoch,
                row["roster_sha256"],
                notice.enrollment_id,
            ),
        )


def record_delivery_error(
    db_path: Path | str, *, enrollment_id: str, code: str
) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) is None:
        raise RetirementError("invalid retirement error code")
    with _transaction(db_path) as conn:
        conn.execute(
            f"UPDATE {HOME_TABLE} SET last_error=? WHERE enrollment_id=? AND state IN ('closed','ready')",
            (code, enrollment_id),
        )


def home_status(
    db_path: Path | str, *, room_id: str | None = None
) -> list[dict[str, Any]]:
    with _transaction(db_path) as conn:
        _block_stale_home_locked(conn)
        rows = conn.execute(
            f"SELECT enrollment_id,room_id,target_install_id,authority_gateway_id,authority_epoch,state,frozen_at,closed_at,acknowledged_at,last_error,superseded_by FROM {HOME_TABLE} WHERE (? IS NULL OR room_id=?) ORDER BY CASE WHEN state IN ('acknowledged','superseded','revoked') THEN 1 ELSE 0 END,created_at,enrollment_id LIMIT ?",
            (room_id, room_id, MAX_PENDING_ENROLLMENTS * 2),
        )
        return [dict(row) for row in rows]
