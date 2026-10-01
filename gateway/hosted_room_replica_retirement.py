"""Owner-enrolled, one-purpose retirement of passive Group Chat copies.

The home prepares a durable obligation and an Ed25519 public verifier for one destination copy
and enrollment generation. Only that verifier crosses to the participant for operator enrollment;
the private seed remains derived from the home's secret and private enrollment nonce. Canonical
Disband closes the obligation transactionally, then the home signs the exact retirement notice.
Pinned, encrypted Room proof protects delivery and replies independently of member-grant expiry
or revocation. The participant verifies the notice again inside its destination-scoped writer.
A retired copy accepts no further history, retains its room namespace, and releases payload under
its retention policy. Retirement is local copy state, never a fabricated canonical log event.
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
from gateway.hosted_room_peer import validate_room_link_url
from gateway.hosted_room_work_records import roster_digest
from gateway.hosted_rooms_common import compact_json, table_exists

HOME_TABLE = "hosted_room_replica_retirement_home"
ENROLLMENT_TABLE = "hosted_room_replica_retirement_enrollments"
RETIREMENT_TABLE = "hosted_room_replica_retirements"
MAX_PENDING_ENROLLMENTS = 512
_PUBLIC_FIELDS = (
    "enrollment_id", "room_id", "authority_gateway_id", "authority_epoch", "target_install_id",
    "roster_sha256", "commitment")
_SCOPE_FIELDS = _PUBLIC_FIELDS[:-1]
_DOMAIN = b"hermes.group.replica.retirement.seed.v2\0"
_NOTICE_DOMAIN = b"hermes.group.replica.retirement.notice.v2\0"
_DONE = ("acknowledged", "superseded", "revoked")


class RetirementError(rooms.HostedRoomError):
    """Invalid or unavailable copy-retirement operation."""

    reason = "invalid_replica_retirement"


class RetirementConflictError(RetirementError):
    """An immutable enrollment, room namespace or expected state differs."""

    reason = "replica_retirement_conflict"


class RetirementAuthorizationError(RetirementError):
    """The signature does not authorize the exact current enrollment."""

    reason = "replica_retirement_not_authorized"


class RetirementCapacityError(RetirementError):
    """Pending cleanup obligations are never dropped to make room."""

    reason = "replica_retirement_capacity"


class RetirementKeyUnavailable(RetirementError):
    """The home secret behind an enrollment is no longer available."""

    reason = "replica_retirement_key_unavailable"


class RetirementProofUnavailable(RetirementError):
    """Installation proof was not retained before the route disappeared."""

    reason = 'replica_retirement_proof_unavailable'


@dataclass(frozen=True)
class RetirementNotice:
    enrollment_id: str
    room_id: str
    authority_gateway_id: str
    authority_epoch: int
    target_install_id: str
    endpoint: str
    value: str = field(repr=False)
    proof_grant: str | None = field(default=None, repr=False)

    def payload(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in _SCOPE_FIELDS if name != "roster_sha256"}


def _identifier(value: Any, name: str) -> str:
    return rooms._validate_identifier(value, label=name, max_chars=128)


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RetirementError(f"invalid {name}")
    return value


def _public(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _PUBLIC_FIELDS}


def _signing_seed(secret: bytes, row: Mapping[str, Any]) -> bytes:
    """Private per-enrollment key material; never returned to the participant."""
    if not isinstance(secret, bytes) or len(secret) < 32:
        raise RetirementKeyUnavailable("retirement key is unavailable")
    material = {key: row[key] for key in _SCOPE_FIELDS}
    material["nonce"] = row["nonce"]
    return hmac.new(secret, _DOMAIN + compact_json(material).encode("utf-8"), hashlib.sha256).digest()


def _commitment(seed: bytes) -> str:
    """The public Ed25519 verifier enrolled by the participant operator."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    return Ed25519PrivateKey.from_private_bytes(seed).public_key().public_bytes(Encoding.Raw, PublicFormat.Raw).hex()


def _notice_message(row):
    return _NOTICE_DOMAIN + compact_json({key: row[key] for key in _SCOPE_FIELDS}).encode('utf-8')


def _sign_notice(seed, row):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    signature = Ed25519PrivateKey.from_private_bytes(seed).sign(_notice_message(row))
    return 'ed25519-v2.' + base64.urlsafe_b64encode(signature).decode('ascii').rstrip('=')


def _valid_signature(value, row):
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    if not isinstance(value, str) or re.fullmatch(r'ed25519-v2\.[A-Za-z0-9_-]{86}', value) is None:
        return False
    try:
        signature = base64.urlsafe_b64decode(value.split('.', 1)[1] + '==')
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(row['commitment'])).verify(signature, _notice_message(row))
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def _initialize(conn: sqlite3.Connection) -> None:
    from gateway.hosted_room_replicas import _initialize_replica_schema
    from gateway.hosted_room_work_records import initialize_retirement_guards
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {HOME_TABLE} (
        enrollment_id TEXT PRIMARY KEY, room_id TEXT NOT NULL,
        authority_gateway_id TEXT NOT NULL, authority_epoch INTEGER NOT NULL,
        target_install_id TEXT NOT NULL, roster_sha256 TEXT NOT NULL,
        endpoint TEXT NOT NULL, nonce TEXT NOT NULL, commitment TEXT NOT NULL,
        is_current INTEGER NOT NULL DEFAULT 1, state TEXT NOT NULL DEFAULT 'prepared',
        created_at REAL NOT NULL, closed_at REAL, acknowledged_at REAL,
        closing_value TEXT, proof_grant TEXT, last_error TEXT, superseded_by TEXT)""")
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
    _initialize_replica_schema(conn)
    # A retired copy refuses every write, by this code or an older one sharing the store. Both the
    # old and the new room id of an update count, so nothing can be moved into a retired identity.
    for table, suffix in (("hosted_room_replicas", "room"), ("hosted_room_replica_events", "event")):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_{suffix}_insert
            BEFORE INSERT ON {table}
            WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id=NEW.room_id)
            BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_event_update
        BEFORE UPDATE ON hosted_room_replica_events
        WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id IN (OLD.room_id,NEW.room_id))
        BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    # Byte and timestamp bookkeeping stay writable; the copy's identity and coverage do not.
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_replica_retired_room_update
        BEFORE UPDATE ON hosted_room_replicas
        WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id IN (OLD.room_id,NEW.room_id))
          AND (NEW.room_id IS NOT OLD.room_id OR NEW.name IS NOT OLD.name
            OR NEW.members_json IS NOT OLD.members_json
            OR NEW.authority_gateway_id IS NOT OLD.authority_gateway_id
            OR NEW.authority_epoch IS NOT OLD.authority_epoch
            OR NEW.last_seq IS NOT OLD.last_seq OR NEW.latest_seq IS NOT OLD.latest_seq
            OR NEW.disbanded_at IS NOT OLD.disbanded_at)
        BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    initialize_retirement_guards(conn)


@contextmanager
def _transaction(db_path: Path | str):
    with rooms._transaction(db_path, immediate=True) as conn:
        _initialize(conn)
        yield conn


# -- the home ----------------------------------------------------------------------------------


def _proof_grant_locked(conn, enrollment):
    if not table_exists(conn, 'hosted_room_links'):
        return None
    from gateway.hosted_room_links import StoredRoomLink
    from gateway.hosted_room_peer import unverified_room_grant_claims
    for row in conn.execute('SELECT * FROM hosted_room_links WHERE room_id=? ORDER BY member_id',
                            (enrollment['room_id'],)):
        link = StoredRoomLink.from_record(dict(row))
        if (link.catalog.installation_id != enrollment['target_install_id']
                or link.target_url != enrollment['endpoint']):
            continue
        claims = unverified_room_grant_claims(link.grant)
        if all(claims.get(k) == enrollment[k] for k in
               ('room_id', 'authority_gateway_id', 'authority_epoch', 'target_install_id')):
            return link.grant  # this route already passed its installation-authenticated registration
    return None


def bind_home_proof(db_path, *, room_id):
    """Retain installation proof material privately before publishing or removing routes."""
    with rooms._transaction(db_path, immediate=True) as conn:
        if not table_exists(conn, HOME_TABLE):
            return  # plain text rooms have no retirement custody to maintain
        for row in conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE room_id=? AND proof_grant IS NULL "
                                "AND state NOT IN ('acknowledged','superseded','revoked')", (room_id,)).fetchall():
            grant = _proof_grant_locked(conn, row)
            if grant is not None:
                conn.execute(f'UPDATE {HOME_TABLE} SET proof_grant=? WHERE enrollment_id=?', (grant, row['enrollment_id']))


def prepare_home_enrollment(
    db_path: Path | str, *, room_id: str, target_install_id: str, endpoint: str, local_gateway_id: str,
    secret: bytes, enrollment_id: str | None = None, replace_enrollment_id: str | None = None,
) -> dict[str, Any]:
    """Reserve a cleanup obligation for one participant installation; idempotent per setup.

    Refused once the room is disbanded, quarantined or no longer this gateway's. A Disband
    committed afterwards closes the obligation in its own transaction.
    """
    room_id = _identifier(room_id, "room_id")
    target_install_id = _identifier(target_install_id, "target_install_id")
    local_gateway_id = _identifier(local_gateway_id, "local_gateway_id")
    endpoint, _ = validate_room_link_url(endpoint)
    if len(endpoint) > 2048 or re.search(r"/p/[^/]+$", endpoint):
        raise RetirementError("retirement requires an installation endpoint, not a profile endpoint")
    if enrollment_id is not None:
        enrollment_id = _identifier(enrollment_id, "enrollment_id")
    if replace_enrollment_id is not None and enrollment_id is None:
        raise RetirementError("replacement requires a new idempotent enrollment_id")
    with _transaction(db_path) as conn:
        room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
        if room is None or room["disbanded_at"] is not None:
            raise RetirementConflictError("Group Chat is not active")
        if room["authority_gateway_id"] != local_gateway_id:
            raise RetirementConflictError("retirement setup requires the current authority")
        if conn.execute("SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone():
            raise RetirementConflictError("Group Chat is quarantined")
        members = json.loads(room["members_json"])
        if target_install_id == local_gateway_id or not any(
            isinstance(m.get("target"), dict) and m["target"].get("kind") == "peer"
            and m["target"].get("installation_id") == target_install_id for m in members
        ):
            raise RetirementConflictError("target is not a participant gateway")
        scope = dict(room_id=room_id, authority_gateway_id=local_gateway_id, authority_epoch=room["authority_epoch"],
                     target_install_id=target_install_id, roster_sha256=roster_digest(members))
        existing = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)).fetchone()
        if existing is not None:
            if (not existing["is_current"] or any(existing[k] != v for k, v in scope.items())
                    or existing["endpoint"] != endpoint):
                raise RetirementConflictError("enrollment_id already belongs to another setup")
            _require_same_key(secret, existing)
            return _public(existing)
        current = conn.execute(
            f"SELECT * FROM {HOME_TABLE} WHERE room_id=? AND target_install_id=? AND is_current=1",
            (room_id, target_install_id)).fetchone()
        if current is not None and replace_enrollment_id is None:
            if any(current[k] != v for k, v in scope.items()) or current["endpoint"] != endpoint:
                raise RetirementConflictError("retirement enrollment changed")
            _require_same_key(secret, current)
            return _public(current)
        if replace_enrollment_id is not None and (current is None or current["enrollment_id"] != replace_enrollment_id):
            raise RetirementConflictError("retirement enrollment replacement lost its expected state")
        open_obligations = conn.execute(
            f"SELECT COUNT(*) FROM {HOME_TABLE} WHERE state NOT IN {_DONE}").fetchone()[0]
        if open_obligations >= MAX_PENDING_ENROLLMENTS:
            raise RetirementCapacityError("pending retirement delivery capacity is full")
        row = dict(scope, enrollment_id=enrollment_id or secrets.token_hex(16), endpoint=endpoint,
                   nonce=secrets.token_hex(16))
        row["commitment"] = _commitment(_signing_seed(secret, row))
        if current is not None:
            conn.execute(f"UPDATE {HOME_TABLE} SET is_current=0 WHERE enrollment_id=?", (current["enrollment_id"],))
        row['proof_grant'] = _proof_grant_locked(conn, row)
        fields = (*_PUBLIC_FIELDS, "endpoint", "nonce", "proof_grant")
        conn.execute(f"INSERT INTO {HOME_TABLE} ({','.join(fields)},created_at) "
                     f"VALUES ({','.join('?' for _ in fields)},?)", (*[row[k] for k in fields], time.time()))
        return _public(row)


def _require_same_key(secret: bytes, row: Mapping[str, Any]) -> None:
    if not hmac.compare_digest(_commitment(_signing_seed(secret, row)), row["commitment"]):
        raise RetirementKeyUnavailable("retirement key no longer matches enrollment")


def current_home_enrollment(db_path: Path | str, *, room_id: str, target_install_id: str) -> dict[str, Any] | None:
    with rooms._transaction(db_path) as conn:
        if not table_exists(conn, HOME_TABLE):
            return None
        row = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE room_id=? AND target_install_id=? AND is_current=1",
                           (room_id, target_install_id)).fetchone()
        return {**_public(row), "state": row["state"]} if row is not None else None


def confirm_home_enrollment(db_path: Path | str, *, enrollment_id: str, proof: Any) -> bool:
    """Record that the participant reports exactly this enrollment active (from its probe)."""
    with _transaction(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=? AND is_current=1 "
                           "AND state IN ('prepared','enrolled')", (enrollment_id,)).fetchone()
        if row is None:
            return False
        if (not isinstance(proof, Mapping) or proof.get("state") != "active"
                or type(proof.get("authority_epoch")) is not int
                or any(proof.get(k) != row[k] for k in _PUBLIC_FIELDS)):
            conn.execute(f"UPDATE {HOME_TABLE} SET last_error='retirement_enrollment_unconfirmed' "
                         "WHERE enrollment_id=?", (enrollment_id,))
            return False
        conn.execute(f"UPDATE {HOME_TABLE} SET state='enrolled',last_error=NULL WHERE enrollment_id=?",
                     (enrollment_id,))
        return True


def reconcile_home_close_locked(conn: sqlite3.Connection, room_id: str) -> None:
    """Close this room's obligations on the canonical Disband proof, in the caller's transaction."""
    if not table_exists(conn, HOME_TABLE):
        return
    room = conn.execute("SELECT authority_gateway_id,authority_epoch,disbanded_at FROM hosted_rooms WHERE room_id=?",
                        (room_id,)).fetchone()
    if room is not None and room["disbanded_at"] is not None:
        conn.execute(f"""UPDATE {HOME_TABLE} SET state='closed',closed_at=? WHERE room_id=? AND authority_gateway_id=?
            AND authority_epoch=? AND state IN ('prepared','enrolled')""",
                     (room["disbanded_at"], room_id, room["authority_gateway_id"], room["authority_epoch"]))


def _block_stale_home_locked(conn):
    conn.execute(f"""UPDATE {HOME_TABLE} SET state='blocked_authority',last_error='stale_authority'
        WHERE state NOT IN ('acknowledged','superseded','revoked','blocked_authority')
          AND EXISTS (SELECT 1 FROM hosted_rooms r WHERE r.room_id={HOME_TABLE}.room_id
            AND (r.authority_gateway_id!={HOME_TABLE}.authority_gateway_id
                 OR r.authority_epoch!={HOME_TABLE}.authority_epoch))""")


def has_open_obligations(conn: sqlite3.Connection) -> bool:
    """Whether a retirement still has to be delivered (or may become due): the publisher must run."""
    return table_exists(conn, HOME_TABLE) and conn.execute(
        f"SELECT 1 FROM {HOME_TABLE} WHERE state NOT IN {(*_DONE, 'blocked_authority')} LIMIT 1").fetchone() is not None


def pending_notice_ids(db_path: Path | str, *, local_gateway_id: str, limit: int = MAX_PENDING_ENROLLMENTS) -> list[str]:
    if type(limit) is not int or not 1 <= limit <= MAX_PENDING_ENROLLMENTS:
        raise RetirementError("invalid retirement notice limit")
    with _transaction(db_path) as conn:
        _block_stale_home_locked(conn)
        return [row[0] for row in conn.execute(
            f"""SELECT enrollment_id FROM {HOME_TABLE} WHERE authority_gateway_id=? AND state IN ('closed','ready')
                ORDER BY created_at,enrollment_id LIMIT ?""", (local_gateway_id, limit))]


def notice_route(db_path: Path | str, *, enrollment_id: str, local_gateway_id: str) -> dict[str, str] | None:
    """Read immutable routing without deriving or materializing a signature."""
    with _transaction(db_path) as conn:
        row = conn.execute(f"""SELECT room_id,target_install_id,endpoint FROM {HOME_TABLE} WHERE enrollment_id=?
            AND authority_gateway_id=? AND state IN ('closed','ready')""", (enrollment_id, local_gateway_id)).fetchone()
        return dict(row) if row is not None else None


def materialize_notice(
    db_path: Path | str, *, enrollment_id: str, local_gateway_id: str, secret_loader: Callable[[], bytes],
) -> RetirementNotice:
    """Sign the exact notice only after canonical Disband closes this obligation.

    The exact signature is kept until acknowledged, so lost delivery can retry even if the
    home signing key becomes unavailable.
    """
    with _transaction(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)).fetchone()
        if row is None or row["authority_gateway_id"] != local_gateway_id:
            raise RetirementConflictError("retirement notice is not owned by this gateway")
        _block_stale_home_locked(conn)
        reconcile_home_close_locked(conn, row["room_id"])
        row = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (enrollment_id,)).fetchone()
        if row["state"] not in {"closed", "ready"} or row["closed_at"] is None:
            raise RetirementConflictError("canonical Group Chat disband has not completed")
        value = row["closing_value"]
        if not value:
            try:
                value = _sign_notice(_signing_seed(secret_loader(), row), row)
            except (OSError, ValueError) as exc:
                raise RetirementKeyUnavailable("retirement key is unavailable") from exc
        if not _valid_signature(value, row):
            raise RetirementKeyUnavailable("retirement key no longer matches enrollment")
        conn.execute(f"UPDATE {HOME_TABLE} SET state='ready',closing_value=? WHERE enrollment_id=?",
                     (value, enrollment_id))
        return RetirementNotice(
            **{k: row[k] for k in ("enrollment_id", "room_id", "authority_gateway_id", "authority_epoch",
                                   "target_install_id", "endpoint")}, value=value, proof_grant=row["proof_grant"])


def acknowledge_notice(db_path: Path | str, *, notice: RetirementNotice, response: Mapping[str, Any]) -> None:
    """Complete an obligation on the participant's exact retirement receipt."""
    expected = notice.payload()
    if (not isinstance(response, Mapping) or response.get("retired") is not True
            or type(response.get("authority_epoch")) is not int
            or any(response.get(k) != v for k, v in expected.items())):
        raise RetirementConflictError("retirement acknowledgment differs from its enrollment")
    with _transaction(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {HOME_TABLE} WHERE enrollment_id=?", (notice.enrollment_id,)).fetchone()
        if row is None or any(row[k] != v for k, v in expected.items()) or row["state"] not in {"ready", "acknowledged"}:
            raise RetirementConflictError("retirement acknowledgment is no longer current")
        if response.get("commitment") != row["commitment"]:
            raise RetirementConflictError("retirement acknowledgment commitment differs")
        conn.execute(f"""UPDATE {HOME_TABLE} SET state='acknowledged',closing_value=NULL,proof_grant=NULL,
            acknowledged_at=COALESCE(acknowledged_at,?),last_error=NULL WHERE enrollment_id=?""",
                     (time.time(), notice.enrollment_id))
        # The copy is gone: earlier obligations for the same copy have nothing left to retire.
        conn.execute(f"""UPDATE {HOME_TABLE} SET state='superseded',closing_value=NULL,proof_grant=NULL,superseded_by=?
            WHERE room_id=? AND target_install_id=? AND authority_gateway_id=? AND authority_epoch=?
              AND roster_sha256=? AND enrollment_id!=? AND state NOT IN {_DONE}""",
                     (notice.enrollment_id, notice.room_id, notice.target_install_id, notice.authority_gateway_id,
                      notice.authority_epoch, row["roster_sha256"], notice.enrollment_id))


def record_delivery_error(db_path: Path | str, *, enrollment_id: str, code: str) -> None:
    if re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code) is None:
        raise RetirementError("invalid retirement error code")
    with _transaction(db_path) as conn:
        conn.execute(f"UPDATE {HOME_TABLE} SET last_error=? WHERE enrollment_id=? AND state IN ('closed','ready')",
                     (code, enrollment_id))


def home_status(db_path: Path | str, *, room_id: str | None = None) -> list[dict[str, Any]]:
    with _transaction(db_path) as conn:
        _block_stale_home_locked(conn)
        rows = conn.execute(f"""SELECT enrollment_id,room_id,target_install_id,authority_gateway_id,authority_epoch,
            state,closed_at,acknowledged_at,last_error,superseded_by FROM {HOME_TABLE} WHERE (? IS NULL OR room_id=?)
            ORDER BY CASE WHEN state IN {_DONE} THEN 1 ELSE 0 END,created_at,enrollment_id LIMIT ?""",
                            (room_id, room_id, MAX_PENDING_ENROLLMENTS * 2))
        return [dict(row) for row in rows]


# -- the participant ---------------------------------------------------------------------------


def _validate_enrollment(value: Any, target_install_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(_PUBLIC_FIELDS):
        raise RetirementError("invalid retirement enrollment fields")
    value = dict(value)
    for key in ("enrollment_id", "room_id", "authority_gateway_id", "target_install_id"):
        value[key] = _identifier(value[key], key)
    if type(value["authority_epoch"]) is not int or value["authority_epoch"] != 1:
        raise RetirementConflictError("a verified initial authority is required")
    if value["target_install_id"] != target_install_id or value["authority_gateway_id"] == target_install_id:
        raise RetirementConflictError("retirement enrollment targets another installation")
    _digest(value["roster_sha256"], "roster_sha256")
    _digest(value["commitment"], "commitment")
    return value


def _check_replica_namespace(conn: sqlite3.Connection, value: Mapping[str, Any]) -> sqlite3.Row | None:
    """The copy this enrollment names, if any; never a local room, quarantined or different copy."""
    from gateway.hosted_room_replicas import _audit_existing_replicas_locked

    _audit_existing_replicas_locked(conn)
    room_id = value["room_id"]
    reservation = conn.execute("SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?",
                               (room_id,)).fetchone()
    if ((reservation is not None and reservation[0] == "authority")
            or conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
            or conn.execute("SELECT 1 FROM hosted_room_retired_ids WHERE room_id=?", (room_id,)).fetchone()):
        raise RetirementConflictError("Group Chat is locally authoritative")
    if conn.execute("SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone():
        raise RetirementConflictError("quarantined Group Chat evidence must be preserved")
    replica = conn.execute("SELECT * FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
    if replica is not None and (
        replica["quarantine_reason"] is not None
        or replica["authority_gateway_id"] != value["authority_gateway_id"]
        or replica["authority_epoch"] != value["authority_epoch"]
        or roster_digest(json.loads(replica["members_json"])) != value["roster_sha256"]
    ):
        raise RetirementConflictError("retirement enrollment differs from the retained copy")
    if replica is None and reservation is not None:
        raise RetirementConflictError("retired copy identity must be preserved")
    return replica


def enroll_target(
    db_path: Path | str, *, enrollment: Mapping[str, Any], target_install_id: str,
    expected_enrollment_id: str | None = None, expected_state: str = "active",
) -> dict[str, Any]:
    """Enroll the home's public verifier by participant-operator authority, never a room grant.

    Replacing an enrollment names the one it replaces and its state, so a stale replacement can't
    reactivate a revoked one.
    """
    value = _validate_enrollment(enrollment, target_install_id)
    if expected_state not in {"active", "revoked"}:
        raise RetirementError("invalid expected enrollment state")
    with _transaction(db_path) as conn:
        if copy_retired_locked(conn, value["room_id"]):
            raise RetirementConflictError("a retired Group Chat copy cannot be reenrolled")
        _check_replica_namespace(conn, value)
        live = conn.execute("""SELECT authority_gateway_id,authority_epoch FROM hosted_room_peer_reservations
            WHERE room_id=? AND revoked_at IS NULL AND expires_at>?""", (value["room_id"], time.time())).fetchall()
        if any((r[0], r[1]) != (value["authority_gateway_id"], value["authority_epoch"]) for r in live):
            raise RetirementConflictError("retirement enrollment differs from a live room reservation")
        existing = conn.execute(f"SELECT * FROM {ENROLLMENT_TABLE} WHERE enrollment_id=?",
                                (value["enrollment_id"],)).fetchone()
        if existing is not None:
            if _public(existing) != value or not existing["is_current"]:
                raise RetirementConflictError("enrollment_id already belongs to another setup")
            if existing["state"] != "active":
                raise RetirementAuthorizationError("retirement capability has been revoked")
            return {**_public(existing), "state": "active"}
        current = conn.execute(f"SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND is_current=1",
                               (value["room_id"],)).fetchone()
        if current is not None:
            if current["enrollment_id"] != expected_enrollment_id or current["state"] != expected_state:
                raise RetirementConflictError("target enrollment replacement lost its expected state")
            if any(current[k] != value.get(k) for k in _SCOPE_FIELDS if k != "enrollment_id"):
                raise RetirementConflictError("implicit enrollment lineage changes are not allowed")
        elif expected_enrollment_id is not None:
            raise RetirementConflictError("expected target enrollment does not exist")
        active = conn.execute(f"SELECT COUNT(*) FROM {ENROLLMENT_TABLE} WHERE is_current=1 AND state='active'").fetchone()[0]
        if (current is None or current["state"] != "active") and active >= MAX_PENDING_ENROLLMENTS:
            raise RetirementCapacityError("target enrollment capacity is full")
        if current is not None:
            conn.execute(f"UPDATE {ENROLLMENT_TABLE} SET is_current=0 WHERE enrollment_id=?",
                         (current["enrollment_id"],))
        conn.execute(f"INSERT INTO {ENROLLMENT_TABLE} ({','.join(_PUBLIC_FIELDS)},created_at) "
                     f"VALUES ({','.join('?' for _ in _PUBLIC_FIELDS)},?)",
                     (*[value[k] for k in _PUBLIC_FIELDS], time.time()))
        return {**value, "state": "active"}


def revoke_target_enrollment(db_path: Path | str, *, room_id: str, enrollment_id: str) -> dict[str, Any]:
    """The participant operator withdraws future retirement only; member grants are untouched."""
    with _transaction(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND enrollment_id=? AND is_current=1",
                           (_identifier(room_id, "room_id"), _identifier(enrollment_id, "enrollment_id"))).fetchone()
        if row is None:
            raise RetirementConflictError("retirement enrollment is no longer current")
        conn.execute(f"UPDATE {ENROLLMENT_TABLE} SET state='revoked' WHERE enrollment_id=?", (enrollment_id,))
        return {"room_id": room_id, "enrollment_id": enrollment_id, "state": "revoked"}


def current_target_enrollment(
    db_path: Path | str, *, room_id: str, authority_gateway_id: str, authority_epoch: int,
) -> dict[str, Any] | None:
    """The active enrollment a probe reports, so the home can confirm it."""
    with rooms._transaction(db_path) as conn:
        if not table_exists(conn, ENROLLMENT_TABLE):
            return None
        row = conn.execute(f"""SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND authority_gateway_id=?
            AND authority_epoch=? AND is_current=1 AND state='active'""",
                           (room_id, authority_gateway_id, authority_epoch)).fetchone()
        return {**_public(row), "state": "active"} if row is not None else None


def copy_retired_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    return table_exists(conn, RETIREMENT_TABLE) and conn.execute(
        f"SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id=?", (room_id,)).fetchone() is not None


def copy_scope_matches_locked(
    conn: sqlite3.Connection, *, room_id: str, authority_gateway_id: str, authority_epoch: int, members_json: str,
) -> bool:
    """An enrollment fences what a sender may write into this copy, from the first page on."""
    row = conn.execute(f"SELECT * FROM {ENROLLMENT_TABLE} WHERE room_id=? AND is_current=1", (room_id,)).fetchone() \
        if table_exists(conn, ENROLLMENT_TABLE) else None
    if row is None:
        return True
    return (row["authority_gateway_id"], row["authority_epoch"], row["roster_sha256"]) == (
        authority_gateway_id, authority_epoch, hashlib.sha256(members_json.encode("utf-8")).hexdigest())


def _retired_response(row) -> dict[str, Any]:
    return {"retired": True, **dict(row)}


def retire_copy(db_path: Path | str, *, payload: Mapping[str, Any], value: str, local_gateway_id: str) -> dict[str, Any]:
    """Retire this participant's copy with its signed notice; idempotent per enrollment.

    An invalid signature never takes the writer or initializes anything: authorization is checked
    against the stored public verifier read-only first, then again inside the writer.
    """
    fields = {"enrollment_id", "room_id", "authority_gateway_id", "authority_epoch", "target_install_id"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise RetirementError("invalid retirement notice fields")
    payload = dict(payload)
    for key in fields - {"authority_epoch"}:
        payload[key] = _identifier(payload[key], key)
    if type(payload["authority_epoch"]) is not int or payload["authority_epoch"] != 1:
        raise RetirementError("invalid retirement authority epoch")
    if not isinstance(value, str) or re.fullmatch(r"ed25519-v2\.[A-Za-z0-9_-]{86}", value) is None:
        raise RetirementAuthorizationError("invalid retirement capability")
    if not Path(db_path).is_file():
        raise RetirementAuthorizationError("retirement enrollment is unavailable")
    with closing(sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=0.25)) as read:
        read.row_factory = sqlite3.Row
        if not table_exists(read, ENROLLMENT_TABLE):
            raise RetirementAuthorizationError("retirement enrollment is unavailable")
        enrolled = read.execute(f"SELECT * FROM {ENROLLMENT_TABLE} WHERE enrollment_id=?",
                                (payload["enrollment_id"],)).fetchone()
        if enrolled is None or not _valid_signature(value, enrolled):
            raise RetirementAuthorizationError("invalid retirement capability")
        if any(payload[k] != enrolled[k] for k in fields) or enrolled["target_install_id"] != local_gateway_id:
            raise RetirementAuthorizationError("retirement capability scope differs")
        if table_exists(read, RETIREMENT_TABLE):
            done = read.execute(f"SELECT * FROM {RETIREMENT_TABLE} WHERE room_id=?", (payload["room_id"],)).fetchone()
            if done is not None:
                if done["enrollment_id"] != payload["enrollment_id"]:
                    raise RetirementConflictError("copy was retired under another enrollment")
                return _retired_response(done)
    with _transaction(db_path) as conn:
        row = conn.execute(f"SELECT * FROM {ENROLLMENT_TABLE} WHERE enrollment_id=?",
                           (payload["enrollment_id"],)).fetchone()
        if row is None or not _valid_signature(value, row):
            raise RetirementAuthorizationError("invalid retirement capability")
        if any(payload[k] != row[k] for k in fields) or row["target_install_id"] != local_gateway_id:
            raise RetirementAuthorizationError("retirement capability scope differs")
        retired = conn.execute(f"SELECT * FROM {RETIREMENT_TABLE} WHERE room_id=?", (row["room_id"],)).fetchone()
        if retired is not None:
            if retired["enrollment_id"] != row["enrollment_id"]:
                raise RetirementConflictError("copy was retired under another enrollment")
            return _retired_response(retired)
        if not row["is_current"] or row["state"] != "active":
            raise RetirementAuthorizationError("retirement capability has been revoked or replaced")
        replica = _check_replica_namespace(conn, row)
        now = time.time()
        conn.execute("INSERT OR IGNORE INTO hosted_room_id_reservations(room_id,owner_kind,reserved_at) "
                     "VALUES (?,'replica',?)", (row["room_id"], now))
        conn.execute(f"""INSERT INTO {RETIREMENT_TABLE} (room_id,enrollment_id,authority_gateway_id,authority_epoch,
            target_install_id,commitment,retired_at,stored_seq,source_latest_seq) VALUES (?,?,?,?,?,?,?,?,?)""",
                     (row["room_id"], row["enrollment_id"], row["authority_gateway_id"], row["authority_epoch"],
                      row["target_install_id"], row["commitment"], now,
                      int(replica["last_seq"]) if replica is not None else 0,
                      int(replica["latest_seq"]) if replica is not None else 0))
        conn.execute(f"UPDATE {ENROLLMENT_TABLE} SET state='retired' WHERE enrollment_id=?", (row["enrollment_id"],))
        return _retired_response(conn.execute(f"SELECT * FROM {RETIREMENT_TABLE} WHERE room_id=?",
                                              (row["room_id"],)).fetchone())
