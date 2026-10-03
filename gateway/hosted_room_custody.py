"""Custody of a Group Chat's history: custodians, successors, watermarks and the tail at risk.

Every member installation that keeps its copy (``replicate``), and any custodian-only installation
the room owner added, is a custodian: it holds the room's whole history. The room's custodians are
a log event, ``custody.configured``, carrying each one's pinned room identity key, endpoint, display
name and whether it may continue the group (``successor``): only when its own operator consented
(grant permission ``successor``) and the room owner designated it. No member is ever picked by an
algorithm, and nothing here votes.

Each copy, and the authority's own room, has a durable watermark ``(epoch, seq, event_hash)``:
``event_hash`` chains every event of that exact prefix. Custodians acknowledge each page with it, and
the authority keeps the acknowledgments that match its own chain. With every page and heartbeat the
host also sends a **head** it signs, ``{room_id, host, epoch, seq, chain_hash}``, and each custodian
keeps the latest head that vouches for its copy; a host that continues its own group at a fresh epoch
keeps its own last head of the epoch it leaves. Copies pass history between themselves only as far
as such a head vouches for it, so no custodian can add to the group's history, keys or voters.
``at_risk_after_seq`` is the highest seq at least one eligible successor durably holds; every later
event is at risk of being lost with this host, and clients say so. Nothing waits for a copy:
replication degrades to local acknowledgement and marks the tail at risk. A task made ready for
dispatch is announced with ``task.admitted`` in the same transaction, so a successor can reconcile it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import secrets
import socket
import sqlite3
import time
from contextlib import closing
from typing import Any, Mapping

from gateway import hosted_room_identity as identity
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import DbPath, compact_json, open_sqlite, table_exists, utf8_len
from gateway.hosted_rooms_common import display_label as common_display_label

logger = logging.getLogger(__name__)

CONFIGURED = "custody.configured"
TASK_ADMITTED = "task.admitted"
CUSTODIANS_TABLE = "hosted_room_custodians"
WATERMARKS_TABLE = "hosted_room_custody_watermarks"
REPORTS_TABLE = "hosted_room_custody_reports"
CHAIN_TABLE = "hosted_room_custody_chain"
CONSENT_TABLE = "hosted_room_custody_consent"
ROUTES_TABLE = "hosted_room_custody_routes"
HEADS_TABLE = "hosted_room_custody_heads"
# The domain of the head a host signs for its own room's prefix.
HEAD_DOMAIN = b"hermes.group.custody.head.v1"
_HEAD_FIELDS = frozenset({"room_id", "host", "epoch", "seq", "chain_hash"})
# The member id a custodian-only grant carries: it names no Bot, and such a grant never runs work.
CUSTODY_MEMBER_ID = "custody:installation"
ROLES = frozenset({"authority", "custodian", "custodian_only"})
MAX_CUSTODIANS = rooms.MAX_MEMBERS + 16
_CHAIN_DOMAIN = b"hermes.room.custody.chain.v1\0"
_CHECKPOINT_EVERY = 128
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_CUSTODIAN_FIELDS = frozenset({"install_id", "public_key", "endpoint", "role", "successor", "name", "operator_name"})


class CustodyError(rooms.HostedRoomError):
    """A custody record, watermark or configuration is invalid or unavailable."""

    reason = "room_custody_invalid"


def initialize_locked(conn: sqlite3.Connection) -> None:
    identity.initialize_locked(conn)
    # The home's custody enrollments. ``allowed``: the installation's operator allowed it to continue
    # the group; ``designated``: the room owner chose it. Only both make it a successor.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {CUSTODIANS_TABLE} (
        room_id TEXT NOT NULL, install_id TEXT NOT NULL, role TEXT NOT NULL, state TEXT NOT NULL,
        endpoint TEXT, name TEXT, operator_name TEXT, allowed INTEGER NOT NULL DEFAULT 0,
        designated INTEGER NOT NULL DEFAULT 0, enrolled_at REAL NOT NULL, updated_at REAL NOT NULL,
        PRIMARY KEY (room_id, install_id))""")
    # The copy-only route on the host to an installation no member route reaches: a custodian-only
    # installation (no Bot), or a custodian whose Bots the host can't reach, such as the previous host.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {ROUTES_TABLE} (
        room_id TEXT NOT NULL, install_id TEXT NOT NULL, target_url TEXT NOT NULL, target_profile TEXT NOT NULL,
        grant TEXT NOT NULL, catalog_json TEXT NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (room_id, install_id))""")
    # On a member installation: whether its operator allows it to continue each room it keeps.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {CONSENT_TABLE} (
        room_id TEXT PRIMARY KEY, allowed INTEGER NOT NULL, host_allowed INTEGER, updated_at REAL NOT NULL)""")
    # Acknowledged watermarks that matched this authority's own chain.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {WATERMARKS_TABLE} (
        room_id TEXT NOT NULL, install_id TEXT NOT NULL, epoch INTEGER NOT NULL, seq INTEGER NOT NULL,
        event_hash TEXT NOT NULL, acknowledged_at REAL NOT NULL, state TEXT NOT NULL DEFAULT 'verified',
        PRIMARY KEY (room_id, install_id))""")
    # What the authority last told this custodian about the room's tail at risk.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {REPORTS_TABLE} (
        room_id TEXT PRIMARY KEY, at_risk_after_seq INTEGER NOT NULL, reported_at REAL NOT NULL)""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {CHAIN_TABLE} (
        room_id TEXT NOT NULL, seq INTEGER NOT NULL, event_hash TEXT NOT NULL, PRIMARY KEY (room_id, seq))""")
    # On a custodian: per epoch, the latest head that host signed and that vouches for this copy.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {HEADS_TABLE} (
        room_id TEXT NOT NULL, epoch INTEGER NOT NULL, host TEXT NOT NULL, seq INTEGER NOT NULL,
        chain_hash TEXT NOT NULL, signature TEXT NOT NULL, received_at REAL NOT NULL, PRIMARY KEY (room_id, epoch))""")


def display_label(value: Any) -> str | None:
    """A bounded, printable display label; never used for identity or authorization."""
    return common_display_label(value, max_chars=rooms.MAX_ACTOR_LABEL_CHARS)


def local_names() -> tuple[str | None, str | None]:
    """This installation's display name (``gateway.display_name``, else the host name) and owner name."""
    try:
        from gateway.run import _load_gateway_config
        section = _load_gateway_config().get("gateway") or {}
    except Exception:
        section = {}
    section = section if isinstance(section, Mapping) else {}
    try:
        host = socket.gethostname()
    except OSError:
        host = None
    return display_label(section.get("display_name")) or display_label(host), display_label(section.get("owner_name"))


# -- the hash chain ----------------------------------------------------------------------------


def _genesis(room_id: str) -> str:
    return hashlib.sha256(_CHAIN_DOMAIN + room_id.encode("utf-8")).hexdigest()


def _material(row: Mapping[str, Any]) -> bytes:
    """One event as both sides store it: normalized like a copy's ingest normalizes a page."""
    kind = str(row["kind"])
    try:
        _, actor_json = rooms._validate_actor(json.loads(row["actor_json"]), kind=kind)
    except (rooms.HostedRoomError, ValueError, TypeError, RecursionError):
        actor_json = str(row["actor_json"])
    try:
        payload_json = rooms._payload_json(json.loads(row["payload_json"]))
    except (rooms.HostedRoomError, ValueError, TypeError, RecursionError):
        payload_json = str(row["payload_json"])
    epoch = row["authority_epoch"]
    return compact_json([
        int(row["seq"]), str(row["event_id"]), kind, actor_json, None if epoch is None else int(epoch),
        payload_json, float(row["created_at"])], ensure_ascii=False).encode("utf-8")


def _fold(previous: str, row: Mapping[str, Any]) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + _material(row)).hexdigest()


def events_table_locked(conn: sqlite3.Connection, room_id: str) -> str | None:
    """Where this store keeps the room's log: its own room, or a copy of another gateway's."""
    if conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone():
        return "hosted_room_events"
    if table_exists(conn, "hosted_room_replicas") and conn.execute(
            "SELECT 1 FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone():
        return "hosted_room_replica_events"
    return None


def chain_hash_locked(
    conn: sqlite3.Connection, room_id: str, seq: int, *, table: str | None = None, store: bool = True,
) -> str:
    """``event_hash`` of the exact prefix ``1..seq`` held here; checkpoints keep each call bounded.

    ``store`` keeps new checkpoints, inside a writer only.
    """
    if seq == 0:
        return _genesis(room_id)
    table = table or events_table_locked(conn, room_id)
    if table is None:
        raise CustodyError("no history of this Group Chat is held here")
    start, value = 0, _genesis(room_id)
    if table_exists(conn, CHAIN_TABLE):
        row = conn.execute(f"SELECT seq, event_hash FROM {CHAIN_TABLE} WHERE room_id=? AND seq<=? "
                           "ORDER BY seq DESC LIMIT 1", (room_id, seq)).fetchone()
        if row is not None:
            start, value = int(row[0]), str(row[1])
    expected, checkpoints = start + 1, []
    for event in conn.execute(
            f"""SELECT seq, event_id, kind, actor_json, authority_epoch, payload_json, created_at FROM {table}
                WHERE room_id=? AND seq>? AND seq<=? ORDER BY seq""", (room_id, start, seq)):
        if int(event["seq"]) != expected:
            raise CustodyError("the held history prefix is incomplete")
        value = _fold(value, event)
        if expected % _CHECKPOINT_EVERY == 0:
            checkpoints.append((room_id, expected, value))
        expected += 1
    if expected != seq + 1:
        raise CustodyError("the held history prefix is incomplete")
    if store and checkpoints and table_exists(conn, CHAIN_TABLE):
        conn.executemany(f"INSERT OR IGNORE INTO {CHAIN_TABLE} (room_id, seq, event_hash) VALUES (?,?,?)",
                         checkpoints)
    return value


def reset_chain_locked(conn: sqlite3.Connection, room_id: str, *, after_seq: int) -> None:
    """Forget derived hashes and heads past ``after_seq``: required wherever stored history is rewritten."""
    if table_exists(conn, CHAIN_TABLE):
        conn.execute(f"DELETE FROM {CHAIN_TABLE} WHERE room_id=? AND seq>?", (room_id, after_seq))
    if table_exists(conn, HEADS_TABLE):
        conn.execute(f"DELETE FROM {HEADS_TABLE} WHERE room_id=? AND seq>?", (room_id, after_seq))


def custody_watermark_locked(conn: sqlite3.Connection, room_id: str, *, store: bool = True) -> dict[str, Any] | None:
    """This store's durable watermark for the room, read inside the caller's transaction."""
    hosted = conn.execute("SELECT authority_epoch, next_seq FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if hosted is not None:
        table, epoch, seq = "hosted_room_events", int(hosted["authority_epoch"]), int(hosted["next_seq"]) - 1
    else:
        copy = conn.execute("SELECT authority_epoch, last_seq FROM hosted_room_replicas WHERE room_id=?",
                            (room_id,)).fetchone() if table_exists(conn, "hosted_room_replicas") else None
        if copy is None:
            return None
        table, epoch, seq = "hosted_room_replica_events", int(copy["authority_epoch"]), int(copy["last_seq"])
    if seq:
        last = conn.execute(f"SELECT authority_epoch FROM {table} WHERE room_id=? AND seq=?", (room_id, seq)).fetchone()
        if last is not None and last[0] is not None:
            epoch = int(last[0])
    return {"epoch": epoch, "seq": seq, "event_hash": chain_hash_locked(conn, room_id, seq, table=table, store=store)}


def validate_watermark(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"epoch", "seq", "event_hash"}:
        raise CustodyError("watermark must carry exactly epoch, seq and event_hash")
    epoch, seq, event_hash = value["epoch"], value["seq"], value["event_hash"]
    if type(epoch) is not int or not 1 <= epoch < 2**63 or type(seq) is not int or not 0 <= seq < 2**63:
        raise CustodyError("watermark coordinates are invalid")
    if not isinstance(event_hash, str) or _HASH_RE.fullmatch(event_hash) is None:
        raise CustodyError("watermark event_hash is invalid")
    return {"epoch": epoch, "seq": seq, "event_hash": event_hash}


# -- heads the host signs ----------------------------------------------------------------------


def _head_statement(head: Any) -> dict[str, Any]:
    """A head's signed statement ``{room_id, host, epoch, seq, chain_hash}``, in its exact shape."""
    if not isinstance(head, Mapping) or set(head) != _HEAD_FIELDS | {"signature"}:
        raise CustodyError("a head carries exactly room_id, host, epoch, seq, chain_hash and signature")
    statement = {key: head[key] for key in sorted(_HEAD_FIELDS)}
    if (not isinstance(statement["room_id"], str) or not isinstance(statement["host"], str)
            or type(statement["epoch"]) is not int or not 1 <= statement["epoch"] < 2**63
            or type(statement["seq"]) is not int or not 0 <= statement["seq"] < 2**63
            or not isinstance(statement["chain_hash"], str) or _HASH_RE.fullmatch(statement["chain_hash"]) is None):
        raise CustodyError("head fields are invalid")
    return statement


def sign_head_locked(conn: sqlite3.Connection, room_id: str, *, seq: int | None = None) -> dict[str, Any]:
    """The head this host signs for its own room: its epoch, and the chain hash of the prefix ``1..seq``.

    ``seq`` defaults to the latest event. A custodian keeps the head that vouches for its copy, and
    custodians pass history between themselves only as far as a head vouches for it.
    """
    host = rooms.local_authority_gateway_id()
    room = conn.execute("SELECT authority_gateway_id, authority_epoch, next_seq FROM hosted_rooms WHERE room_id=?",
                        (room_id,)).fetchone()
    if room is None or room["authority_gateway_id"] != host:
        raise CustodyError("only the Group Chat's host signs its head")
    latest = int(room["next_seq"]) - 1
    seq = latest if seq is None else seq
    if type(seq) is not int or not 0 <= seq <= latest:
        raise CustodyError("a head names only history its host holds")
    statement = {"room_id": room_id, "host": host, "epoch": int(room["authority_epoch"]), "seq": seq,
                 "chain_hash": chain_hash_locked(conn, room_id, seq, table="hosted_room_events", store=False)}
    return {**statement, "signature": identity.sign(HEAD_DOMAIN, statement)}


def verify_head_locked(conn: sqlite3.Connection, room_id: str, head: Any) -> dict[str, Any]:
    """A head's statement, once its signature checks against the key this store pinned for its ``host``.

    It says nothing about lineage: the caller checks that ``(host, epoch)`` is an authority it follows.
    """
    statement = _head_statement(head)
    if statement["room_id"] != room_id or not identity.verify_locked(
            conn, room_id, statement["host"], HEAD_DOMAIN, statement, head["signature"]):
        raise CustodyError("the head is not signed with its host's pinned key")
    return statement


def _copy_head_locked(conn: sqlite3.Connection, room_id: str) -> tuple[str, int, int] | None:
    """``(authority, epoch, last_seq)`` of the copy held here; None without one, or for a quarantined copy."""
    if not table_exists(conn, "hosted_room_replicas"):
        return None
    row = conn.execute("""SELECT authority_gateway_id, authority_epoch, last_seq FROM hosted_room_replicas
        WHERE room_id=? AND quarantine_reason IS NULL""", (room_id,)).fetchone()
    return (str(row[0]), int(row[1]), int(row[2])) if row is not None else None


def record_head_locked(conn: sqlite3.Connection, room_id: str, head: Any) -> bool:
    """Keep a head that vouches for this copy's own prefix; returns whether it was kept.

    It must be signed by the host this copy follows (``verify_head_locked``), name that authority and
    epoch, and match this copy's chain at ``seq``. The latest head per epoch is kept; one from an
    earlier epoch stays valid for its prefix.
    """
    initialize_locked(conn)
    copy = _copy_head_locked(conn, room_id)
    try:
        statement = verify_head_locked(conn, room_id, head)
        if (copy is None or (statement["host"], statement["epoch"]) != copy[:2] or statement["seq"] > copy[2]
                or chain_hash_locked(conn, room_id, statement["seq"], table="hosted_room_replica_events")
                != statement["chain_hash"]):
            return False
    except CustodyError:
        return False
    _store_head_locked(conn, room_id, statement, head["signature"])
    return True


def _store_head_locked(conn: sqlite3.Connection, room_id: str, statement: Mapping[str, Any], signature: str) -> None:
    """Keep the latest head per epoch: a later head of the same epoch replaces an earlier one."""
    conn.execute(f"""INSERT INTO {HEADS_TABLE} (room_id, epoch, host, seq, chain_hash, signature, received_at)
        VALUES (?,?,?,?,?,?,?) ON CONFLICT(room_id, epoch) DO UPDATE SET host=excluded.host, seq=excluded.seq,
        chain_hash=excluded.chain_hash, signature=excluded.signature, received_at=excluded.received_at
        WHERE excluded.seq >= {HEADS_TABLE}.seq""",
                 (room_id, statement["epoch"], statement["host"], statement["seq"], statement["chain_hash"],
                  signature, time.time()))


def record_head(db_path: DbPath, room_id: str, head: Any) -> bool:
    """``record_head_locked`` in its own writer."""
    with rooms._transaction(db_path, immediate=True) as conn:
        return record_head_locked(conn, rooms._room_id(room_id), head)


def keep_own_head_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any]:
    """On the host, in the writer that continues its own group at a fresh epoch: keep its own head
    for the epoch it leaves, and return it.

    Call it before the change of host is appended, so the head names this host's current epoch and
    its latest event, the one the change directly follows. A host signs heads for its current epoch
    only, and its copies may have missed that epoch's last pushes: kept here, the head still vouches
    for them (``heads_locked``), so a copy behind the change can catch up across it.
    """
    initialize_locked(conn)
    head = sign_head_locked(conn, room_id)
    _store_head_locked(conn, room_id, head, head["signature"])
    return head


def vouched_head_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any] | None:
    """The head that vouches for the history held here, for receipts and catch-up; None without one.

    On the host, a head it signs now for its latest event. On a copy, the stored head of the host it
    follows, as long as it still matches the copy's chain.
    """
    room = conn.execute("SELECT authority_gateway_id FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL",
                        (room_id,)).fetchone()
    try:
        if room is not None:
            return sign_head_locked(conn, room_id) if room[0] == rooms.local_authority_gateway_id() else None
        copy = _copy_head_locked(conn, room_id)
        if copy is None or not table_exists(conn, HEADS_TABLE):
            return None
        row = conn.execute(f"SELECT * FROM {HEADS_TABLE} WHERE room_id=? AND epoch=? AND host=?",
                           (room_id, copy[1], copy[0])).fetchone()
        if row is None or int(row["seq"]) > copy[2] or chain_hash_locked(
                conn, room_id, int(row["seq"]), table="hosted_room_replica_events", store=False) != row["chain_hash"]:
            return None
    except (CustodyError, identity.RoomIdentityError):
        return None
    return {"room_id": room_id, "host": row["host"], "epoch": int(row["epoch"]), "seq": int(row["seq"]),
            "chain_hash": row["chain_hash"], "signature": row["signature"]}


def heads_locked(conn: sqlite3.Connection, room_id: str) -> list[dict[str, Any]]:
    """Every stored head that still vouches for the history held here, one per epoch, oldest first.

    A head from an earlier epoch stays valid for its prefix: a copy that has moved on can still vouch
    for that part of the history to a copy that hasn't.
    """
    table = events_table_locked(conn, room_id)
    if table is None or not table_exists(conn, HEADS_TABLE):
        return []
    kept = []
    for row in conn.execute(f"SELECT * FROM {HEADS_TABLE} WHERE room_id=? ORDER BY epoch", (room_id,)).fetchall():
        try:
            if chain_hash_locked(conn, room_id, int(row["seq"]), table=table, store=False) != row["chain_hash"]:
                continue
        except CustodyError:
            continue
        kept.append({"room_id": room_id, "host": row["host"], "epoch": int(row["epoch"]), "seq": int(row["seq"]),
                     "chain_hash": row["chain_hash"], "signature": row["signature"]})
    return kept


# -- configurations ----------------------------------------------------------------------------


def _custodian(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CUSTODIAN_FIELDS:
        raise CustodyError("a custodian carries exactly install_id, public_key, endpoint, role, successor and name")
    endpoint = value["endpoint"]
    if endpoint is not None:
        from gateway.hosted_room_peer import HostedRoomPeerError, validate_room_link_url
        try:
            endpoint, _ = validate_room_link_url(endpoint)
        except HostedRoomPeerError as exc:
            raise CustodyError("custodian endpoint is invalid") from exc
    if value["role"] not in ROLES or type(value["successor"]) is not bool:
        raise CustodyError("custodian role or successor flag is invalid")
    if value["role"] == "authority" and value["successor"]:
        raise CustodyError("the current host is not its own successor")
    for label in ("name", "operator_name"):
        if value[label] is not None and display_label(value[label]) != value[label]:
            raise CustodyError(f"custodian {label} is not a clean display label")
    try:
        public_key = identity.public_key_of(value["public_key"])
    except identity.RoomIdentityError as exc:
        raise CustodyError("custodian key is invalid") from exc
    return {"install_id": rooms._validate_identifier(value["install_id"], label="install_id", max_chars=128),
            "public_key": public_key, "endpoint": endpoint, "role": value["role"],
            "successor": value["successor"], "name": value["name"], "operator_name": value["operator_name"]}


def parse_configuration(payload: Any) -> dict[str, Any]:
    """A ``custody.configured`` payload: ``{custodians, owner_name}``, custodians sorted and distinct."""
    if not isinstance(payload, Mapping) or set(payload) != {"custodians", "owner_name"}:
        raise CustodyError("custody configuration fields are invalid")
    custodians, owner_name = payload["custodians"], payload["owner_name"]
    if not isinstance(custodians, list) or not 1 <= len(custodians) <= MAX_CUSTODIANS:
        raise CustodyError("custody configuration custodians are invalid")
    if owner_name is not None and display_label(owner_name) != owner_name:
        raise CustodyError("owner name is not a clean display label")
    parsed = [_custodian(custodian) for custodian in custodians]
    ids = [custodian["install_id"] for custodian in parsed]
    if ids != sorted(set(ids)) or sum(custodian["role"] == "authority" for custodian in parsed) != 1:
        raise CustodyError("custodians must be sorted and distinct, with exactly one current host")
    return {"custodians": parsed, "owner_name": owner_name}


def configurations_locked(conn: sqlite3.Connection, room_id: str) -> list[dict[str, Any]]:
    """Every configuration in this store's log for the room, in log order."""
    table = events_table_locked(conn, room_id)
    if table is None:
        return []
    return [{"seq": int(row["seq"]), **parse_configuration(json.loads(row["payload_json"]))}
            for row in conn.execute(f"SELECT seq, payload_json FROM {table} WHERE room_id=? AND kind=? ORDER BY seq",
                                    (room_id, CONFIGURED))]


def configuration_locked(conn: sqlite3.Connection, room_id: str) -> dict[str, Any]:
    """The latest configuration in this store's log: ``{configuration_seq, custodians, owner_name}``."""
    configurations = configurations_locked(conn, room_id)
    if not configurations:
        return {"configuration_seq": 0, "custodians": [], "owner_name": None}
    latest = configurations[-1]
    return {"configuration_seq": latest["seq"], "custodians": latest["custodians"], "owner_name": latest["owner_name"]}


def has_custody_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    """Whether another installation keeps the room's history beside its host."""
    return len(configuration_locked(conn, room_id)["custodians"]) > 1


def at_risk_after_locked(conn: sqlite3.Connection, room_id: str) -> int:
    """On the authority: the highest seq at least one eligible successor durably holds (0: none)."""
    successors = {custodian["install_id"] for custodian in configuration_locked(conn, room_id)["custodians"]
                  if custodian["successor"]}
    if not successors or not table_exists(conn, WATERMARKS_TABLE):
        return 0
    held = [int(row[1]) for row in conn.execute(
        f"SELECT install_id, seq FROM {WATERMARKS_TABLE} WHERE room_id=? AND state='verified'", (room_id,))
        if str(row[0]) in successors]
    return max(held, default=0)


# -- the home's custody records ----------------------------------------------------------------


def _append_system_event_locked(
    conn: sqlite3.Connection, room_id: str, *, event_id: str, kind: str, actor_id: str, payload: Mapping[str, Any],
    now: float,
) -> int:
    """Append one system event in the caller's transaction; the same event id is idempotent."""
    actor_json, payload_json = rooms._system_actor_json(actor_id), rooms._payload_json(dict(payload))
    existing = conn.execute("SELECT seq, kind, payload_json FROM hosted_room_events WHERE room_id=? AND event_id=?",
                            (room_id, event_id)).fetchone()
    if existing is not None:
        if (existing["kind"], existing["payload_json"]) != (kind, payload_json):
            raise rooms.EventConflictError("event_id already exists with different content")
        return int(existing["seq"])
    room = conn.execute("""SELECT next_seq, event_bytes, authority_gateway_id, authority_epoch FROM hosted_rooms
        WHERE room_id=? AND disbanded_at IS NULL""", (room_id,)).fetchone()
    if room is None:
        raise rooms.RoomNotFoundError("hosted room not found")
    seq = int(room["next_seq"])
    added = rooms._insert_event(conn, room, room_id, seq, event_id, kind, actor_json, int(room["authority_epoch"]),
                                payload_json, now)
    conn.execute("UPDATE hosted_rooms SET next_seq=?, event_bytes=event_bytes+?, updated_at=? WHERE room_id=? "
                 "AND next_seq=?", (seq + 1, added, now, room_id, seq))
    return seq


def enroll_custodian(
    db_path: DbPath, *, room_id: str, install_id: str, public_key: str | None, endpoint: str | None, name: Any,
    operator_name: Any = None, role: str, active: bool, allowed: bool, designated: bool | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Record one installation's custody enrollment on the home and pin its identity key.

    ``active`` is False when its operator opted out of the copy: it stays listed, without a copy to
    count. ``allowed`` is its operator's consent to continue the group; ``designated``, when given, is
    the room owner's choice (otherwise the earlier one stands). An installation that offers no key
    runs an older Hermes: it is ``unsupported``, and never counts as holding history.
    """
    if role not in ROLES - {"authority"}:
        raise CustodyError("custody role is invalid")
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        if public_key is not None:
            identity.pin_locked(conn, room_id=room_id, install_id=install_id, public_key=public_key,
                                source="custody_enrollment")
        state = "opted_out" if not active else "active" if public_key is not None else "unsupported"
        conn.execute(f"""INSERT INTO {CUSTODIANS_TABLE} (room_id, install_id, role, state, endpoint, name,
            operator_name, allowed, designated, enrolled_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(room_id, install_id) DO UPDATE SET
            role=CASE WHEN {CUSTODIANS_TABLE}.role='custodian' THEN 'custodian' ELSE excluded.role END,
            state=excluded.state, endpoint=excluded.endpoint, name=excluded.name, operator_name=excluded.operator_name,
            allowed=excluded.allowed,
            designated=CASE WHEN ? THEN excluded.designated ELSE {CUSTODIANS_TABLE}.designated END,
            updated_at=excluded.updated_at""",
                     (room_id, install_id, role, state, endpoint, display_label(name), display_label(operator_name),
                      int(allowed), int(bool(designated)), now, now, designated is not None))
        return dict(conn.execute(f"SELECT * FROM {CUSTODIANS_TABLE} WHERE room_id=? AND install_id=?",
                                 (room_id, install_id)).fetchone())


def designate_successor(db_path: DbPath, *, room_id: str, install_id: str, successor: bool,
                        now: float | None = None) -> dict[str, Any]:
    """The room owner designates (or no longer designates) one custodian to continue the group."""
    if type(successor) is not bool:
        raise CustodyError("successor must be a boolean")
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        if conn.execute(f"UPDATE {CUSTODIANS_TABLE} SET designated=?, updated_at=? WHERE room_id=? AND install_id=? "
                        "AND state!='withdrawn'", (int(successor), now, room_id, install_id)).rowcount != 1:
            raise CustodyError("this installation keeps no copy of the Group Chat")
        return dict(conn.execute(f"SELECT * FROM {CUSTODIANS_TABLE} WHERE room_id=? AND install_id=?",
                                 (room_id, install_id)).fetchone())


def record_allowed(db_path: DbPath, *, room_id: str, install_id: str, allowed: bool, now: float | None = None) -> bool:
    """A custodian reported its operator's current consent (an acknowledgment or a probe); True if it changed."""
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        return conn.execute(f"UPDATE {CUSTODIANS_TABLE} SET allowed=?, updated_at=? WHERE room_id=? AND install_id=? "
                            "AND allowed!=?", (int(allowed), now, room_id, install_id, int(allowed))).rowcount == 1


def mark_unsupported(db_path: DbPath, *, room_id: str, install_id: str, now: float | None = None) -> None:
    """A custodian acknowledged a page without a custody watermark: an older Hermes, never counted."""
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        conn.execute(f"UPDATE {CUSTODIANS_TABLE} SET state='unsupported', updated_at=? WHERE room_id=? AND install_id=? "
                     "AND state='active'", (now, room_id, install_id))
        conn.execute(f"DELETE FROM {WATERMARKS_TABLE} WHERE room_id=? AND install_id=?", (room_id, install_id))


def save_custody_route(
    db_path: DbPath, *, room_id: str, install_id: str, target_url: str, target_profile: str, grant: str,
    catalog: Mapping[str, Any], now: float | None = None,
) -> None:
    """Keep, on the host, the copy-only grant and endpoint of an installation no member route reaches.

    That is a custodian-only installation, or a custodian whose Bots the host has no route to, such
    as the previous host after a move: the host copies the history there on this grant. Before it
    runs out, the custodian hands back a renewed one in an acknowledgment, kept here the same way.
    """
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        conn.execute(f"""INSERT INTO {ROUTES_TABLE} (room_id, install_id, target_url, target_profile, grant,
            catalog_json, updated_at) VALUES (?,?,?,?,?,?,?) ON CONFLICT(room_id, install_id) DO UPDATE SET
            target_url=excluded.target_url, target_profile=excluded.target_profile, grant=excluded.grant,
            catalog_json=excluded.catalog_json, updated_at=excluded.updated_at""",
                     (room_id, install_id, target_url, target_profile, grant, compact_json(dict(catalog)), now))


def remove_custody_route(db_path: DbPath, *, room_id: str, install_id: str, now: float | None = None) -> bool:
    """Stop keeping a copy on one custodian-only installation; members leave only by opting out.

    The copy there is deleted through copy retirement, which its operator enrolled.
    """
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        withdrawn = conn.execute(f"""UPDATE {CUSTODIANS_TABLE} SET state='withdrawn', designated=0, updated_at=?
            WHERE room_id=? AND install_id=? AND role='custodian_only' AND state!='withdrawn'""",
                                 (now, room_id, install_id)).rowcount == 1
        if not withdrawn:
            raise CustodyError("this installation is not a custodian-only installation of the Group Chat")
        conn.execute(f"DELETE FROM {ROUTES_TABLE} WHERE room_id=? AND install_id=?", (room_id, install_id))
        return True


def maintain_configuration(
    db_path: DbPath, *, room_id: str, local_gateway_id: str, public_key: str, endpoint: str | None,
    name: str | None = None, owner_name: str | None = None, now: float | None = None,
) -> dict[str, Any] | None:
    """Append the room's custodians when they changed; returns ``{configuration_seq, **payload}``, else None.

    A room only its host keeps needs none: it behaves exactly as before.
    """
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        room = conn.execute("SELECT authority_gateway_id FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL",
                            (room_id,)).fetchone()
        if (room is None or room["authority_gateway_id"] != local_gateway_id
                or conn.execute("SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone()):
            return None
        custodians = [{"install_id": local_gateway_id, "public_key": public_key, "endpoint": endpoint,
                       "role": "authority", "successor": False, "name": display_label(name),
                       "operator_name": display_label(owner_name)}]
        for row in conn.execute(f"""SELECT c.*, p.public_key FROM {CUSTODIANS_TABLE} c
                JOIN {identity.PINS_TABLE} p ON p.room_id=c.room_id AND p.install_id=c.install_id
                WHERE c.room_id=? AND c.state='active' AND c.install_id!=? ORDER BY c.install_id""",
                                (room_id, local_gateway_id)):
            custodians.append({"install_id": row["install_id"], "public_key": row["public_key"],
                               "endpoint": row["endpoint"], "role": row["role"],
                               "successor": bool(row["allowed"] and row["designated"]), "name": row["name"],
                               "operator_name": row["operator_name"]})
        custodians.sort(key=lambda custodian: custodian["install_id"])
        configurations = configurations_locked(conn, room_id)
        if not configurations and len(custodians) == 1:
            return None
        payload = parse_configuration({"custodians": custodians, "owner_name": display_label(owner_name)})
        if configurations and {key: configurations[-1][key] for key in payload} == payload:
            return None
        seq = _append_system_event_locked(
            conn, room_id, event_id=f"system:custody-configured:{len(configurations) + 1}", kind=CONFIGURED,
            actor_id="custody-control", payload=payload, now=now)
        for custodian in payload["custodians"]:
            identity.pin_locked(conn, room_id=room_id, install_id=custodian["install_id"],
                                public_key=custodian["public_key"], source="configuration")
        return {"configuration_seq": seq, **payload}


def reconfigure_after_transition_locked(
    conn: sqlite3.Connection, room_id: str, *, successor: str, previous_host: str, now: float | None = None,
) -> dict[str, Any]:
    """On the new host, in the writer that follows a verified transition: its first configuration.

    The successor becomes the authority (never its own successor); the previous host keeps its copy
    as a custodian that is not a successor; every other custodian keeps its fields. The custody
    records the previous host kept are recreated here from that configuration, so a later change on
    this host starts from the same custodians and never shrinks them. Returns ``{configuration_seq,
    **payload}``.
    """
    initialize_locked(conn)
    now = time.time() if now is None else float(now)
    configurations = configurations_locked(conn, room_id)
    if not configurations:
        raise CustodyError("the Group Chat has no custodians to carry over")
    custodians = {custodian["install_id"]: dict(custodian) for custodian in configurations[-1]["custodians"]}
    if custodians.get(previous_host, {}).get("role") != "authority":
        raise CustodyError("the previous host is not the Group Chat's current host")
    if successor not in custodians or successor == previous_host:
        raise CustodyError("the successor keeps no copy of the Group Chat")
    custodians[successor].update(role="authority", successor=False)
    custodians[previous_host].update(role="custodian", successor=False)
    payload = parse_configuration({"custodians": [custodians[key] for key in sorted(custodians)],
                                   "owner_name": configurations[-1]["owner_name"]})
    seq = _append_system_event_locked(
        conn, room_id, event_id=f"system:custody-configured:{len(configurations) + 1}", kind=CONFIGURED,
        actor_id="custody-control", payload=payload, now=now)
    for custodian in payload["custodians"]:
        identity.pin_locked(conn, room_id=room_id, install_id=custodian["install_id"],
                            public_key=custodian["public_key"], source="configuration")
        if custodian["install_id"] == successor:
            continue
        # A successor stays one: its operator allowed it and the owner designated it before.
        conn.execute(f"""INSERT INTO {CUSTODIANS_TABLE} (room_id, install_id, role, state, endpoint, name, operator_name,
            allowed, designated, enrolled_at, updated_at) VALUES (?,?,?,'active',?,?,?,?,?,?,?)
            ON CONFLICT(room_id, install_id) DO UPDATE SET role=excluded.role, state='active',
            endpoint=excluded.endpoint, name=excluded.name, operator_name=excluded.operator_name,
            allowed=excluded.allowed, designated=excluded.designated, updated_at=excluded.updated_at""",
                     (room_id, custodian["install_id"], custodian["role"], custodian["endpoint"], custodian["name"],
                      custodian["operator_name"], int(custodian["successor"]), int(custodian["successor"]), now, now))
    conn.execute(f"DELETE FROM {CUSTODIANS_TABLE} WHERE room_id=? AND install_id=?", (room_id, successor))
    return {"configuration_seq": seq, **payload}


def record_acknowledgment(
    db_path: DbPath, *, room_id: str, install_id: str, watermark: Any, now: float | None = None,
) -> str:
    """Keep a custodian's acknowledged watermark when it matches this authority's own chain.

    Returns ``acknowledged`` (kept even when lower than before: a copy that lost history no longer
    counts for it), ``stale`` (no longer this gateway's room), ``unverifiable`` (this log no longer
    holds that prefix itself: not counted) or ``divergent`` (a prefix this log does not contain:
    never counted).
    """
    watermark = validate_watermark(watermark)
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        room = conn.execute("SELECT authority_gateway_id, authority_epoch, next_seq FROM hosted_rooms "
                            "WHERE room_id=?", (room_id,)).fetchone()
        if room is None or room["authority_gateway_id"] != rooms.local_authority_gateway_id():
            return "stale"
        seq = watermark["seq"]
        held = conn.execute("SELECT authority_epoch FROM hosted_room_events WHERE room_id=? AND seq=?",
                            (room_id, seq)).fetchone() if seq else None
        epoch = int(held[0]) if held is not None and held[0] is not None else int(room["authority_epoch"])
        try:
            own = chain_hash_locked(conn, room_id, min(seq, int(room["next_seq"]) - 1), table="hosted_room_events")
        except CustodyError:
            return "unverifiable"
        matches = (seq <= int(room["next_seq"]) - 1 and (seq == 0 or held is not None)
                   and watermark["epoch"] == epoch and own == watermark["event_hash"])
        conn.execute(f"""INSERT INTO {WATERMARKS_TABLE} (room_id, install_id, epoch, seq, event_hash, acknowledged_at,
            state) VALUES (?,?,?,?,?,?,?) ON CONFLICT(room_id, install_id) DO UPDATE SET epoch=excluded.epoch,
            seq=excluded.seq, event_hash=excluded.event_hash, acknowledged_at=excluded.acknowledged_at,
            state=excluded.state""",
                     (room_id, install_id, watermark["epoch"], seq, watermark["event_hash"], now,
                      "verified" if matches else "divergent"))
        return "acknowledged" if matches else "divergent"


def report_locked(
    conn: sqlite3.Connection, room_id: str, install_id: str, *, head_seq: int | None = None,
) -> dict[str, Any]:
    """What the authority tells one custodian with each page: the tail at risk, the configuration,
    the consent it recorded for that custodian, and the head it signs for the page's end
    (``head_seq``), which the custodian keeps to vouch for its copy."""
    allowed = conn.execute(f"SELECT allowed FROM {CUSTODIANS_TABLE} WHERE room_id=? AND install_id=?",
                           (room_id, install_id)).fetchone() if table_exists(conn, CUSTODIANS_TABLE) else None
    report = {"at_risk_after_seq": at_risk_after_locked(conn, room_id),
              "configuration_seq": configuration_locked(conn, room_id)["configuration_seq"],
              **({"allowed": bool(allowed[0])} if allowed is not None else {})}
    if head_seq is not None:
        try:
            report["head"] = sign_head_locked(conn, room_id, seq=head_seq)
        except (CustodyError, identity.RoomIdentityError):
            logger.warning("custody head unavailable for a Group Chat; its page goes unvouched")
    return report


# -- a custodian's copy ------------------------------------------------------------------------


def local_consent_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    """Whether this installation's operator allows it to continue ``room_id``."""
    if not table_exists(conn, CONSENT_TABLE):
        return False
    row = conn.execute(f"SELECT allowed FROM {CONSENT_TABLE} WHERE room_id=?", (room_id,)).fetchone()
    return row is not None and bool(row[0])


def local_consent(db_path: DbPath, room_id: str) -> bool:
    """Whether this installation's operator allows it to continue ``room_id``: checked before acting."""
    with closing(open_sqlite(db_path, timeout=1)) as conn:
        return local_consent_locked(conn, room_id)


def set_local_consent(db_path: DbPath, *, room_id: str, allowed: bool, now: float | None = None) -> dict[str, Any]:
    """The operator allows (or no longer allows) this installation to continue ``room_id``.

    Takes effect here at once. The host learns it from the next acknowledgment or probe;
    ``confirmed`` turns true once the host's report shows it recorded the same choice.
    """
    if type(allowed) is not bool:
        raise CustodyError("successor must be a boolean")
    now = time.time() if now is None else float(now)
    with rooms._transaction(db_path, immediate=True) as conn:
        initialize_locked(conn)
        conn.execute(f"""INSERT INTO {CONSENT_TABLE} (room_id, allowed, updated_at) VALUES (?,?,?)
            ON CONFLICT(room_id) DO UPDATE SET allowed=excluded.allowed, updated_at=excluded.updated_at""",
                     (room_id, int(allowed), now))
        host = conn.execute(f"SELECT host_allowed FROM {CONSENT_TABLE} WHERE room_id=?", (room_id,)).fetchone()[0]
    return {"room_id": room_id, "install_id": rooms.local_authority_gateway_id(), "allowed": allowed,
            "confirmed": host is not None and bool(host) == allowed}


def pin_configured_locked(conn: sqlite3.Connection, room_id: str, events: list[Mapping[str, Any]]) -> None:
    """Pin the keys of the custodians each stored ``custody.configured`` among ``events`` names."""
    initialize_locked(conn)
    for event in events:
        if event["kind"] != CONFIGURED:
            continue
        for custodian in parse_configuration(json.loads(event["payload_json"]))["custodians"]:
            identity.pin_locked(conn, room_id=room_id, install_id=custodian["install_id"],
                                public_key=custodian["public_key"], source="configuration")


def after_ingest_locked(
    conn: sqlite3.Connection, room_id: str, events: list[Mapping[str, Any]], *, report: Any = None,
) -> dict[str, Any]:
    """Pin the custodians a newly stored configuration names, keep the authority's report, return the watermark.

    Runs inside the copy's writer, so an acknowledgment never names history that is not durable.
    """
    pin_configured_locked(conn, room_id, events)
    if report is not None:
        at_risk_after = report.get("at_risk_after_seq") if isinstance(report, Mapping) else None
        if type(at_risk_after) is not int or at_risk_after < 0:
            raise CustodyError("custody report is invalid")
        conn.execute(f"""INSERT INTO {REPORTS_TABLE} (room_id, at_risk_after_seq, reported_at) VALUES (?,?,?)
            ON CONFLICT(room_id) DO UPDATE SET at_risk_after_seq=excluded.at_risk_after_seq,
            reported_at=excluded.reported_at""", (room_id, at_risk_after, time.time()))
        if isinstance(report.get("allowed"), bool):
            # The consent the host recorded for this installation: confirms a change made here.
            conn.execute(f"UPDATE {CONSENT_TABLE} SET host_allowed=? WHERE room_id=?",
                         (int(report["allowed"]), room_id))
        if report.get("head") is not None:
            # Kept only when it vouches for this copy: the host it follows signed it, and it matches.
            record_head_locked(conn, room_id, report["head"])
    watermark = custody_watermark_locked(conn, room_id)
    if watermark is None:  # pragma: no cover - the caller just stored this copy
        raise CustodyError("no copy of this Group Chat is held here")
    return watermark


def custody_status(db_path: DbPath, room_id: str) -> dict[str, Any]:
    """Custodians with their watermarks, ``at_risk_after_seq`` and the configuration, from this store.

    On the authority the watermarks are the custodians' verified acknowledgments; on a custodian they
    come from its own copy and the authority's last report, and other custodians' are unknown (None).
    ``head`` is the head that vouches for the history held here (``vouched_head_locked``).
    """
    room_id = rooms._room_id(room_id)
    with closing(open_sqlite(db_path)) as conn:
        conn.execute("BEGIN")  # one snapshot; nothing is written
        table = events_table_locked(conn, room_id)
        if table is None:
            raise rooms.RoomNotFoundError("no history of this Group Chat is held here")
        configuration = configuration_locked(conn, room_id)
        own = custody_watermark_locked(conn, room_id, store=False)
        vouched = vouched_head_locked(conn, room_id)
        local = rooms.local_authority_gateway_id()
        listed = {custodian["install_id"]: custodian for custodian in configuration["custodians"]
                  if custodian["role"] != "authority"}
        custodians: list[dict[str, Any]] = []
        if table == "hosted_room_events":
            enrolled = {str(row["install_id"]): row for row in conn.execute(
                f"SELECT * FROM {CUSTODIANS_TABLE} WHERE room_id=? ORDER BY install_id", (room_id,))} \
                if table_exists(conn, CUSTODIANS_TABLE) else {}
            acked = {str(row["install_id"]): row for row in conn.execute(
                f"SELECT * FROM {WATERMARKS_TABLE} WHERE room_id=?", (room_id,))} \
                if table_exists(conn, WATERMARKS_TABLE) else {}
            for install_id in sorted((set(enrolled) | set(listed)) - {local}):
                row, ack, entry = enrolled.get(install_id), acked.get(install_id), listed.get(install_id, {})
                custodians.append({
                    "install_id": install_id, "role": row["role"] if row is not None else entry.get("role"),
                    "state": row["state"] if row is not None else "active",
                    "name": row["name"] if row is not None else entry.get("name"),
                    "operator_name": row["operator_name"] if row is not None else entry.get("operator_name"),
                    "successor": bool(entry.get("successor")),
                    "allowed": bool(row["allowed"]) if row is not None else None,
                    "designated": bool(row["designated"]) if row is not None else None,
                    "opted_out": row is not None and row["state"] == "opted_out",
                    "watermark": {"epoch": int(ack["epoch"]), "seq": int(ack["seq"]), "event_hash": ack["event_hash"]}
                    if ack is not None and ack["state"] == "verified" else None,
                    "acknowledged_at": float(ack["acknowledged_at"]) if ack is not None else None,
                    "divergent": ack is not None and ack["state"] == "divergent"})
            at_risk_after = at_risk_after_locked(conn, room_id)
        else:
            report = conn.execute(f"SELECT at_risk_after_seq FROM {REPORTS_TABLE} WHERE room_id=?",
                                  (room_id,)).fetchone() if table_exists(conn, REPORTS_TABLE) else None
            for install_id, entry in sorted(listed.items()):
                custodians.append({
                    "install_id": install_id, "role": entry["role"], "state": "active", "name": entry["name"],
                    "operator_name": entry["operator_name"], "successor": entry["successor"], "allowed": None,
                    "designated": None, "opted_out": False,
                    "watermark": own if install_id == local else None, "acknowledged_at": None, "divergent": False})
            at_risk_after = int(report[0]) if report is not None else 0
        conn.rollback()
    return {"room_id": room_id, "role": "authority" if table == "hosted_room_events" else "custodian",
            "custodians": custodians, "at_risk_after_seq": at_risk_after,
            "configuration_seq": configuration["configuration_seq"], "configuration": configuration, "watermark": own,
            "head": vouched}


def wait_protected(db_path: DbPath, room_id: str, seq: int, timeout: float, *, poll_seconds: float = 0.05) -> bool:
    """Wait until an eligible successor holds ``seq`` of this hosted room; False once ``timeout`` passes.

    Phase 1 never calls it on the dispatch path: nothing waits for a copy. A designated backup
    (phase 2) acknowledges acceptance through it.
    """
    deadline = time.monotonic() + max(0.0, float(timeout))
    while True:
        with closing(open_sqlite(db_path, timeout=1)) as conn:
            if at_risk_after_locked(conn, room_id) >= seq:
                return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


# -- the history held here -------------------------------------------------------------------


def read_copy_page(conn: sqlite3.Connection, room_id: str, *, after_seq: int, limit: int) -> dict[str, Any]:
    """One bounded page of the history held here, the room itself or a copy of it.

    Returns ``{room_name, members, page}``. The page has ``read_events``' shape and bounds, and its
    authority is this store's own: the room's, or the authority the copy verified.
    """
    from gateway.hosted_rooms import _bounded_page, _event_from_row, _page_rows
    table = events_table_locked(conn, room_id)
    if table is None:
        raise rooms.RoomNotFoundError("no history of this Group Chat is held here")
    head = conn.execute(
        "SELECT name, members_json, authority_gateway_id, authority_epoch, next_seq - 1 AS latest "
        "FROM hosted_rooms WHERE room_id=?" if table == "hosted_room_events" else
        "SELECT name, members_json, authority_gateway_id, authority_epoch, last_seq AS latest "
        "FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
    latest = int(head["latest"])
    if type(after_seq) is not int or not 0 <= after_seq <= latest:
        raise CustodyError("the history held here does not reach that sequence")
    if type(limit) is not int or not 1 <= limit <= rooms.MAX_LOG_LIMIT:
        raise CustodyError("page limit is invalid")
    authority = {"gateway_id": str(head["authority_gateway_id"]), "epoch": int(head["authority_epoch"])}
    events = [_event_from_row(row) for row in _page_rows(conn, table, room_id, after_seq, limit)]
    return {"room_name": head["name"], "members": json.loads(head["members_json"]),
            "page": _bounded_page(events, after_seq, latest, authority)}


# -- catch-up from another custodian -------------------------------------------------------------

PAGES_PATH = "/v1/room-members/custody/pages"
PAGES_DOMAIN = b"hermes.group.custody.pages.v1"
PAGES_REPLY_DOMAIN = b"hermes.group.custody.pages-reply.v1"
# A signed request stays valid this long either side of its issue time.
REQUEST_SKEW_SECONDS = 300.0
_PAGES_REQUEST_FIELDS = frozenset({
    "room_id", "requester_install_id", "source_install_id", "after_seq", "limit", "issued_at", "nonce"})
_PAGES_REPLY_FIELDS = frozenset({
    "room_id", "source_install_id", "requester_install_id", "nonce", "room_name", "members", "page", "head"})
_ANSWERS = ("room_id", "source_install_id", "requester_install_id", "nonce")
_NONCE_RE = re.compile(r"[0-9a-f]{32}")


class CustodyAuthorizationError(CustodyError):
    """A catch-up request or reply is not signed by a custodian of the Group Chat."""

    reason = "custody_not_authorized"


def fetch_custodian_pages(
    db_path: DbPath, *, room_id: str, source_install_id: str, after_seq: int, limit: int, timeout: float = 10.0,
) -> dict[str, Any]:
    """One page of another custodian's history of the room, from its own room or copy.

    The request names both installations and is signed with this installation's room identity key;
    the source answers only an installation its configuration lists, and its signed reply is checked
    here against the key pinned for it. Returns ``{room_id, room_name, members, page, head,
    source_install_id}``: ``head`` is the host-signed head that vouches for the source's history (or
    None). Nothing here is stored: ``catch_up_from_custodian`` stores only what a head vouches for.
    """
    from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient
    room_id = rooms._room_id(room_id)
    with closing(open_sqlite(db_path, timeout=1)) as conn:
        entry = next((custodian for custodian in configuration_locked(conn, room_id)["custodians"]
                      if custodian["install_id"] == source_install_id), None)
        pinned = identity.pinned_key_locked(conn, room_id=room_id, install_id=source_install_id)
    if entry is None or entry["endpoint"] is None or pinned is None:
        raise CustodyError("that custodian of the Group Chat is unknown here or has no endpoint")
    request = {"room_id": room_id, "requester_install_id": rooms.local_authority_gateway_id(),
               "source_install_id": source_install_id, "after_seq": after_seq, "limit": limit,
               "issued_at": time.time(), "nonce": secrets.token_hex(16)}
    client = PeerRunsHTTPClient(base_url=entry["endpoint"], api_key="", timeout_seconds=timeout,
                                proof_install_id=source_install_id)
    reply = dict(client.custody_pages(body={**request, "signature": identity.sign(PAGES_DOMAIN, request)}))
    signature = reply.pop("signature", None)
    if set(reply) != _PAGES_REPLY_FIELDS or any(reply[key] != request[key] for key in _ANSWERS):
        raise CustodyError("the custodian's reply does not answer this request")
    with closing(open_sqlite(db_path, timeout=1)) as conn:
        if not identity.verify_locked(conn, room_id, source_install_id, PAGES_REPLY_DOMAIN, reply, signature):
            raise CustodyAuthorizationError("the custodian's reply is not signed with its pinned key")
    return {"room_id": room_id, "room_name": reply["room_name"], "members": reply["members"], "page": reply["page"],
            "head": reply["head"], "source_install_id": source_install_id}


def serve_custodian_pages(db_path: DbPath, body: Any, *, now: float | None = None) -> dict[str, Any]:
    """Answer one custodian's signed catch-up request with a signed page of the history held here.

    Only an installation that the room's latest configuration here lists may ask, signed with the
    key pinned for it, in a request issued within ``REQUEST_SKEW_SECONDS`` that names this
    installation as its source. The reply relays the head that vouches for this history.
    """
    if not isinstance(body, Mapping) or set(body) != _PAGES_REQUEST_FIELDS | {"signature"}:
        raise CustodyError("catch-up request fields are invalid")
    request = {key: body[key] for key in _PAGES_REQUEST_FIELDS}
    room_id = rooms._room_id(request["room_id"])
    issued_at, now = request["issued_at"], time.time() if now is None else float(now)
    if (request["source_install_id"] != rooms.local_authority_gateway_id()
            or not isinstance(request["requester_install_id"], str)
            or isinstance(issued_at, bool) or not isinstance(issued_at, (int, float)) or not math.isfinite(issued_at)
            or abs(now - issued_at) > REQUEST_SKEW_SECONDS
            or not isinstance(request["nonce"], str) or _NONCE_RE.fullmatch(request["nonce"]) is None):
        raise CustodyAuthorizationError("catch-up request is not current or not addressed to this installation")
    with closing(open_sqlite(db_path, timeout=1)) as conn:
        conn.execute("BEGIN")  # one snapshot; nothing is written
        listed = {custodian["install_id"] for custodian in configuration_locked(conn, room_id)["custodians"]}
        if request["requester_install_id"] not in listed or not identity.verify_locked(
                conn, room_id, request["requester_install_id"], PAGES_DOMAIN, request, body["signature"]):
            raise CustodyAuthorizationError("catch-up request is not signed by a custodian of the Group Chat")
        page = read_copy_page(conn, room_id, after_seq=request["after_seq"], limit=request["limit"])
        head = vouched_head_locked(conn, room_id)
        conn.rollback()
    reply = {**{key: request[key] for key in _ANSWERS}, "room_id": room_id, **page, "head": head}
    return {**reply, "signature": identity.sign(PAGES_REPLY_DOMAIN, reply)}


def _wire_event(room_id: str, event: Mapping[str, Any]) -> dict[str, Any]:
    """A normalized event back in the replay-page shape ``ingest_page`` takes."""
    return {"room_id": room_id, "seq": event["seq"], "event_id": event["event_id"], "kind": event["kind"],
            "actor": json.loads(event["actor_json"]), "authority_epoch": event["authority_epoch"],
            "payload": json.loads(event["payload_json"]), "created_at": event["created_at"]}


def catch_up_from_custodian(
    db_path: DbPath, *, room_id: str, source_install_id: str, head: Any = None, page_limit: int = rooms.MAX_LOG_LIMIT,
    timeout: float = 10.0, _verify_transition: Any = None, _fetch: Any = None,
) -> dict[str, Any]:
    """Catch this copy up from another custodian, exactly as far as a head its host signed vouches.

    ``head`` (by default the one the source relays) must be signed by the host this copy follows, and
    the range then holds no change of host; or by the successor of a change of host that is the first
    event fetched, which ``_verify_transition`` then checks. Its signature checks against keys this
    copy pinned before the range. The source's pages are fetched from this copy's last seq up to
    ``head.seq``, never further, so an unvouched tail is dropped; and their chain must reach
    ``head.chain_hash`` before anything is stored, so keys and configurations are only ever taken
    from inside a host-signed prefix. On any mismatch nothing is stored (``CustodyError``).

    Returns ``{room_id, stored_seq, watermark, head}``; the head is then kept here too.
    ``_fetch(db_path, *, room_id, source_install_id, after_seq, limit)`` defaults to
    ``fetch_custodian_pages``.
    """
    from gateway import hosted_room_replicas as replicas
    fetch = _fetch or (lambda path, **request: fetch_custodian_pages(path, timeout=timeout, **request))
    room_id = rooms._room_id(room_id)
    with closing(open_sqlite(db_path, timeout=1)) as conn:
        copy = _copy_head_locked(conn, room_id)
        if copy is None:
            raise CustodyError("no usable copy of this Group Chat is held here")
        own = chain_hash_locked(conn, room_id, copy[2], table="hosted_room_replica_events", store=False)
    host, epoch, last = copy

    def checked(target: Any) -> dict[str, Any]:
        """The head's statement, signed by the host this copy follows or by a later host."""
        with closing(open_sqlite(db_path, timeout=1)) as conn:
            statement = verify_head_locked(conn, room_id, target)
        if (statement["host"], statement["epoch"]) != (host, epoch) and statement["epoch"] <= epoch:
            raise CustodyError("the head is signed neither by the host this copy follows nor by its successor")
        return statement

    statement = checked(head) if head is not None else None
    events, cursor, held, fetched = [], last, 0, None
    while statement is None or cursor < statement["seq"]:
        fetched = fetch(db_path, room_id=room_id, source_install_id=source_install_id, after_seq=cursor,
                        limit=page_limit)
        if not isinstance(fetched, Mapping) or not {"room_name", "members", "page"} <= set(fetched):
            raise CustodyError("the custodian's catch-up page is invalid")
        if statement is None:
            if fetched.get("head") is None:
                raise CustodyError("the custodian holds no head its host signed")
            head = fetched["head"]
            statement = checked(head)
            if statement["seq"] <= cursor:
                break
        page_events, _, _ = replicas._validate_page(fetched["page"])
        new = [event for event in page_events if event["seq"] > cursor]
        if not new or new[0]["seq"] != cursor + 1:
            raise CustodyError("the custodian holds less of the history than the head vouches for")
        held += sum(utf8_len(event["event_id"], event["kind"], event["actor_json"], event["payload_json"])
                    for event in new)
        if held > rooms.MAX_GATEWAY_EVENT_BYTES:
            raise CustodyError("catching up exceeds this gateway's history budget")
        events.extend(new)
        cursor = new[-1]["seq"]
    if statement["seq"] <= last:
        # Nothing to fetch: the head must still describe this copy's own prefix.
        with closing(open_sqlite(db_path, timeout=1)) as conn:
            if chain_hash_locked(conn, room_id, statement["seq"], table="hosted_room_replica_events",
                                 store=False) != statement["chain_hash"]:
                raise CustodyError("this copy differs from the history its host signed")
            return {"room_id": room_id, "stored_seq": last, "head": dict(head),
                    "watermark": custody_watermark_locked(conn, room_id, store=False)}
    vouched = events[:statement["seq"] - last]  # the unvouched tail is dropped
    transitions = [index for index, event in enumerate(vouched) if event["kind"] == "authority.transition"]
    if (statement["host"], statement["epoch"]) == (host, epoch):
        if transitions:
            raise CustodyError("a head of the host this copy follows vouches for no change of host")
    else:
        if transitions != [0]:
            raise CustodyError("a later host's head vouches only for history that starts with its change of host")
        moved = json.loads(vouched[0]["payload_json"])
        if (moved.get("from_epoch"), moved.get("to_epoch"), moved.get("successor_gateway_id"),
                vouched[0]["authority_epoch"]) != (epoch, statement["epoch"], statement["host"], statement["epoch"]):
            raise CustodyError("the head follows another change of host than this copy's next")
    chain = own
    for event in vouched:
        chain = _fold(chain, event)
    if chain != statement["chain_hash"]:
        raise CustodyError("the custodian's history differs from the history its host signed")
    # Proven to be what the host signed; stored in page-sized parts, the change of host first.
    authority = {"gateway_id": statement["host"], "epoch": statement["epoch"]}
    remaining, start = [_wire_event(room_id, event) for event in vouched], last
    while remaining:
        page = rooms._bounded_page(remaining[:rooms.MAX_LOG_LIMIT], start, statement["seq"], authority)
        replicas.ingest_page(db_path, room_id=room_id, room_name=fetched["room_name"], members=fetched["members"],
                             page=page, _verify_transition=_verify_transition, _from_custodian=True)
        remaining, start = remaining[len(page["events"]):], page["cursor"]
    record_head(db_path, room_id, head)
    with closing(open_sqlite(db_path, timeout=1)) as conn:
        return {"room_id": room_id, "stored_seq": statement["seq"], "head": dict(head),
                "watermark": custody_watermark_locked(conn, room_id, store=False)}


# -- admissions --------------------------------------------------------------------------------


def _admission_event_id(task_id: str, generation: int) -> str:
    return f"system:task-admitted:{hashlib.sha256(task_id.encode('utf-8')).hexdigest()[:32]}:{generation}"


def announce_queued_task_locked(conn: sqlite3.Connection, task: Mapping[str, Any], *, now: float) -> int | None:
    """Announce the generation a queued task will dispatch, in the transaction that queued it.

    Returns the ``task.admitted`` event's seq, or None where only the host keeps the room: such a
    room behaves exactly as before. Dispatch never waits for the announcement to be copied.
    """
    if task["status"] != "queued" or not has_custody_locked(conn, str(task["room_id"])):
        return None
    room_id, generation = str(task["room_id"]), int(task["execution_generation"]) + 1
    payload = json.loads(task["payload_json"])
    member_id = str(payload.get("target_member_id") or payload["target_profile"])
    room = conn.execute("SELECT authority_gateway_id, members_json FROM hosted_rooms WHERE room_id=?",
                        (room_id,)).fetchone()
    target = next((member.get("target") for member in json.loads(room["members_json"])
                   if isinstance(member, Mapping) and member.get("member_id") == member_id), None)
    install_id = (target.get("installation_id") if isinstance(target, Mapping) and target.get("kind") == "peer"
                  else room["authority_gateway_id"])
    return _append_system_event_locked(
        conn, room_id, event_id=_admission_event_id(str(task["task_id"]), generation), kind=TASK_ADMITTED,
        actor_id="room-driver", now=now, payload={
            "task": {key: str(task[key]) for key in ("room_id", "task_id", "thread_id", "turn_id")},
            "execution_generation": generation, "target_member_id": member_id, "target_install_id": install_id,
            "source_event_seq": int(task["source_event_seq"])})
