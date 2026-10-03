"""Custody of a Group Chat's history: custodians, successors, watermarks and the tail at risk.

Every member installation that keeps its copy (``replicate``), and any custodian-only installation
the room owner added, is a custodian: it holds the room's whole history. The room's custodians are
a log event, ``custody.configured``, carrying each one's pinned room identity key, endpoint, display
name and whether it may continue the group (``successor``): only when its own operator consented
(grant permission ``successor``) and the room owner designated it. No member is ever picked by an
algorithm, and nothing here votes.

Each copy, and the authority's own room, has a durable watermark ``(epoch, seq, event_hash)``:
``event_hash`` chains every event of that exact prefix. Custodians acknowledge each page with it, and
the authority keeps the acknowledgments that match its own chain. ``at_risk_after_seq`` is the
highest seq at least one eligible successor durably holds; every later event is at risk of being
lost with this host, and clients say so. Nothing waits for a copy: replication degrades to local
acknowledgement and marks the tail at risk. A task made ready for dispatch is announced with
``task.admitted`` in the same transaction, so a successor can reconcile it.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import sqlite3
import time
from contextlib import closing
from typing import Any, Mapping

from gateway import hosted_room_identity as identity
from gateway import hosted_rooms as rooms
from gateway.hosted_rooms_common import DbPath, compact_json, open_sqlite, table_exists
from gateway.hosted_rooms_common import display_label as common_display_label

CONFIGURED = "custody.configured"
TASK_ADMITTED = "task.admitted"
CUSTODIANS_TABLE = "hosted_room_custodians"
WATERMARKS_TABLE = "hosted_room_custody_watermarks"
REPORTS_TABLE = "hosted_room_custody_reports"
CHAIN_TABLE = "hosted_room_custody_chain"
CONSENT_TABLE = "hosted_room_custody_consent"
ROUTES_TABLE = "hosted_room_custody_routes"
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
    """Forget derived hashes past ``after_seq``: required wherever stored history is rewritten."""
    if table_exists(conn, CHAIN_TABLE):
        conn.execute(f"DELETE FROM {CHAIN_TABLE} WHERE room_id=? AND seq>?", (room_id, after_seq))


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


def report_locked(conn: sqlite3.Connection, room_id: str, install_id: str) -> dict[str, Any]:
    """What the authority tells one custodian with each page: the tail at risk, the configuration,
    and the consent it recorded for that custodian."""
    allowed = conn.execute(f"SELECT allowed FROM {CUSTODIANS_TABLE} WHERE room_id=? AND install_id=?",
                           (room_id, install_id)).fetchone() if table_exists(conn, CUSTODIANS_TABLE) else None
    return {"at_risk_after_seq": at_risk_after_locked(conn, room_id),
            "configuration_seq": configuration_locked(conn, room_id)["configuration_seq"],
            **({"allowed": bool(allowed[0])} if allowed is not None else {})}


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


def set_local_consent(db_path: DbPath, *, room_id: str, allowed: bool, now: float | None = None,
                      conn: sqlite3.Connection | None = None) -> dict[str, Any]:
    """The operator allows (or no longer allows) this installation to continue ``room_id``.

    Takes effect here at once. The host learns it from the next acknowledgment or probe;
    ``confirmed`` turns true once the host's report shows it recorded the same choice. ``conn``: a
    caller's open write transaction on the same store (a recoverable setup issuance), joined, not nested.
    """
    if type(allowed) is not bool:
        raise CustodyError("successor must be a boolean")
    from contextlib import nullcontext
    now = time.time() if now is None else float(now)
    with (nullcontext(conn) if conn is not None else rooms._transaction(db_path, immediate=True)) as conn:
        initialize_locked(conn)
        conn.execute(f"""INSERT INTO {CONSENT_TABLE} (room_id, allowed, updated_at) VALUES (?,?,?)
            ON CONFLICT(room_id) DO UPDATE SET allowed=excluded.allowed, updated_at=excluded.updated_at""",
                     (room_id, int(allowed), now))
        host = conn.execute(f"SELECT host_allowed FROM {CONSENT_TABLE} WHERE room_id=?", (room_id,)).fetchone()[0]
    return {"room_id": room_id, "install_id": rooms.local_authority_gateway_id(), "allowed": allowed,
            "confirmed": host is not None and bool(host) == allowed}


def after_ingest_locked(
    conn: sqlite3.Connection, room_id: str, events: list[Mapping[str, Any]], *, report: Any = None,
) -> dict[str, Any]:
    """Pin the custodians a newly stored configuration names, keep the authority's report, return the watermark.

    Runs inside the copy's writer, so an acknowledgment never names history that is not durable.
    """
    initialize_locked(conn)
    for event in events:
        if event["kind"] != CONFIGURED:
            continue
        for custodian in parse_configuration(json.loads(event["payload_json"]))["custodians"]:
            identity.pin_locked(conn, room_id=room_id, install_id=custodian["install_id"],
                                public_key=custodian["public_key"], source="configuration")
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
    watermark = custody_watermark_locked(conn, room_id)
    if watermark is None:  # pragma: no cover - the caller just stored this copy
        raise CustodyError("no copy of this Group Chat is held here")
    return watermark


def custody_status(db_path: DbPath, room_id: str) -> dict[str, Any]:
    """Custodians with their watermarks, ``at_risk_after_seq`` and the configuration, from this store.

    On the authority the watermarks are the custodians' verified acknowledgments; on a custodian they
    come from its own copy and the authority's last report, and other custodians' are unknown (None).
    """
    room_id = rooms._room_id(room_id)
    with closing(open_sqlite(db_path)) as conn:
        conn.execute("BEGIN")  # one snapshot; nothing is written
        table = events_table_locked(conn, room_id)
        if table is None:
            raise rooms.RoomNotFoundError("no history of this Group Chat is held here")
        configuration = configuration_locked(conn, room_id)
        own = custody_watermark_locked(conn, room_id, store=False)
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
            "configuration_seq": configuration["configuration_seq"], "configuration": configuration, "watermark": own}


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
