"""Private local-session identity captured by an admitted in-process room turn.

These rows are not replicated, do not contain conversation contents, and grant
no execution or takeover rights. A missing binding is never title-based proof
that an original-local Bot can rejoin a different coordinator.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from typing import Any

from gateway import hosted_room_driver as driver, hosted_rooms
from gateway.hosted_room_authority_history import read_history_locked
from gateway.hosted_room_route_schema import require_room_work_open
from gateway.hosted_rooms_common import DbPath, table_exists

_TABLE = "hosted_room_local_sessions"
_COLUMNS = {
    "room_id", "member_id", "profile", "gateway_id", "session_id",
    "session_started_at", "first_task_id", "first_execution_generation", "created_at",
    "last_session_id", "last_session_started_at",
}
_DDL = f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
    room_id TEXT NOT NULL, member_id TEXT NOT NULL, profile TEXT NOT NULL,
    gateway_id TEXT NOT NULL, session_id TEXT NOT NULL, session_started_at REAL NOT NULL,
    first_task_id TEXT NOT NULL, first_execution_generation INTEGER NOT NULL,
    created_at REAL NOT NULL, last_session_id TEXT NOT NULL, last_session_started_at REAL NOT NULL,
    PRIMARY KEY(room_id, member_id),
    UNIQUE(room_id, profile), UNIQUE(profile, session_id),
    FOREIGN KEY(room_id) REFERENCES hosted_rooms(room_id) ON DELETE CASCADE)"""


class LocalSessionBindingError(RuntimeError):
    """No model work may begin with an unverified local context association."""

    not_admitted = True


def _validate_schema(conn) -> None:
    columns = conn.execute("SELECT name,pk FROM pragma_table_info(?)", (_TABLE,)).fetchall()
    if {row["name"] for row in columns} != _COLUMNS or {
        row["name"]: row["pk"] for row in columns if row["pk"]
    } != {"room_id": 1, "member_id": 2}:
        raise LocalSessionBindingError("Local Group Chat session registry needs repair.")
    unique = {
        tuple(item["name"] for item in conn.execute(
            "SELECT name FROM pragma_index_info(?) ORDER BY seqno", (row["name"],)))
        for row in conn.execute("SELECT name FROM pragma_index_list(?) WHERE [unique]=1 AND partial=0", (_TABLE,))
    }
    foreign = conn.execute("SELECT * FROM pragma_foreign_key_list(?)", (_TABLE,)).fetchall()
    if not {("room_id", "profile"), ("profile", "session_id")} <= unique or not any(
        row["table"] == "hosted_rooms" and row["from"] == "room_id"
        and row["to"] == "room_id" and row["on_delete"] == "CASCADE" for row in foreign
    ):
        raise LocalSessionBindingError("Local Group Chat session registry constraints changed.")


def _local_member(conn, room_id: str, profile: str, gateway_id: str):
    room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if room is None:
        raise LocalSessionBindingError("The Group Chat is no longer available.")
    epoch = int(room["authority_epoch"])
    history = read_history_locked(conn, room_id, gateway_id=room["authority_gateway_id"], epoch=epoch)
    if history is None and epoch > 1:
        claim = conn.execute("""SELECT payload_json FROM hosted_room_events
            WHERE room_id=? AND kind='authority.claimed' ORDER BY seq DESC LIMIT 1""", (room_id,)).fetchone()
        if (claim is not None and json.loads(claim["payload_json"]).get("previous_gateway_id") == "legacy"
                and room["authority_gateway_id"] == gateway_id):
            return room, None  # Keep ordinary legacy adoption; do not certify its origin.
        raise LocalSessionBindingError("The Bot's original Group Chat host is unverified.")
    origin = history[0]["gateway_id"] if history else room["authority_gateway_id"]
    if origin != gateway_id:
        raise LocalSessionBindingError("The Bot belongs to another Group Chat host.")
    members = json.loads(room["members_json"])
    if not isinstance(members, list) or any(not isinstance(member, dict) for member in members):
        raise LocalSessionBindingError("The saved Group Chat membership is invalid.")
    matches = [member for member in members if member.get("profile") == profile
               and (member.get("target") or {}).get("kind", "local") == "local"]
    if len(matches) != 1 or not isinstance(matches[0].get("member_id"), str):
        raise LocalSessionBindingError("The local Bot does not match the saved membership.")
    return room, matches[0]["member_id"]


def lookup_binding(db_path: DbPath, *, room_id: str, profile: str) -> dict[str, Any] | None:
    """Read identity for cleanup/resume too; this does not authorize new work."""
    gateway_id = hosted_rooms.local_authority_gateway_id()
    with driver._transaction(db_path) as conn:
        room, member_id = _local_member(conn, room_id, profile, gateway_id)
        if member_id is None:
            return None
        row = None
        if table_exists(conn, _TABLE):
            _validate_schema(conn)
            row = conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=? AND member_id=?",
                               (room_id, member_id)).fetchone()
        if row is not None:
            if row["profile"] != profile or row["gateway_id"] != gateway_id:
                raise LocalSessionBindingError("The saved Bot session belongs to another identity.")
            return dict(row)
        if room["authority_gateway_id"] != gateway_id or room["disbanded_at"] is not None:
            raise LocalSessionBindingError("The original Bot session was not recorded for recovery.")
        return None


def record_binding(
    db_path: DbPath, *, task: driver.TaskIdentity, execution_generation: int,
    profile: str, session_id: str, session_started_at: float, context_chain=None, clock=time.time,
) -> dict[str, Any] | None:
    """Capture identity at the real submit boundary, fenced with current task state."""
    if type(execution_generation) is not int or execution_generation < 1:
        raise LocalSessionBindingError("The Group Chat task generation is invalid.")
    session_id = driver._identifier(session_id, label="session_id")
    if (isinstance(session_started_at, bool) or not isinstance(session_started_at, (int, float))
            or not math.isfinite(session_started_at) or session_started_at < 0):
        raise LocalSessionBindingError("The stored Bot session identity is invalid.")
    chain = list(context_chain) if context_chain is not None else [(session_id, session_started_at)]
    if not chain or len(chain) > 101 or chain[0] != (session_id, session_started_at):
        raise LocalSessionBindingError("The private Bot continuation is invalid.")
    seen = set()
    for key, started_at in chain:
        driver._identifier(key, label="session_id")
        if (key in seen or isinstance(started_at, bool) or not isinstance(started_at, (int, float))
                or not math.isfinite(started_at) or started_at < 0):
            raise LocalSessionBindingError("The private Bot continuation is invalid.")
        seen.add(key)
    gateway_id = hosted_rooms.local_authority_gateway_id()
    with driver._transaction(db_path) as conn:
        room, member_id = _local_member(conn, task.room_id, profile, gateway_id)
        if member_id is None:
            return None
        driver._require_room_authority(conn, task.room_id, gateway_id, int(room["authority_epoch"]))
        require_room_work_open(conn, task.room_id, error=LocalSessionBindingError)
        current = driver._load_task(conn, task)
        payload = json.loads(current["payload_json"])
        if (current["status"] != "running" or int(current["execution_generation"]) != execution_generation
                or payload.get("target_profile") != profile
                or payload.get("target_member_id", profile) != member_id):
            raise LocalSessionBindingError("The Group Chat task is no longer the admitted local turn.")
        lease_row = conn.execute("SELECT * FROM hosted_room_driver_leases WHERE room_id=?", (task.room_id,)).fetchone()
        if lease_row is None:
            raise LocalSessionBindingError("The Group Chat worker lease is unavailable.")
        lease = driver._lease_from_row(lease_row)
        now = driver._timestamp(clock)
        driver._require_active_lease(conn, lease, now=now)
        if (current["run_gateway_id"], current["run_process_generation"], current["run_lease_generation"]) != driver._run_fence(lease):
            raise LocalSessionBindingError("The Group Chat worker changed before session binding.")
        conn.execute(_DDL)
        _validate_schema(conn)
        prior = conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=? AND member_id=?",
                             (task.room_id, member_id)).fetchone()
        expected = (profile, gateway_id, session_id, session_started_at)
        if prior is not None:
            if tuple(prior[key] for key in ("profile", "gateway_id", "session_id", "session_started_at")) != expected:
                raise LocalSessionBindingError("Another private Bot session is already bound to this Group Chat.")
            if (prior["last_session_id"], prior["last_session_started_at"]) not in chain:
                raise LocalSessionBindingError("The private Bot continuation was lost or replaced.")
            conn.execute(f"UPDATE {_TABLE} SET last_session_id=?,last_session_started_at=? WHERE room_id=? AND member_id=?",
                         (*chain[-1], task.room_id, member_id))
            return dict(conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=? AND member_id=?",
                                     (task.room_id, member_id)).fetchone())
        try:
            conn.execute(f"""INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                         (task.room_id, member_id, *expected, task.task_id, execution_generation, now, *chain[-1]))
        except sqlite3.IntegrityError as exc:
            raise LocalSessionBindingError("The private Bot session already belongs to another Group Chat.") from exc
        return dict(conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=? AND member_id=?",
                                 (task.room_id, member_id)).fetchone())
