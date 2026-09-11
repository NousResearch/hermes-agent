"""Owner-requested local commit fence for manual same-group reconciliation.

This preserves the old room namespace and history. It neither grants successor
authority nor proves that an already-admitted local or remote tool has stopped.
The final manual recovery controller consumes the receipt before reconciliation;
the public takeover RPCs remain disabled until that controller is complete.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from gateway import hosted_rooms as rooms, hosted_room_safety as safety
from gateway.hosted_room_authority_history import read_history_locked
from gateway.hosted_room_route_schema import require_room_work_open
from gateway.hosted_rooms_common import DbPath, table_exists
from gateway.hosted_room_work_records import encode

_TABLE = "hosted_room_coordinator_freezes"
_REASON = "manual_coordinator_freeze"
_DDL = f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
    room_id TEXT PRIMARY KEY, recovery_id TEXT NOT NULL UNIQUE,
    source_gateway_id TEXT NOT NULL, source_epoch INTEGER NOT NULL,
    successor_gateway_id TEXT NOT NULL, history_seq INTEGER NOT NULL,
    history_sha256 TEXT NOT NULL, roster_sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(room_id) REFERENCES hosted_rooms(room_id) ON DELETE CASCADE)"""


class CoordinatorFreezeError(rooms.HostedRoomError):
    """The explicit request no longer matches the local coordinator state."""


def _receipt(row, *, idempotent):
    return {
        "object": "hermes.group_recovery.local_freeze", **dict(row), "idempotent": idempotent,
        "local_commits_fenced": True, "successor_authorized": False,
        "accepted_work_stopped": False,
    }


def _history_digest(conn, room_id):
    digest = hashlib.sha256()
    for row in conn.execute("""SELECT seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at
        FROM hosted_room_events WHERE room_id=? ORDER BY seq""", (room_id,)):
        digest.update(encode([row["seq"], row["event_id"], row["kind"], json.loads(row["actor_json"]),
                              row["authority_epoch"], json.loads(row["payload_json"]), row["created_at"]]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def freeze_local_coordinator(
    db_path: DbPath, *, room_id: str, recovery_id: str, expected_gateway_id: str,
    expected_epoch: int, expected_history_seq: int, successor_gateway_id: str,
    confirm: bool, now: float | None = None,
) -> dict[str, Any]:
    """Fence only this coordinator after an explicit owner decision.

    The caller must already authenticate the local owner. The successor field
    records the intended destination, not permission to operate on that host.
    No peer calls, pending questions, canonical events or replacement rooms.
    """
    if confirm is not True:
        raise CoordinatorFreezeError("Confirm stopping this host from coordinating the Group Chat.")
    room_id, recovery_id = rooms._room_id(room_id), rooms._event_id(recovery_id)
    expected_gateway_id = rooms._actor_id(expected_gateway_id, "expected_gateway_id")
    successor_gateway_id = rooms._actor_id(successor_gateway_id, "successor_gateway_id")
    rooms._require_positive_int(expected_epoch, "expected_epoch")
    if type(expected_history_seq) is not int or expected_history_seq < 0:
        raise CoordinatorFreezeError("The selected recovery point is invalid.")
    local_gateway_id = rooms.local_authority_gateway_id()
    if expected_gateway_id != local_gateway_id or successor_gateway_id == local_gateway_id:
        raise CoordinatorFreezeError("The request does not move coordination away from this host.")
    requested = (recovery_id, expected_gateway_id, expected_epoch, successor_gateway_id, expected_history_seq)
    with rooms._transaction(db_path, immediate=True) as conn:
        row = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
        if row is None or row["disbanded_at"] is not None:
            raise CoordinatorFreezeError("This Group Chat is no longer available.")
        if table_exists(conn, _TABLE):
            prior = conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=?", (room_id,)).fetchone()
            if prior is not None:
                fields = ("recovery_id", "source_gateway_id", "source_epoch", "successor_gateway_id", "history_seq")
                if tuple(prior[key] for key in fields) != requested:
                    raise CoordinatorFreezeError("This host is already frozen for another recovery decision.")
                fence = conn.execute("SELECT reason FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone()
                if fence is None or fence["reason"] != _REASON:
                    raise CoordinatorFreezeError("The saved coordinator fence is unavailable or changed.")
                return _receipt(prior, idempotent=True)
        safety._raise_if_quarantined(conn, room_id)
        rooms._require_authority(row, expected_gateway_id, expected_epoch, "Group Chat authority changed")
        require_room_work_open(conn, room_id, error=CoordinatorFreezeError)
        if int(row["next_seq"]) - 1 != expected_history_seq:
            raise CoordinatorFreezeError("New Group Chat activity arrived. Refresh before continuing.")
        history = read_history_locked(conn, room_id, gateway_id=expected_gateway_id, epoch=expected_epoch)
        if history is None and expected_epoch > 1:
            raise CoordinatorFreezeError("The original Group Chat host is unverified.")
        origin = history[0]["gateway_id"] if history else expected_gateway_id
        members = json.loads(row["members_json"])
        participants = {origin} | {member["target"]["installation_id"] for member in members
                                   if member.get("target", {}).get("kind") == "peer"}
        if successor_gateway_id not in participants:
            raise CoordinatorFreezeError("The selected host is not a participant in this Group Chat.")
        captured_at = rooms._now(now)
        conn.execute(_DDL)
        try:
            conn.execute(f"INSERT INTO {_TABLE} VALUES(?,?,?,?,?,?,?,?,?)", (
                room_id, *requested, _history_digest(conn, room_id),
                hashlib.sha256(encode(members).encode("utf-8")).hexdigest(), captured_at))
        except sqlite3.IntegrityError as exc:
            raise CoordinatorFreezeError("The recovery decision already belongs to another Group Chat.") from exc
        conn.execute("INSERT INTO hosted_room_quarantine(room_id,reason,detected_at) VALUES(?,?,?)",
                     (room_id, _REASON, captured_at))
        return _receipt(conn.execute(f"SELECT * FROM {_TABLE} WHERE room_id=?", (room_id,)).fetchone(), idempotent=False)
