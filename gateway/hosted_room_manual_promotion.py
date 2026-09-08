"""Move a verified saved Group Chat to a fenced reconciliation state.

Builds on the same-room replica conversion in Teknium's 6af548a1f73f, while
retaining the later namespace, budget and quarantine contracts. The operator
assertion is not independent fencing proof. No execution is enabled here.
"""

import hmac
import json

from gateway import hosted_rooms as rooms, hosted_room_replicas as replicas
from gateway.hosted_room_authority_history import read_history_locked
from gateway.hosted_room_manual_promotion_schema import TABLE, initialize
from gateway.hosted_room_manual_recovery import prepare_recovery_locked
from gateway.hosted_room_work_records import TARGET_TABLE, validate as validate_work_record, _budget
from gateway.hosted_rooms_common import table_exists

_FENCE = "manual_recovery_pending"
_EVENT_COLUMNS = "room_id,seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at"


class ManualRecoveryError(rooms.HostedRoomError):
    """The saved recovery decision is no longer applicable."""


def _receipt(row, *, idempotent):
    return {"object": "hermes.group_recovery.pending", "room_id": row["room_id"],
        "recovery_id": row["recovery_id"], "snapshot_id": row["snapshot_id"],
        "authority_gateway_id": row["target_gateway_id"], "authority_epoch": row["source_epoch"] + 1,
        "saved_through_seq": row["history_seq"], "execution_authorized": False,
        "accepted_tail": "unverified", "status": row["status"], "idempotent": idempotent}


def _move_events(conn, room_id):
    # Move bounded batches so the shared byte budget never temporarily counts
    # both copies. Other connections see either complete state, never a batch.
    while batch := conn.execute(f"SELECT {_EVENT_COLUMNS} FROM hosted_room_replica_events WHERE room_id=? ORDER BY seq LIMIT 128",
                               (room_id,)).fetchall():
        conn.executemany("DELETE FROM hosted_room_replica_events WHERE room_id=? AND seq=?",
                         [(room_id, row["seq"]) for row in batch])
        conn.executemany(f"INSERT INTO hosted_room_events({_EVENT_COLUMNS}) VALUES(?,?,?,?,?,?,?,?)",
                         [tuple(row) for row in batch])


def stage_manual_recovery(db_path, *, room_id, recovery_id, snapshot_id,
                          confirm_previous_host_fenced, confirm_saved_point, now=None):
    """Called only by the authenticated owner controller; public RPC stays disabled.

    This step preserves the copied work ledger and installs a local execution
    quarantine. Reconciliation and fresh scoped routes must precede activation.
    """
    if confirm_previous_host_fenced is not True or confirm_saved_point is not True:
        raise ManualRecoveryError("Confirm that the previous host cannot coordinate and accept the saved recovery point.")
    room_id, recovery_id = rooms._room_id(room_id), rooms._event_id(recovery_id)
    if not isinstance(snapshot_id, str) or len(snapshot_id) != 64 or any(c not in "0123456789abcdef" for c in snapshot_id):
        raise ManualRecoveryError("The saved recovery point is invalid.")
    target = rooms.local_authority_gateway_id()
    with replicas._replica_transaction(db_path) as conn:
        if table_exists(conn, TABLE):
            prior = conn.execute(f"SELECT * FROM {TABLE} WHERE room_id=?", (room_id,)).fetchone()
            if prior is not None:
                if (prior["recovery_id"], prior["snapshot_id"], prior["target_gateway_id"]) != (recovery_id, snapshot_id, target):
                    raise ManualRecoveryError("This Group Chat already has another recovery decision.")
                fence = conn.execute("SELECT reason FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone()
                if prior["status"] != "pending_reconciliation" or fence is None or fence["reason"] != _FENCE:
                    raise ManualRecoveryError("This recovery decision is no longer pending.")
                return _receipt(prior, idempotent=True)
        preview = prepare_recovery_locked(conn, room_id=room_id, target_gateway_id=target)
        if preview["blockers"] or not hmac.compare_digest(preview["snapshot_id"], snapshot_id):
            raise ManualRecoveryError("The saved Group Chat changed or is incomplete. Refresh before continuing.")
        copy = conn.execute("SELECT * FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
        if conn.execute("SELECT COUNT(*) FROM hosted_rooms WHERE disbanded_at IS NULL").fetchone()[0] >= rooms.MAX_ACTIVE_ROOMS:
            raise ManualRecoveryError("This host has reached its Group Chat limit.")
        record = conn.execute(f"SELECT record_json FROM {TARGET_TABLE} WHERE room_id=?", (room_id,)).fetchone()
        if record is None:
            raise ManualRecoveryError("Saved work records are unavailable.")
        validate_work_record(json.loads(record[0]))
        initialize(conn)
        timestamp = rooms._now(now)
        # Transfer the existing metadata charge rather than temporarily keeping
        # two charged copies. The writer transaction restores both on failure.
        conn.execute(f"DELETE FROM {TARGET_TABLE} WHERE room_id=?", (room_id,))
        _budget(conn, TABLE, room_id, record[0])
        conn.execute(f"INSERT INTO {TABLE} VALUES(?,?,?,?,?,?,?,?,?,?)", (
            room_id, recovery_id, snapshot_id, copy["authority_gateway_id"], copy["authority_epoch"],
            target, copy["last_seq"], record[0], timestamp, "transferring"))
        conn.execute("""INSERT INTO hosted_rooms(room_id,name,members_json,authority_gateway_id,authority_epoch,
            next_seq,event_bytes,revision,created_at,updated_at,disbanded_at) VALUES(?,?,?,?,?,?,?,?,?,?,NULL)""", (
            room_id, copy["name"], copy["members_json"], target, copy["authority_epoch"] + 1,
            copy["last_seq"] + 1, copy["event_bytes"], 1, copy["created_at"], timestamp))
        _move_events(conn, room_id)
        conn.execute("DELETE FROM hosted_room_replicas WHERE room_id=?", (room_id,))
        row = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
        epoch = int(copy["authority_epoch"]) + 1
        event_id = f"system:authority-claimed:{epoch}"
        payload = rooms._canonical_json({"previous_gateway_id": copy["authority_gateway_id"],
            "authority_gateway_id": target, "authority_epoch": epoch, "recovery_id": recovery_id,
            "saved_recovery_point": snapshot_id, "accepted_tail": "unverified"},
            label="payload", max_bytes=rooms.MAX_EVENT_JSON_BYTES)
        added = rooms._insert_event(conn, row, room_id, copy["last_seq"] + 1, event_id, "authority.claimed",
            rooms._system_actor_json("authority-control"), epoch, payload, timestamp, allow_control=True)
        conn.execute("UPDATE hosted_rooms SET next_seq=next_seq+1,event_bytes=event_bytes+?,revision=revision+1 WHERE room_id=?",
                     (added, room_id))
        read_history_locked(conn, room_id, gateway_id=target, epoch=epoch)
        conn.execute("INSERT INTO hosted_room_quarantine(room_id,reason,detected_at) VALUES(?,?,?)", (room_id, _FENCE, timestamp))
        conn.execute(f"UPDATE {TABLE} SET status='pending_reconciliation' WHERE room_id=?", (room_id,))
        return _receipt(conn.execute(f"SELECT * FROM {TABLE} WHERE room_id=?", (room_id,)).fetchone(), idempotent=False)
