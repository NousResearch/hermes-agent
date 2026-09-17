"""Bounded room-scoped evidence, never executable driver state or authority."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3

from gateway import hosted_rooms as rooms
from gateway import hosted_room_work_storage as storage
from gateway.hosted_rooms_common import identifier, table_exists

VERSION = 1
PERMISSION = "work_records"
MAX_TASKS = 128
MAX_RECEIPTS = 256
MAX_BYTES = 128 * 1024
MAX_STORE_BYTES = 4 * 1024 * 1024
MAX_STORE_ROWS = 512
SOURCE_TABLE = "hosted_room_work_records_source"
TARGET_TABLE = "hosted_room_work_records_target"
PENDING_TABLE = "hosted_room_work_records_pending"
BLOCKED_DELIVERY_STATUSES = {"rejected", "needs_reauthorization", "invalid_ack", "unsupported_lineage", "invalid_work_evidence"}
LIMITATIONS = ["process_local_approvals_not_captured", "field_journals_not_captured",
               "external_effects_not_captured", "absent_record_is_not_non_admission", "not_execution_checkpoint"]
_PHASES = {"queued", "running", "indeterminate", "stopping", "deferred", "settled", "failed", "cancelled"}
_TASK_FIELDS = {
    "task_id", "thread_id", "turn_id", "source_event_seq", "payload_sha256", "member_id", "profile",
    "execution_generation", "cancel_generation", "phase", "settlement_id", "cancel_id"}
_RECEIPT_FIELDS = {
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
    "target_profile", "task_id", "execution_generation", "run_id", "session_id"}
_FIELDS = {"version", "room_id", "home_install_id", "authority", "roster_sha256", "history", "revision", "digest",
           "availability", "reason", "tasks", "receipts", "stop", "limitations"}


class WorkRecordError(ValueError):
    """Controlled invalid or unavailable work evidence."""


class InvalidStoredWorkRecord(WorkRecordError):
    """Retained bytes cannot be reused; commit their invalid disposition."""


class WorkRecordCapacityError(WorkRecordError):
    """The bounded evidence store is full."""


class WorkRecordPrefixError(WorkRecordError):
    """The matching canonical history prefix is not retained."""


def encode(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def _fields(value, fields):
    if not isinstance(value, dict) or set(value) != fields:
        raise WorkRecordError("work record fields are invalid")


def _integer(value, *, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise WorkRecordError("work record integer is invalid")


def _id(value):
    if identifier(value, label="record identity", error=WorkRecordError, max_chars=256) != value:
        raise WorkRecordError("work record identity is not canonical")


def _sha(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise WorkRecordError("work record digest is invalid")


def validate(record: dict) -> dict:
    v2 = isinstance(record, dict) and type(record.get("version")) is int and record["version"] == 2
    _fields(record, _FIELDS | {"lineage_sha256", "incompleteness"} if v2 else _FIELDS)
    if type(record["version"]) is not int or record["version"] not in {1, 2}:
        raise WorkRecordError("work record version is unsupported")
    _id(record["room_id"])
    _id(record["home_install_id"])
    _fields(record["authority"], {"gateway_id", "epoch"})
    _integer(record["authority"]["epoch"], minimum=1)
    if record["authority"]["gateway_id"] != record["home_install_id"] or (not v2 and record["authority"]["epoch"] != 1):
        raise WorkRecordError("work record authority is unsupported")
    if v2:
        _sha(record["lineage_sha256"])
        if record["incompleteness"] != ["prior_authority_work_unknown"]:
            raise WorkRecordError("work record incompleteness is required")
    _sha(record["roster_sha256"])
    _integer(record["revision"], minimum=1)
    _fields(record["history"], {"seq", "event_sha256"})
    _integer(record["history"]["seq"])
    _sha(record["history"]["event_sha256"])
    if record["limitations"] != LIMITATIONS:
        raise WorkRecordError("work record limitations are required")
    if record["availability"] not in {"available", "unavailable"} or record["reason"] not in {
        None, "task_store_missing", "unsupported_task", "bounds_exceeded"}:
        raise WorkRecordError("work record availability is invalid")
    if (record["availability"] == "available") != (record["reason"] is None):
        raise WorkRecordError("work record availability conflicts")
    tasks, receipts = record["tasks"], record["receipts"]
    if not isinstance(tasks, list) or len(tasks) > MAX_TASKS or not isinstance(receipts, list) or len(receipts) > MAX_RECEIPTS:
        raise WorkRecordError("work record list exceeds its bound")
    if record["availability"] == "unavailable" and (tasks or receipts):
        raise WorkRecordError("unavailable work record must not appear complete")
    task_ids, receipt_ids = set(), set()
    for task in tasks:
        _fields(task, _TASK_FIELDS)
        for name in ("task_id", "thread_id", "turn_id", "member_id", "profile"):
            _id(task[name])
        for name in ("settlement_id", "cancel_id"):
            if task[name] is not None:
                _id(task[name])
        for name in ("execution_generation", "cancel_generation"):
            _integer(task[name])
        _integer(task["source_event_seq"], minimum=1)
        _sha(task["payload_sha256"])
        if task["source_event_seq"] > record["history"]["seq"] or task["phase"] not in _PHASES or task["task_id"] in task_ids:
            raise WorkRecordError("work record task conflicts")
        task_ids.add(task["task_id"])
    for receipt in receipts:
        _fields(receipt, _RECEIPT_FIELDS)
        for name in _RECEIPT_FIELDS - {"authority_epoch", "execution_generation"}:
            _id(receipt[name])
        _integer(receipt["execution_generation"], minimum=1)
        _integer(receipt["authority_epoch"], minimum=1)
        if (receipt["room_id"] != record["room_id"]
                or receipt["authority_gateway_id"] != receipt["home_install_id"]
                or (not v2 and (receipt["home_install_id"] != record["home_install_id"] or receipt["authority_epoch"] != 1))):
            raise WorkRecordError("work record receipt lineage conflicts")
        key = tuple(receipt[k] for k in sorted(_RECEIPT_FIELDS - {"run_id", "session_id"}))
        if key in receipt_ids:
            raise WorkRecordError("work record receipt is duplicated")
        receipt_ids.add(key)
    _fields(record["stop"], {"closing", "revocation_complete", "seq", "cancel_id"})
    for key in ("closing", "revocation_complete"):
        if type(record["stop"][key]) is not bool:
            raise WorkRecordError("work record closing fact is invalid")
    _integer(record["stop"]["seq"])
    if record["stop"]["seq"] > record["history"]["seq"]:
        raise WorkRecordError("work record stop is beyond its prefix")
    if record["stop"]["cancel_id"] is not None:
        _id(record["stop"]["cancel_id"])
    _sha(record["digest"])
    if record["digest"] != digest({k: v for k, v in record.items() if k not in {"revision", "digest"}}):
        raise WorkRecordError("work record content digest conflicts")
    if len(encode(record).encode("utf-8")) > MAX_BYTES:
        raise WorkRecordError("work record exceeds its byte bound")
    return record


def initialize(conn: sqlite3.Connection) -> None:
    storage.initialize(conn)


def _budget(conn, table, proposed):
    total_sql, count_sql = storage.usage_sql((SOURCE_TABLE, PENDING_TABLE, storage.INVALID_TABLE))
    total, count = conn.execute(f"SELECT {total_sql}, {count_sql}").fetchone()
    keys = ["room_id", "producer_gateway_id", "producer_epoch"]
    if table == PENDING_TABLE:
        keys.append("target_install_id")
    where = " AND ".join(f"{k}=?" for k in keys)
    old = conn.execute(f"SELECT {storage.row_size_sql(table)} FROM {table} WHERE {where}",
                       [proposed[k] for k in keys]).fetchone()
    size = conn.execute(f"SELECT {storage.row_size_sql(table)} FROM (SELECT "
                        + ",".join(f"? AS {k}" for k in proposed) + ")", tuple(proposed.values())).fetchone()[0]
    if total - (old[0] if old else 0) + size > MAX_STORE_BYTES or count - bool(old) + 1 > MAX_STORE_ROWS:
        raise WorkRecordCapacityError("work record storage is full")


def history_anchor(conn, table: str, room_id: str, seq: int) -> str:
    if seq == 0:
        return digest([])
    row = conn.execute(f"""SELECT seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at
        FROM {table} WHERE room_id=? AND seq=?""", (room_id, seq)).fetchone()
    if row is None:
        raise WorkRecordPrefixError("work record history prefix is unavailable")
    return digest([row["seq"], row["event_id"], row["kind"], json.loads(row["actor_json"]),
                   row["authority_epoch"], json.loads(row["payload_json"]), float(row["created_at"])])


def _capture_tasks(conn, room_id):
    from gateway import hosted_room_driver as driver
    if not table_exists(conn, "hosted_room_driver_tasks"):
        return [], [], "task_store_missing"
    # Published terminal records already travel in canonical history. Keep
    # outstanding tasks and unacknowledged publications within this small slice.
    published = ""
    if table_exists(conn, "hosted_room_policy_publications"):
        published = """NOT EXISTS (SELECT 1 FROM hosted_room_policy_publications AS p
            WHERE p.room_id=t.room_id AND p.task_id=t.task_id AND p.kind IN ('turn.settled','turn.failed','turn.cancelled'))"""
    task_filter = f" AND (status NOT IN ('settled','failed','cancelled') OR {published})" if published else ""
    rows = conn.execute("""SELECT task_id,thread_id,turn_id,source_event_seq,payload_json,payload_digest,
        status,execution_generation,cancel_generation,settlement_id,cancel_id
        FROM hosted_room_driver_tasks AS t WHERE room_id=?""" + task_filter + " ORDER BY task_id LIMIT ?",
        (room_id, MAX_TASKS + 1)).fetchall()
    receipts = [dict(row) for row in conn.execute(
        f"SELECT {','.join(sorted(_RECEIPT_FIELDS))} FROM hosted_room_remote_runs AS t WHERE room_id=?"
        + (f" AND {published}" if published else "") + " ORDER BY task_id,execution_generation,member_id LIMIT ?",
        (room_id, MAX_RECEIPTS + 1))]
    if len(rows) > MAX_TASKS or len(receipts) > MAX_RECEIPTS:
        return [], [], "bounds_exceeded"
    tasks = []
    for row in rows:
        try:
            payload, encoded, payload_digest = driver._task_payload(json.loads(row["payload_json"]))
        except (ValueError, driver.DriverStateError):
            return [], [], "unsupported_task"
        if payload_digest != row["payload_digest"] or encoded != row["payload_json"] or payload["source_event_seq"] != row["source_event_seq"]:
            return [], [], "unsupported_task"
        tasks.append({**{key: row[key] for key in _TASK_FIELDS - {"payload_sha256", "phase", "member_id", "profile"}},
                      "payload_sha256": payload_digest, "phase": row["status"], "profile": payload["target_profile"],
                      "member_id": payload.get("target_member_id", payload["target_profile"])})
    return tasks, receipts, None


def capture_transition_locked(conn, room_id):
    """Observe committed driver facts, without making evidence an execution gate.

    The caller's transaction owns both facts and evidence. A rejected evidence
    write rolls back only its savepoint; malformed retained bytes are classified
    in place, never replaced with a fabricated current snapshot.
    """
    import logging
    owner = conn.execute(
        "SELECT authority_gateway_id FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL",
        (room_id,)).fetchone()
    if owner is None:
        return
    conn.execute("SAVEPOINT work_transition_capture")
    try:
        capture_locked(conn, room_id=room_id, local_gateway_id=owner[0])
    except InvalidStoredWorkRecord:
        logging.getLogger(__name__).warning("Hosted work evidence invalid for room %s", room_id)
    except (WorkRecordError, sqlite3.Error):
        conn.execute("ROLLBACK TO work_transition_capture")
        logging.getLogger(__name__).warning("Hosted work evidence unavailable for room %s", room_id, exc_info=True)
    finally:
        conn.execute("RELEASE work_transition_capture")


def capture(db_path, *, room_id: str, local_gateway_id: str, through_seq: int | None = None) -> dict:
    """Commit a new revision only when one consistent source view changes."""
    with rooms._transaction(db_path, immediate=True) as conn:
        try:
            return capture_locked(conn, room_id=room_id, local_gateway_id=local_gateway_id, through_seq=through_seq)
        except InvalidStoredWorkRecord:
            pass  # Commit this capture's invalid disposition, not new evidence.
    raise InvalidStoredWorkRecord("stored work evidence is invalid")


def capture_locked(conn, *, room_id, local_gateway_id, through_seq=None):
    initialize(conn)
    room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if (room is None or room["authority_gateway_id"] != local_gateway_id or room["disbanded_at"] is not None):
        raise WorkRecordError("work record source is unavailable")
    seq = room["next_seq"] - 1
    if through_seq is not None and seq > through_seq:
        raise WorkRecordPrefixError("history delivery must precede work records")
    tasks, receipts, reason = _capture_tasks(conn, room_id)
    fence = None  # This runtime has no durable peer revocation completion proof.
    stop = conn.execute("""SELECT seq,payload_json FROM hosted_room_events
        WHERE room_id=? AND kind='room.stop_requested' ORDER BY seq DESC LIMIT 1""", (room_id,)).fetchone()
    content = {
        "version": 1 if room["authority_epoch"] == 1 else 2, "room_id": room_id, "home_install_id": local_gateway_id,
        "authority": {"gateway_id": local_gateway_id, "epoch": room["authority_epoch"]},
        "roster_sha256": roster_digest(json.loads(room["members_json"])),
        "history": {"seq": seq, "event_sha256": history_anchor(conn, "hosted_room_events", room_id, seq)},
        "availability": "unavailable" if reason else "available", "reason": reason,
        "tasks": tasks, "receipts": receipts, "limitations": LIMITATIONS,
        "stop": {"closing": fence is not None, "revocation_complete": fence is not None and fence[0] is not None,
                 "seq": stop["seq"] if stop else 0, "cancel_id": json.loads(stop["payload_json"]).get("cancel_id") if stop else None},
    }
    spans = None
    if content["version"] == 2:
        from gateway import hosted_room_work_lineage as lineage
        spans, content["lineage_sha256"] = lineage.source_prefix_locked(conn, room_id, content["authority"], seq)
        content["incompleteness"] = ["prior_authority_work_unknown"]
    previous = conn.execute(f"SELECT * FROM {SOURCE_TABLE} WHERE room_id=? AND producer_gateway_id=? AND producer_epoch=?",
                            storage.scope(content)).fetchone()
    previous_record = storage.validate_stored_locked(conn, SOURCE_TABLE, previous) if previous is not None else None
    revision = previous_record["revision"] + 1 if previous_record else 1
    record = {**content, "revision": revision, "digest": digest(content)}
    if len(encode(record).encode("utf-8")) > MAX_BYTES:
        content.update(tasks=[], receipts=[], availability="unavailable", reason="bounds_exceeded")
        record = {**content, "revision": revision, "digest": digest(content)}
    try:
        validate(record)
        _validate_roster(record, json.loads(room["members_json"]))
        if spans is not None:
            lineage.validate_provenance(record, spans)
    except WorkRecordError:
        if reason is not None:
            raise
        content.update(tasks=[], receipts=[], availability="unavailable", reason="unsupported_task")
        record = validate({**content, "revision": revision, "digest": digest(content)})
    if previous_record is not None and previous_record["digest"] == record["digest"]:
        return previous_record
    storage.save_locked(conn, SOURCE_TABLE, record)
    return record


def _validate_roster(record, members):
    if roster_digest(members) != record["roster_sha256"]:
        raise WorkRecordError("work record roster conflicts")
    roster = {m.get("member_id", m.get("profile")): m for m in members}
    for task in record["tasks"]:
        if roster.get(task["member_id"], {}).get("profile") != task["profile"]:
            raise WorkRecordError("work record task target conflicts")
    for receipt in record["receipts"]:
        target = roster.get(receipt["member_id"], {}).get("target", {})
        if (target.get("kind") != "peer" or target.get("installation_id") != receipt["target_install_id"]
                or target.get("profile") != receipt["target_profile"]):
            raise WorkRecordError("work record receipt target conflicts")


def prepare_delivery_locked(conn, *, room_id, target_install_id, route_generation, local_gateway_id, through_seq):
    """Freeze a source view now; expose it only after its history is acknowledged."""
    initialize(conn)
    current = conn.execute("SELECT authority_gateway_id,authority_epoch FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if current is None or current["authority_gateway_id"] != local_gateway_id:
        raise WorkRecordError("work record source is unavailable")
    key = (room_id, current["authority_gateway_id"], current["authority_epoch"], target_install_id)
    old = conn.execute(f"SELECT * FROM {PENDING_TABLE} WHERE room_id=? AND producer_gateway_id=? AND producer_epoch=? AND target_install_id=?", key).fetchone()
    old_record = storage.validate_stored_locked(conn, PENDING_TABLE, old) if old is not None else None
    if old is not None and old["disposition"] != "current":
        raise WorkRecordError("pending work record is not current")
    if old is not None and old["status"] != "acked":
        if old["route_generation"] == route_generation and old["status"] in BLOCKED_DELIVERY_STATUSES:
            return None
        record = old_record
        current = conn.execute("SELECT authority_gateway_id,authority_epoch,members_json FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
        if (current is None or record["home_install_id"] != local_gateway_id
                or record["authority"] != {"gateway_id": current["authority_gateway_id"], "epoch": current["authority_epoch"]}
                or record["roster_sha256"] != roster_digest(json.loads(current["members_json"]))):
            raise WorkRecordError("pending work record source changed")
    else:
        record = capture_locked(conn, room_id=room_id, local_gateway_id=local_gateway_id)
        if old is not None and (old["revision"], old["digest"]) == (record["revision"], record["digest"]):
            return None
    if old is not None and old["status"] != "acked":
        # A legitimate migrated JSON spelling is still immutable pending bytes.
        conn.execute(f"""UPDATE {PENDING_TABLE} SET route_generation=?,status='pending'
            WHERE room_id=? AND producer_gateway_id=? AND producer_epoch=? AND target_install_id=?
            AND disposition='current'""", (route_generation, *key))
    else:
        storage.save_locked(conn, PENDING_TABLE, record, target_install_id=target_install_id, route_generation=route_generation)
    # Persist the anchor even while history is behind. Recapturing the moving
    # source tip on each attempt can otherwise starve a busy group indefinitely.
    return record if record["history"]["seq"] <= through_seq else None


def delivery_status_locked(conn, *, room_id, target_install_id, route_generation, record, status):
    initialize(conn)
    row = conn.execute(f"SELECT * FROM {PENDING_TABLE} WHERE room_id=? AND target_install_id=? "
                       "AND producer_gateway_id=? AND producer_epoch=? AND disposition='current'",
                       (room_id, target_install_id, record["authority"]["gateway_id"], record["authority"]["epoch"])).fetchone()
    if row is None:
        return False
    try:
        if storage.validate_stored_locked(conn, PENDING_TABLE, row) != record:
            return False
    except InvalidStoredWorkRecord:
        return False
    return conn.execute(f"""UPDATE {PENDING_TABLE} SET status=? WHERE room_id=? AND target_install_id=?
        AND route_generation=? AND revision=? AND digest=? AND producer_gateway_id=? AND producer_epoch=?
        AND disposition='current'""",
        (status, room_id, target_install_id, route_generation, record["revision"], record["digest"],
         record["authority"]["gateway_id"], record["authority"]["epoch"])).rowcount == 1



def roster_digest(members):
    _, encoded = rooms._validate_members(members)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def acknowledgement(record):
    """Exact passive ACK shape; never an admission or execution receipt."""
    ack = {"room_id": record["room_id"], "revision": record["revision"],
           "digest": record["digest"], "passive": True}
    if record["version"] == 2:
        ack.update(version=2, authority=record["authority"], lineage_sha256=record["lineage_sha256"])
    return ack


def acknowledge_locked(conn, *, room_id, target_install_id, route_generation, record, ack):
    if encode(ack) != encode(acknowledgement(record)):
        return False
    return delivery_status_locked(conn, room_id=room_id, target_install_id=target_install_id,
                                  route_generation=route_generation, record=record, status="acked")
