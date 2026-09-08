"""Bounded room-scoped evidence, never executable driver state or authority."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter

from gateway import hosted_rooms as rooms
from gateway.hosted_room_replica_retirement import copy_retired_locked, roster_digest
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
BLOCKED_DELIVERY_STATUSES = {"rejected", "needs_reauthorization", "invalid_ack"}
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
    _fields(record, _FIELDS)
    if type(record["version"]) is not int or record["version"] != VERSION:
        raise WorkRecordError("work record version is unsupported")
    _id(record["room_id"])
    _id(record["home_install_id"])
    _fields(record["authority"], {"gateway_id", "epoch"})
    if record["authority"] != {"gateway_id": record["home_install_id"], "epoch": 1} or type(record["authority"]["epoch"]) is not int:
        raise WorkRecordError("work record authority is unsupported")
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
        if (receipt["room_id"] != record["room_id"] or receipt["home_install_id"] != record["home_install_id"]
                or receipt["authority_gateway_id"] != record["home_install_id"]
                or type(receipt["authority_epoch"]) is not int or receipt["authority_epoch"] != 1):
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
    for table, parent in ((SOURCE_TABLE, "hosted_rooms"), (TARGET_TABLE, "hosted_room_replicas")):
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            room_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
            FOREIGN KEY(room_id) REFERENCES {parent}(room_id) ON DELETE CASCADE)""")
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {PENDING_TABLE} (
        room_id TEXT NOT NULL, target_install_id TEXT NOT NULL, route_generation TEXT NOT NULL,
        revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
        status TEXT NOT NULL, PRIMARY KEY(room_id,target_install_id),
        FOREIGN KEY(room_id) REFERENCES hosted_rooms(room_id) ON DELETE CASCADE)""")
    for operation in ("INSERT", "UPDATE"):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_records_active_{operation.lower()}
            BEFORE {operation} ON {TARGET_TABLE}
            WHEN NOT EXISTS (SELECT 1 FROM hosted_room_replicas WHERE room_id=NEW.room_id
                AND disbanded_at IS NULL AND quarantine_reason IS NULL)
              OR EXISTS (SELECT 1 FROM hosted_room_quarantine WHERE room_id=NEW.room_id)
              OR EXISTS (SELECT 1 FROM hosted_rooms WHERE room_id=NEW.room_id)
            BEGIN SELECT RAISE(ABORT, 'passive work record target is unavailable'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_records_disband_cleanup
        AFTER UPDATE OF disbanded_at ON hosted_room_replicas WHEN NEW.disbanded_at IS NOT NULL
        BEGIN DELETE FROM {TARGET_TABLE} WHERE room_id=NEW.room_id; END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_records_history_cleanup
        AFTER DELETE ON hosted_room_replicas
        BEGIN DELETE FROM {TARGET_TABLE} WHERE room_id=OLD.room_id; END""")
    initialize_retirement_guards(conn)


def initialize_retirement_guards(conn):
    """Either owner may initialize first; persist guards for older SQLite writers."""
    from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
    if not table_exists(conn, TARGET_TABLE) or not table_exists(conn, RETIREMENT_TABLE):
        return
    for operation in ("INSERT", "UPDATE"):
        ids = "NEW.room_id" if operation == "INSERT" else "NEW.room_id,OLD.room_id"
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_records_retired_{operation.lower()}
            BEFORE {operation} ON {TARGET_TABLE}
            WHEN EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id IN ({ids}))
            BEGIN SELECT RAISE(ABORT, 'replica copy is retired'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_records_retired_cleanup
        AFTER INSERT ON {RETIREMENT_TABLE}
        BEGIN DELETE FROM {TARGET_TABLE} WHERE room_id=NEW.room_id; END""")


def _budget(conn, table, room_id, data, target_install_id=None):
    from gateway.hosted_room_work_record_budget import stores
    total, count = 0, 0
    owners = stores(conn)
    for name, column, _keys in owners:
        row = conn.execute(f"SELECT COALESCE(SUM(length(CAST({column} AS BLOB))),0),COUNT(*) FROM {name}").fetchone()
        total, count = total + row[0], count + row[1]
    where, args = "room_id=?", (room_id,)
    if target_install_id is not None:
        where, args = where + " AND target_install_id=?", (*args, target_install_id)
    column = next(column for name, column, _keys in owners if name == table)
    old = conn.execute(f"SELECT length(CAST({column} AS BLOB)) FROM {table} WHERE {where}", args).fetchone()
    if total - (old[0] if old else 0) + len(data.encode("utf-8")) > MAX_STORE_BYTES or (old is None and count >= MAX_STORE_ROWS):
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


def capture(db_path, *, room_id: str, local_gateway_id: str, through_seq: int | None = None) -> dict:
    """Commit a new revision only when one consistent source view changes."""
    with rooms._transaction(db_path, immediate=True) as conn:
        return capture_locked(conn, room_id=room_id, local_gateway_id=local_gateway_id, through_seq=through_seq)


def capture_locked(conn, *, room_id, local_gateway_id, through_seq=None):
    initialize(conn)
    room = conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
    if (room is None or room["authority_gateway_id"] != local_gateway_id or room["authority_epoch"] != 1
            or conn.execute("SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone()):
        raise WorkRecordError("work record source is unavailable")
    seq = room["next_seq"] - 1
    if through_seq is not None and seq > through_seq:
        raise WorkRecordPrefixError("history delivery must precede work records")
    tasks, receipts, reason = _capture_tasks(conn, room_id)
    fence = conn.execute("SELECT revocation_complete_at FROM hosted_room_disband_fences WHERE room_id=?", (room_id,)).fetchone()
    stop = conn.execute("""SELECT seq,payload_json FROM hosted_room_events
        WHERE room_id=? AND kind='room.stop_requested' ORDER BY seq DESC LIMIT 1""", (room_id,)).fetchone()
    content = {
        "version": VERSION, "room_id": room_id, "home_install_id": local_gateway_id,
        "authority": {"gateway_id": local_gateway_id, "epoch": 1},
        "roster_sha256": roster_digest(json.loads(room["members_json"])),
        "history": {"seq": seq, "event_sha256": history_anchor(conn, "hosted_room_events", room_id, seq)},
        "availability": "unavailable" if reason else "available", "reason": reason,
        "tasks": tasks, "receipts": receipts, "limitations": LIMITATIONS,
        "stop": {"closing": fence is not None, "revocation_complete": fence is not None and fence[0] is not None,
                 "seq": stop["seq"] if stop else 0, "cancel_id": json.loads(stop["payload_json"]).get("cancel_id") if stop else None},
    }
    previous = conn.execute(f"SELECT * FROM {SOURCE_TABLE} WHERE room_id=?", (room_id,)).fetchone()
    revision = previous["revision"] + 1 if previous else 1
    record = {**content, "revision": revision, "digest": digest(content)}
    if len(encode(record).encode("utf-8")) > MAX_BYTES:
        content.update(tasks=[], receipts=[], availability="unavailable", reason="bounds_exceeded")
        record = {**content, "revision": revision, "digest": digest(content)}
    try:
        validate(record)
        _validate_roster(record, json.loads(room["members_json"]))
    except WorkRecordError:
        if reason is not None:
            raise
        content.update(tasks=[], receipts=[], availability="unavailable", reason="unsupported_task")
        record = validate({**content, "revision": revision, "digest": digest(content)})
    if previous is not None and previous["digest"] == record["digest"]:
        return json.loads(previous["record_json"])
    data = encode(record)
    _budget(conn, SOURCE_TABLE, room_id, data)
    conn.execute(f"INSERT OR REPLACE INTO {SOURCE_TABLE} VALUES (?,?,?,?)", (room_id, revision, record["digest"], data))
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


def ingest(db_path, *, record: dict, token: str, secret: bytes, target_install_id: str, target_profile: str) -> dict:
    from gateway import hosted_room_replicas as replicas
    checked = validate(record)
    with replicas._replica_transaction(db_path) as conn:
        row = conn.execute("SELECT * FROM hosted_room_replicas WHERE room_id=?", (checked["room_id"],)).fetchone()
        if row is None or row["quarantine_reason"] is None:
            return _ingest_audited_locked(conn, checked=checked, row=row, token=token, secret=secret,
                                          target_install_id=target_install_id, target_profile=target_profile)
        # Commit the existing auditor's quarantine, not a metadata write. Raising
        # inside the transaction would roll back that newly discovered evidence.
    raise WorkRecordError("passive work record target is quarantined")


def _ingest_audited_locked(conn, *, checked, row, token, secret, target_install_id, target_profile):
    from gateway.hosted_room_replica_ingress import authorize_granted_room
    initialize(conn)
    room_id = checked["room_id"]
    if (row is None or row["disbanded_at"] is not None or copy_retired_locked(conn, room_id)
            or conn.execute("SELECT 1 FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone()
            or conn.execute("SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()):
        raise WorkRecordError("passive work record target is unavailable")
    members = json.loads(row["members_json"])
    authorize_granted_room(
        token=token, secret=secret, target_install_id=target_install_id, target_profile=target_profile,
        room_id=room_id, members=members, authority=checked["authority"], permission=PERMISSION,
    )(conn)
    if checked["authority"] != {"gateway_id": row["authority_gateway_id"], "epoch": row["authority_epoch"]}:
        raise WorkRecordError("work record lineage conflicts")
    _validate_roster(checked, members)
    prefix = checked["history"]
    if prefix["seq"] > row["last_seq"] or prefix["event_sha256"] != history_anchor(
        conn, "hosted_room_replica_events", room_id, prefix["seq"],
    ):
        raise WorkRecordPrefixError("work record history prefix conflicts")
    old = conn.execute(f"SELECT * FROM {TARGET_TABLE} WHERE room_id=?", (room_id,)).fetchone()
    if old is not None:
        previous = json.loads(old["record_json"])
        if checked["revision"] < old["revision"] or (checked["revision"] == old["revision"] and checked["digest"] != old["digest"]):
            raise WorkRecordError("work record revision conflicts")
        if prefix["seq"] < previous["history"]["seq"]:
            raise WorkRecordError("work record history regresses")
    data = encode(checked)
    _budget(conn, TARGET_TABLE, room_id, data)
    conn.execute(f"INSERT OR REPLACE INTO {TARGET_TABLE} VALUES (?,?,?,?)",
                 (room_id, checked["revision"], checked["digest"], data))
    return {"room_id": room_id, "revision": checked["revision"], "digest": checked["digest"], "passive": True}


def discard_retired_locked(conn, room_id):
    if table_exists(conn, TARGET_TABLE):
        conn.execute(f"DELETE FROM {TARGET_TABLE} WHERE room_id=?", (room_id,))


def pending_delivery_is_anchored_locked(conn, *, room_id, target_install_id, through_seq):
    """Routing hint only; transmission still revalidates the exact pending record."""
    if not table_exists(conn, PENDING_TABLE):
        return False
    row = conn.execute(f"SELECT status,record_json FROM {PENDING_TABLE} WHERE room_id=? AND target_install_id=?",
                       (room_id, target_install_id)).fetchone()
    if row is None or row["status"] == "acked":
        return False
    try:
        record = validate(json.loads(row["record_json"]))
    except (WorkRecordError, ValueError, TypeError):
        return False
    return record["history"]["seq"] <= through_seq


def prepare_delivery_locked(conn, *, room_id, target_install_id, route_generation, local_gateway_id, through_seq):
    """Freeze a source view now; expose it only after its history is acknowledged."""
    initialize(conn)
    key = (room_id, target_install_id)
    old = conn.execute(f"SELECT * FROM {PENDING_TABLE} WHERE room_id=? AND target_install_id=?", key).fetchone()
    if old is not None and old["status"] != "acked":
        if old["route_generation"] == route_generation and old["status"] in BLOCKED_DELIVERY_STATUSES:
            return None
        record = validate(json.loads(old["record_json"]))
        current = conn.execute("SELECT authority_gateway_id,authority_epoch,members_json FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()
        if (current is None or record["home_install_id"] != local_gateway_id
                or record["authority"] != {"gateway_id": current["authority_gateway_id"], "epoch": current["authority_epoch"]}
                or record["roster_sha256"] != roster_digest(json.loads(current["members_json"]))):
            raise WorkRecordError("pending work record source changed")
    else:
        record = capture_locked(conn, room_id=room_id, local_gateway_id=local_gateway_id)
        if old is not None and (old["revision"], old["digest"]) == (record["revision"], record["digest"]):
            return None
    data = encode(record)
    _budget(conn, PENDING_TABLE, room_id, data, target_install_id)
    conn.execute(f"INSERT OR REPLACE INTO {PENDING_TABLE} VALUES (?,?,?,?,?,?,?)",
                 (*key, route_generation, record["revision"], record["digest"], data, "pending"))
    # Persist the anchor even while history is behind. Recapturing the moving
    # source tip on each attempt can otherwise starve a busy group indefinitely.
    return record if record["history"]["seq"] <= through_seq else None


def delivery_status_locked(conn, *, room_id, target_install_id, route_generation, record, status):
    return conn.execute(f"""UPDATE {PENDING_TABLE} SET status=? WHERE room_id=? AND target_install_id=?
        AND route_generation=? AND revision=? AND digest=?""",
                        (status, room_id, target_install_id, route_generation, record["revision"], record["digest"])).rowcount == 1


def delivery_summaries_locked(conn, room_id=None):
    if not table_exists(conn, PENDING_TABLE):
        return []
    return [dict(row) for row in conn.execute(f"""SELECT room_id,target_install_id,revision,digest,status
        FROM {PENDING_TABLE} WHERE (? IS NULL OR room_id=?) ORDER BY room_id,target_install_id""", (room_id, room_id))]


def summary_locked(conn, room_id):
    if not table_exists(conn, TARGET_TABLE):
        return {"availability": "not_retained", "source_loss_safe": False}
    row = conn.execute(f"SELECT record_json FROM {TARGET_TABLE} WHERE room_id=?", (room_id,)).fetchone()
    if row is None:
        return {"availability": "not_retained", "source_loss_safe": False}
    record = validate(json.loads(row[0]))
    return {"mode": "passive_work_records", "source_loss_safe": False,
            **{k: record[k] for k in ("revision", "digest", "history", "availability", "reason", "stop", "limitations")},
            "task_count": len(record["tasks"]), "receipt_count": len(record["receipts"]),
            "phases": dict(Counter(t["phase"] for t in record["tasks"])),
            "tasks": record["tasks"], "receipts": record["receipts"]}
