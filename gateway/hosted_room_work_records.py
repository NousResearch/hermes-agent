"""Bounded room-scoped evidence, never executable driver state or authority."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter

from gateway import hosted_rooms as rooms
from gateway import hosted_room_work_storage as storage
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


def initialize_target_guards(conn):
    columns = [r["name"] for r in conn.execute(f"PRAGMA table_info({TARGET_TABLE})") if r["name"] != "disposition"]
    unchanged = " AND ".join(f'NEW."{k}" IS OLD."{k}" AND typeof(NEW."{k}")=typeof(OLD."{k}")' for k in columns)
    invalidation = f"""OLD.disposition='current' AND NEW.disposition='invalid'
        AND NEW.rowid IS OLD.rowid AND {unchanged}
        AND EXISTS (SELECT 1 FROM hosted_room_replicas WHERE room_id=OLD.room_id AND disbanded_at IS NULL)
        AND NOT EXISTS (SELECT 1 FROM hosted_rooms WHERE room_id=OLD.room_id)"""
    # Replace the old all-update quarantine guard atomically with schema setup.
    # This exception grants only evidence invalidation; retirement still rejects
    # every target update independently below.
    conn.execute("DROP TRIGGER IF EXISTS trg_work_records_active_update")
    for operation in ("INSERT", "UPDATE"):
        exception = f" AND NOT ({invalidation})" if operation == "UPDATE" else ""
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_records_active_{operation.lower()}
            BEFORE {operation} ON {TARGET_TABLE}
            WHEN (NOT EXISTS (SELECT 1 FROM hosted_room_replicas WHERE room_id=NEW.room_id
                AND disbanded_at IS NULL AND quarantine_reason IS NULL)
              OR EXISTS (SELECT 1 FROM hosted_room_quarantine WHERE room_id=NEW.room_id)
              OR EXISTS (SELECT 1 FROM hosted_rooms WHERE room_id=NEW.room_id)){exception}
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
    storage.initialize_lineage_guards(conn)
    if table_exists(conn, storage.INVALID_TABLE):
        storage._invalid_guards(conn)
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


def _budget(conn, table, proposed):
    total_sql, count_sql = storage.usage_sql((SOURCE_TABLE, TARGET_TABLE, PENDING_TABLE, storage.INVALID_TABLE))
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
    if (room is None or room["authority_gateway_id"] != local_gateway_id
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


def ingest(db_path, *, record: dict, token: str, secret: bytes, target_install_id: str, target_profile: str) -> dict:
    from gateway import hosted_room_replicas as replicas
    checked = validate(record)
    error = WorkRecordError("passive work record target is quarantined")
    with replicas._replica_transaction(db_path) as conn:
        row = conn.execute("SELECT * FROM hosted_room_replicas WHERE room_id=?", (checked["room_id"],)).fetchone()
        if row is None or row["quarantine_reason"] is None:
            try:
                return _ingest_audited_locked(conn, checked=checked, row=row, token=token, secret=secret,
                                              target_install_id=target_install_id, target_profile=target_profile)
            except InvalidStoredWorkRecord as exc:
                error = exc  # Preserve the newly discovered invalid disposition.
        # Commit the existing auditor's quarantine, not a metadata write. Raising
        # inside the transaction would roll back that newly discovered evidence.
    raise error


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
    from gateway.hosted_room_passive_lineage import current_locked
    enrolled = current_locked(conn, room_id)
    # The retained prefix may still say epoch 1 after owner enrollment advances.
    # Historical evidence is not writable through an old enrolled sender.
    if enrolled is not None and checked["authority"] != {
        "gateway_id": enrolled["authority_gateway_id"], "epoch": enrolled["authority_epoch"],
    }:
        raise WorkRecordError("unsupported work record lineage: current enrollment requires work-record v2")
    if checked["authority"] != {"gateway_id": row["authority_gateway_id"], "epoch": row["authority_epoch"]}:
        raise WorkRecordError("work record lineage conflicts")
    _validate_roster(checked, members)
    if checked["version"] == 2:
        from gateway.hosted_room_work_lineage import target_prefix_locked
        target_prefix_locked(conn, row, checked)
    prefix = checked["history"]
    if prefix["seq"] > row["last_seq"] or prefix["event_sha256"] != history_anchor(
        conn, "hosted_room_replica_events", room_id, prefix["seq"],
    ):
        raise WorkRecordPrefixError("work record history prefix conflicts")
    old = conn.execute(f"SELECT * FROM {TARGET_TABLE} WHERE room_id=? AND producer_gateway_id=? AND producer_epoch=?", storage.scope(checked)).fetchone()
    if old is not None:
        previous = storage.validate_stored_locked(conn, TARGET_TABLE, old)
        if checked["revision"] < old["revision"] or (checked["revision"] == old["revision"] and checked["digest"] != old["digest"]):
            raise WorkRecordError("work record revision conflicts")
        if prefix["seq"] < previous["history"]["seq"]:
            raise WorkRecordError("work record history regresses")
    storage.save_locked(conn, TARGET_TABLE, checked)
    ack = {"room_id": room_id, "revision": checked["revision"], "digest": checked["digest"], "passive": True}
    if checked["version"] == 2:
        ack.update(version=2, authority=checked["authority"], lineage_sha256=checked["lineage_sha256"])
    return ack


def discard_retired_locked(conn, room_id):
    if table_exists(conn, TARGET_TABLE):
        conn.execute(f"DELETE FROM {TARGET_TABLE} WHERE room_id=?", (room_id,))


def pending_delivery_is_anchored_locked(conn, *, room_id, target_install_id, through_seq):
    """Routing hint only; transmission still revalidates the exact pending record."""
    if not table_exists(conn, PENDING_TABLE):
        return False
    initialize(conn)
    row = conn.execute(f"""SELECT p.* FROM {PENDING_TABLE} p JOIN hosted_rooms r ON r.room_id=p.room_id
        AND r.authority_gateway_id=p.producer_gateway_id AND r.authority_epoch=p.producer_epoch
        WHERE p.room_id=? AND p.target_install_id=? AND p.disposition='current'""",
        (room_id, target_install_id)).fetchone()
    if row is None:
        return False
    try:
        record = storage.validate_stored_locked(conn, PENDING_TABLE, row)
    except InvalidStoredWorkRecord:
        return False
    return row["status"] != "acked" and record["history"]["seq"] <= through_seq


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


def delivery_summaries_locked(conn, room_id=None):
    if not table_exists(conn, PENDING_TABLE):
        return []
    initialize(conn)
    summaries = []
    fields = ("room_id", "target_install_id", "revision", "digest", "status", "producer_gateway_id", "producer_epoch", "disposition")
    for row in conn.execute(f"SELECT * FROM {PENDING_TABLE} WHERE (? IS NULL OR room_id=?) ORDER BY room_id,target_install_id", (room_id, room_id)):
        item = {k: row[k] for k in fields}
        item["source_loss_safe"] = False
        item["disposition"] = storage.scope_disposition(conn, PENDING_TABLE, row)
        try:
            record = storage.validate_stored(row)
            if row["disposition"] == "invalid":
                raise WorkRecordError("invalid evidence")
            item["incompleteness"] = record.get("incompleteness", [])
            if record["availability"] != "available":
                item["incompleteness"] = [*item["incompleteness"], "work_evidence_unknown"]
        except (WorkRecordError, ValueError, TypeError, KeyError):
            item.update(disposition="invalid", incompleteness=["invalid_work_evidence"])
        summaries.append(item)
    for row in conn.execute(f"SELECT * FROM {storage.INVALID_TABLE} WHERE source_table=? AND (? IS NULL OR room_id=?)",
                            (PENDING_TABLE, room_id, room_id)):
        summaries.append({**{k: row[k] for k in fields if k not in {"producer_gateway_id", "producer_epoch"}},
                          "producer_gateway_id": None, "producer_epoch": None,
                          "source_loss_safe": False, "incompleteness": ["invalid_work_evidence"]})
    return summaries


def audit_replica_locked(conn, room_id):
    """Persist invalidity only for the replica explicitly being inspected.

    Quarantine remains owned by the history auditor. Valid rows and immutable
    history are not rewritten; a retired/disbanded copy has no audit write lease.
    """
    if not table_exists(conn, TARGET_TABLE):
        return
    initialize(conn)
    parent = conn.execute("SELECT disbanded_at FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
    if parent is None or parent["disbanded_at"] is not None or copy_retired_locked(conn, room_id):
        return
    for row in conn.execute(f"SELECT * FROM {TARGET_TABLE} WHERE room_id=? AND disposition='current'", (room_id,)).fetchall():
        try:
            storage.validate_stored_locked(conn, TARGET_TABLE, row)
        except InvalidStoredWorkRecord:
            continue  # The caller commits this exact row's classification.


def summary_locked(conn, room_id):
    missing = {"availability": "not_retained", "source_loss_safe": False, "incompleteness": ["work_evidence_unknown"]}
    if not table_exists(conn, TARGET_TABLE):
        return missing
    initialize(conn)
    from gateway.hosted_room_passive_lineage import current_locked, enrolled_history
    enrolled = current_locked(conn, room_id)
    head = conn.execute("SELECT authority_gateway_id,authority_epoch FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
    current = enrolled if enrolled is not None else head
    scopes, selected = [], None
    for row in conn.execute(f"SELECT * FROM {TARGET_TABLE} WHERE room_id=? ORDER BY producer_epoch", (room_id,)):
        producer = {"gateway_id": row["producer_gateway_id"], "epoch": row["producer_epoch"]}
        item = {"producer": producer, "disposition": storage.scope_disposition(conn, TARGET_TABLE, row), "source_loss_safe": False}
        try:
            if row["disposition"] == "invalid":
                raise WorkRecordError("stored work evidence is explicitly invalid")
            record = storage.validate_stored(row)
            item.update({k: record[k] for k in ("revision", "digest", "history", "availability", "reason", "stop", "limitations")})
            item.update(task_count=len(record["tasks"]), receipt_count=len(record["receipts"]),
                phases=dict(Counter(t["phase"] for t in record["tasks"])), tasks=record["tasks"], receipts=record["receipts"],
                incompleteness=record.get("incompleteness", []))
            if record["availability"] != "available":
                item["incompleteness"] = [*item["incompleteness"], "work_evidence_unknown"]
            if record["version"] == 2 and enrolled is not None:
                from gateway.hosted_room_work_lineage import task_origins
                item["task_origins"] = task_origins(record, enrolled_history(enrolled))
        except (WorkRecordError, ValueError, TypeError):
            item.update(availability="invalid", incompleteness=["invalid_work_evidence"])
            if row["disposition"] == "current":
                item["disposition"] = "invalid"
        if item["disposition"] in {"current", "invalid"} and current is not None and producer == {"gateway_id": current["authority_gateway_id"], "epoch": current["authority_epoch"]}:
            selected = item
        scopes.append(item)
    for row in conn.execute(f"SELECT source_table FROM {storage.INVALID_TABLE} WHERE room_id=? AND source_table=?",
                            (room_id, TARGET_TABLE)):
        scopes.append({"producer": None, "disposition": "invalid", "availability": "invalid",
                       "source_loss_safe": False, "incompleteness": ["invalid_work_evidence"]})
    return {"mode": "passive_work_records", **(selected or missing), "scopes": scopes}
