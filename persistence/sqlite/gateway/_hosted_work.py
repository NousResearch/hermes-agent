"""The hosted-room work-record capture: bounded room-scoped evidence of driver tasks and remote-run receipts, recorded inside the transaction that changed them, never executable driver state or authority.

Holds the capture, its storage schema with the budget, immutability and authority-transition triggers, which every capture runs first under a savepoint of the caller's transaction, the lineage check of a successor authority's evidence and the authority-history read. The record rules are ``session_rules.hosted_rooms``'. ``capture_transition_locked`` swallows every work-record failure under its savepoint; ``AuthorityHistoryError`` from the lineage read and ``HostedRoomError`` from the roster check escape and roll the caller's transaction back.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from persistence.gateway.hosted_rooms import AuthorityHistoryError, DriverStateError
from session_rules.hosted_rooms import (
    INVALID_TABLE, LIMITATIONS, MAX_AUTHORITY_SPANS, MAX_BYTES, MAX_RECEIPTS, MAX_STORE_BYTES, MAX_STORE_ROWS,
    MAX_TASKS, PENDING_TABLE, SOURCE_TABLE, TARGET_TABLE, _RECEIPT_FIELDS, _TASK_FIELDS, AuthoritySpan,
    InvalidStoredWorkRecord, WorkRecordCapacityError, WorkRecordError, WorkRecordPrefixError, _gateway,
    _validate_roster, at_sequence, digest, encode, retained_fields, roster_digest, scope, validate,
    validate_history, validate_provenance, validate_stored,
)
from session_rules.hosted_tasks import normalize_task_payload

from ._hosted_rooms_common import table_exists

__all__ = ["capture_transition_locked"]

# Main's logger name for work-record warnings.
logger = logging.getLogger("gateway.hosted_room_work_records")


# ── storage ───────────────────────────────────────────────────────────────────


def row_size_sql(table, prefix=""):
    # Reserve the longest bounded disposition up front so freezing/cleanup can
    # still run at capacity. Every original variable column is charged as bytes.
    return "20+" + "+".join(f"COALESCE(length(CAST({prefix}{k} AS BLOB)),0)" for k in retained_fields(table))


def usage_sql(tables):
    return (" + ".join(f"(SELECT COALESCE(SUM({row_size_sql(t)}),0) FROM {t})" for t in tables),
            " + ".join(f"(SELECT COUNT(*) FROM {t})" for t in tables))


# The schema step under its own savepoint; the caller's transaction is always open here.
def initialize(conn):
    # Nested initialization must never commit or roll back caller-owned work.
    conn.execute("SAVEPOINT work_record_initialize")
    try:
        _initialize_locked(conn)
    except BaseException:
        conn.execute("ROLLBACK TO work_record_initialize")
        conn.execute("RELEASE work_record_initialize")
        raise
    conn.execute("RELEASE work_record_initialize")


# Creates the work-record tables, rebuilds a table that predates the producer columns, then installs the triggers.
def _initialize_locked(conn):
    tables = (SOURCE_TABLE, PENDING_TABLE)
    # Orphans are opaque evidence, not snapshots. No FK or producer can honestly
    # be supplied for them. Preserve every original column and forbid mutation.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {INVALID_TABLE} (
        evidence_id INTEGER PRIMARY KEY, source_table TEXT NOT NULL,
        room_id TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
        target_install_id TEXT, route_generation TEXT, status TEXT,
        disposition TEXT NOT NULL CHECK(disposition='invalid'))""")
    for table in tables:
        pending = table == PENDING_TABLE
        parent = "hosted_room_replicas" if table == TARGET_TABLE else "hosted_rooms"
        columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        legacy = bool(columns) and "producer_gateway_id" not in columns
        # Preserve surrounding cleanup triggers by recreating the same table name
        # inside this transaction, rather than renaming its referents.
        rows = conn.execute(f"SELECT * FROM {table}").fetchall() if legacy else []
        triggers = conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,)).fetchall() if legacy else []
        if legacy:
            conn.execute(f"DROP TABLE {table}")
        extra = "target_install_id TEXT NOT NULL, route_generation TEXT NOT NULL, status TEXT NOT NULL," if pending else ""
        key = ",target_install_id" if pending else ""
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            room_id TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
            producer_gateway_id TEXT NOT NULL, producer_epoch INTEGER NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('current','superseded_authority','historical','invalid')),
            {extra} PRIMARY KEY(room_id,producer_gateway_id,producer_epoch{key}),
            FOREIGN KEY(room_id) REFERENCES {parent}(room_id) ON DELETE CASCADE)""")
        for row in rows:
            if not conn.execute(f"SELECT 1 FROM {parent} WHERE room_id=?", (row["room_id"],)).fetchone():
                fields = ["source_table", "room_id", "revision", "digest", "record_json", "disposition"]
                values = [table, row["room_id"], row["revision"], row["digest"], row["record_json"], "invalid"]
                if pending:
                    fields += ["target_install_id", "route_generation", "status"]
                    values += [row[k] for k in fields[-3:]]
                conn.execute(f"INSERT INTO {INVALID_TABLE} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
                continue
            producer, epoch, disposition = "", 0, "invalid"
            try:
                record = validate(json.loads(row["record_json"]))
                if (record["version"] != 1 or record["room_id"] != row["room_id"]
                        or record["revision"] != row["revision"] or record["digest"] != row["digest"]):
                    raise WorkRecordError("legacy work scope conflicts")
                producer, epoch, disposition = record["home_install_id"], 1, "current"
            except (ValueError, TypeError, KeyError):
                pass  # Explicitly invalid, with original bytes and no invented producer.
            fields = ["room_id", "revision", "digest", "record_json", "producer_gateway_id", "producer_epoch", "disposition"]
            values = [row[k] for k in fields[:4]] + [producer, epoch, disposition]
            if pending:
                fields += ["target_install_id", "route_generation", "status"]
                values += [row[k] for k in fields[-3:]]
            values[6] = scope_disposition(conn, table, dict(zip(fields, values)))
            conn.execute(f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
        for trigger in triggers:
            conn.execute(trigger[0])
    _guards(conn, tables)
    _invalid_guards(conn)
    # Ordinary initialization owns schema/guards only. Each consuming operation
    # validates its own row; authority/enrollment triggers freeze exact-room old
    # scopes. Legacy dispositions were resolved at insertion above.


def scope_disposition(conn, table, row):
    """Read-only authority projection, also used when inserting migrated rows."""
    if row["disposition"] != "current":
        return row["disposition"]
    current = None
    current = conn.execute("SELECT authority_gateway_id,authority_epoch FROM hosted_rooms WHERE room_id=?",
                           (row["room_id"],)).fetchone()
    if current is None or tuple(current) == (row["producer_gateway_id"], row["producer_epoch"]):
        return "current"
    return "superseded_authority" if table == PENDING_TABLE and row["status"] != "acked" else "historical"


# The budget, immutability, replace and authority-transition triggers of the work-record tables.
def _guards(conn, tables):
    total, count = usage_sql((*tables, INVALID_TABLE))
    for table in tables:
        for operation in ("INSERT", "UPDATE"):
            key = "room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch"
            if table == PENDING_TABLE:
                key += " AND target_install_id=NEW.target_install_id"
            old_size = (f"COALESCE((SELECT {row_size_sql(table)} FROM {table} WHERE {key}),0)"
                        if operation == "INSERT" else row_size_sql(table, "OLD."))
            new_size = row_size_sql(table, "NEW.")
            old_count = f"(SELECT COUNT(*) FROM {table} WHERE {key})" if operation == "INSERT" else "1"
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_budget_v2_{operation.lower()}")
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_budget_v3_{operation.lower()}")
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_budget_v4_{operation.lower()}")
            conn.execute(f"""CREATE TRIGGER trg_{table}_budget_v4_{operation.lower()}
                BEFORE {operation} ON {table} WHEN (
                    (({new_size})>({old_size}) AND ({total})-({old_size})+({new_size})>{MAX_STORE_BYTES})
                    OR ({old_count}=0 AND ({total}>{MAX_STORE_BYTES} OR {count}+1>{MAX_STORE_ROWS})))
                BEGIN SELECT RAISE(ABORT, 'work record storage is full'); END""")
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_immutable_v2
            BEFORE UPDATE ON {table} WHEN
                NEW.room_id!=OLD.room_id OR NEW.producer_gateway_id!=OLD.producer_gateway_id
                OR NEW.producer_epoch!=OLD.producer_epoch
                OR (OLD.disposition!='current' AND (NEW.record_json!=OLD.record_json OR NEW.revision!=OLD.revision
                    OR NEW.digest!=OLD.digest OR NEW.disposition!=OLD.disposition))
            BEGIN SELECT RAISE(ABORT, 'historical work record is immutable'); END""")
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_delete_v2 BEFORE DELETE ON {table}
            WHEN OLD.disposition!='current' AND EXISTS (SELECT 1 FROM hosted_rooms
                WHERE room_id=OLD.room_id AND disbanded_at IS NULL)
            BEGIN SELECT RAISE(ABORT, 'historical work record is immutable'); END""")
        # INSERT OR REPLACE performs a delete internally. Refuse it before the
        # uniqueness conflict can remove an immutable snapshot.
        extra = " AND p.target_install_id=NEW.target_install_id" if table == PENDING_TABLE else ""
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_replace_v2
            BEFORE INSERT ON {table} WHEN EXISTS (SELECT 1 FROM {table} p WHERE p.room_id=NEW.room_id
                AND p.producer_gateway_id=NEW.producer_gateway_id AND p.producer_epoch=NEW.producer_epoch
                {extra} AND p.disposition!='current')
            BEGIN SELECT RAISE(ABORT, 'historical work record is immutable'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_pending_outcome_v2 BEFORE UPDATE ON {PENDING_TABLE}
        WHEN OLD.disposition!='current' AND (NEW.status!=OLD.status OR NEW.route_generation!=OLD.route_generation
            OR NEW.target_install_id!=OLD.target_install_id)
        BEGIN SELECT RAISE(ABORT, 'historical delivery outcome is immutable'); END""")
    # A raw authority update by an older owner must also freeze old evidence.
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_authority_transition_v2
        AFTER UPDATE OF authority_gateway_id,authority_epoch ON hosted_rooms BEGIN
        UPDATE {SOURCE_TABLE} SET disposition='historical' WHERE room_id=NEW.room_id AND disposition='current'
            AND (producer_gateway_id!=NEW.authority_gateway_id OR producer_epoch!=NEW.authority_epoch);
        UPDATE {PENDING_TABLE} SET disposition=CASE WHEN status='acked' THEN 'historical' ELSE 'superseded_authority' END
            WHERE room_id=NEW.room_id AND disposition='current'
            AND (producer_gateway_id!=NEW.authority_gateway_id OR producer_epoch!=NEW.authority_epoch);
        END""")


def validate_stored_locked(conn, table, row):
    """Validate the consumed row; invalidate only its unchanged current version.

    The savepoint joins the caller's write transaction. Historical rows are never rewritten.
    """
    try:
        return validate_stored(row)
    except InvalidStoredWorkRecord:
        if row["disposition"] == "current":
            # Match all examined values, not merely a key that may since have
            # been replaced. IS preserves NULL/BLOB distinctions in old stores.
            fields = list(row.keys())
            where = " AND ".join(f'"{k}" IS ?' for k in fields)
            conn.execute("SAVEPOINT work_record_invalidate")
            try:
                conn.execute(f"UPDATE {table} SET disposition='invalid' WHERE {where}", [row[k] for k in fields])
            except BaseException:
                conn.execute("ROLLBACK TO work_record_invalidate")
                conn.execute("RELEASE work_record_invalidate")
                raise
            conn.execute("RELEASE work_record_invalidate")
        raise


# Upserts the current record of its scope after the budget check; a delivery row starts ``pending``.
def save_locked(conn, table, record, *, target_install_id=None, route_generation=None):
    data = encode(record)
    fields = ["room_id", "producer_gateway_id", "producer_epoch", "revision", "digest", "record_json", "disposition"]
    values = [*scope(record), record["revision"], record["digest"], data, "current"]
    keys = fields[:3]
    if target_install_id is not None:
        fields += ["target_install_id", "route_generation", "status"]
        values += [target_install_id, route_generation, "pending"]
        keys += ["target_install_id"]
    _budget(conn, table, dict(zip(fields, values)))
    updates = ','.join(f'{k}=excluded.{k}' for k in fields if k not in keys)
    conn.execute(f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)}) "
                 f"ON CONFLICT({','.join(keys)}) DO UPDATE SET {updates}", values)


# Refuses every insert, update and delete on the invalid-evidence table.
def _invalid_guards(conn):
    # There is no reviewed retirement grant in this runtime. Opaque evidence
    # cannot acquire a first-writer owner or be reclaimed by an ordinary writer.
    for operation in ("INSERT", "UPDATE", "DELETE"):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_invalid_{operation.lower()}
            BEFORE {operation} ON {INVALID_TABLE}
            BEGIN SELECT RAISE(ABORT, 'invalid work evidence is immutable'); END""")


# ── authority history ─────────────────────────────────────────────────────────


#: Committed authority transfers, each the first event of its span. ``authority.claimed`` is the new
#: owner's record; ``authority.lost`` is the record the outgoing owner commits when it is fenced to a
#: proven newer epoch. Both carry the same previous/next/epoch payload and are checked alike, so a store
#: whose log holds its own fence followed by a later claim reads as one unbroken chain.
TRANSFER_KINDS = ("authority.claimed", "authority.lost")


def read_history_locked(conn, room_id: str, *, gateway_id: str, epoch: int) -> list[dict[str, Any]] | None:
    """Read committed authority transfers within the caller's room-state transaction.

    Legacy adoption is left on its existing path; it is not retroactively
    certified as a history of physical-host transfers.
    """
    rows = conn.execute(
        f"""SELECT seq,authority_epoch,actor_json,payload_json FROM hosted_room_events
           WHERE room_id=? AND kind IN {TRANSFER_KINDS} ORDER BY seq LIMIT ?""",
        (room_id, MAX_AUTHORITY_SPANS),
    ).fetchall()
    if len(rows) >= MAX_AUTHORITY_SPANS:
        raise AuthorityHistoryError("authority history exceeds its bound")
    if not rows:
        if epoch == 1:
            return None
        raise AuthorityHistoryError("current authority has no committed lineage")
    first = json.loads(rows[0]["payload_json"])
    if isinstance(first, dict) and first.get("previous_gateway_id") == "legacy":
        return None
    if not isinstance(first, dict) or "previous_gateway_id" not in first:
        raise AuthorityHistoryError("authority history has no previous owner")
    history = [AuthoritySpan(_gateway(first["previous_gateway_id"]), 1, 0)]
    for row in rows:
        actor, payload = json.loads(row["actor_json"]), json.loads(row["payload_json"])
        if actor != {"kind": "system", "id": "authority-control"} or not isinstance(payload, dict):
            raise AuthorityHistoryError("authority claim provenance is invalid")
        expected = history[-1].epoch + 1
        if (payload.get("previous_gateway_id") != history[-1].gateway_id
                or type(payload.get("authority_epoch")) is not int or payload["authority_epoch"] != expected
                or row["authority_epoch"] != expected):
            raise AuthorityHistoryError("authority claim does not extend its predecessor")
        history.append(AuthoritySpan(_gateway(payload.get("authority_gateway_id")), expected, int(row["seq"])))
    result = [span.as_mapping() for span in history]
    validate_history(result, gateway_id=gateway_id, epoch=epoch)
    return result


# ── lineage ───────────────────────────────────────────────────────────────────


# Verifies the room log up to ``seq`` against the authority history; returns the spans and the history digest.
def source_prefix_locked(conn, room_id, authority, seq):
    history = read_history_locked(conn, room_id, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
    if history is None:
        raise WorkRecordPrefixError("successor work evidence has no verified lineage")
    spans = validate_history(history, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
    count = 0
    for event in conn.execute("SELECT seq,authority_epoch,actor_json,kind FROM hosted_room_events WHERE room_id=? AND seq<=? ORDER BY seq", (room_id, seq)):
        count += 1
        span = at_sequence(spans, count)
        actor = json.loads(event["actor_json"])
        boundary = span.epoch > 1 and count == span.from_seq
        if (event["seq"] != count or event["authority_epoch"] != span.epoch
                or (actor.get("kind") == "gateway" and actor.get("id") != span.gateway_id)
                or (event["kind"] in TRANSFER_KINDS) != boundary):
            raise WorkRecordPrefixError("work record history prefix is not verified")
    if count != seq:
        raise WorkRecordPrefixError("work record history prefix is unavailable")
    return spans, digest(history)


# ── capture ───────────────────────────────────────────────────────────────────


# Refuses a write that would push the work-record tables past their byte or row budget.
def _budget(conn, table, proposed):
    total_sql, count_sql = usage_sql((SOURCE_TABLE, PENDING_TABLE, INVALID_TABLE))
    total, count = conn.execute(f"SELECT {total_sql}, {count_sql}").fetchone()
    keys = ["room_id", "producer_gateway_id", "producer_epoch"]
    if table == PENDING_TABLE:
        keys.append("target_install_id")
    where = " AND ".join(f"{k}=?" for k in keys)
    old = conn.execute(f"SELECT {row_size_sql(table)} FROM {table} WHERE {where}",
                       [proposed[k] for k in keys]).fetchone()
    size = conn.execute(f"SELECT {row_size_sql(table)} FROM (SELECT "
                        + ",".join(f"? AS {k}" for k in proposed) + ")", tuple(proposed.values())).fetchone()[0]
    if total - (old[0] if old else 0) + size > MAX_STORE_BYTES or count - bool(old) + 1 > MAX_STORE_ROWS:
        raise WorkRecordCapacityError("work record storage is full")


# The digest of the room event at ``seq``, or of an empty history when ``seq`` is 0.
def history_anchor(conn, table: str, room_id: str, seq: int) -> str:
    if seq == 0:
        return digest([])
    row = conn.execute(f"""SELECT seq,event_id,kind,actor_json,authority_epoch,payload_json,created_at
        FROM {table} WHERE room_id=? AND seq=?""", (room_id, seq)).fetchone()
    if row is None:
        raise WorkRecordPrefixError("work record history prefix is unavailable")
    return digest([row["seq"], row["event_id"], row["kind"], json.loads(row["actor_json"]),
                   row["authority_epoch"], json.loads(row["payload_json"]), float(row["created_at"])])


# The room's outstanding tasks and receipts within their bounds, or empty lists and why they are unavailable.
def _capture_tasks(conn, room_id):
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
            payload, encoded, payload_digest = normalize_task_payload(json.loads(row["payload_json"]))
        except (ValueError, DriverStateError):
            return [], [], "unsupported_task"
        if payload_digest != row["payload_digest"] or encoded != row["payload_json"] or payload["source_event_seq"] != row["source_event_seq"]:
            return [], [], "unsupported_task"
        tasks.append({**{key: row[key] for key in _TASK_FIELDS - {"payload_sha256", "phase", "member_id", "profile"}},
                      "payload_sha256": payload_digest, "phase": row["status"], "profile": payload["target_profile"],
                      "member_id": payload.get("target_member_id", payload["target_profile"])})
    return tasks, receipts, None


def capture_transition_locked(conn: sqlite3.Connection, room_id: str) -> None:
    """Observe committed driver facts, without making evidence an execution gate.

    The caller's transaction owns both facts and evidence. A rejected evidence
    write rolls back only its savepoint; malformed retained bytes are classified
    in place, never replaced with a fabricated current snapshot.
    """
    owner = conn.execute(
        "SELECT authority_gateway_id FROM hosted_rooms WHERE room_id=? AND disbanded_at IS NULL",
        (room_id,)).fetchone()
    if owner is None:
        return
    conn.execute("SAVEPOINT work_transition_capture")
    try:
        capture_locked(conn, room_id=room_id, local_gateway_id=owner[0])
    except InvalidStoredWorkRecord:
        logger.warning("Hosted work evidence invalid for room %s", room_id)
    except (WorkRecordError, sqlite3.Error):
        conn.execute("ROLLBACK TO work_transition_capture")
        logger.warning("Hosted work evidence unavailable for room %s", room_id, exc_info=True)
    finally:
        conn.execute("RELEASE work_transition_capture")


def capture_locked(conn, *, room_id, local_gateway_id, through_seq=None):
    """Commit a new revision only when one consistent source view changes."""
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
        spans, content["lineage_sha256"] = source_prefix_locked(conn, room_id, content["authority"], seq)
        content["incompleteness"] = ["prior_authority_work_unknown"]
    previous = conn.execute(f"SELECT * FROM {SOURCE_TABLE} WHERE room_id=? AND producer_gateway_id=? AND producer_epoch=?",
                            scope(content)).fetchone()
    previous_record = validate_stored_locked(conn, SOURCE_TABLE, previous) if previous is not None else None
    revision = previous_record["revision"] + 1 if previous_record else 1
    record = {**content, "revision": revision, "digest": digest(content)}
    if len(encode(record).encode("utf-8")) > MAX_BYTES:
        content.update(tasks=[], receipts=[], availability="unavailable", reason="bounds_exceeded")
        record = {**content, "revision": revision, "digest": digest(content)}
    try:
        validate(record)
        _validate_roster(record, json.loads(room["members_json"]))
        if spans is not None:
            validate_provenance(record, spans)
    except WorkRecordError:
        if reason is not None:
            raise
        content.update(tasks=[], receipts=[], availability="unavailable", reason="unsupported_task")
        record = validate({**content, "revision": revision, "digest": digest(content)})
    if previous_record is not None and previous_record["digest"] == record["digest"]:
        return previous_record
    save_locked(conn, SOURCE_TABLE, record)
    return record
