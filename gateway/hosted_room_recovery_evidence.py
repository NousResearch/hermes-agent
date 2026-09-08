"""Room-local immutable recovery inventory; never an execution capability."""

import hashlib
import json

from gateway import hosted_room_work_records as work
from gateway import hosted_room_work_storage as storage
from gateway import hosted_room_passive_lineage as lineage
from gateway.hosted_room_replica_retirement import roster_digest

OBJECT = "hermes.group_recovery.evidence"
ENROLLMENT_FIELDS = ("enrollment_id", "room_id", "authority_gateway_id", "authority_epoch",
    "target_install_id", "roster_sha256", "version", "is_current", "state",
    "authority_history_json", "lineage_sha256")


def columns(table):
    if table == storage.INVALID_TABLE:
        return ("evidence_id", "source_table", "room_id", "revision", "digest", "record_json",
                "target_install_id", "route_generation", "status", "disposition")
    if table == work.TARGET_TABLE:
        return ("room_id", "revision", "digest", "record_json", "producer_gateway_id",
                "producer_epoch", "disposition")
    raise work.WorkRecordError("Unsupported recovery evidence table.")


def fingerprint(value):
    return hashlib.sha256(work.encode(value).encode("utf-8")).hexdigest()


def inventory_locked(conn, room_id):
    result = {}
    for table in (work.TARGET_TABLE, storage.INVALID_TABLE):
        expected = columns(table)
        actual = tuple(r["name"] for r in conn.execute(f"PRAGMA table_info({table})"))
        if not actual:
            result[table] = []
            continue
        if set(actual) != set(expected):
            raise work.WorkRecordError("Unsupported recovery evidence columns.")
        where = "room_id=?"
        args = [room_id]
        if table == storage.INVALID_TABLE:
            where += " AND source_table=?"
            args.append(work.TARGET_TABLE)
        rows = [dict(row) for row in conn.execute(f"SELECT * FROM {table} WHERE {where}", args)]
        for row in rows:
            # JSON preserves these SQLite values exactly. Other storage classes
            # require a future encoding, not coercion or silent column loss.
            if any(type(v) not in (str, int, type(None)) for v in row.values()):
                raise work.WorkRecordError("Unsupported recovery evidence storage type.")
        result[table] = sorted(rows, key=work.encode)
    return result


def selection_locked(conn, state, target):
    room_id = state["room_id"]
    rows = inventory_locked(conn, room_id)
    copy = conn.execute("SELECT * FROM hosted_room_replicas WHERE room_id=?", (room_id,)).fetchone()
    enrolled = lineage.current_locked(conn, room_id)
    enrollment = {k: enrolled[k] for k in ENROLLMENT_FIELDS} if enrolled is not None else None
    blockers = []
    if enrolled is not None:
        if (enrolled["state"] != "active" or enrolled["is_current"] != 1
                or enrolled["target_install_id"] != target
                or enrolled["roster_sha256"] != roster_digest(state["members"])):
            blockers.append("enrollment_unavailable")
    if copy["replica_version"] == 2 or (enrolled is not None and lineage.is_v2(enrolled)):
        if (state.get("lineage_status") != "verified" or enrolled is None
                or state["authority"] != state.get("source_authority")):
            blockers.append("lineage_unverified")
    elif state["authority"]["epoch"] != 1:
        blockers.append("lineage_unverified")
    current = (enrolled["authority_gateway_id"], enrolled["authority_epoch"]) if enrolled else (
        state["authority"]["gateway_id"], state["authority"]["epoch"])
    found = False
    for row in rows[work.TARGET_TABLE]:
        if (row["producer_gateway_id"], row["producer_epoch"]) != current:
            continue
        try:
            record = storage.validate_stored(row)
            if row["disposition"] != "current" or record["availability"] != "available":
                raise work.WorkRecordError("Current work unavailable")
            work._validate_roster(record, state["members"])
            if record["version"] == 2:
                from gateway.hosted_room_work_lineage import target_prefix_locked
                target_prefix_locked(conn, copy, record)
            elif current[1] != 1:
                raise work.WorkRecordError("Legacy successor evidence")
            prefix = record["history"]
            if prefix["seq"] > copy["last_seq"] or prefix["event_sha256"] != work.history_anchor(
                    conn, "hosted_room_replica_events", room_id, prefix["seq"]):
                raise work.WorkRecordError("Current work prefix conflicts")
            found = True
        except (work.WorkRecordError, ValueError, TypeError):
            pass
    if not found:
        blockers.append("work_records_unavailable")
    origin = None
    history = state.get("authority_history")
    if history:
        origin = history[0]["gateway_id"]
    elif state["authority"]["epoch"] == 1:
        origin = state["authority"]["gateway_id"]
    origins = []
    for member in state["members"]:
        destination = member.get("target") or {}
        peer = destination.get("kind") == "peer"
        origins.append({"member_id": member["member_id"],
            "installation_id": destination.get("installation_id") if peer else origin,
            "profile": destination.get("profile", member.get("profile")),
            "resolved": bool(destination.get("installation_id") if peer else origin)})
    return rows, enrollment, origins, blockers


def envelope(binding, rows):
    return {"object": OBJECT, "version": 2, "binding": binding, "rows": rows}


def validate_decision(row):
    """Read-only floor check, including surviving rows with lost evidence."""
    try:
        value = json.loads(row["work_record_json"])
        if value.get("object") != OBJECT:
            # Preserve unwrapped legacy bytes, but never grant them new
            # activation compatibility or let damaged v2 evidence downgrade.
            raise ValueError("Legacy recovery evidence requires separate reconciliation")
        if type(value.get("version")) is not int or value["version"] != 2:
            raise ValueError("unsupported envelope")
        binding, rows = value["binding"], value["rows"]
        if set(rows) != {work.TARGET_TABLE, storage.INVALID_TABLE} or not rows[work.TARGET_TABLE]:
            raise ValueError("missing inventory")
        for table, items in rows.items():
            if not isinstance(items, list) or items != sorted(items, key=work.encode):
                raise ValueError("unordered inventory")
            seen = set()
            for item in items:
                if set(item) != set(columns(table)) or item["room_id"] != row["room_id"]:
                    raise ValueError("wrong scope")
                if any(type(v) not in (str, int, type(None)) for v in item.values()):
                    raise ValueError("wrong storage type")
                key = item["evidence_id"] if table == storage.INVALID_TABLE else (item["producer_gateway_id"], item["producer_epoch"])
                if key in seen or (table == storage.INVALID_TABLE and item["source_table"] != work.TARGET_TABLE):
                    raise ValueError("duplicate or foreign inventory")
                seen.add(key)
        if (binding["inventory_sha256"] != fingerprint(rows) or fingerprint(binding) != row["snapshot_id"]
                or binding["room_id"] != row["room_id"] or binding["target_gateway_id"] != row["target_gateway_id"]
                or binding["saved_through_seq"] != row["history_seq"]
                or binding["source_authority"] != {"gateway_id": row["source_gateway_id"], "epoch": row["source_epoch"]}):
            raise ValueError("changed binding")
        return value
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        raise RuntimeError("The Group Chat recovery record evidence is missing or changed.") from exc


def exact_row_sql(table, value="item.value", prefix="OLD."):
    """SQL storage-class equality; no text/numeric or NULL coercion credit."""
    return " AND ".join(
        f"typeof({prefix}{k})=json_type({value},'$.{k}') AND "
        f"{prefix}{k} IS json_extract({value},'$.{k}')" for k in columns(table))


def transfer_permission_sql(conn, table):
    from gateway.hosted_rooms_common import table_exists
    from gateway.hosted_room_manual_promotion_schema import TABLE
    if not table_exists(conn, TABLE):
        return "0"
    scope = f" AND OLD.source_table='{work.TARGET_TABLE}'" if table == storage.INVALID_TABLE else ""
    return f"""EXISTS (SELECT 1 FROM {TABLE} decision,
        json_each(decision.work_record_json,'$.rows.{table}') item
        WHERE decision.room_id=OLD.room_id AND decision.status='transferring' {scope}
        AND json_extract(decision.work_record_json,'$.object')='{OBJECT}'
        AND json_extract(decision.work_record_json,'$.version')=2
        AND ({exact_row_sql(table)}))"""
