"""Immutable, identity-bound membership for interrupted session adoption."""

import hashlib
import json
import math
import time
from pathlib import Path

from hermes_state_lineage import SessionLineageError

FORMAT = 2


class AdoptionConflict(SessionLineageError):
    """Saved transfer evidence cannot safely authorize this adoption."""

    code = "session_adoption_conflict"


def validate_handles(*handles):
    for handle in handles:
        if handle._db_replaced or handle._db_file_was_replaced():
            raise AdoptionConflict("Adoption store was replaced; reopen and verify its identity before recovery")


def store_identity(db):
    """Bind the generation actually opened, including in-place file replacement."""
    validate_handles(db)
    physical = db._db_file_identity
    generation = int(db._db_file_application_id or 0)
    if physical is None and not generation:
        raise AdoptionConflict("Cannot establish the adoption store's file generation; verify it before recovery")
    return [str(db.db_path.resolve(strict=True)), list(physical) if physical else None, generation]


def _table_exists(conn):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='adoption_transfers'"
    ).fetchone() is not None


def _check_schema(conn):
    columns = {row[1] for row in conn.execute("PRAGMA table_info(adoption_transfers)")}
    required = {"transfer_id", "member_id", "member_order", "donor_store", "target_store",
                "status", "format_version", "manifest_json", "created_at", "updated_at"}
    if not required.issubset(columns):
        raise AdoptionConflict("Unsupported adoption record schema; inspect the saved transfer before recovery")


def _valid_state(value):
    if not isinstance(value, list) or len(value) != 3:
        return False
    parent, ended, reason = value
    return ((parent is None or isinstance(parent, str))
            and (ended is None or type(ended) in (int, float) and math.isfinite(ended))
            and (reason is None or isinstance(reason, str)))


def _validate_member(member):
    if not isinstance(member, dict) or not isinstance(member.get("id"), str) or not member["id"]:
        raise AdoptionConflict("Invalid adoption member identity")
    if not _valid_state(member.get("source")) or not _valid_state(member.get("target")):
        raise AdoptionConflict("Invalid saved adoption lifecycle evidence")
    witness = member.get("target_active_digest")
    source_count = member.get("source_count")
    if ("target_active_digest" in member or "source_count" in member) and (
            type(source_count) is not int or source_count < 0
            or not isinstance(witness, str) or len(witness) != 64
            or any(ch not in "0123456789abcdef" for ch in witness)):
        raise AdoptionConflict("Invalid saved adoption activity witness")
    count, digest = member.get("target_count"), member.get("target_digest")
    if type(count) is not int or count < 0 or (source_count is not None and source_count > count):
        raise AdoptionConflict("Invalid saved adoption transcript length")
    if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise AdoptionConflict("Invalid saved adoption transcript fingerprint")


def _read_manifest(first):
    if first[6] != FORMAT:
        raise AdoptionConflict("Unsupported adoption record version; inspect the saved transfer before recovery")
    try:
        manifest = json.loads(first[7])
    except (TypeError, ValueError) as exc:
        raise AdoptionConflict("Invalid saved adoption manifest; inspect the transfer before recovery") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
        raise AdoptionConflict("Unsupported saved adoption manifest")
    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise AdoptionConflict("Incomplete adoption membership; inspect the saved transfer")
    for member in members:
        _validate_member(member)
    ids = [member["id"] for member in members]
    if len(set(ids)) != len(ids):
        raise AdoptionConflict("Invalid adoption member identities; inspect the saved transfer")
    return manifest


def _decode_record(rows, donor_db, target_db):
    if not rows:
        raise AdoptionConflict("Adoption record disappeared while reading its membership")
    first = rows[0]
    manifest = _read_manifest(first)
    if manifest.get("donor") != store_identity(donor_db) or manifest.get("target") != store_identity(target_db):
        raise AdoptionConflict("Adoption record belongs to a different donor or target store; verify the requested pair")
    ids = [member["id"] for member in manifest["members"]]
    if len(rows) != len(ids):
        raise AdoptionConflict("Adoption record lost members; recover the complete saved transfer")
    expected_id = transfer_id(manifest)
    for order, row in enumerate(rows):
        expected = (expected_id, ids[order], order, manifest["donor"][0], manifest["target"][0],
                    first[5], FORMAT, first[7] if order == 0 else None)
        if tuple(row) != expected:
            raise AdoptionConflict("Inconsistent adoption record; inspect the complete saved transfer")
    if first[5] not in {"copied", "pending", "complete"}:
        raise AdoptionConflict("Invalid adoption progress state")
    return {"transfer_id": expected_id, "member_ids": ids, "manifest": manifest, "status": first[5]}


def find_in(conn, session_id, donor_db, target_db):
    """Absence is distinct from a failed read or an incompatible saved operation."""
    if not _table_exists(conn):
        return None
    _check_schema(conn)
    target_store = str(target_db.db_path.resolve(strict=True))
    candidates = conn.execute(
        "SELECT DISTINCT transfer_id FROM adoption_transfers WHERE member_id=? AND target_store=?",
        (session_id, target_store),
    ).fetchall()
    if not candidates:
        return None
    if len(candidates) != 1:
        raise AdoptionConflict("Several adoption records claim this member; choose the intended transfer explicitly")
    rows = conn.execute(
        "SELECT transfer_id, member_id, member_order, donor_store, target_store, status,"
        " format_version, manifest_json FROM adoption_transfers WHERE transfer_id=? ORDER BY member_order",
        (candidates[0][0],),
    ).fetchall()
    return _decode_record(rows, donor_db, target_db)


def find(db, session_id, donor_db):
    with db._read_ctx() as conn:
        return find_in(conn, session_id, donor_db, db)


def transfer_id(manifest):
    identity = [FORMAT, manifest["donor"], manifest["target"], manifest["members"][0]["id"]]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def _history(conn, session_id, count=None):
    # Stable transcript fields exclude generated row IDs and mutable display flags.
    sql = "SELECT role, content, tool_call_id, tool_calls, tool_name FROM messages WHERE session_id=? ORDER BY id"
    params = [session_id]
    if count is not None:
        sql += " LIMIT ?"
        params.append(count)
    rows = [list(row) for row in conn.execute(sql, params)]
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()
    return len(rows), digest


def activity_digest(conn, session_id, count):
    rows = [row[0] for row in conn.execute(
        "SELECT active FROM messages WHERE session_id=? ORDER BY id LIMIT ?", (session_id, count))]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _lifecycle(row):
    return [row.get("parent_session_id"), row.get("ended_at"), row.get("end_reason")]


def build_manifest(db, donor_db, segments, conn):
    members = []
    for segment in segments:
        mid = segment["id"]
        target = conn.execute(
            "SELECT parent_session_id, ended_at, end_reason FROM sessions WHERE id=?", (mid,),
        ).fetchone()
        if target is None:
            raise AdoptionConflict("Imported target member is missing; adoption was not committed")
        count, digest = _history(conn, mid)
        with donor_db._read_ctx() as donor_conn:
            source_count = donor_conn.execute(
                "SELECT count(*) FROM messages WHERE session_id=?", (mid,)).fetchone()[0]
        members.append({"id": mid, "source": _lifecycle(segment), "target": list(target),
                        "target_count": count, "target_digest": digest, "source_count": source_count,
                        "target_active_digest": activity_digest(conn, mid, source_count)})
    return {"format": FORMAT, "donor": store_identity(donor_db),
            "target": store_identity(db), "members": members}


def ensure_schema(conn):
    """Create the optional journal for adoption or a recovery destination."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS adoption_transfers ("
        " transfer_id TEXT NOT NULL, member_id TEXT NOT NULL, member_order INTEGER NOT NULL,"
        " donor_store TEXT NOT NULL, target_store TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',"
        " format_version INTEGER NOT NULL, manifest_json TEXT,"
        " created_at REAL NOT NULL, updated_at REAL NOT NULL, PRIMARY KEY (transfer_id, member_id))"
    )
    _check_schema(conn)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_adoption_transfers_member ON adoption_transfers(member_id, target_store)")


def insert(conn, manifest, *, retire_donor=True):
    """Membership is immutable; transaction callers verify an existing operation first."""
    ensure_schema(conn)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    identity = transfer_id(manifest)
    now = time.time()
    status = "pending" if retire_donor else "copied"
    conn.executemany(
        "INSERT INTO adoption_transfers (transfer_id,member_id,member_order,donor_store,target_store,"
        "status,format_version,manifest_json,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(identity, member["id"], order, manifest["donor"][0], manifest["target"][0], status, FORMAT,
          encoded if order == 0 else None, now, now)
         for order, member in enumerate(manifest["members"])],
    )
    return identity


def validate_members(db, donor_db, record):
    """Validate copies before granting a record's additional ended-pair permission."""
    for member in record["manifest"]["members"]:
        source = donor_db.get_session(member["id"])
        target = db.get_session(member["id"])
        if source is None or target is None:
            raise AdoptionConflict("Saved adoption member is missing from a store; recover the complete transfer")
        source_state = _lifecycle(source)
        expected = member["source"]
        retired = source_state[1] is not None and source_state[2] == "adopted_by_profile"
        if source_state[0] != expected[0] or (not retired and source_state != expected):
            raise AdoptionConflict("Donor member lifecycle differs from the saved transfer")
        target_state = _lifecycle(target)
        if target_state[0] != member["target"][0]:
            raise AdoptionConflict("Target member lineage differs from the saved transfer")
        if record["status"] != "complete" and target_state != member["target"]:
            raise AdoptionConflict("Target member lifecycle differs from the pending transfer")
        with db._read_ctx() as conn:
            history = _history(conn, member["id"], member["target_count"])
            if ("target_active_digest" in member and activity_digest(
                    conn, member["id"], member["source_count"]) != member["target_active_digest"]):
                raise AdoptionConflict("Saved adoption activity witness differs; inspect history before recovery")
        if history != (member["target_count"], member["target_digest"]):
            raise AdoptionConflict("Target history differs from the saved transfer; inspect it before retiring the donor")


def ownership(record, donor_path, target_path):
    from hermes_state_lineage import AdoptionOwnership
    states = {member["id"]: (member["source"][1] is None, member["source"][2],
                              member["target"][1] is None, member["target"][2])
              for member in record["manifest"]["members"]}
    return AdoptionOwnership(Path(donor_path), Path(target_path), states)


def validate_fresh(db, segments):
    for segment in segments:
        if segment.get("end_reason") == "adopted_by_profile":
            raise AdoptionConflict("Retired donor has no complete adoption record; explicit recovery membership is required")
        target = db.get_session(segment["id"])
        if target is not None and (segment.get("ended_at") is not None or target.get("ended_at") is not None):
            raise AdoptionConflict("Ended target copy has no matching adoption record; verify its lineage before recovery")


def mark_complete(db, record, donor_db):
    def write(conn):
        current = find_in(conn, record["member_ids"][0], donor_db, db)
        if current is None or current["manifest"] != record["manifest"]:
            raise AdoptionConflict("Adoption membership changed before completion")
        conn.execute("UPDATE adoption_transfers SET status='complete', updated_at=? WHERE transfer_id=?",
                     (time.time(), record["transfer_id"]))
    db._execute_write(write)


def set_retirement_intent(conn, record, requested):
    if record["status"] != "complete":
        conn.execute("UPDATE adoption_transfers SET status=?, updated_at=? WHERE transfer_id=?",
                     ("pending" if requested else "copied", time.time(), record["transfer_id"]))


def _pending_component_records(conn, session_id, db):
    """Reject damaged pending membership connected to the requested target row."""
    if not _table_exists(conn):
        return []
    _check_schema(conn)
    members = [row[0] for row in conn.execute(
        "WITH RECURSIVE component(id) AS (SELECT ? UNION "
        "SELECT s.parent_session_id FROM sessions s JOIN component c ON s.id=c.id "
        "WHERE s.parent_session_id IS NOT NULL UNION "
        "SELECT s.id FROM sessions s JOIN component c ON s.parent_session_id=c.id) "
        "SELECT id FROM component LIMIT 8193", (session_id,))]
    if len(members) > 8192:
        raise AdoptionConflict("Pending adoption component exceeds the verification bound")
    markers = ",".join("?" for _ in members)
    target_store = str(db.db_path.resolve(strict=True))
    transfers = conn.execute(
        f"SELECT DISTINCT transfer_id FROM adoption_transfers WHERE member_id IN ({markers}) "
        "AND status='pending' LIMIT 8193", members).fetchall()
    if len(transfers) > 8192:
        raise AdoptionConflict("Pending adoption inventory exceeds the verification bound")
    valid = []
    for (transfer,) in transfers:
        rows = conn.execute(
            "SELECT transfer_id,member_id,member_order,donor_store,target_store,status,"
            "format_version,manifest_json FROM adoption_transfers WHERE transfer_id=? ORDER BY member_order LIMIT 8193",
            (transfer,)).fetchall()
        if not rows:
            raise AdoptionConflict("Pending adoption record disappeared during verification")
        if len(rows) > 8192:
            raise AdoptionConflict("Pending adoption membership exceeds the verification bound")
        manifest = _read_manifest(rows[0])
        ids = [member["id"] for member in manifest["members"]]
        donor = manifest.get("donor")
        if (not isinstance(donor, list) or len(donor) != 3
                or not isinstance(donor[0], str) or not donor[0]
                or (donor[1] is not None and (not isinstance(donor[1], list) or len(donor[1]) != 2
                    or any(type(part) is not int for part in donor[1])))
                or type(donor[2]) is not int or donor[2] < 0
                or (donor[1] is None and donor[2] == 0)):
            raise AdoptionConflict("Invalid pending adoption donor identity; inspect the saved transfer")
        if manifest.get("target") != store_identity(db) or len(rows) != len(ids):
            raise AdoptionConflict("Incomplete pending adoption membership; inspect the saved transfer")
        if any(mid not in members for mid in ids):
            raise AdoptionConflict("Pending adoption member is missing from its target component")
        for order, row in enumerate(rows):
            expected = (transfer_id(manifest), ids[order], order, manifest["donor"][0], target_store,
                        "pending", FORMAT, rows[0][7] if order == 0 else None)
            if tuple(row) != expected:
                raise AdoptionConflict("Inconsistent pending adoption membership; inspect the saved transfer")
        valid.append((manifest, rows[0][3]))
    return valid


def retirement_pending_source(db, session_id):
    """Find a pending transfer after verifying its connected component's membership."""
    with db._read_ctx() as conn:
        records = _pending_component_records(conn, session_id, db)
        sources = [source for manifest, source in records
                   if session_id in {member["id"] for member in manifest["members"]}]
    if len(sources) > 1:
        raise AdoptionConflict("Several pending adoptions claim this session; inspect the complete transfer records")
    return sources[0] if sources else None
