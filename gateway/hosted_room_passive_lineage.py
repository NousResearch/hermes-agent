"""Strict passive wire lineage over the shared authored authority-history helper.

A pinned descriptor authorizes a copy source; only contiguous canonical claims
verify its transitions. Neither fact enables execution or proves exclusivity.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gateway.hosted_room_authority_history import (
    AuthorityHistoryError, AuthoritySpan, at_sequence, read_history_locked, validate_history,
)
from gateway.hosted_rooms import HostedRoomError
from gateway.hosted_rooms_common import table_exists

MAX_SPANS = 1023
MAX_DESCRIPTOR_BYTES = 8 * 1024
MAX_STORED_DESCRIPTOR_BYTES = 4 * 1024 * 1024
MAX_STORED_DESCRIPTORS = 512
ENROLLMENTS = "hosted_room_replica_retirement_enrollments"
HOME = "hosted_room_replica_retirement_home"


class PassiveLineageError(HostedRoomError):
    """The copy source or retained prefix cannot prove the requested lineage."""

    reason = "replica_lineage_unverified"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def descriptor(value: Any, *, gateway_id: str, epoch: int) -> tuple[tuple[AuthoritySpan, ...], str, str]:
    try:
        if not isinstance(value, list) or not 1 <= len(value) <= MAX_SPANS:
            raise ValueError("authority history is absent or exceeds its bound")
        encoded = canonical(value)
        if len(encoded.encode("utf-8")) > MAX_DESCRIPTOR_BYTES:
            raise ValueError("authority history exceeds its byte bound")
        spans = validate_history(value, gateway_id=gateway_id, epoch=epoch)
    except (ValueError, TypeError, RecursionError) as exc:
        raise PassiveLineageError(str(exc)) from exc
    return spans, encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def source_locked(conn, room_id: str, authority: dict) -> tuple[list[dict], str]:
    try:
        history = read_history_locked(conn, room_id, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
        if history is None:
            if authority["epoch"] != 1:
                raise AuthorityHistoryError("legacy adoption is not passive lineage")
            history = [AuthoritySpan(authority["gateway_id"], 1, 0).as_mapping()]
        _, _, digest = descriptor(history, gateway_id=authority["gateway_id"], epoch=authority["epoch"])
    except (AuthorityHistoryError, ValueError, TypeError) as exc:
        raise PassiveLineageError(str(exc)) from exc
    return history, digest


def is_v2(value) -> bool:
    return "version" in value.keys() and type(value["version"]) is int and value["version"] == 2


def enrolled_history(row) -> tuple[AuthoritySpan, ...]:
    if not is_v2(row):
        raise PassiveLineageError("v2 owner enrollment is required")
    try:
        value = json.loads(row["authority_history_json"])
        spans, encoded, digest = descriptor(value, gateway_id=row["authority_gateway_id"], epoch=row["authority_epoch"])
        if encoded != row["authority_history_json"] or digest != row["lineage_sha256"]:
            raise ValueError("enrolled lineage digest differs")
    except (TypeError, ValueError, KeyError, IndexError) as exc:
        raise PassiveLineageError("enrolled lineage metadata is invalid") from exc
    return spans


def current_locked(conn, room_id: str):
    if not table_exists(conn, ENROLLMENTS):
        return None
    return conn.execute(f"SELECT * FROM {ENROLLMENTS} WHERE room_id=? AND is_current=1", (room_id,)).fetchone()


def event_span(spans, event):
    """Validate mandatory transition facts without discarding canonical metadata."""
    seq = event["seq"]
    if type(seq) is not int or not 1 <= seq < 2**63:
        raise PassiveLineageError("invalid lineage event sequence")
    span = at_sequence(spans, seq)
    actor = json.loads(event["actor_json"]) if "actor_json" in event.keys() else event["actor"]
    payload = json.loads(event["payload_json"]) if "payload_json" in event.keys() else event["payload"]
    if type(event["authority_epoch"]) is not int or event["authority_epoch"] != span.epoch:
        raise PassiveLineageError("event is outside its authority span")
    if actor.get("kind") == "gateway" and actor.get("id") != span.gateway_id:
        raise PassiveLineageError("gateway actor differs from historical authority")
    boundary = span.epoch > 1 and seq == span.from_seq
    if (event["kind"] == "authority.claimed") != boundary:
        raise PassiveLineageError("missing or extra authority claim")
    if boundary:
        previous = spans[span.epoch - 2]
        if (actor != {"kind": "system", "id": "authority-control"}
                or not isinstance(payload, dict)
                or payload.get("previous_gateway_id") != previous.gateway_id
                or payload.get("authority_gateway_id") != span.gateway_id
                or type(payload.get("authority_epoch")) is not int
                or payload["authority_epoch"] != span.epoch):
            raise PassiveLineageError("authority claim differs from its descriptor")
    return span


def replica_history_locked(conn, row):
    """Separate current sender scope from the header of the verified prefix."""
    enrolled = current_locked(conn, row["room_id"])
    spans = enrolled_history(enrolled) if enrolled is not None else None
    if (spans is None or row["lineage_sha256"] != enrolled["lineage_sha256"]
            or row["replica_version"] != 2):
        raise PassiveLineageError("replica is missing its pinned lineage")
    head = at_sequence(spans, row["last_seq"])
    if (row["authority_gateway_id"], row["authority_epoch"]) != (head.gateway_id, head.epoch):
        raise PassiveLineageError("replica prefix header differs from retained claims")
    return spans


def state_fields_locked(conn, row) -> dict:
    if "replica_version" not in row.keys() or row["replica_version"] is None:
        return {}
    # Matching descriptor/header metadata cannot overrule a failed prefix audit.
    if row["quarantine_reason"] is not None:
        return {}
    try:
        spans = replica_history_locked(conn, row)
    except PassiveLineageError:
        return {}  # Existing quarantine, never a fabricated verified descriptor.
    return {
        "replica_version": 2, "lineage_sha256": row["lineage_sha256"],
        "authority_history": [s.as_mapping() for s in spans],
        "source_authority": {"gateway_id": spans[-1].gateway_id, "epoch": spans[-1].epoch},
        "lineage_status": status(spans, row["last_seq"]),
    }


def status(spans, last_seq):
    return "verified" if last_seq >= spans[-1].from_seq else "pending"


def compatible_extension(conn, room_id, spans, previous=None, replica=None):
    """Refuse to cut a known old tail off when an owner replaces copy scope."""
    old = None
    if previous is not None:
        old = (enrolled_history(previous) if is_v2(previous) else
               (AuthoritySpan(previous["authority_gateway_id"], previous["authority_epoch"], 0),))
    elif replica is not None:
        old = (replica_history_locked(conn, replica) if replica["replica_version"] == 2 else
               (AuthoritySpan(replica["authority_gateway_id"], replica["authority_epoch"], 0),))
    if old is not None:
        if len(spans) < len(old) or tuple(spans[:len(old)]) != tuple(old):
            raise PassiveLineageError("enrollment is not a descriptor extension")
        if replica is not None and len(spans) > len(old) and spans[len(old)].from_seq <= replica["latest_seq"]:
            raise PassiveLineageError("descriptor extension discards a known old tail")
    if replica is not None:
        for event in conn.execute("SELECT * FROM hosted_room_replica_events WHERE room_id=? ORDER BY seq", (room_id,)):
            event_span(spans, event)


def ensure_descriptor_capacity(conn, encoded):
    count, size = 0, 0
    for table in (HOME, ENROLLMENTS):
        row = conn.execute(f"SELECT COUNT(*),COALESCE(SUM(LENGTH(CAST(authority_history_json AS BLOB))),0) FROM {table} WHERE authority_history_json IS NOT NULL").fetchone()
        count += row[0]
        size += row[1]
    if count >= MAX_STORED_DESCRIPTORS or size + len(encoded.encode("utf-8")) > MAX_STORED_DESCRIPTOR_BYTES:
        from gateway.hosted_room_replica_retirement import RetirementCapacityError
        raise RetirementCapacityError("retained lineage descriptor capacity exhausted")


def initialize(conn):
    """Additive metadata; v1 remains NULL and byte-for-byte unversioned."""
    for table in (HOME, ENROLLMENTS):
        columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in (("version", "INTEGER"), ("lineage_sha256", "TEXT"), ("authority_history_json", "TEXT")):
            if name not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
    # Old initializers may add their own guards, but cannot drop these v2 guards.
    total_size = " + ".join(f"(SELECT COALESCE(SUM(LENGTH(CAST(authority_history_json AS BLOB))),0) FROM {t})" for t in (HOME, ENROLLMENTS))
    total_count = " + ".join(f"(SELECT COUNT(*) FROM {t} WHERE authority_history_json IS NOT NULL)" for t in (HOME, ENROLLMENTS))
    for table in (HOME, ENROLLMENTS):
        for operation in ("INSERT", "UPDATE"):
            old_size = "0" if operation == "INSERT" else "COALESCE(LENGTH(CAST(OLD.authority_history_json AS BLOB)),0)"
            old_count = "0" if operation == "INSERT" else "(OLD.authority_history_json IS NOT NULL)"
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_lineage_{operation.lower()}_v2
                BEFORE {operation} ON {table} WHEN NEW.authority_history_json IS NOT NULL AND (
                    {total_size} - {old_size} + LENGTH(CAST(NEW.authority_history_json AS BLOB)) > {MAX_STORED_DESCRIPTOR_BYTES}
                    OR {total_count} - {old_count} + 1 > {MAX_STORED_DESCRIPTORS}
                    OR LENGTH(CAST(NEW.authority_history_json AS BLOB)) > {MAX_DESCRIPTOR_BYTES})
                BEGIN SELECT RAISE(ABORT, 'lineage descriptor capacity exhausted'); END""")
