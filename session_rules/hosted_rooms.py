"""The pure hosted-room rules the store applies inside its hosted-room transactions, shared by the gateway validators and every backend.

The leaf validators and canonical JSON of the hosted-room modules, the room limits and event-kind tables, the actor and payload encoders and validators, the room settings record, the grant scope key, the replica limits, control-event codecs and page validator, the work-record validators and the authority-history rules. The error classes (``HostedRoomError``, ``ReplicaError``, ``ReplicaGapError``, ``AuthorityHistoryError``) are the persistence contract's. Standard library and ``persistence.gateway.hosted_rooms`` only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

from persistence.gateway.hosted_rooms import AuthorityHistoryError, HostedRoomError, ReplicaError, ReplicaGapError


IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


def identifier(
    value: Any, *, label: str, error: type[Exception], max_chars: int = 128,
    pattern: re.Pattern[str] | None = IDENTIFIER_RE, invalid: str | None = None) -> str:
    """Strip and validate a bounded string; ``pattern=None`` skips the shape check."""
    if not isinstance(value, str):
        raise error(f"{label} must be a string")
    value = value.strip()
    if not value or len(value) > max_chars or (pattern is not None and not pattern.fullmatch(value)):
        raise error(invalid or f"invalid {label}")
    return value


def bounded_int(value: Any, *, error: type[Exception], message: str, low: int = 0, high: int | None = None) -> int:
    """Reject bools, non-ints and values outside ``[low, high]`` (``message`` is the exact text)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < low or (high is not None and value > high):
        raise error(message)
    return value


def exact_fields(
    value: Any, *, label: str, required: frozenset[str] | set[str], optional: frozenset[str] | set[str] = frozenset(),
    error: type[Exception], not_object: str | None = None, missing_fmt: str = "{label} is missing fields: {fields}",
    unknown_fmt: str = "{label} has unknown fields: {fields}") -> Mapping[str, Any]:
    """Require exactly ``required`` (+ any ``optional``) keys; formats name the offenders sorted."""
    if not isinstance(value, Mapping):
        raise error(not_object or f"{label} must be an object")
    keys = frozenset(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise error(missing_fmt.format(label=label, fields=", ".join(sorted(missing))))
    if unknown:
        raise error(unknown_fmt.format(label=label, fields=", ".join(sorted(unknown))))
    return value


def text(value: Any, *, error: type[Exception], label: str, max_bytes: int, strip: bool = True) -> str:
    """Non-blank string bounded by ``max_bytes`` of UTF-8; ``strip=False`` keeps and measures the raw text."""
    if not isinstance(value, str):
        raise error(f"{label} must be a string")
    if not value.strip():
        raise error(f"{label} must not be empty")
    if strip:
        value = value.strip()
    if len(value.encode("utf-8")) > max_bytes:
        raise error(f"{label} is too large")
    return value


def compact_json(value: Any, *, ensure_ascii: bool = True) -> str:
    """Sorted-key, separator-free JSON (the digest/storage canonical form)."""
    return json.dumps(value, ensure_ascii=ensure_ascii, sort_keys=True, separators=(",", ":"))


def canonical_json(value: Any, *, error: type[Exception], label: str, max_bytes: int, ensure_ascii: bool) -> str:
    """``compact_json`` bounded by ``max_bytes`` of UTF-8; unserializable input raises ``error``."""
    try:
        encoded = compact_json(value, ensure_ascii=ensure_ascii)
    except (TypeError, ValueError, RecursionError) as exc:
        raise error(f"{label} must be JSON-serializable") from exc
    if len(encoded.encode("utf-8")) > max_bytes:
        raise error(f"{label} is too large")
    return encoded


def utf8_len(*parts: str) -> int:
    return len("".join(parts).encode("utf-8"))


def clock(now: float | None) -> float:
    """``now`` as a float, or the current wall clock when ``None``."""
    return time.time() if now is None else float(now)


PROTOCOL_VERSION = 2
MAX_ROOM_ID_CHARS = 128
MAX_EVENT_ID_CHARS = 128
MAX_ROOM_NAME_CHARS = 200
MAX_EVENT_KIND_CHARS = 64
MAX_ACTOR_ID_CHARS = 128
MAX_ACTOR_LABEL_CHARS = 200
MAX_MEMBERS = 128
MAX_MEMBERS_JSON_BYTES = 128 * 1024
MAX_EVENT_JSON_BYTES = 256 * 1024
MAX_LOG_LIMIT = 500
MAX_LOG_PAGE_BYTES = 2 * 1024 * 1024
MAX_ROOM_LIST_LIMIT = 500
MAX_ACTIVE_ROOMS = 256
MAX_DISBANDED_ROOM_TOMBSTONES = 512
DISBANDED_ROOM_RETENTION_SECONDS = 90 * 24 * 60 * 60
MAX_EVENTS_PER_ROOM = 50_000
MAX_ROOM_EVENT_BYTES = 256 * 1024 * 1024
# Leave substantial headroom below the pre-update state.db snapshot ceiling: event accounting excludes
# SQLite indexes and repeated room ids, so the logical budget must stay well below the physical-file limit.
MAX_GATEWAY_EVENT_BYTES = 16 * 1024 * 1024
CONTROL_EVENT_COUNT_RESERVE = 64
CONTROL_EVENT_BYTE_RESERVE = 1024 * 1024
_JOURNAL_MODE_LOCK_RETRIES = 8

_EVENT_KIND_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_CONTROL_EVENT_KINDS = frozenset({"authority.claimed", "authority.lost", "room.disbanded", "room.stop_requested"})
_EVENT_KINDS_BY_ACTOR = {
    "user": frozenset({"message.user"}), "member": frozenset({"message.member"}),
    "gateway": frozenset({
        "member.unavailable", "room.activity", "room.assignment", "room.stop_requested", "turn.deferred", "turn.reassigned",
        "turn.cancelled", "turn.failed", "turn.settled", "turn.started"}),
    "system": frozenset({
        "authority.claimed", "authority.lost", "room.context_set", "room.created", "room.disbanded",
        "room.members_changed", "room.renamed"})}
_OPTIONAL_ACTOR_FIELDS = (
    ("display_name", MAX_ACTOR_LABEL_CHARS), ("profile", MAX_ACTOR_ID_CHARS), ("connection_id", MAX_ACTOR_ID_CHARS))
_ACTOR_FIELDS = frozenset({"kind", "id", *(field for field, _ in _OPTIONAL_ACTOR_FIELDS)})


_canonical_json = partial(canonical_json, error=HostedRoomError, ensure_ascii=False)
_validate_identifier = partial(identifier, error=HostedRoomError)
_room_id = partial(_validate_identifier, label="room_id", max_chars=MAX_ROOM_ID_CHARS)
_event_id = partial(_validate_identifier, label="event_id", max_chars=MAX_EVENT_ID_CHARS)
_actor_json = partial(_canonical_json, label="actor", max_bytes=4 * 1024)
_payload_json = partial(_canonical_json, label="payload", max_bytes=MAX_EVENT_JSON_BYTES)
_bounded_int = partial(bounded_int, error=HostedRoomError)
_validate_room_name = partial(
    _validate_identifier, label="name", max_chars=MAX_ROOM_NAME_CHARS, pattern=None, invalid="invalid room name")
_validate_event_kind = partial(
    _validate_identifier, label="kind", max_chars=MAX_EVENT_KIND_CHARS, pattern=_EVENT_KIND_RE,
    invalid="invalid event kind")


def _actor_id(value: Any, label: str) -> str:
    return _validate_identifier(value, label=label, max_chars=MAX_ACTOR_ID_CHARS)


def _require_positive_int(value: Any, label: str) -> int:
    return _bounded_int(value, message=f"{label} must be a positive integer", low=1)


def _bounded_limit(value: Any, maximum: int) -> int:
    return _bounded_int(value, message=f"limit must be between 1 and {maximum}", low=1, high=maximum)


def _non_negative(value: Any, label: str) -> int:
    return _bounded_int(value, message=f"{label} must be a non-negative integer")


def _system_actor_json(actor_id: str) -> str:
    return _actor_json({"kind": "system", "id": actor_id})


def _claim_payload_json(previous_gateway_id: str, new_gateway_id: str, epoch: int) -> str:
    return _payload_json(
        {"previous_gateway_id": previous_gateway_id, "authority_gateway_id": new_gateway_id, "authority_epoch": epoch})


def user_event_id(client_event_id: Any) -> str:
    """Map a client retry key into the server-owned user-event namespace."""
    return f"user:{hashlib.sha256(_event_id(client_event_id).encode('utf-8')).hexdigest()}"


def _validate_members(value: Any) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(value, list):
        raise HostedRoomError("members must be a list")
    if len(value) > MAX_MEMBERS:
        raise HostedRoomError("too many room members")
    if not all(isinstance(member, dict) for member in value):
        raise HostedRoomError("each room member must be an object")
    members = [dict(member) for member in value]
    return members, _canonical_json(members, label="members", max_bytes=MAX_MEMBERS_JSON_BYTES)


def _legacy_members_match(existing_json: str, proposed: list[dict[str, Any]]) -> bool:
    """Allow adoption to add routing metadata an older room could not store."""
    try:
        existing = json.loads(existing_json)
    except (TypeError, ValueError):
        return False
    if not isinstance(existing, list) or len(existing) != len(proposed):
        return False
    for previous, current in zip(existing, proposed, strict=True):
        if not isinstance(previous, dict):
            return False
        previous, current = dict(previous), dict(current)
        previous_target, current_target = previous.pop("target", None), current.pop("target", None)
        if previous != current or (previous_target not in (None, {}) and previous_target != current_target):
            return False
    return True


def _validate_actor(value: Any, *, kind: str) -> tuple[dict[str, str], str]:
    if not isinstance(value, dict):
        raise HostedRoomError("actor must be an object")
    unknown = set(value) - _ACTOR_FIELDS
    if unknown:
        raise HostedRoomError(f"unknown actor fields: {', '.join(sorted(unknown))}")
    actor_kind = value.get("kind")
    if not isinstance(actor_kind, str) or actor_kind not in _EVENT_KINDS_BY_ACTOR:
        raise HostedRoomError("invalid actor.kind")
    if kind not in _EVENT_KINDS_BY_ACTOR[actor_kind]:
        raise HostedRoomError(f"actor kind '{actor_kind}' cannot append '{kind}'")
    actor = {"kind": actor_kind, "id": _actor_id(value.get("id"), "actor.id")}
    for field, max_chars in _OPTIONAL_ACTOR_FIELDS:
        field_value = value.get(field)
        if field_value is None:
            continue
        if not isinstance(field_value, str):
            raise HostedRoomError(f"actor.{field} must be a string")
        if len(field_value := field_value.strip()) > max_chars:
            raise HostedRoomError(f"actor.{field} is too long")
        if field_value:
            actor[field] = field_value
    return actor, _actor_json(actor)


# Room settings: purpose, shared notes and routing, one ``room.context_set`` event by the room-control actor.
CONTEXT_EVENT_KIND = "room.context_set"
_CONTEXT_ACTOR = {"kind": "system", "id": "room-control"}
MAX_PURPOSE_CHARS = 500
MAX_NOTES_BYTES = 4 * 1024
ROUTING_MODES = ("everyone", "lead")
_EMPTY_CONTEXT = {"purpose": "", "notes": "", "routing": "everyone", "lead_member_id": ""}


def validate_context(purpose: Any, notes: Any, routing: Any = "everyone", lead_member_id: Any = "") -> dict[str, str]:
    """Normalize one room settings record; purpose and notes may be empty (clears them). ``lead`` routing
    needs a lead; ``everyone`` routing never keeps one."""
    if not all(isinstance(value, str) for value in (purpose, notes, routing, lead_member_id)):
        raise HostedRoomError("purpose, notes, routing and lead_member_id must be strings")
    purpose = " ".join(purpose.split())
    notes = notes.strip()
    if len(purpose) > MAX_PURPOSE_CHARS:
        raise HostedRoomError(f"purpose must be at most {MAX_PURPOSE_CHARS} characters")
    if len(notes.encode("utf-8")) > MAX_NOTES_BYTES:
        raise HostedRoomError(f"notes must be at most {MAX_NOTES_BYTES} bytes")
    if routing not in ROUTING_MODES:
        raise HostedRoomError("routing must be 'everyone' or 'lead'")
    if routing == "lead" and not lead_member_id.strip():
        raise HostedRoomError("lead routing needs a lead_member_id")
    return {"purpose": purpose, "notes": notes, "routing": routing,
            "lead_member_id": lead_member_id.strip() if routing == "lead" else ""}


def context_from_payload(payload: Mapping[str, Any] | None) -> dict[str, str]:
    """The settings record of a stored settings payload; None (a room with no settings event) is the empty one."""
    if payload is None:
        return dict(_EMPTY_CONTEXT)
    return {key: str(payload.get(key) or default) for key, default in _EMPTY_CONTEXT.items()}


def retry_fields(context: Mapping[str, str], original: Mapping[str, Any]) -> dict[str, str]:
    """``context`` in the schema of ``original``, the first event of a retried save: one saved before routing
    existed has only purpose and notes, so a field still at its default is left out rather than read as a
    different save."""
    return {key: value for key, value in context.items() if key in original or value != _EMPTY_CONTEXT[key]}


def _claim_values(claims: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, str]:
    return {key: str(claims.get(key) or "") for key in keys}


def room_grant_scope_key(claims: Mapping[str, Any]) -> str:
    """Return a stable non-secret key for one room/home/target/profile scope."""
    fields = _claim_values(claims, (
        "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id", "target_install_id",
        "target_profile"))
    if not all(fields.values()):
        raise HostedRoomError("room grant scope is incomplete")
    return hashlib.sha256(compact_json(fields).encode("utf-8")).hexdigest()


MAX_REPLICA_ROOMS = 256
MAX_REPLICA_EVENT_BYTES = 256 * 1024 * 1024
_SYSTEM_ACTOR = {"kind": "system", "id": "authority-control"}


_positive_int = partial(bounded_int, error=ReplicaError, low=1)


def _control_event(kind: str, epoch: int, payload: dict[str, Any]) -> tuple[str, str, str, str]:
    """(event_id, kind, actor_json, payload_json) of the system ``authority.<kind>`` control event for ``epoch``."""
    return f"system:authority-{kind}:{epoch}", f"authority.{kind}", _actor_json(_SYSTEM_ACTOR), _payload_json(payload)


def _event_bytes(event: dict[str, Any]) -> int:
    return utf8_len(
        str(event["event_id"]), str(event["kind"]),
        json.dumps(event["actor"], ensure_ascii=False, separators=(",", ":")),
        json.dumps(event["payload"], ensure_ascii=False, separators=(",", ":")))


def _validate_page(page: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(page, dict):
        raise ReplicaError("page must be an object")
    events, authority = page.get("events"), page.get("authority")
    if not isinstance(events, list):
        raise ReplicaError("page.events must be a list")
    if not isinstance(authority, dict):
        raise ReplicaError("page.authority is required for replication")
    gateway_id = _validate_identifier(
        authority.get("gateway_id"), label="page.authority.gateway_id", max_chars=MAX_ACTOR_ID_CHARS)
    epoch = _positive_int(authority.get("epoch"), message="page.authority.epoch must be a positive integer")
    previous_seq: int | None = None
    for event in events:
        if not isinstance(event, dict):
            raise ReplicaError("page events must be objects")
        seq = _positive_int(event.get("seq"), message="event.seq must be a positive integer")
        if previous_seq is not None and seq != previous_seq + 1:
            raise ReplicaGapError("page events must be contiguous")
        previous_seq = seq
        for field in ("event_id", "kind"):
            if not isinstance(event.get(field), str) or not event[field]:
                raise ReplicaError(f"event.{field} must be a non-empty string")
        if not isinstance(event.get("actor"), dict):
            raise ReplicaError("event.actor must be an object")
        if "payload" not in event:
            raise ReplicaError("event.payload is required")
    return events, {"gateway_id": gateway_id, "epoch": epoch}


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


def _validate_tasks(record, tasks):
    task_ids = set()
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


def _validate_receipts(record, receipts, v2):
    receipt_ids = set()
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


def _validate_stop(record):
    _fields(record["stop"], {"closing", "revocation_complete", "seq", "cancel_id"})
    for key in ("closing", "revocation_complete"):
        if type(record["stop"][key]) is not bool:
            raise WorkRecordError("work record closing fact is invalid")
    _integer(record["stop"]["seq"])
    if record["stop"]["seq"] > record["history"]["seq"]:
        raise WorkRecordError("work record stop is beyond its prefix")
    if record["stop"]["cancel_id"] is not None:
        _id(record["stop"]["cancel_id"])


def _validate_availability(record):
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
    _validate_availability(record)
    _validate_tasks(record, record["tasks"])
    _validate_receipts(record, record["receipts"], v2)
    _validate_stop(record)
    _sha(record["digest"])
    if record["digest"] != digest({k: v for k, v in record.items() if k not in {"revision", "digest"}}):
        raise WorkRecordError("work record content digest conflicts")
    if len(encode(record).encode("utf-8")) > MAX_BYTES:
        raise WorkRecordError("work record exceeds its byte bound")
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


def roster_digest(members):
    _, encoded = _validate_members(members)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


INVALID_TABLE = "hosted_room_work_records_invalid"


def retained_fields(table):
    fields = ["room_id", "revision", "digest", "record_json"]
    fields += ["source_table"] if table == INVALID_TABLE else ["producer_gateway_id", "producer_epoch"]
    if table in (PENDING_TABLE, INVALID_TABLE):
        fields += ["target_install_id", "route_generation", "status"]
    return fields


def row_size_sql(table, prefix=""):
    # Reserve the longest bounded disposition up front so freezing/cleanup can
    # still run at capacity. Every original variable column is charged as bytes.
    return "20+" + "+".join(f"COALESCE(length(CAST({prefix}{k} AS BLOB)),0)" for k in retained_fields(table))


def usage_sql(tables):
    return (" + ".join(f"(SELECT COALESCE(SUM({row_size_sql(t)}),0) FROM {t})" for t in tables),
            " + ".join(f"(SELECT COUNT(*) FROM {t})" for t in tables))


def validate_stored(row):
    try:
        record = validate(json.loads(row["record_json"]))
        if (row["disposition"] == "invalid"
                or scope(record) != (row["room_id"], row["producer_gateway_id"], row["producer_epoch"])
                or record["revision"] != row["revision"] or record["digest"] != row["digest"]):
            raise WorkRecordError("stored work metadata conflicts with record scope")
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise InvalidStoredWorkRecord("stored work evidence is invalid") from exc
    return record


def scope(record):
    return record["room_id"], record["authority"]["gateway_id"], record["authority"]["epoch"]


def validate_provenance(record, spans):
    # Execution generation remains an independent per-task counter.
    for task in record["tasks"]:
        at_sequence(spans, task["source_event_seq"])
    for receipt in record["receipts"]:
        if not any(s.gateway_id == receipt["home_install_id"] == receipt["authority_gateway_id"]
                   and s.epoch == receipt["authority_epoch"] for s in spans):
            raise WorkRecordError("work record receipt has no original authority span")


MAX_AUTHORITY_SPANS = 1024


@dataclass(frozen=True)
class AuthoritySpan:
    gateway_id: str
    epoch: int
    from_seq: int

    def as_mapping(self) -> dict[str, Any]:
        return {"gateway_id": self.gateway_id, "epoch": self.epoch, "from_seq": self.from_seq}


def _gateway(value: Any) -> str:
    result = identifier(value, label="authority gateway", error=AuthorityHistoryError, max_chars=128)
    if result != value:
        raise AuthorityHistoryError("authority gateway must be canonical")
    return result


def validate_history(value: Any, *, gateway_id: str, epoch: int) -> tuple[AuthoritySpan, ...]:
    """Validate contextual epochs; possession of this data grants no control rights."""
    if value is None:
        return (AuthoritySpan(gateway_id, epoch, 0),)
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_AUTHORITY_SPANS:
        raise AuthorityHistoryError("authority history exceeds its bound or is absent")
    spans = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"gateway_id", "epoch", "from_seq"}:
            raise AuthorityHistoryError("authority history fields are invalid")
        source = _gateway(item["gateway_id"])
        generation, start = item["epoch"], item["from_seq"]
        if type(generation) is not int or not 1 <= generation < 2**63 or type(start) is not int or not 0 <= start < 2**63:
            raise AuthorityHistoryError("authority history coordinates are invalid")
        if not spans:
            if generation != 1 or start != 0:
                raise AuthorityHistoryError("authority history has no verified starting point")
        elif generation != spans[-1].epoch + 1 or start <= spans[-1].from_seq:
            raise AuthorityHistoryError("authority history has a gap or unordered boundary")
        spans.append(AuthoritySpan(source, generation, start))
    if (spans[-1].gateway_id, spans[-1].epoch) != (gateway_id, epoch):
        raise AuthorityHistoryError("authority history does not reach the current owner")
    return tuple(spans)


def at_sequence(history: tuple[AuthoritySpan, ...], seq: int) -> AuthoritySpan:
    for span in reversed(history):
        if seq >= span.from_seq:
            return span
    raise AuthorityHistoryError("event precedes the known authority history")
