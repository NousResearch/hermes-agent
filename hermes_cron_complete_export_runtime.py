"""Owner-scoped runtime for the fixed stock-cron complete-export contract.

The runtime resolves one active profile, takes the qualified profile custody
lease, validates the known Cron tree, and keeps the lease plus any SQLite
``BEGIN IMMEDIATE`` fences until the bounded record iterator is fully consumed.
It intentionally does not import ``cron`` for a fresh profile: producer
constructors create directories and repair state, which is outside a capture.

The resulting stream is a source component.  Main/native drain, all producer
lease adoption, and the serialized publisher remain separate composition work.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Iterator, Mapping

from hermes_cron_complete_export import (
    CRON_ADMISSIONS_SOURCE_ID,
    CRON_BOT_CHAT_PENDING_SOURCE_ID,
    CRON_CONTROL_MARKERS_SOURCE_ID,
    CRON_DELIVERIES_SOURCE_ID,
    CRON_EXECUTIONS_SOURCE_ID,
    CRON_JOBS_SOURCE_ID,
    CRON_NOTEPAD_SOURCE_ID,
    CRON_OUTPUTS_SOURCE_ID,
    CRON_RECOVERY_TELEMETRY_SOURCE_ID,
    CRON_SUGGESTIONS_SOURCE_ID,
    CRON_USAGE_AUDIT_SOURCE_ID,
    STOCK_CRON_SOURCE_IDS,
    CronCaptureBudget,
    CronCaptureFailed,
    CronCaptureField,
    CronCaptureManifest,
    CronCaptureRecord,
    CronCaptureRefused,
    CronCaptureUnavailable,
    CronCaptureRelation,
    CronStateCapture,
    CronFieldDisposition,
)
from hermes_cron_complete_export_files import (
    FileCaptureRefused,
    FileSnapshot,
    assert_current as assert_file_current,
    bounded_children,
    encoded_size,
    json_lines,
    parse_json,
    read_bytes,
    safe_component,
    scalar_bytes,
    snapshot,
)
from hermes_cron_complete_export_sqlite import (
    SQLiteCaptureHandle,
    SQLiteCaptureRefused,
    SQLiteColumnSpec,
    SQLiteIndexSpec,
    SQLiteTableSpec,
)


_MARKERS = (
    "ticker_heartbeat",
    "ticker_last_success",
    "ticker_last_error",
    "catch_up_occurrences",
)
_TELEMETRY = (
    "persisted_error_recoveries.jsonl",
    "timezone_migration_catchups.jsonl",
    "inflight_forced_releases.jsonl",
)
_TRANSIENT_CRON_FILES = {".jobs.lock", ".tick.lock"}
_MAX_JSON_FILE_BYTES = 512 * 1024 * 1024
_DATABASES = {"executions.db", "deliveries.db", "notepad.db"}
_DATABASE_SIDECARS = {
    f"{database}{suffix}": database
    for database in _DATABASES
    for suffix in ("-wal", "-shm", "-journal")
}
_ALLOWED_CRON_CHILDREN = frozenset(
    {
        "jobs.json", "output", "executions.db", "deliveries.db", "admissions", "notepad.db",
        "suggestions.json", "bot_chat_pending", "external-workers", "usage_audit.jsonl",
        *_MARKERS, *_TELEMETRY,
    }
)


def _column(name: str, declared: str, not_null: int, primary_key: int, default: str | None = None) -> SQLiteColumnSpec:
    return SQLiteColumnSpec(name, declared, not_null, primary_key, default)


_EXECUTIONS_SPEC = SQLiteTableSpec(
    "executions",
    (
        _column("id", "TEXT", 0, 1),
        _column("job_id", "TEXT", 1, 0),
        _column("source", "TEXT", 1, 0),
        _column("process_id", "TEXT", 1, 0),
        _column("pid", "INTEGER", 1, 0),
        _column("process_started_at", "INTEGER", 0, 0),
        _column("status", "TEXT", 1, 0),
        _column("handoff_pending", "INTEGER", 1, 0, "0"),
        _column("handoff_started_at", "REAL", 0, 0),
        _column("claimed_at", "TEXT", 1, 0),
        _column("started_at", "TEXT", 0, 0),
        _column("finished_at", "TEXT", 0, 0),
        _column("error", "TEXT", 0, 0),
        _column("delivery_outcome", "TEXT", 0, 0),
        _column("scheduled_instant", "TEXT", 0, 0),
        _column("progress_at", "TEXT", 0, 0),
    ),
    (
        SQLiteIndexSpec("idx_executions_job_claimed", ("job_id", "claimed_at", "id"), (False, True, True), False, False, ("BINARY", "BINARY", "BINARY")),
        SQLiteIndexSpec("idx_executions_status_claimed", ("status", "claimed_at", "id"), (False, True, True), False, False, ("BINARY", "BINARY", "BINARY")),
        SQLiteIndexSpec("idx_executions_occurrence", ("job_id", "scheduled_instant"), (False, False), True, False, ("BINARY", "BINARY"), "status='completed'"),
    ),
    ddl_fragments=(
        "create table",
        "status text not null check",
        "'claimed','running','completed','failed','unknown'",
        "handoff_pending integer not null default 0",
    ),
    ddl_exact=(
        "create table executions ( id text primary key, job_id text not null, source text not null, "
        "process_id text not null, pid integer not null, process_started_at integer, status text not null "
        "check(status in ('claimed','running','completed','failed','unknown')), "
        "handoff_pending integer not null default 0, handoff_started_at real, claimed_at text not null, "
        "started_at text, finished_at text, error text , delivery_outcome text, scheduled_instant text, "
        "progress_at text)"
    ),
)
_INCIDENTS_SPEC = SQLiteTableSpec(
    "cron_incidents",
    (
        _column("id", "TEXT", 0, 1),
        _column("job_id", "TEXT", 1, 0),
        _column("error_sig", "TEXT", 1, 0),
        _column("state", "TEXT", 1, 0),
        _column("failure_type", "TEXT", 1, 0, "'unknown'"),
        _column("first_seen_at", "TEXT", 1, 0),
        _column("last_seen_at", "TEXT", 1, 0),
        _column("acked_at", "TEXT", 0, 0),
        _column("alerted_at", "TEXT", 0, 0),
        _column("closed_at", "TEXT", 0, 0),
        _column("error", "TEXT", 1, 0),
        _column("output_file", "TEXT", 0, 0),
    ),
    (
        SQLiteIndexSpec("idx_cron_incidents_job", ("job_id",), (False,), False, False, ("BINARY",)),
        SQLiteIndexSpec("idx_cron_incidents_state", ("state",), (False,), False, False, ("BINARY",)),
    ),
    ddl_fragments=(
        "create table",
        "failure_type text not null default 'unknown'",
    ),
    ddl_exact=(
        "create table cron_incidents ( id text primary key, job_id text not null, error_sig text not null, "
        "state text not null, failure_type text not null default 'unknown', first_seen_at text not null, "
        "last_seen_at text not null, acked_at text, alerted_at text, closed_at text, error text not null, "
        "output_file text )"
    ),
)
_DELIVERIES_SPEC = SQLiteTableSpec(
    "deliveries",
    (
        _column("execution_id", "TEXT", 0, 1),
        _column("job_json", "TEXT", 1, 0),
        _column("content", "TEXT", 1, 0),
        _column("for_failure", "INTEGER", 1, 0, "0"),
        _column("status", "TEXT", 1, 0),
        _column("owner_process_id", "TEXT", 0, 0),
        _column("owner_pid", "INTEGER", 0, 0),
        _column("owner_started_at", "INTEGER", 0, 0),
        _column("created_at", "TEXT", 1, 0),
        _column("finished_at", "TEXT", 0, 0),
        _column("error", "TEXT", 0, 0),
    ),
    ddl_fragments=(
        "create table",
        "status text not null check",
        "'pending','delivering','delivered','failed','unknown','suppressed'",
    ),
    ddl_exact=(
        "create table deliveries ( execution_id text primary key, job_json text not null, content text not null, "
        "for_failure integer not null default 0, status text not null check(status in "
        "('pending','delivering','delivered','failed','unknown','suppressed')), owner_process_id text, "
        "owner_pid integer, owner_started_at integer, created_at text not null, finished_at text, error text )"
    ),
)
_TOMBSTONES_SPEC = SQLiteTableSpec(
    "delivery_tombstones",
    (
        _column("execution_id", "TEXT", 0, 1),
        _column("terminal_status", "TEXT", 1, 0),
        _column("finished_at", "TEXT", 0, 0),
    ),
    ddl_fragments=(
        "create table",
        "terminal_status text not null check",
        "'delivered','failed','unknown','suppressed'",
    ),
    ddl_exact=(
        "create table delivery_tombstones ( execution_id text primary key, terminal_status text not null "
        "check(terminal_status in ('delivered','failed','unknown','suppressed')), finished_at text )"
    ),
)
_NOTEPAD_SPEC = SQLiteTableSpec(
    "cron_notepad",
    (
        _column("job_id", "TEXT", 1, 1),
        _column("key", "TEXT", 1, 2),
        _column("value", "TEXT", 1, 0),
        _column("updated_at", "TEXT", 1, 0),
    ),
    ddl_fragments=("create table", "primary key (job_id, key)"),
    ddl_exact=(
        "create table cron_notepad ( job_id text not null, key text not null, value text not null, "
        "updated_at text not null, primary key (job_id, key) )"
    ),
)


def _deadline(budget: CronCaptureBudget) -> float:
    return time.monotonic() + float(budget.timeout_seconds)


def _resolve_profile_home() -> Path:
    """Resolve the source owner's current profile without importing ``cron``."""

    try:
        from hermes_constants import get_hermes_home

        jobs_module = sys.modules.get("cron.jobs")
        if jobs_module is not None:
            current = getattr(jobs_module, "_current_cron_store", None)
            if not callable(current):
                raise CronCaptureRefused("cron_capture_store_owner_unavailable")
            store = current()
            root = Path(store.cron_dir).parent
        else:
            root = Path(get_hermes_home())
    except CronCaptureRefused:
        raise
    except (AttributeError, TypeError, ValueError, OSError) as exc:
        raise CronCaptureRefused("cron_capture_profile_unavailable") from exc
    if not root.is_absolute():
        raise CronCaptureRefused("cron_capture_profile_invalid")
    return Path(os.path.abspath(os.fspath(root)))


@dataclass(slots=True)
class _PreparedFile:
    path: Path
    source_id: str
    record_type: str
    key: str
    raw: bytes
    value: Any = None
    snapshot: FileSnapshot | None = None


class _PreflightBudget:
    """One global bound for eager file reads before lazy SQLite paging."""

    def __init__(self, budget: CronCaptureBudget, deadline: float) -> None:
        self.max_bytes = budget.max_bytes
        self.max_files = budget.max_records
        self.deadline = deadline
        self.bytes = 0
        self.files = 0
        self.directory_entries = 0
        self.directory_membership: dict[Path, tuple[str, ...]] = {}

    def check(self) -> None:
        if time.monotonic() > self.deadline:
            raise CronCaptureRefused("cron_capture_deadline_exceeded")

    def reserve(self, saved: FileSnapshot) -> None:
        self.check()
        if self.files >= self.max_files:
            raise CronCaptureRefused("cron_capture_record_budget_exceeded")
        if self.bytes + max(0, saved.size) > self.max_bytes:
            raise CronCaptureRefused("cron_capture_byte_budget_exceeded")
        self.files += 1

    def commit(self, size: int) -> None:
        self.check()
        self.bytes += size
        if self.bytes > self.max_bytes:
            raise CronCaptureRefused("cron_capture_byte_budget_exceeded")

    def children(self, path: Path) -> tuple[Path, ...]:
        self.check()
        remaining = self.max_files - self.directory_entries
        if remaining <= 0:
            raise CronCaptureRefused("cron_capture_record_budget_exceeded")
        children = bounded_children(
            path,
            max_entries=remaining,
            deadline=self.deadline,
        )
        self.directory_entries += len(children)
        self.directory_membership.setdefault(
            path,
            tuple(
                child.name
                for child in children
                if child.name not in _DATABASE_SIDECARS or not child.name.endswith("-shm")
            ),
        )
        self.check()
        return children


@dataclass(slots=True)
class _PreparedCapture:
    root: Path
    cron_dir: Path | None
    files: list[_PreparedFile]
    jobs: tuple[dict[str, Any], ...]
    job_envelope: Any
    sqlite: dict[str, SQLiteCaptureHandle]
    present: set[str]
    snapshots: list[FileSnapshot]
    directories: tuple[tuple[Path, tuple[str, ...]], ...]


class _BudgetState:
    def __init__(self, budget: CronCaptureBudget, deadline: float) -> None:
        self.budget = budget
        self.deadline = deadline
        self.records = 0
        self.bytes = 0

    def check(self) -> None:
        if time.monotonic() > self.deadline:
            raise CronCaptureFailed("cron_capture_deadline_exceeded")

    def observe(self, record: CronCaptureRecord) -> None:
        self.check()
        size = len(record.raw_bytes or b"")
        size += encoded_size((record.source_id, record.record_type, record.key))
        for field in record.fields:
            size += encoded_size((field.name, field.disposition.value)) + encoded_size(field.value)
            if field.raw_bytes is not None:
                size += len(field.raw_bytes)
        for relation in record.relations:
            size += encoded_size((relation.relation, relation.source_key, relation.target_key))
        if self.records >= self.budget.max_records:
            raise CronCaptureFailed("cron_capture_record_budget_exceeded")
        if self.bytes + size > self.budget.max_bytes:
            raise CronCaptureFailed("cron_capture_byte_budget_exceeded")
        self.records += 1
        self.bytes += size


_CONTROL_FIELDS = frozenset({
    "enabled", "state", "next_run_at", "pending_slot", "monitor_state", "repeat",
    "handoff_pending", "handoff_started_at", "progress_at", "pause_recorded",
    "status", "failure_type", "terminal_status", "owner_process_id", "owner_pid",
    "owner_started_at",
})
_IDENTITY_FIELDS = frozenset({
    "id", "job_id", "execution_id", "process_id", "pid", "process_started_at",
    "error_sig", "request_id", "session_id", "admission_id",
})
_DELIVERY_FIELDS = frozenset({"for_failure", "created_at", "finished_at", "error"})
_DERIVED_FIELDS = frozenset({
    "table", "columns", "indexes", "ddl", "journal_mode", "schema_version", "user_version", "application_id",
})


def _source_disposition(source_id: str, name: str, record_type: str) -> CronFieldDisposition | None:
    if source_id == CRON_RECOVERY_TELEMETRY_SOURCE_ID:
        return CronFieldDisposition.CONTROL_RECOVERY
    if source_id == CRON_USAGE_AUDIT_SOURCE_ID:
        return CronFieldDisposition.BUSINESS_PAYLOAD if name == "error" else CronFieldDisposition.BUSINESS_METADATA
    if source_id == CRON_BOT_CHAT_PENDING_SOURCE_ID:
        if name in {"home", "id"}:
            return CronFieldDisposition.SECURITY_IDENTITY
        if name in {"status", "for_failure", "sequence"}:
            return CronFieldDisposition.DELIVERY_METADATA
        return CronFieldDisposition.BUSINESS_PAYLOAD
    if source_id == CRON_CONTROL_MARKERS_SOURCE_ID:
        return CronFieldDisposition.CONTROL_RECOVERY
    if source_id == CRON_EXECUTIONS_SOURCE_ID and record_type == "cron_incidents":
        if name == "job_id":
            return CronFieldDisposition.BUSINESS_METADATA
        if name == "error_sig":
            return CronFieldDisposition.SECURITY_IDENTITY
        return CronFieldDisposition.BUSINESS_PAYLOAD if name == "error" else CronFieldDisposition.BUSINESS_METADATA
    if source_id == CRON_EXECUTIONS_SOURCE_ID and name == "job_id":
        return CronFieldDisposition.BUSINESS_METADATA
    if source_id == CRON_NOTEPAD_SOURCE_ID and name in {"job_id", "updated_at"}:
        return CronFieldDisposition.BUSINESS_METADATA
    if source_id == CRON_DELIVERIES_SOURCE_ID:
        if name in {"owner_process_id", "owner_pid", "owner_started_at"}:
            return CronFieldDisposition.CONTROL_RECOVERY
        if name in {"job_json", "content"}:
            return CronFieldDisposition.BUSINESS_PAYLOAD
        return CronFieldDisposition.DELIVERY_METADATA
    if source_id == CRON_EXECUTIONS_SOURCE_ID:
        if name in {"handoff_pending", "handoff_started_at", "progress_at", "pending_slot"}:
            return CronFieldDisposition.CONTROL_RECOVERY
        return CronFieldDisposition.BUSINESS_METADATA
    return None


def _disposition(source_id: str, name: str, record_type: str = "") -> CronFieldDisposition:
    if name in _DERIVED_FIELDS:
        return CronFieldDisposition.DERIVED_INTERNAL
    source_result = _source_disposition(source_id, name, record_type)
    if source_result is not None:
        return source_result
    if name in _IDENTITY_FIELDS:
        return CronFieldDisposition.SECURITY_IDENTITY
    if name in _CONTROL_FIELDS:
        return CronFieldDisposition.CONTROL_RECOVERY
    if name in _DELIVERY_FIELDS:
        return CronFieldDisposition.DELIVERY_METADATA
    return CronFieldDisposition.BUSINESS_PAYLOAD


def _field(source_id: str, record_type: str, name: str, value: Any) -> CronCaptureField:
    return CronCaptureField(name, _disposition(source_id, name, record_type), value, scalar_bytes(value))


def _fields(source_id: str, record_type: str, values: Mapping[str, Any]) -> tuple[CronCaptureField, ...]:
    return tuple(_field(source_id, record_type, str(name), value) for name, value in values.items())


def _record(
    source_id: str,
    record_type: str,
    key: str,
    values: Mapping[str, Any] = (),
    *,
    raw: bytes | None = None,
    relations: Iterable[CronCaptureRelation] = (),
) -> CronCaptureRecord:
    return CronCaptureRecord(
        source_id=source_id,
        record_type=record_type,
        key=key,
        fields=_fields(source_id, record_type, values),
        relations=tuple(relations),
        raw_bytes=raw,
    )


def _read_file(path: Path, *, max_bytes: int, resource: _PreflightBudget) -> tuple[bytes, FileSnapshot]:
    try:
        resource.reserve(snapshot(path))
    except FileCaptureRefused as exc:
        raise CronCaptureRefused(exc.code) from exc
    try:
        return read_bytes(path, max_bytes=max_bytes, deadline=resource.deadline)
    except FileCaptureRefused as exc:
        raise CronCaptureRefused(exc.code) from exc


def _read_prepared(path: Path, source_id: str, record_type: str, key: str, *, max_bytes: int, resource: _PreflightBudget, parse: bool = False) -> _PreparedFile:
    raw, saved = _read_file(path, max_bytes=max_bytes, resource=resource)
    resource.commit(len(raw))
    value = parse_json(raw, deadline=resource.deadline) if parse else None
    resource.check()
    return _PreparedFile(path, source_id, record_type, key, raw, value, saved)


def _validate_jobs(
    prepared: _PreparedFile,
    *,
    max_records: int | None = None,
    deadline: float | None = None,
) -> tuple[tuple[dict[str, Any], ...], Any]:
    data = prepared.value
    if isinstance(data, dict):
        jobs = data.get("jobs")
        if not isinstance(jobs, list):
            raise CronCaptureRefused("cron_capture_jobs_shape_invalid")
    elif isinstance(data, list):
        jobs = data
    else:
        raise CronCaptureRefused("cron_capture_jobs_shape_invalid")
    found: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for number, value in enumerate(jobs, 1):
        if deadline is not None and time.monotonic() > deadline:
            raise CronCaptureRefused("cron_capture_deadline_exceeded")
        if max_records is not None and number > max_records:
            raise CronCaptureRefused("cron_capture_record_budget_exceeded")
        if not isinstance(value, dict):
            raise CronCaptureRefused("cron_capture_job_record_invalid")
        job_id = value.get("id")
        if not safe_component(job_id) or job_id in identifiers:
            raise CronCaptureRefused("cron_capture_job_identity_invalid")
        identifiers.add(job_id)
        found.append(value)
    return tuple(found), data


def _validate_suggestions(
    prepared: _PreparedFile,
    *,
    max_records: int | None = None,
    deadline: float | None = None,
) -> tuple[dict[str, Any], ...]:
    data = prepared.value
    values = data.get("suggestions") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not isinstance(values, list):
        raise CronCaptureRefused("cron_capture_suggestions_shape_invalid")
    if max_records is not None and len(values) > max_records:
        raise CronCaptureRefused("cron_capture_record_budget_exceeded")
    for value in values:
        if deadline is not None and time.monotonic() > deadline:
            raise CronCaptureRefused("cron_capture_deadline_exceeded")
        if not isinstance(value, dict):
            raise CronCaptureRefused("cron_capture_suggestion_record_invalid")
    return tuple(values)


def _validate_admission(path: Path, value: Any) -> None:
    if not isinstance(value, dict) or not isinstance(value.get("params"), dict):
        raise CronCaptureRefused("cron_capture_admission_malformed")
    params = value["params"]
    job_id, request_id = params.get("job_id"), params.get("request_id")
    if not isinstance(job_id, str) or not job_id or not isinstance(request_id, str) or not request_id:
        raise CronCaptureRefused("cron_capture_admission_identity_invalid")
    expected = hashlib.sha256(json.dumps([job_id, request_id]).encode("utf-8")).hexdigest() + ".json"
    if path.name != expected:
        raise CronCaptureRefused("cron_capture_admission_filename_mismatch")


def _marker_bad() -> None:
    raise CronCaptureRefused("cron_capture_marker_malformed")


def _marker_epoch(fields: list[str], count: int | frozenset[int]) -> None:
    if len(fields) not in (count if isinstance(count, frozenset) else {count}):
        _marker_bad()
    try:
        stamp = float(fields[0])
    except (TypeError, ValueError) as exc:
        raise CronCaptureRefused("cron_capture_marker_malformed") from exc
    if not math.isfinite(stamp):
        _marker_bad()


def _validate_heartbeat(value: str) -> None:
    fields = value.split()
    _marker_epoch(fields, frozenset({1, 2}))
    if len(fields) == 2:
        try:
            pid = int(fields[1])
        except (TypeError, ValueError) as exc:
            raise CronCaptureRefused("cron_capture_marker_malformed") from exc
        if pid <= 0:
            _marker_bad()


def _validate_last_success(value: str) -> None:
    _marker_epoch(value.split(), 1)


def _validate_last_error(value: str) -> None:
    lines = value.splitlines()
    if len(lines) < 2:
        _marker_bad()
    _marker_epoch([lines[0].strip()], 1)


def _validate_catch_up(value: str) -> None:
    try:
        count = int(value.strip())
    except (TypeError, ValueError) as exc:
        raise CronCaptureRefused("cron_capture_marker_malformed") from exc
    if count < 0:
        _marker_bad()


_MARKER_VALIDATORS = {
    "ticker_heartbeat": _validate_heartbeat,
    "ticker_last_success": _validate_last_success,
    "ticker_last_error": _validate_last_error,
    "catch_up_occurrences": _validate_catch_up,
}


def _marker_text(name: str, raw: bytes, *, deadline: float | None = None) -> str:
    if deadline is not None and time.monotonic() > deadline:
        raise CronCaptureRefused("cron_capture_deadline_exceeded")
    try:
        value = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CronCaptureRefused("cron_capture_marker_encoding_invalid") from exc
    validator = _MARKER_VALIDATORS.get(name)
    if validator is not None:
        validator(value)
    if deadline is not None and time.monotonic() > deadline:
        raise CronCaptureRefused("cron_capture_deadline_exceeded")
    return value


def _validate_bot(value: Any) -> None:
    if not isinstance(value, dict):
        raise CronCaptureRefused("cron_capture_bot_record_invalid")
    for name in ("id", "home", "content"):
        if not isinstance(value.get(name), str) or not value[name]:
            raise CronCaptureRefused("cron_capture_bot_record_invalid")
    if not isinstance(value.get("job") or {}, dict):
        raise CronCaptureRefused("cron_capture_bot_record_invalid")
    if value.get("status") not in {"queued", "claimed", "ambiguous", "suppressed", "transferred"}:
        raise CronCaptureRefused("cron_capture_bot_status_invalid")
    if "for_failure" in value and not isinstance(value["for_failure"], bool):
        raise CronCaptureRefused("cron_capture_bot_record_invalid")
    if "sequence" in value and (
        isinstance(value["sequence"], bool)
        or not isinstance(value["sequence"], int)
        or value["sequence"] < 0
    ):
        raise CronCaptureRefused("cron_capture_bot_record_invalid")
    if "error" in value and value["error"] is not None and not isinstance(value["error"], str):
        raise CronCaptureRefused("cron_capture_bot_record_invalid")


def _jsonl_text(value: Any, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise CronCaptureRefused("cron_capture_jsonl_record_invalid")


def _jsonl_number(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise CronCaptureRefused("cron_capture_jsonl_record_invalid")


def _validate_jsonl_record(filename: str, value: Any) -> None:
    if not isinstance(value, dict):
        raise CronCaptureRefused("cron_capture_jsonl_record_invalid")
    if filename == "persisted_error_recoveries.jsonl":
        for name in ("job_id", "name", "previous_next_run_at", "rearmed_at"):
            _jsonl_text(value.get(name), name)
    elif filename == "timezone_migration_catchups.jsonl":
        for name in ("job_id", "name", "stored_next_run_at", "normalized_next_run_at", "fired_at"):
            _jsonl_text(value.get(name), name)
        if value.get("expr") is not None:
            _jsonl_text(value.get("expr"), "expr", allow_empty=True)
    elif filename == "inflight_forced_releases.jsonl":
        for name in ("job_id", "name", "at"):
            _jsonl_text(value.get(name), name)
        _jsonl_number(value.get("age_seconds"), "age_seconds")
        _jsonl_number(value.get("allowance_seconds"), "allowance_seconds")


def _check_child_directory(path: Path, snapshots: list[FileSnapshot]) -> None:
    try:
        saved = snapshot(path, regular=False)
    except FileCaptureRefused as exc:
        raise CronCaptureRefused("cron_capture_child_directory_alias_unsupported") from exc
    snapshots.append(saved)


def _check_regular(path: Path, snapshots: list[FileSnapshot]) -> FileSnapshot:
    try:
        saved = snapshot(path)
    except FileCaptureRefused as exc:
        raise CronCaptureRefused(exc.code) from exc
    snapshots.append(saved)
    return saved


def _is_fire_lock(name: str) -> bool:
    prefix, suffix = ".fire-", ".lock"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    token = name[len(prefix):-len(suffix)]
    return len(token) == 32 and all(char in "0123456789abcdef" for char in token)


def _prepare_external_workers(
    child: Path,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    tracked_dirs: list[Path],
) -> None:
    _check_child_directory(child, snapshots)
    tracked_dirs.append(child)
    for number, handoff in enumerate(resource.children(child), 1):
        resource.check()
        if number > resource.max_files:
            raise CronCaptureRefused("cron_capture_record_budget_exceeded")
        if not handoff.is_file():
            raise CronCaptureRefused("cron_capture_unknown_external_worker_artifact")
        suffix = next(
            (value for value in (".json", ".ready", ".stderr") if handoff.name.endswith(value)),
            None,
        )
        if suffix is None:
            raise CronCaptureRefused("cron_capture_unknown_external_worker_artifact")
        token = handoff.name[:-len(suffix)]
        if len(token) != 32 or any(char not in "0123456789abcdef" for char in token):
            raise CronCaptureRefused("cron_capture_external_worker_identity_invalid")
        _check_regular(handoff, snapshots)


def _prepare_output(
    child: Path,
    root: Path,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    prepared_files: list[_PreparedFile],
    tracked_dirs: list[Path],
) -> bool:
    tracked_dirs.append(child)
    found = False
    for job_dir in resource.children(child):
        resource.check()
        if not safe_component(job_dir.name):
            raise CronCaptureRefused("cron_capture_output_directory_invalid")
        _check_child_directory(job_dir, snapshots)
        tracked_dirs.append(job_dir)
        for output in resource.children(job_dir):
            resource.check()
            if output.name != "monitor_last_output.txt" and not output.name.endswith(".md"):
                raise CronCaptureRefused("cron_capture_unknown_output")
            if not output.is_file():
                raise CronCaptureRefused("cron_capture_output_not_regular")
            item = _read_prepared(
                output,
                CRON_OUTPUTS_SOURCE_ID,
                "monitor_snapshot" if output.name == "monitor_last_output.txt" else "output",
                str(output.relative_to(root)),
                max_bytes=_MAX_JSON_FILE_BYTES,
                resource=resource,
            )
            prepared_files.append(item)
            snapshots.append(item.snapshot)  # type: ignore[arg-type]
            found = True
    return found


def _prepare_admissions(
    child: Path,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    prepared_files: list[_PreparedFile],
    tracked_dirs: list[Path],
) -> bool:
    _check_child_directory(child, snapshots)
    tracked_dirs.append(child)
    found = False
    for journal in resource.children(child):
        resource.check()
        if journal.suffix != ".json" or not journal.is_file():
            raise CronCaptureRefused("cron_capture_unknown_admission")
        item = _read_prepared(
            journal,
            CRON_ADMISSIONS_SOURCE_ID,
            "admission",
            journal.name,
            max_bytes=_MAX_JSON_FILE_BYTES,
            resource=resource,
            parse=True,
        )
        _validate_admission(journal, item.value)
        prepared_files.append(item)
        snapshots.append(item.snapshot)  # type: ignore[arg-type]
        found = True
    return found


def _prepare_bot_chat(
    child: Path,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    prepared_files: list[_PreparedFile],
    tracked_dirs: list[Path],
) -> bool:
    _check_child_directory(child, snapshots)
    tracked_dirs.append(child)
    found = False
    for pending in resource.children(child):
        resource.check()
        if pending.suffix != ".json" or not pending.is_file():
            raise CronCaptureRefused("cron_capture_unknown_bot_record")
        item = _read_prepared(
            pending,
            CRON_BOT_CHAT_PENDING_SOURCE_ID,
            "bot_chat_pending",
            pending.name,
            max_bytes=_MAX_JSON_FILE_BYTES,
            resource=resource,
            parse=True,
        )
        _validate_bot(item.value)
        if pending.stem != item.value["id"]:
            raise CronCaptureRefused("cron_capture_bot_identity_invalid")
        prepared_files.append(item)
        snapshots.append(item.snapshot)  # type: ignore[arg-type]
        found = True
    return found


def _prepare_jsonl(
    child: Path,
    source_id: str,
    record_type: str,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    prepared_files: list[_PreparedFile],
) -> None:
    raw, saved = _read_file(child, max_bytes=_MAX_JSON_FILE_BYTES, resource=resource)
    resource.commit(len(raw))
    line_count = 0
    try:
        for _line, value in json_lines(
            raw,
            code="cron_capture_jsonl_malformed",
            deadline=resource.deadline,
        ):
            resource.check()
            line_count += 1
            if line_count > resource.max_files:
                raise CronCaptureRefused("cron_capture_record_budget_exceeded")
            _validate_jsonl_record(child.name, value)
    except FileCaptureRefused as exc:
        raise CronCaptureRefused(exc.code) from exc
    prepared_files.append(_PreparedFile(child, source_id, record_type, child.name, raw, None, saved))
    snapshots.append(saved)


def _prepare_child(
    child: Path,
    root: Path,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    prepared_files: list[_PreparedFile],
    present: set[str],
    tracked_dirs: list[Path],
    jobs: tuple[dict[str, Any], ...],
    job_envelope: Any,
) -> tuple[tuple[dict[str, Any], ...], Any]:
    if child.name == "jobs.json":
        item = _read_prepared(
            child, CRON_JOBS_SOURCE_ID, "file", "jobs.json",
            max_bytes=_MAX_JSON_FILE_BYTES, resource=resource, parse=True,
        )
        jobs, job_envelope = _validate_jobs(
            item,
            max_records=resource.max_files,
            deadline=resource.deadline,
        )
        prepared_files.append(item)
        snapshots.append(item.snapshot)  # type: ignore[arg-type]
        present.add(CRON_JOBS_SOURCE_ID)
    elif child.name == "output":
        if _prepare_output(child, root, resource, snapshots, prepared_files, tracked_dirs):
            present.add(CRON_OUTPUTS_SOURCE_ID)
    elif child.name == "admissions":
        if _prepare_admissions(child, resource, snapshots, prepared_files, tracked_dirs):
            present.add(CRON_ADMISSIONS_SOURCE_ID)
    elif child.name == "bot_chat_pending":
        if _prepare_bot_chat(child, resource, snapshots, prepared_files, tracked_dirs):
            present.add(CRON_BOT_CHAT_PENDING_SOURCE_ID)
    elif child.name in _MARKERS:
        present.add(CRON_CONTROL_MARKERS_SOURCE_ID)
        item = _read_prepared(
            child, CRON_CONTROL_MARKERS_SOURCE_ID, "marker", child.name,
            max_bytes=_MAX_JSON_FILE_BYTES, resource=resource,
        )
        _marker_text(child.name, item.raw, deadline=resource.deadline)
        prepared_files.append(item)
        snapshots.append(item.snapshot)  # type: ignore[arg-type]
    elif child.name in _TELEMETRY:
        present.add(CRON_RECOVERY_TELEMETRY_SOURCE_ID)
        _prepare_jsonl(child, CRON_RECOVERY_TELEMETRY_SOURCE_ID, "telemetry", resource, snapshots, prepared_files)
    elif child.name == "usage_audit.jsonl":
        present.add(CRON_USAGE_AUDIT_SOURCE_ID)
        _prepare_jsonl(child, CRON_USAGE_AUDIT_SOURCE_ID, "usage_audit", resource, snapshots, prepared_files)
    elif child.name == "suggestions.json":
        item = _read_prepared(
            child, CRON_SUGGESTIONS_SOURCE_ID, "file", "suggestions.json",
            max_bytes=_MAX_JSON_FILE_BYTES, resource=resource, parse=True,
        )
        _validate_suggestions(
            item,
            max_records=resource.max_files,
            deadline=resource.deadline,
        )
        prepared_files.append(item)
        snapshots.append(item.snapshot)  # type: ignore[arg-type]
        present.add(CRON_SUGGESTIONS_SOURCE_ID)
    elif child.name in _DATABASES:
        present.add({
            "executions.db": CRON_EXECUTIONS_SOURCE_ID,
            "deliveries.db": CRON_DELIVERIES_SOURCE_ID,
            "notepad.db": CRON_NOTEPAD_SOURCE_ID,
        }[child.name])
    return jobs, job_envelope


def _close_sqlite_handles(handles: Mapping[str, SQLiteCaptureHandle]) -> None:
    for handle in handles.values():
        try:
            handle.close()
        except SQLiteCaptureRefused:
            continue


def _prepare_tree_child(
    child: Path,
    child_names: set[str],
    root: Path,
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
    prepared_files: list[_PreparedFile],
    present: set[str],
    tracked_dirs: list[Path],
    jobs: tuple[dict[str, Any], ...],
    job_envelope: Any,
) -> tuple[tuple[dict[str, Any], ...], Any]:
    resource.check()
    if child.name in _DATABASE_SIDECARS:
        owner = _DATABASE_SIDECARS[child.name]
        if owner not in child_names:
            raise CronCaptureRefused("cron_capture_orphan_sqlite_sidecar")
        if child.name.endswith("-shm"):
            try:
                snapshot(child)
            except FileCaptureRefused as exc:
                raise CronCaptureRefused(exc.code) from exc
        else:
            _check_regular(child, snapshots)
        return jobs, job_envelope
    if child.name in _TRANSIENT_CRON_FILES or _is_fire_lock(child.name):
        _check_regular(child, snapshots)
        return jobs, job_envelope
    if child.name not in _ALLOWED_CRON_CHILDREN:
        raise CronCaptureRefused("cron_capture_unknown_artifact")
    directory_names = {"output", "admissions", "bot_chat_pending", "external-workers"}
    try:
        child_identity = snapshot(child, regular=child.name not in directory_names)
    except FileCaptureRefused as exc:
        raise CronCaptureRefused(exc.code) from exc
    snapshots.append(child_identity)
    if child.name == "external-workers":
        _prepare_external_workers(child, resource, snapshots, tracked_dirs)
        return jobs, job_envelope
    return _prepare_child(
        child, root, resource, snapshots, prepared_files, present,
        tracked_dirs, jobs, job_envelope,
    )


def _prepare_tree_children(
    root: Path,
    cron_dir: Path,
    children: tuple[Path, ...],
    resource: _PreflightBudget,
    snapshots: list[FileSnapshot],
) -> tuple[list[_PreparedFile], set[str], tuple[dict[str, Any], ...], Any, list[Path]]:
    child_names = {child.name for child in children}
    prepared_files: list[_PreparedFile] = []
    present: set[str] = set()
    jobs: tuple[dict[str, Any], ...] = ()
    job_envelope: Any = None
    tracked_dirs: list[Path] = [cron_dir]
    for child in children:
        jobs, job_envelope = _prepare_tree_child(
            child, child_names, root, resource, snapshots, prepared_files,
            present, tracked_dirs, jobs, job_envelope,
        )
    expected_monitor = {
        str(job["id"])
        for job in jobs
        if isinstance(job.get("monitor_state"), dict)
        and (job["monitor_state"].get("last_output_hash") or job["monitor_state"].get("last_changed_at"))
    }
    observed_monitor = {
        item.path.parent.name
        for item in prepared_files
        if item.source_id == CRON_OUTPUTS_SOURCE_ID and item.record_type == "monitor_snapshot"
    }
    if expected_monitor - observed_monitor:
        raise CronCaptureRefused("cron_capture_configured_output_missing")
    return prepared_files, present, jobs, job_envelope, tracked_dirs


def _open_tree_sqlite(
    cron_dir: Path,
    child_names: set[str],
    deadline: float,
    max_bytes: int,
) -> dict[str, SQLiteCaptureHandle]:
    sqlite_handles: dict[str, SQLiteCaptureHandle] = {}
    try:
        if "executions.db" in child_names:
            sqlite_handles[CRON_EXECUTIONS_SOURCE_ID] = _open_sqlite_handle(
                cron_dir / "executions.db",
                {item.name: item for item in (_EXECUTIONS_SPEC, _INCIDENTS_SPEC)},
                deadline,
                max_bytes,
            )
        if "deliveries.db" in child_names:
            sqlite_handles[CRON_DELIVERIES_SOURCE_ID] = _open_sqlite_handle(
                cron_dir / "deliveries.db",
                {item.name: item for item in (_DELIVERIES_SPEC, _TOMBSTONES_SPEC)},
                deadline,
                max_bytes,
            )
        if "notepad.db" in child_names:
            sqlite_handles[CRON_NOTEPAD_SOURCE_ID] = _open_sqlite_handle(
                cron_dir / "notepad.db", {"cron_notepad": _NOTEPAD_SPEC}, deadline, max_bytes,
            )
    except CronCaptureUnavailable:
        _close_sqlite_handles(sqlite_handles)
        raise
    except SQLiteCaptureRefused as exc:
        _close_sqlite_handles(sqlite_handles)
        raise CronCaptureRefused(exc.code) from exc
    return sqlite_handles


def _directory_entries(
    paths: Iterable[Path],
    resource: _PreflightBudget,
) -> tuple[tuple[Path, tuple[str, ...]], ...]:
    entries: list[tuple[Path, tuple[str, ...]]] = []
    for path in paths:
        resource.check()
        try:
            names = resource.directory_membership[path]
        except KeyError as exc:
            raise CronCaptureRefused("cron_capture_tree_changed") from exc
        entries.append((path, names))
    return tuple(entries)


def _expected_directory_names(
    path: Path,
    names: tuple[str, ...],
    sqlite_handles: Mapping[str, SQLiteCaptureHandle],
) -> tuple[str, ...]:
    expected = set(names)
    for handle in sqlite_handles.values():
        if handle.path.parent == path and handle.capture_empty_wal is not None:
            expected.add(handle.path.name + "-wal")
    return tuple(sorted(expected))


def _prepare_tree(root: Path, budget: CronCaptureBudget, deadline: float) -> _PreparedCapture:
    cron_dir = root / "cron"
    snapshots: list[FileSnapshot] = []
    try:
        cron_snapshot = snapshot(cron_dir, regular=False)
    except FileCaptureRefused as exc:
        if exc.code == "cron_capture_file_missing":
            return _PreparedCapture(root, None, [], (), None, {}, set(), snapshots, ())
        if exc.code == "cron_capture_file_reparse":
            raise CronCaptureRefused("cron_capture_child_directory_alias_unsupported") from exc
        raise CronCaptureRefused(exc.code) from exc
    resource = _PreflightBudget(budget, deadline)
    snapshots.append(cron_snapshot)
    children = resource.children(cron_dir)
    prepared_files, present, jobs, job_envelope, tracked_dirs = _prepare_tree_children(
        root, cron_dir, children, resource, snapshots,
    )
    resource.check()
    directory_entries = _directory_entries(tracked_dirs, resource)
    sqlite_handles = _open_tree_sqlite(
        cron_dir,
        {child.name for child in children},
        deadline,
        budget.max_bytes,
    )
    return _PreparedCapture(
        root, cron_dir, prepared_files, jobs, job_envelope, sqlite_handles,
        present, snapshots, directory_entries,
    )


def _open_sqlite_handle(
    path: Path,
    specs: Mapping[str, SQLiteTableSpec],
    deadline: float,
    max_bytes: int,
) -> SQLiteCaptureHandle:
    """Open the contextmanager implementation without importing any producer."""

    try:
        from hermes_cron_complete_export_sqlite import _open_handle
    except (ImportError, ModuleNotFoundError) as exc:
        raise CronCaptureUnavailable("cron_capture_sqlite_backend_unavailable") from exc

    try:
        return _open_handle(path, specs, deadline=deadline, max_bytes=max_bytes)
    except (ImportError, ModuleNotFoundError) as exc:
        raise CronCaptureUnavailable("cron_capture_sqlite_backend_unavailable") from exc


def _schema_records(
    source_id: str,
    handle: SQLiteCaptureHandle,
    *,
    deadline: float | None = None,
) -> Iterator[CronCaptureRecord]:
    for kind, name, sql in handle.schema_sql:
        if deadline is not None and time.monotonic() > deadline:
            raise CronCaptureFailed("cron_capture_deadline_exceeded")
        values: dict[str, Any] = {
            "table": name,
            "ddl": sql,
            "journal_mode": handle.journal_mode,
            "schema_version": handle.schema_version,
            "user_version": handle.user_version,
            "application_id": handle.application_id,
        }
        if kind == "table":
            table_schema = next(
                (
                    item
                    for item in handle.schema_values(deadline=deadline)
                    if item["table"] == name
                ),
                None,
            )
            if table_schema is None:
                raise CronCaptureFailed("cron_capture_sqlite_schema_missing")
            values["columns"] = table_schema["columns"]
            values["indexes"] = table_schema["indexes"]
        yield _record(source_id, "schema", f"{handle.path.name}:{kind}:{name}", values)


_EXECUTION_STATUSES = frozenset({"claimed", "running", "completed", "failed", "unknown"})
_INCIDENT_STATES = frozenset({"detected", "alerted", "resolved", "closed"})
_FAILURE_TYPES = frozenset({"rate_limit", "timeout", "auth", "delivery", "config", "script", "agent", "unknown"})
_DELIVERY_STATUSES = frozenset({"pending", "delivering", "delivered", "failed", "unknown", "suppressed"})
_TOMBSTONE_STATUSES = frozenset({"delivered", "failed", "unknown", "suppressed"})


def _row_text(row: Mapping[str, Any], name: str, *, required: bool = True, allow_empty: bool = False) -> str | None:
    value = row.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (not allow_empty and not value):
        raise CronCaptureRefused("cron_capture_sqlite_row_value_invalid")
    return value


def _row_integer(row: Mapping[str, Any], name: str, *, required: bool = True, minimum: int | None = None, allowed: frozenset[int] | None = None) -> int | None:
    value = row.get(name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CronCaptureRefused("cron_capture_sqlite_row_value_invalid")
    if minimum is not None and value < minimum:
        raise CronCaptureRefused("cron_capture_sqlite_row_value_invalid")
    if allowed is not None and value not in allowed:
        raise CronCaptureRefused("cron_capture_sqlite_row_value_invalid")
    return value


def _row_number(row: Mapping[str, Any], name: str, *, required: bool = False) -> int | float | None:
    value = row.get(name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CronCaptureRefused("cron_capture_sqlite_row_value_invalid")
    return value


def _validate_sqlite_row(
    table: str,
    row: Mapping[str, Any],
    *,
    deadline: float | None = None,
) -> dict[str, Any] | None:
    if table == "executions":
        for name in ("id", "job_id", "source", "process_id", "claimed_at"):
            _row_text(row, name)
        _row_integer(row, "pid", minimum=1)
        _row_integer(row, "process_started_at", required=False, minimum=0)
        status = _row_text(row, "status")
        if status not in _EXECUTION_STATUSES:
            raise CronCaptureRefused("cron_capture_sqlite_enum_invalid")
        _row_integer(row, "handoff_pending", allowed=frozenset({0, 1}))
        _row_number(row, "handoff_started_at")
        for name in ("started_at", "finished_at", "error", "delivery_outcome", "scheduled_instant", "progress_at"):
            _row_text(row, name, required=False, allow_empty=True)
        return None
    if table == "cron_incidents":
        for name in ("id", "job_id", "error_sig", "state", "failure_type", "first_seen_at", "last_seen_at"):
            _row_text(row, name)
        state = row["state"]
        failure_type = row["failure_type"]
        if state not in _INCIDENT_STATES or failure_type not in _FAILURE_TYPES:
            raise CronCaptureRefused("cron_capture_sqlite_enum_invalid")
        for name in ("acked_at", "alerted_at", "closed_at"):
            _row_text(row, name, required=False, allow_empty=True)
        _row_text(row, "error", allow_empty=True)
        output_file = _row_text(row, "output_file", required=False)
        return {"output_file": output_file} if output_file is not None else None
    if table == "deliveries":
        execution_id = _row_text(row, "execution_id")
        job_json = _row_text(row, "job_json", allow_empty=True)
        _row_text(row, "content", allow_empty=True)
        _row_integer(row, "for_failure", allowed=frozenset({0, 1}))
        status = _row_text(row, "status")
        if status not in _DELIVERY_STATUSES:
            raise CronCaptureRefused("cron_capture_sqlite_enum_invalid")
        _row_text(row, "owner_process_id", required=False)
        _row_integer(row, "owner_pid", required=False, minimum=1)
        _row_integer(row, "owner_started_at", required=False, minimum=0)
        _row_text(row, "created_at")
        _row_text(row, "finished_at", required=False, allow_empty=True)
        _row_text(row, "error", required=False, allow_empty=True)
        try:
            payload = parse_json((job_json or "").encode("utf-8"), deadline=deadline)
        except (UnicodeEncodeError, FileCaptureRefused) as exc:
            raise CronCaptureRefused("cron_capture_sqlite_json_invalid") from exc
        if not isinstance(payload, dict):
            raise CronCaptureRefused("cron_capture_sqlite_json_invalid")
        job_id = payload.get("id")
        return {"execution_id": execution_id, "job_id": job_id} if isinstance(job_id, str) and job_id else {"execution_id": execution_id}
    if table == "delivery_tombstones":
        execution_id = _row_text(row, "execution_id")
        status = _row_text(row, "terminal_status")
        if status not in _TOMBSTONE_STATUSES:
            raise CronCaptureRefused("cron_capture_sqlite_enum_invalid")
        _row_text(row, "finished_at", required=False, allow_empty=True)
        return {"execution_id": execution_id}
    if table == "cron_notepad":
        job_id = _row_text(row, "job_id")
        key = _row_text(row, "key")
        value = _row_text(row, "value", allow_empty=True)
        _row_text(row, "updated_at")
        if len(key or "") > 128 or len((value or "").encode("utf-8")) > 16 * 1024:
            raise CronCaptureRefused("cron_capture_sqlite_row_value_invalid")
        return {"job_id": job_id}
    raise CronCaptureRefused("cron_capture_sqlite_unknown_table")


def _sqlite_records(
    source_id: str,
    handle: SQLiteCaptureHandle,
    *,
    deadline: float | None = None,
) -> Iterator[CronCaptureRecord]:
    yield from _schema_records(source_id, handle, deadline=deadline)
    for table in handle.tables:
        for row in handle.iter_rows(table, deadline=deadline):
            if deadline is not None and time.monotonic() > deadline:
                raise CronCaptureFailed("cron_capture_deadline_exceeded")
            extra = _validate_sqlite_row(table, row, deadline=deadline)
            if table == "cron_notepad":
                key = f"{row['job_id']}:{row['key']}"
            else:
                key_value = row.get("id") or row.get("execution_id")
                if not isinstance(key_value, str) or not key_value:
                    raise CronCaptureRefused("cron_capture_sqlite_row_identity_invalid")
                key = key_value
            relations: list[CronCaptureRelation] = []
            if table in {"executions", "cron_incidents", "cron_notepad"} and isinstance(row.get("job_id"), str):
                relations.append(CronCaptureRelation("job", key, row["job_id"]))
            if table in {"deliveries", "delivery_tombstones"} and isinstance(row.get("execution_id"), str):
                relations.append(CronCaptureRelation("execution", key, row["execution_id"]))
            if table == "deliveries" and extra is not None and isinstance(extra.get("job_id"), str) and extra["job_id"]:
                relations.append(CronCaptureRelation("job", key, extra["job_id"]))
            if table == "cron_incidents" and extra is not None and isinstance(extra.get("output_file"), str):
                relations.append(CronCaptureRelation("output", key, extra["output_file"]))
            yield _record(source_id, table, key, row, relations=relations)


def _job_file_records(item: _PreparedFile, prepared: _PreparedCapture) -> Iterator[CronCaptureRecord]:
    values = item.value if isinstance(item.value, dict) else {"jobs": item.value}
    yield _record(item.source_id, item.record_type, item.key, values, raw=item.raw)
    for job in prepared.jobs:
        yield _record(item.source_id, "job", str(job["id"]), job)


def _suggestion_file_records(item: _PreparedFile, budget: _BudgetState) -> Iterator[CronCaptureRecord]:
    values = item.value if isinstance(item.value, dict) else {"suggestions": item.value}
    yield _record(item.source_id, item.record_type, item.key, values, raw=item.raw)
    for number, suggestion in enumerate(_validate_suggestions(item, deadline=budget.deadline), 1):
        key = str(suggestion.get("dedup_key") or suggestion.get("id") or number)
        yield _record(item.source_id, "suggestion", key, suggestion)


def _output_file_records(item: _PreparedFile) -> Iterator[CronCaptureRecord]:
    relations: tuple[CronCaptureRelation, ...] = ()
    job_id = item.path.parent.name
    if safe_component(job_id):
        relations = (CronCaptureRelation("output-job", item.key, job_id),)
    yield _record(item.source_id, item.record_type, item.key, {"path": item.key}, raw=item.raw, relations=relations)


def _admission_relations(item: _PreparedFile) -> tuple[CronCaptureRelation, ...]:
    params = item.value.get("params") if isinstance(item.value, dict) else None
    if not isinstance(params, dict):
        return ()
    edges: list[CronCaptureRelation] = []
    job_id, request_id = params.get("job_id"), params.get("request_id")
    if isinstance(job_id, str) and job_id:
        edges.append(CronCaptureRelation("job", item.key, job_id))
    if isinstance(request_id, str) and request_id:
        edges.append(CronCaptureRelation("execution", item.key, request_id))
    return tuple(edges)


def _bot_relations(item: _PreparedFile) -> tuple[CronCaptureRelation, ...]:
    job = item.value.get("job") if isinstance(item.value, dict) else None
    job_id = job.get("id") if isinstance(job, dict) else None
    if isinstance(job_id, str) and job_id:
        return (CronCaptureRelation("job", item.key, job_id),)
    return ()


def _pending_file_records(item: _PreparedFile) -> Iterator[CronCaptureRecord]:
    relations = _admission_relations(item) if item.source_id == CRON_ADMISSIONS_SOURCE_ID else _bot_relations(item)
    yield _record(item.source_id, item.record_type, item.key, item.value, raw=item.raw, relations=relations)


def _jsonl_file_records(item: _PreparedFile, budget: _BudgetState) -> Iterator[CronCaptureRecord]:
    try:
        lines = json_lines(
            item.raw,
            code="cron_capture_jsonl_malformed",
            deadline=budget.deadline,
        )
        for number, (line, value) in enumerate(lines, 1):
            budget.check()
            _validate_jsonl_record(item.key, value)
            relations: tuple[CronCaptureRelation, ...] = ()
            job_id = value.get("job_id")
            if isinstance(job_id, str) and job_id:
                relations = (CronCaptureRelation("job", f"{item.key}:{number}", job_id),)
            yield _record(item.source_id, item.record_type, f"{item.key}:{number}", value, raw=line, relations=relations)
    except FileCaptureRefused as exc:
        raise CronCaptureFailed(exc.code) from exc


def _marker_file_records(item: _PreparedFile, budget: _BudgetState) -> Iterator[CronCaptureRecord]:
    value = _marker_text(item.key, item.raw, deadline=budget.deadline)
    yield _record(
        item.source_id,
        item.record_type,
        item.key,
        {"value": value},
        raw=item.raw,
        relations=(CronCaptureRelation("profile-marker", item.key, "stock-cron"),),
    )


def _file_records(
    item: _PreparedFile,
    prepared: _PreparedCapture,
    budget: _BudgetState,
) -> Iterator[CronCaptureRecord]:
    if item.source_id == CRON_JOBS_SOURCE_ID:
        yield from _job_file_records(item, prepared)
    elif item.source_id == CRON_SUGGESTIONS_SOURCE_ID:
        yield from _suggestion_file_records(item, budget)
    elif item.source_id == CRON_OUTPUTS_SOURCE_ID:
        yield from _output_file_records(item)
    elif item.source_id in {CRON_ADMISSIONS_SOURCE_ID, CRON_BOT_CHAT_PENDING_SOURCE_ID}:
        yield from _pending_file_records(item)
    elif item.source_id in {CRON_RECOVERY_TELEMETRY_SOURCE_ID, CRON_USAGE_AUDIT_SOURCE_ID}:
        yield from _jsonl_file_records(item, budget)
    elif item.source_id == CRON_CONTROL_MARKERS_SOURCE_ID:
        yield from _marker_file_records(item, budget)
    else:
        yield _record(item.source_id, item.record_type, item.key, {"path": item.key}, raw=item.raw)


class _LiveCapture(CronStateCapture):
    def __init__(self, prepared: _PreparedCapture, budget: CronCaptureBudget, deadline: float, lease: Any) -> None:
        self._prepared = prepared
        self._budget = _BudgetState(budget, deadline)
        self._lease = lease
        self._started = False
        self._complete = False
        self._closed = False
        absent = tuple(source for source in STOCK_CRON_SOURCE_IDS if source not in prepared.present)
        self._manifest = CronCaptureManifest(absent_source_ids=absent)

    @property
    def manifest(self) -> CronCaptureManifest:
        return self._manifest

    def _records(self) -> Iterator[CronCaptureRecord]:
        for item in self._prepared.files:
            self._budget.check()
            yield from _file_records(item, self._prepared, self._budget)
        for source_id in (CRON_EXECUTIONS_SOURCE_ID, CRON_DELIVERIES_SOURCE_ID, CRON_NOTEPAD_SOURCE_ID):
            handle = self._prepared.sqlite.get(source_id)
            if handle is not None:
                yield from _sqlite_records(source_id, handle, deadline=self._budget.deadline)

    def iter_records(self) -> Iterator[CronCaptureRecord]:
        if self._closed:
            raise CronCaptureFailed("cron_capture_released")
        if self._started:
            raise CronCaptureFailed("cron_capture_iterator_reused")
        self._started = True
        try:
            for record in self._records():
                self._budget.observe(record)
                yield record
            self._complete = True
        except (CronCaptureFailed, CronCaptureRefused):
            raise
        except (FileCaptureRefused, SQLiteCaptureRefused) as exc:
            raise CronCaptureFailed(exc.code) from exc
        except (GeneratorExit, KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            raise CronCaptureFailed("cron_capture_consumption_failed") from exc

    def finish(self) -> None:
        if not self._started or not self._complete:
            raise CronCaptureFailed("cron_capture_partial_consumption")
        self._budget.check()
        self._lease.assert_current()
        self._budget.check()
        if self._prepared.cron_dir is None:
            try:
                snapshot(self._prepared.root / "cron", regular=False)
            except FileCaptureRefused as exc:
                if exc.code == "cron_capture_file_missing":
                    self._budget.check()
                    return
                raise CronCaptureFailed(exc.code) from exc
            raise CronCaptureFailed("cron_capture_tree_created")
        for saved in self._prepared.snapshots:
            self._budget.check()
            try:
                assert_file_current(saved, deadline=self._budget.deadline)
            except FileCaptureRefused as exc:
                raise CronCaptureFailed(exc.code) from exc
            self._budget.check()
        remaining_entries = self._budget.budget.max_records
        for path, names in self._prepared.directories:
            self._budget.check()
            try:
                current_children = bounded_children(
                    path,
                    max_entries=remaining_entries,
                    deadline=self._budget.deadline,
                )
            except FileCaptureRefused as exc:
                raise CronCaptureFailed(exc.code) from exc
            remaining_entries -= len(current_children)
            current_names = tuple(
                item.name
                for item in current_children
                if item.name not in _DATABASE_SIDECARS or not item.name.endswith("-shm")
            )
            expected_names = _expected_directory_names(path, names, self._prepared.sqlite)
            if current_names != expected_names:
                raise CronCaptureFailed("cron_capture_tree_changed")
            self._budget.check()
        for handle in self._prepared.sqlite.values():
            self._budget.check()
            try:
                handle.assert_current(deadline=self._budget.deadline)
            except SQLiteCaptureRefused as exc:
                raise CronCaptureFailed(exc.code) from exc
            self._budget.check()
        self._budget.check()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: BaseException | None = None
        for handle in tuple(self._prepared.sqlite.values()):
            try:
                handle.close()
            except SQLiteCaptureRefused as exc:
                if error is None:
                    error = exc
        self._prepared.sqlite.clear()
        if error is not None:
            raise CronCaptureFailed("cron_capture_cleanup_failed") from error


@contextmanager
def open_existing_cron_complete_capture(
    *, budget: CronCaptureBudget = CronCaptureBudget()
) -> Iterator[CronStateCapture]:
    if not isinstance(budget, CronCaptureBudget):
        raise TypeError("budget must be a CronCaptureBudget")
    root = _resolve_profile_home()
    deadline = _deadline(budget)
    try:
        from hermes_cron_export_custody import CronCustodyRefused, cron_capture_custody
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        token = set_hermes_home_override(root)
        capture: _LiveCapture | None = None
        try:
            with ExitStack() as stack:
                lease = stack.enter_context(
                    cron_capture_custody(timeout_seconds=float(budget.timeout_seconds))
                )
                try:
                    prepared = _prepare_tree(root, budget, deadline)
                    capture = _LiveCapture(prepared, budget, deadline, lease)
                except CronCaptureUnavailable:
                    raise
                except (CronCaptureRefused, SQLiteCaptureRefused, FileCaptureRefused) as exc:
                    raise CronCaptureRefused(getattr(exc, "code", "cron_capture_refused")) from exc
                # The lease and explicit source handles retain the owner root.  The
                # caller's ambient configuration must be restored before its body
                # runs, especially when jobs.use_cron_store() selected a sibling.
                reset_hermes_home_override(token)
                token = None
                try:
                    yield capture
                except BaseException:
                    raise
                else:
                    try:
                        capture.finish()
                    except SQLiteCaptureRefused as exc:
                        raise CronCaptureFailed(exc.code) from exc
                finally:
                    capture.close()
        except CronCustodyRefused as exc:
            raise CronCaptureRefused(exc.code) from exc
        finally:
            if token is not None:
                reset_hermes_home_override(token)
    except (CronCaptureRefused, CronCaptureUnavailable):
        raise
    except (ImportError, ModuleNotFoundError) as exc:
        raise CronCaptureUnavailable("cron_capture_unavailable") from exc
    except (OSError, TypeError, ValueError) as exc:
        raise CronCaptureRefused("cron_capture_unavailable") from exc


__all__ = ["open_existing_cron_complete_capture"]
