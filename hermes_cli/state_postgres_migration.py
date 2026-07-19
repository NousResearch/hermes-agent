"""Safe, adapter-driven SQLite-to-PostgreSQL state migration orchestration.

This module deliberately has no CLI, configuration, SQLite connection, or
PostgreSQL driver dependency. Callers provide adapters for those concerns.
The source adapter's ``snapshot_via_sqlite_backup`` method is the only
snapshot boundary: its implementation must use ``sqlite3.Connection.backup``;
copying ``state.db-wal`` or ``state.db-shm`` is never an acceptable snapshot.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import base64
import hashlib
import json
import math
import re
import time
import uuid
from typing import Any, Callable, ContextManager, Iterable, Mapping, Optional, Protocol, Sequence


MAX_BATCH_SIZE = 10_000


@dataclass(frozen=True)
class TableSpec:
    """A durable source table and its stable keyset pagination key."""

    name: str
    key_columns: tuple[str, ...]
    optional: bool = False


# Search indexes, SQLite schema bookkeeping, and expired compression leases are
# intentionally absent. Optional Telegram tables only exist after explicit opt-in.
DEFAULT_DURABLE_TABLES: tuple[TableSpec, ...] = (
    TableSpec("sessions", ("id",)),
    TableSpec("messages", ("id",)),
    TableSpec(
        "session_model_usage",
        ("session_id", "model", "billing_provider", "billing_base_url", "billing_mode", "task"),
    ),
    TableSpec("state_meta", ("key",)),
    TableSpec("gateway_routing", ("scope", "session_key")),
    TableSpec("compression_locks", ("session_id",)),
    TableSpec("async_delegations", ("delegation_id",)),
    TableSpec("telegram_dm_topic_mode", ("chat_id",), optional=True),
    TableSpec("telegram_dm_topic_bindings", ("chat_id", "thread_id"), optional=True),
)


class MigrationPhase(str, Enum):
    PREFLIGHT = "preflight"
    DRY_RUN = "dry_run"
    SNAPSHOT = "snapshot"
    COPYING = "copying"
    VERIFYING = "verifying"
    PUBLISHING = "publishing"
    CUTOVER = "cutover"
    COMPLETE = "complete"
    FAILED = "failed"


class MigrationSafetyError(RuntimeError):
    """The source or target is unsafe to migrate."""


class MigrationValidationError(RuntimeError):
    """Copied data does not match the source snapshot."""


@dataclass(frozen=True)
class MigrationRequest:
    """Inputs which contain no target credential or plaintext connection URL."""

    apply: bool = False
    run_id: Optional[str] = None
    batch_size: int = 1_000
    replace: bool = False


@dataclass
class TableVerification:
    source_count: int = 0
    source_digest: str = ""
    target_count: int = 0
    target_digest: str = ""


@dataclass
class MigrationReport:
    """Redacted, resumable migration state suitable for operator reporting."""

    run_id: str
    target_identity: str
    apply: bool
    phase: MigrationPhase = MigrationPhase.PREFLIGHT
    manifest: list[str] = field(default_factory=list)
    tables: dict[str, TableVerification] = field(default_factory=dict)
    failure: Optional[str] = None
    resume_run_id: Optional[str] = None
    published: bool = False
    cutover_complete: bool = False
    recovered_published_run: bool = False

    @property
    def succeeded(self) -> bool:
        return self.phase in (MigrationPhase.DRY_RUN, MigrationPhase.COMPLETE)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["phase"] = self.phase.value
        return result


class SourceSnapshot(Protocol):
    """Read-only view of one SQLite backup snapshot."""

    def available_tables(self) -> Iterable[str]:
        """List tables present in this immutable backup."""

    def columns(self, table: str) -> Sequence[str]:
        """Return durable columns in a stable order."""

    def fetchmany_keyset(
        self,
        table: str,
        key_columns: Sequence[str],
        after_key: Optional[tuple[Any, ...]],
        batch_size: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Fetch at most ``batch_size`` rows after a keyset cursor."""

    def close(self) -> None:
        """Release the temporary snapshot."""


class SQLiteSourceAdapter(Protocol):
    """Preflight and snapshot operations for the existing SQLite state store."""

    def writer_fence(self, run_id: str) -> ContextManager[None]:
        """Quiesce all state writers until the migration leaves the context."""

    def available_tables(self) -> Iterable[str]:
        """List source tables without mutating the source."""

    def sqlite_integrity_errors(self) -> Sequence[str]:
        """Return failed SQLite integrity-check results, if any."""

    def sqlite_foreign_key_violations(self) -> Sequence[str]:
        """Return SQLite foreign-key violations, if any."""

    def active_writer_reasons(self) -> Sequence[str]:
        """Return active process/lease reasons which make a snapshot unsafe."""

    def active_compression_leases(self, now: float) -> Sequence[str]:
        """Return non-expired compression leases."""

    def active_delegations(self) -> Sequence[str]:
        """Return in-flight async delegations."""

    def active_handoffs(self) -> Sequence[str]:
        """Return pending or running session handoffs."""

    def snapshot_via_sqlite_backup(self, run_id: str) -> SourceSnapshot:
        """Create a consistent snapshot with sqlite3.Connection.backup()."""


class PostgreSQLTargetAdapter(Protocol):
    """Target operations; implementations own database connectivity and SQL."""

    @property
    def redacted_identity(self) -> str:
        """A credential-free target label for reports and diagnostics."""

    def publish_schema_is_empty(self) -> bool:
        """Return whether the final target schema has no durable state."""

    def acquire_advisory_lock(self, run_id: str) -> None:
        """Acquire a migration-wide advisory lock."""

    def release_advisory_lock(self, run_id: str) -> None:
        """Release the migration-wide advisory lock."""

    def create_or_resume_staging(
        self, run_id: str, manifest: Sequence[TableSpec]
    ) -> str:
        """Create a fresh isolated staging schema for this run."""

    def begin_session_parent_link_copy(self, staging_schema: str) -> None:
        """Defer session self-FKs or stage parent links before session batches."""

    def copy_rows(
        self, staging_schema: str, table: TableSpec, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Idempotently copy one bounded batch into the staging schema."""

    def finalize_session_parent_links(self, staging_schema: str) -> None:
        """Apply/validate staged parent links after every session row is copied."""

    def reset_identity(
        self, staging_schema: str, table: TableSpec, identity_column: str
    ) -> None:
        """Advance an identity sequence after explicit IDs are imported."""

    def fetchmany_keyset(
        self,
        staging_schema: str,
        table: TableSpec,
        columns: Sequence[str],
        after_key: Optional[tuple[Any, ...]],
        batch_size: int,
    ) -> Sequence[Mapping[str, Any]]:
        """Fetch target rows in the same key order without loading a table."""

    def read_row(
        self,
        staging_schema: str,
        table: TableSpec,
        key: tuple[Any, ...],
        columns: Sequence[str],
    ) -> Optional[Mapping[str, Any]]:
        """Read one representative row by its durable primary key."""

    def validate_foreign_keys(self, staging_schema: str) -> Sequence[str]:
        """Return FK violations discovered in staging."""

    def validate_session_lineage(self, staging_schema: str) -> Sequence[str]:
        """Return dangling/cyclic session-parent lineage violations."""

    def atomic_publish(
        self,
        staging_schema: str,
        run_id: str,
        verified_source: Mapping[str, Any],
    ) -> None:
        """Publish and persist ``verified_source`` in the same transaction."""

    def published_run_report(self, run_id: str) -> Optional[Mapping[str, Any]]:
        """Return the authoritative report for an already-published run."""

    def record_success(self, report: Mapping[str, Any]) -> None:
        """Persist a redacted completed-run report."""

    def record_failure(self, report: Mapping[str, Any]) -> None:
        """Persist a redacted failed-run report for safe resume."""


CutoverCallback = Callable[[MigrationReport], None]
Clock = Callable[[], float]


class _Digest:
    def __init__(self, table: str, columns: Sequence[str]) -> None:
        self._table = table
        self._columns = tuple(columns)
        self._hash = hashlib.sha256()
        self.count = 0

    def update(self, row: Mapping[str, Any]) -> None:
        payload = {
            "table": self._table,
            "row": [[column, _canonical_value(row.get(column))] for column in self._columns],
        }
        self._hash.update(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        self._hash.update(b"\n")
        self.count += 1

    @property
    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def _canonical_value(value: Any) -> Any:
    """Normalize common DB-driver value differences into stable digest input."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"$float": repr(value)}
        return {"$number": _canonical_number(value)}
    if isinstance(value, Decimal):
        return {"$number": _canonical_number(value)}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, Mapping):
        return {
            "$mapping": [
                [str(key), _canonical_value(item)]
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            ]
        }
    if isinstance(value, (list, tuple)):
        return {"$sequence": [_canonical_value(item) for item in value]}
    if hasattr(value, "isoformat"):
        return {"$time": value.isoformat()}
    return {"$repr": str(value)}


def _canonical_number(value: Any) -> str:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return repr(value)
    if not decimal.is_finite():
        return repr(value)
    normalized = decimal.normalize()
    text = format(normalized, "f")
    return "0" if text in ("", "-0") else text


_URL_CREDENTIALS = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.-]*://)(?:[^@\r\n]*)@"
)
_SECRET_NAME = (
    r"(?:pass(?:word)?|pwd|pg[ _-]?(?:password|user)|"
    r"api(?:[ _-]?key)?|access(?:[ _-]?(?:key|token))?|"
    r"secret(?:[ _-]?(?:key|token))?|client[ _-]?secret|"
    r"private[ _-]?key|(?:auth|bearer|refresh|session)?[ _-]?token)"
)
_SECRET_ASSIGNMENT = re.compile(
    rf"""(?isx)
    (?P<name>\b{_SECRET_NAME}\b)
    (?P<separator>\s*(?:=|:)\s*)
    (?P<value>
        "(?:\\.|[^"\\])*"
        | '(?:\\.|[^'\\])*'
        | .*?(?=
            \s+(?:{_SECRET_NAME}|host|port|user|dbname|database|sslmode)\b\s*(?:=|:)
            | [;,&\r\n]
            | $
        )
    )
    """,
)
_AUTHORIZATION_HEADER = re.compile(
    r"(?im)(\b(?:proxy-)?authorization\s*:\s*)(?:bearer|basic)\s+[^\r\n]+"
)


def _redact_text(value: str) -> str:
    """Conservatively scrub URL userinfo and connection-style secret values."""
    value = _URL_CREDENTIALS.sub(r"\1***@", value)
    value = _AUTHORIZATION_HEADER.sub(r"\1***", value)
    return _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group('name')}{match.group('separator')}***",
        value,
    )


def _key_for(row: Mapping[str, Any], table: TableSpec) -> tuple[Any, ...]:
    try:
        return tuple(row[column] for column in table.key_columns)
    except KeyError as exc:
        raise MigrationValidationError(
            f"{table.name} row is missing key column {exc.args[0]!r}"
        ) from exc


def _is_copyable(table: TableSpec, row: Mapping[str, Any], now: float) -> bool:
    # Expired lock records are safe to omit and must not revive stale ownership.
    if table.name != "compression_locks":
        return True
    expires_at = row.get("expires_at")
    return expires_at is not None and float(expires_at) >= now


class StatePostgresMigration:
    """Orchestrate a no-driver, backend-neutral migration with strict ordering."""

    def __init__(
        self,
        source: SQLiteSourceAdapter,
        target: PostgreSQLTargetAdapter,
        *,
        cutover: Optional[CutoverCallback] = None,
        manifest: Sequence[TableSpec] = DEFAULT_DURABLE_TABLES,
        clock: Clock = time.time,
    ) -> None:
        self._source = source
        self._target = target
        self._cutover = cutover
        self._manifest = tuple(manifest)
        self._clock = clock

    def run(self, request: MigrationRequest = MigrationRequest()) -> MigrationReport:
        """Run preflight, optional copy, verification, publication, then cutover."""
        self._validate_request(request)
        run_id = request.run_id or uuid.uuid4().hex
        report = MigrationReport(
            run_id=run_id,
            target_identity=_redact_text(self._target.redacted_identity),
            apply=request.apply,
            resume_run_id=run_id if request.apply else None,
        )

        try:
            # The fence starts before preflight and remains held through target
            # publication, config cutover, report persistence, and lock release.
            with self._source.writer_fence(run_id):
                self._run_while_fenced(request, report)
        except Exception as exc:
            self._mark_failure(report, exc)
            if request.apply:
                self._record_failure_safely(report)
        return report

    def _run_while_fenced(
        self, request: MigrationRequest, report: MigrationReport
    ) -> None:
        snapshot: Optional[SourceSnapshot] = None
        lock_held = False
        failure_recorded = False
        try:
            manifest = self._preflight_source()
            report.manifest = [table.name for table in manifest]
            if not request.apply:
                published = self._target.published_run_report(report.run_id)
                if published is not None:
                    self._hydrate_published_run(
                        report, self._published_source_evidence(published)
                    )
                elif not self._target.publish_schema_is_empty():
                    raise MigrationSafetyError(
                        "target publish schema is not empty; destructive replacement is refused"
                    )
                report.phase = MigrationPhase.DRY_RUN
                return

            self._target.acquire_advisory_lock(report.run_id)
            lock_held = True

            # A writer can appear between source preflight and fence/advisory-lock
            # admission. Check again before taking the immutable SQLite backup.
            self._ensure_source_admission()
            published = self._target.published_run_report(report.run_id)
            if published is not None:
                verified_source = self._published_source_evidence(published)
                self._hydrate_published_run(report, verified_source)
                report.phase = MigrationPhase.SNAPSHOT
                snapshot = self._source.snapshot_via_sqlite_backup(report.run_id)
                recovery_manifest = self._resolve_manifest(
                    self._manifest, snapshot.available_tables()
                )
                self._verify_recovery_source(
                    snapshot,
                    recovery_manifest,
                    verified_source,
                    request.batch_size,
                )
                self._complete_cutover(report)
                return
            if not self._target.publish_schema_is_empty():
                raise MigrationSafetyError(
                    "target publish schema is not empty; destructive replacement is refused"
                )

            report.phase = MigrationPhase.SNAPSHOT
            snapshot = self._source.snapshot_via_sqlite_backup(report.run_id)

            # Re-read table availability from the immutable backup, not live SQLite.
            manifest = self._resolve_manifest(manifest, snapshot.available_tables())
            report.manifest = [table.name for table in manifest]
            staging_schema = self._target.create_or_resume_staging(report.run_id, manifest)
            self._target.begin_session_parent_link_copy(staging_schema)

            report.phase = MigrationPhase.COPYING
            probes, columns_by_table = self._copy_snapshot(
                snapshot, staging_schema, manifest, request.batch_size, report
            )
            self._target.finalize_session_parent_links(staging_schema)
            self._target.reset_identity(staging_schema, TableSpec("messages", ("id",)), "id")

            report.phase = MigrationPhase.VERIFYING
            self._verify_target(
                staging_schema,
                manifest,
                report,
                probes,
                columns_by_table,
                request.batch_size,
            )

            report.phase = MigrationPhase.PUBLISHING
            self._target.atomic_publish(
                staging_schema,
                report.run_id,
                self._verified_source_evidence(report),
            )
            report.published = True
            self._complete_cutover(report)
        except Exception as exc:
            self._mark_failure(report, exc)
            if request.apply and lock_held:
                self._record_failure_safely(report)
                failure_recorded = True
        finally:
            if snapshot is not None:
                try:
                    snapshot.close()
                except Exception as exc:
                    self._mark_failure(report, exc)
                    if request.apply and lock_held and not failure_recorded:
                        self._record_failure_safely(report)
                        failure_recorded = True
            if lock_held:
                try:
                    self._target.release_advisory_lock(report.run_id)
                except Exception as exc:
                    self._mark_failure(report, exc)
                    if request.apply and not failure_recorded:
                        self._record_failure_safely(report)

    def _complete_cutover(self, report: MigrationReport) -> None:
        """Run an idempotent config cutover after the target publish marker exists."""
        report.phase = MigrationPhase.CUTOVER
        if self._cutover is not None:
            self._cutover(report)
        report.cutover_complete = True
        report.phase = MigrationPhase.COMPLETE
        self._target.record_success(report.to_dict())

    @staticmethod
    def _mark_failure(report: MigrationReport, exc: Exception) -> None:
        report.phase = MigrationPhase.FAILED
        if report.failure is None:
            report.failure = _redact_text(str(exc)) or exc.__class__.__name__

    def _record_failure_safely(self, report: MigrationReport) -> None:
        try:
            self._target.record_failure(report.to_dict())
        except Exception:
            # Preserve the original migration failure; a later retry can
            # reconcile from the publish marker when one was committed.
            pass

    def _hydrate_published_run(
        self, report: MigrationReport, verified_source: Mapping[str, Any]
    ) -> None:
        """Restore a verified marker without target recopying or republishing."""
        report.published = True
        report.recovered_published_run = True
        manifest = verified_source.get("manifest")
        if isinstance(manifest, list) and all(isinstance(item, str) for item in manifest):
            report.manifest = list(manifest)
        saved_tables = verified_source.get("tables")
        if not isinstance(saved_tables, Mapping):
            return
        for name, saved in saved_tables.items():
            if not isinstance(name, str) or not isinstance(saved, Mapping):
                continue
            report.tables[name] = TableVerification(
                source_count=int(saved.get("source_count", 0)),
                source_digest=str(saved.get("source_digest", "")),
                target_count=int(saved.get("target_count", 0)),
                target_digest=str(saved.get("target_digest", "")),
            )

    def _verified_source_evidence(self, report: MigrationReport) -> dict[str, Any]:
        """Build evidence that atomic_publish must commit with the publish marker."""
        tables: dict[str, dict[str, Any]] = {}
        for table_name in report.manifest:
            verification = report.tables.get(table_name)
            if verification is None:
                raise MigrationValidationError(
                    f"{table_name} is missing verified source evidence"
                )
            if (
                verification.source_count != verification.target_count
                or verification.source_digest != verification.target_digest
            ):
                raise MigrationValidationError(
                    f"{table_name} was not fully verified before publication"
                )
            tables[table_name] = {
                "source_count": verification.source_count,
                "source_digest": verification.source_digest,
                "target_count": verification.target_count,
                "target_digest": verification.target_digest,
            }
        return {
            "run_id": report.run_id,
            "manifest": list(report.manifest),
            "tables": tables,
        }

    def _published_source_evidence(
        self, published: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Validate the immutable source evidence attached to a publish marker."""
        evidence = published.get("verified_source")
        if not isinstance(evidence, Mapping):
            raise MigrationSafetyError(
                "published run lacks durable verified source evidence; "
                "refuse cutover recovery"
            )
        manifest = evidence.get("manifest")
        tables = evidence.get("tables")
        known_tables = {table.name for table in self._manifest}
        if (
            not isinstance(manifest, list)
            or not manifest
            or len(set(manifest)) != len(manifest)
            or not all(isinstance(name, str) and name in known_tables for name in manifest)
            or not isinstance(tables, Mapping)
            or set(tables) != set(manifest)
        ):
            raise MigrationSafetyError(
                "published run has malformed verified source evidence; "
                "refuse cutover recovery"
            )
        for name in manifest:
            table = tables[name]
            if not isinstance(table, Mapping):
                raise MigrationSafetyError(
                    "published run has malformed verified source table evidence"
                )
            count = table.get("source_count")
            digest = table.get("source_digest")
            target_count = table.get("target_count")
            target_digest = table.get("target_digest")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                or isinstance(target_count, bool)
                or not isinstance(target_count, int)
                or target_count != count
                or not isinstance(digest, str)
                or not isinstance(target_digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or target_digest != digest
            ):
                raise MigrationSafetyError(
                    "published run has unverifiable source digest evidence"
                )
        return evidence

    def _verify_recovery_source(
        self,
        snapshot: SourceSnapshot,
        manifest: Sequence[TableSpec],
        verified_source: Mapping[str, Any],
        batch_size: int,
    ) -> None:
        """Require the fenced current source to equal the published source."""
        expected_manifest = verified_source["manifest"]
        current_manifest = [table.name for table in manifest]
        if set(current_manifest) != set(expected_manifest):
            raise MigrationSafetyError(
                "current source manifest differs from verified publish evidence; "
                "require a new migration"
            )
        current = self._fingerprint_snapshot(snapshot, manifest, batch_size)
        expected_tables = verified_source["tables"]
        for table_name in expected_manifest:
            actual = current[table_name]
            expected = expected_tables[table_name]
            if (
                actual.source_count != expected["source_count"]
                or actual.source_digest != expected["source_digest"]
            ):
                raise MigrationSafetyError(
                    f"current source differs from verified publish evidence for "
                    f"{table_name}; require a new migration"
                )

    def _fingerprint_snapshot(
        self,
        snapshot: SourceSnapshot,
        manifest: Sequence[TableSpec],
        batch_size: int,
    ) -> dict[str, TableVerification]:
        """Stream the source snapshot into counts/digests without target writes."""
        now = self._clock()
        result: dict[str, TableVerification] = {}
        for table in manifest:
            columns = tuple(snapshot.columns(table.name))
            if not columns:
                raise MigrationValidationError(f"{table.name} has no readable columns")
            digest = _Digest(table.name, columns)
            after_key: Optional[tuple[Any, ...]] = None
            while True:
                rows = snapshot.fetchmany_keyset(
                    table.name, table.key_columns, after_key, batch_size
                )
                if len(rows) > batch_size:
                    raise MigrationValidationError(
                        f"{table.name} source adapter returned more than batch_size rows"
                    )
                if not rows:
                    break
                last_key = _key_for(rows[-1], table)
                for row in rows:
                    if _is_copyable(table, row, now):
                        digest.update(row)
                if after_key is not None and last_key <= after_key:
                    raise MigrationValidationError(
                        f"{table.name} source keyset did not advance from {after_key!r}"
                    )
                after_key = last_key
            result[table.name] = TableVerification(
                source_count=digest.count,
                source_digest=digest.hexdigest,
                target_count=digest.count,
                target_digest=digest.hexdigest,
            )
        return result

    def _validate_request(self, request: MigrationRequest) -> None:
        if request.replace:
            raise ValueError("destructive --replace is not supported for state migration")
        if not 1 <= request.batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")

    def _preflight_source(self) -> tuple[TableSpec, ...]:
        failures: list[str] = []
        failures.extend(f"SQLite integrity: {item}" for item in self._source.sqlite_integrity_errors())
        failures.extend(
            f"SQLite foreign key: {item}" for item in self._source.sqlite_foreign_key_violations()
        )
        failures.extend(self._source_admission_failures())
        if failures:
            raise MigrationSafetyError("; ".join(failures))
        return self._resolve_manifest(self._manifest, self._source.available_tables())

    def _ensure_source_admission(self) -> None:
        failures = self._source_admission_failures()
        if failures:
            raise MigrationSafetyError("; ".join(failures))

    def _source_admission_failures(self) -> list[str]:
        now = self._clock()
        failures = [f"active writer: {item}" for item in self._source.active_writer_reasons()]
        failures.extend(
            f"active compression lease: {item}"
            for item in self._source.active_compression_leases(now)
        )
        failures.extend(
            f"active delegation: {item}" for item in self._source.active_delegations()
        )
        failures.extend(f"active handoff: {item}" for item in self._source.active_handoffs())
        return failures

    @staticmethod
    def _resolve_manifest(
        manifest: Sequence[TableSpec], available_tables: Iterable[str]
    ) -> tuple[TableSpec, ...]:
        available = set(available_tables)
        selected: list[TableSpec] = []
        missing: list[str] = []
        for table in manifest:
            if table.name in available:
                selected.append(table)
            elif not table.optional:
                missing.append(table.name)
        if missing:
            raise MigrationSafetyError(
                "source snapshot is missing durable table(s): " + ", ".join(sorted(missing))
            )
        return tuple(selected)

    def _copy_snapshot(
        self,
        snapshot: SourceSnapshot,
        staging_schema: str,
        manifest: Sequence[TableSpec],
        batch_size: int,
        report: MigrationReport,
    ) -> tuple[
        dict[str, list[tuple[tuple[Any, ...], Mapping[str, Any], tuple[str, ...]]]],
        dict[str, tuple[str, ...]],
    ]:
        probes: dict[str, list[tuple[tuple[Any, ...], Mapping[str, Any], tuple[str, ...]]]] = {}
        columns_by_table: dict[str, tuple[str, ...]] = {}
        now = self._clock()
        for table in manifest:
            columns = tuple(snapshot.columns(table.name))
            if not columns:
                raise MigrationValidationError(f"{table.name} has no readable columns")
            columns_by_table[table.name] = columns
            digest = _Digest(table.name, columns)
            after_key: Optional[tuple[Any, ...]] = None
            table_probes: list[tuple[tuple[Any, ...], Mapping[str, Any], tuple[str, ...]]] = []
            while True:
                rows = snapshot.fetchmany_keyset(table.name, table.key_columns, after_key, batch_size)
                if len(rows) > batch_size:
                    raise MigrationValidationError(
                        f"{table.name} source adapter returned more than batch_size rows"
                    )
                if not rows:
                    break
                last_key = _key_for(rows[-1], table)
                copied_rows = [row for row in rows if _is_copyable(table, row, now)]
                for row in copied_rows:
                    digest.update(row)
                    if not table_probes:
                        table_probes.append((_key_for(row, table), dict(row), columns))
                    elif len(table_probes) == 1:
                        table_probes.append((_key_for(row, table), dict(row), columns))
                    else:
                        table_probes[-1] = (_key_for(row, table), dict(row), columns)
                if copied_rows:
                    self._target.copy_rows(staging_schema, table, copied_rows)
                if after_key is not None and last_key <= after_key:
                    raise MigrationValidationError(
                        f"{table.name} source keyset did not advance from {after_key!r}"
                    )
                after_key = last_key
            report.tables[table.name] = TableVerification(
                source_count=digest.count, source_digest=digest.hexdigest
            )
            probes[table.name] = table_probes
        return probes, columns_by_table

    def _verify_target(
        self,
        staging_schema: str,
        manifest: Sequence[TableSpec],
        report: MigrationReport,
        probes: Mapping[str, Sequence[tuple[tuple[Any, ...], Mapping[str, Any], tuple[str, ...]]]],
        columns_by_table: Mapping[str, tuple[str, ...]],
        batch_size: int,
    ) -> None:
        fk_violations = self._target.validate_foreign_keys(staging_schema)
        if fk_violations:
            raise MigrationValidationError("target foreign keys: " + "; ".join(fk_violations))
        lineage_violations = self._target.validate_session_lineage(staging_schema)
        if lineage_violations:
            raise MigrationValidationError(
                "target session lineage: " + "; ".join(lineage_violations)
            )
        for table in manifest:
            source_result = report.tables[table.name]
            column_names = columns_by_table[table.name]
            digest = _Digest(table.name, column_names)
            after_key: Optional[tuple[Any, ...]] = None
            while True:
                rows = self._target.fetchmany_keyset(
                    staging_schema, table, column_names, after_key, batch_size
                )
                if len(rows) > batch_size:
                    raise MigrationValidationError(
                        f"{table.name} target adapter returned more than batch_size rows"
                    )
                if not rows:
                    break
                last_key = _key_for(rows[-1], table)
                for row in rows:
                    digest.update(row)
                if after_key is not None and last_key <= after_key:
                    raise MigrationValidationError(
                        f"{table.name} target keyset did not advance from {after_key!r}"
                    )
                after_key = last_key
            source_result.target_count = digest.count
            source_result.target_digest = digest.hexdigest
            if (
                source_result.source_count != source_result.target_count
                or source_result.source_digest != source_result.target_digest
            ):
                raise MigrationValidationError(
                    f"{table.name} digest mismatch: source count/digest "
                    f"{source_result.source_count}/{source_result.source_digest}, target "
                    f"{source_result.target_count}/{source_result.target_digest}"
                )
            for key, source_row, probe_columns in probes[table.name]:
                target_row = self._target.read_row(
                    staging_schema, table, key, probe_columns
                )
                if target_row is None:
                    raise MigrationValidationError(
                        f"{table.name} representative row {key!r} is missing"
                    )
                source_digest = _Digest(table.name, probe_columns)
                target_digest = _Digest(table.name, probe_columns)
                source_digest.update(source_row)
                target_digest.update(target_row)
                if source_digest.hexdigest != target_digest.hexdigest:
                    raise MigrationValidationError(
                        f"{table.name} representative row {key!r} differs"
                    )
