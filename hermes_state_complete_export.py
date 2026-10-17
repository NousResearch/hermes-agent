"""Fenced, lossless export of the reviewed Hermes core state tables.

This module owns the official core component only.  It does not own Desktop/Main
lifecycle, addon filesystem stores, or publication.  A supported addon command
may consume this API after it has joined its own sources and may provide a
source-controlled trusted extension descriptor for namespaced core relations.

The capture takes the live ``SessionDB._lock`` and starts ``BEGIN IMMEDIATE`` on
that same writer connection before it validates or reads anything.  The lock and
transaction stay held until the yielded capture has finished its stream.  That
is deliberately a narrow waist: a caller cannot supply an owner-stopped flag,
an authority receipt, a policy path, a source connection, or an output path.
"""

from __future__ import annotations

import atexit
import base64
import json
import math
import numbers
import sqlite3
import struct
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, Protocol, runtime_checkable

from hermes_state_complete_export_policy import (
    CURRENT_WRITER_SOURCE_ID,
    DERIVED_INTERNAL,
    DERIVED_SQLITE_TABLES,
    FIELD_POLICY_VERSION,
    MIXED_REQUIRES_PROJECTION,
    OFFICIAL_SCHEMA_VERSION,
    OFFICIAL_SOURCE_ID,
    ModelConfigProjectionRefused,
    SCHEMA32_SCHEMA_VERSION,
    SCHEMA_POLICIES,
    VersionedSchemaPolicy,
    extension_disposition,
    _read_schema_version,
    policy_table_names,
    project_model_config,
    schema_policy_for_version,
    table_policy_for_mapping,
    validate_schema,
)

CORE_CAPTURE_API_VERSION = "agc1663-core-state-capture/v1"
_PAGE_SIZE = 256
_JSON_LINE_SUFFIX = b"\n"

try:
    # Bind the core implementation while this module is loaded.  The addon may
    # legitimately replace ``hermes_state.SessionDB`` for its own migrations;
    # this factory must never inherit that mutable binding.
    from hermes_state import SessionDB as _BOUND_OFFICIAL_SESSION_DB
except ImportError:  # pragma: no cover - stripped installs cannot use the factory
    _BOUND_OFFICIAL_SESSION_DB = None


def _official_session_db_class() -> type:
    session_db = _BOUND_OFFICIAL_SESSION_DB
    if (
        session_db is None
        or session_db.__module__ != "hermes_state"
        or session_db.__qualname__ != "SessionDB"
    ):
        raise CoreCaptureRefused(
            "complete capture requires the immutable official SessionDB implementation"
        )
    return session_db


class CoreCaptureError(RuntimeError):
    """Base class for a capture that cannot produce a complete stream."""


class CoreCaptureRefused(CoreCaptureError):
    """The schema, extension registry, budget, or source is unsafe."""


class CoreCaptureUnavailable(CoreCaptureError):
    """The safe existing-profile factory is not available in this component yet."""


class CoreCaptureFailed(CoreCaptureError):
    """A capture failed after opening and must close without publishing."""


class FieldDisposition(StrEnum):
    BUSINESS_PAYLOAD = "business_payload"
    BUSINESS_METADATA = "business_metadata"
    CONTROL_RECOVERY = "control_recovery"
    DELIVERY_METADATA = "delivery_metadata"
    SECURITY_IDENTITY = "security_identity"
    MIXED_REQUIRES_PROJECTION = "mixed_requires_projection"
    DERIVED_INTERNAL = "derived_internal"


@dataclass(frozen=True)
class CaptureBudget:
    """Bound rows, bytes, and the lifetime of one core capture."""

    max_rows: int = 5_000_000
    max_bytes: int = 512 * 1024 * 1024
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name, value in (("max_rows", self.max_rows), ("max_bytes", self.max_bytes)):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
                raise ValueError(f"core capture {name} must be a positive integer")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, numbers.Real)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("core capture timeout_seconds must be a finite positive real")


@dataclass(frozen=True)
class CoreTableExtension:
    """A table declaration returned by trusted source-controlled extension code."""

    relation_name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class CoreExtensionDescriptor:
    """Static declarations for namespaced relations owned outside official core.

    This nominal descriptor remains the validate-only registration shape.  A
    source-owned typed descriptor must be consumed through the live extension
    reader before a capture with registered row tables can complete; merely
    providing column names never authorizes the core sink to dump them.
    """

    source_name: str
    tables: tuple[CoreTableExtension, ...] = ()
    relations: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ExtensionRegistry:
    """The nominal registrations captured before the live SQLite fence."""

    tables: dict[str, tuple[str, ...]]
    relations: tuple[str, ...]
    sources: tuple[str, ...]
    source_tables: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]]

    def __iter__(self) -> Iterator[object]:
        # Keep the old private helper's three-value unpacking shape for callers
        # that only need validate-only registration data.
        yield self.tables
        yield self.relations
        yield self.sources


@dataclass(frozen=True)
class _ExtensionFieldSpec:
    name: str
    disposition: FieldDisposition
    preserve_raw_bytes: bool


@dataclass(frozen=True)
class _ExtensionRelationSpec:
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    label: str


@dataclass(frozen=True)
class _ExtensionTableSpec:
    relation_name: str
    fields: tuple[_ExtensionFieldSpec, ...]
    primary_key: tuple[str, ...]
    relations: tuple[_ExtensionRelationSpec, ...]


@runtime_checkable
class TrustedCoreExtension(Protocol):
    """Source-controlled extension provider, not a caller-supplied schema object."""

    def describe_core_extension(self) -> CoreExtensionDescriptor:
        """Return the provider's compiled, reviewed relation declarations."""
        ...


@dataclass(frozen=True)
class CoreCaptureManifest:
    """Stable provenance for a core stream; it is not an authority receipt."""

    api_version: str = CORE_CAPTURE_API_VERSION
    field_policy_version: str = FIELD_POLICY_VERSION
    official_schema_source: str = OFFICIAL_SOURCE_ID
    policy_writer_provenance: str = CURRENT_WRITER_SOURCE_ID
    schema_version: int = OFFICIAL_SCHEMA_VERSION
    core_tables: tuple[str, ...] = ()
    optional_tables: tuple[str, ...] = ()
    extension_sources: tuple[str, ...] = ()
    validated_extension_tables: tuple[str, ...] = ()
    declared_relations: tuple[str, ...] = ()
    derived_relations: tuple[str, ...] = ()
    schema_census_sha256: str = ""
    field_policy_review_sha256: str = ""


@dataclass(frozen=True)
class CoreFieldValue:
    """One field with its reviewed disposition and lossless scalar/raw value."""

    column: str
    disposition: FieldDisposition
    value: Any
    raw_bytes: bytes | None = None


@dataclass(frozen=True)
class CoreRelation:
    relation: str
    source_key: str
    target_key: str


@dataclass(frozen=True)
class CoreStateRecord:
    """One bounded core row or separate derived-evidence record."""

    table: str
    primary_key: tuple[Any, ...]
    fields: tuple[CoreFieldValue, ...]
    relations: tuple[CoreRelation, ...] = ()
    evidence_class: str = "core"


class CoreStateCapture(Protocol):
    """Capture yielded while the canonical internal database fence is held."""

    @property
    def manifest(self) -> CoreCaptureManifest:
        ...

    def iter_records(self) -> Iterable[CoreStateRecord]:
        """Yield every reviewed core row/edge exactly once, in stable order."""
        ...

    def iter_extension_records(
        self,
        *,
        descriptor: object,
        budget: object,
    ) -> Iterable[CoreStateRecord]:
        """Yield one registered extension's rows under this live Core fence."""
        ...

    def write_to(self, stream: BinaryIO) -> CoreCaptureReceipt:
        """Write the canonical bounded stream to a caller-owned temporary stream."""
        ...


@dataclass(frozen=True)
class CoreCaptureReceipt:
    """Receipt returned after the core stream was fully written."""

    manifest: CoreCaptureManifest
    record_count: int
    byte_count: int


def _extension_registry(
    extensions: Iterable[TrustedCoreExtension],
    *,
    core_table_names: Iterable[str] | None = None,
) -> _ExtensionRegistry:
    """Materialize only source-controlled extension declarations for validation."""
    tables: dict[str, tuple[str, ...]] = {}
    relations: set[str] = set()
    sources: list[str] = []
    source_tables: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {}
    if core_table_names is None:
        core_table_names = tuple(
            name
            for schema_policy in SCHEMA_POLICIES.values()
            for name in policy_table_names(table_policies=schema_policy.table_policies)
        )
    core_names = set(core_table_names) | set(DERIVED_SQLITE_TABLES)
    for provider in extensions:
        descriptor = provider.describe_core_extension()
        if not isinstance(descriptor, CoreExtensionDescriptor):
            raise CoreCaptureRefused("trusted core extension returned an invalid descriptor")
        if (
            not isinstance(descriptor.source_name, str)
            or not descriptor.source_name
            or descriptor.source_name in sources
        ):
            raise CoreCaptureRefused("trusted core extension source must be unique and non-empty")
        sources.append(descriptor.source_name)
        registered_tables: list[tuple[str, tuple[str, ...]]] = []
        for table in descriptor.tables:
            if (
                not isinstance(table.relation_name, str)
                or not table.relation_name
                or table.relation_name in core_names
                or table.relation_name in tables
            ):
                raise CoreCaptureRefused(
                    "trusted extension relation collides with an existing relation: "
                    f"{table.relation_name!r}"
                )
            if (
                not isinstance(table.columns, tuple)
                or not table.columns
                or any(not isinstance(column, str) or not column for column in table.columns)
                or len(set(table.columns)) != len(table.columns)
            ):
                raise CoreCaptureRefused(
                    f"trusted extension relation has no unique reviewed columns: {table.relation_name}"
                )
            if table.relation_name in relations:
                raise CoreCaptureRefused(f"extension relation is both a table and relation: {table.relation_name}")
            columns = tuple(table.columns)
            tables[table.relation_name] = columns
            registered_tables.append((table.relation_name, columns))
        for relation in descriptor.relations:
            if (
                not isinstance(relation, str)
                or not relation
                or relation in core_names
                or relation in relations
                or relation in tables
            ):
                raise CoreCaptureRefused(f"duplicate or invalid trusted extension relation: {relation!r}")
            relations.add(relation)
        source_tables[descriptor.source_name] = tuple(registered_tables)
    return _ExtensionRegistry(
        tables=tables,
        relations=tuple(sorted(relations)),
        sources=tuple(sources),
        source_tables=source_tables,
    )


def _quote_identifier(identifier: str) -> str:
    if not identifier:
        raise CoreCaptureRefused("empty SQLite identifier")
    return '"' + identifier.replace('"', '""') + '"'


def _source_snapshot(session_db: Any) -> tuple[Any, tuple[tuple[str, Any], ...], int]:
    sidecars = getattr(session_db, "_db_sidecar_identity", {}) or {}
    return (
        getattr(session_db, "_db_file_identity", None),
        tuple(sorted((str(key), value) for key, value in sidecars.items())),
        int(getattr(session_db, "_db_file_application_id", 0) or 0),
    )


def _check_source_current(session_db: Any, conn: sqlite3.Connection, expected: tuple[Any, ...]) -> None:
    """Reuse SessionDB's real replacement/WAL-generation guard while its lock is held."""
    if getattr(session_db, "_conn", None) is not conn:
        raise CoreCaptureRefused("SessionDB connection changed during complete capture")
    corrupt_checker = getattr(session_db, "_raise_if_db_corrupt", None)
    if callable(corrupt_checker):
        try:
            corrupt_checker(storage=True)
        except Exception as exc:
            raise CoreCaptureRefused("SessionDB is quarantined after a storage-corruption observation") from exc
    checker = getattr(session_db, "_raise_if_db_replaced", None)
    if not callable(checker):
        raise CoreCaptureRefused("complete capture requires the canonical SessionDB generation guard")
    try:
        checker()
    except Exception as exc:
        raise CoreCaptureRefused("SessionDB source generation is no longer current") from exc
    if _source_snapshot(session_db) != expected:
        raise CoreCaptureRefused("SessionDB source generation changed during complete capture")


def _check_deadline(deadline: float, message: str = "complete capture exceeded its time budget") -> None:
    if time.monotonic() > deadline:
        raise CoreCaptureRefused(message)


def _count_relation_with_deadline(
    conn: sqlite3.Connection,
    relation: str,
    deadline: float | Any,
) -> int:
    """Run one derived count with a cooperative SQLite VM deadline."""
    def current_deadline() -> float:
        return deadline() if callable(deadline) else deadline

    def progress() -> int:
        return int(time.monotonic() > current_deadline())

    try:
        conn.set_progress_handler(progress, 1_000)
        row = conn.execute(f"SELECT COUNT(*) FROM {_quote_identifier(relation)}").fetchone()
        _check_deadline(current_deadline())
        if not row or isinstance(row[0], bool) or not isinstance(row[0], int):
            raise CoreCaptureFailed(f"derived relation count is not an integer: {relation}")
        return row[0]
    except sqlite3.Error as exc:
        if time.monotonic() > current_deadline():
            raise CoreCaptureRefused("complete capture exceeded its time budget") from exc
        raise CoreCaptureFailed(f"could not count derived relation {relation}") from exc
    finally:
        try:
            conn.set_progress_handler(None, 0)
        except sqlite3.Error as exc:
            raise CoreCaptureFailed("could not clear the derived-count progress handler") from exc


def _begin_immediate(conn: sqlite3.Connection, deadline: float) -> None:
    timeout_row = conn.execute("PRAGMA busy_timeout").fetchone()
    if not timeout_row or timeout_row[0] is None:
        raise CoreCaptureFailed("could not observe the SessionDB busy timeout")
    try:
        previous_timeout = int(timeout_row[0])
    except (TypeError, ValueError) as exc:
        raise CoreCaptureFailed("SessionDB busy timeout is not an integer") from exc
    # sqlite3's connection timeout installs a busy handler that can sleep past the
    # capture deadline.  Disable it for the bounded BEGIN loop, then restore the
    # exact caller-owned setting even when acquisition fails.
    conn.execute("PRAGMA busy_timeout=0")
    try:
        while True:
            _check_deadline(
                deadline,
                "SessionDB writer lock did not become available within the capture budget",
            )
            try:
                conn.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if "locked" not in message and "busy" not in message:
                    raise CoreCaptureFailed("could not begin the fenced SessionDB transaction") from exc
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CoreCaptureRefused(
                        "SessionDB writer lock did not become available within the capture budget"
                    ) from exc
                time.sleep(min(0.01, remaining))
    finally:
        conn.execute(f"PRAGMA busy_timeout={previous_timeout}")


def _read_schema_version_bounded(
    conn: sqlite3.Connection,
    deadline: float,
) -> int | None:
    """Read the pre-fence marker without inheriting SQLite's unbounded busy wait."""
    timeout_row = conn.execute("PRAGMA busy_timeout").fetchone()
    if not timeout_row or timeout_row[0] is None:
        raise CoreCaptureFailed("could not observe the SessionDB busy timeout")
    try:
        previous_timeout = int(timeout_row[0])
    except (TypeError, ValueError) as exc:
        raise CoreCaptureFailed("SessionDB busy timeout is not an integer") from exc
    conn.execute("PRAGMA busy_timeout=0")
    try:
        _check_deadline(deadline)
        version = _read_schema_version(conn)
        _check_deadline(deadline)
        return version
    finally:
        conn.execute(f"PRAGMA busy_timeout={previous_timeout}")


def _table_info(conn: sqlite3.Connection, table: str) -> tuple[tuple[str, int], ...]:
    rows = conn.execute(f"PRAGMA table_info({_quote_identifier(table)})").fetchall()
    return tuple((str(row[1]), int(row[5] or 0)) for row in rows)


def _foreign_keys(conn: sqlite3.Connection, table: str) -> tuple[tuple[str, str, str], ...]:
    rows = conn.execute(f"PRAGMA foreign_key_list({_quote_identifier(table)})").fetchall()
    return tuple((str(row[2]), str(row[3]), str(row[4])) for row in rows)


def _relation_labels(
    conn: sqlite3.Connection,
    tables: Iterable[str],
    *,
    allowed_tables: Iterable[str] | None = None,
) -> tuple[str, ...]:
    labels: list[str] = []
    if allowed_tables is None:
        allowed_tables = tuple(
            name
            for schema_policy in SCHEMA_POLICIES.values()
            for name in policy_table_names(table_policies=schema_policy.table_policies)
        )
    allowed = set(allowed_tables) | set(DERIVED_SQLITE_TABLES)
    for table in tables:
        for target, source_column, target_column in _foreign_keys(conn, table):
            if target not in allowed:
                raise CoreCaptureRefused(
                    f"{table}.{source_column} references unregistered relation {target}.{target_column}"
                )
            labels.append(f"{table}.{source_column}->{target}.{target_column}")
    return tuple(sorted(set(labels)))


def _wire_scalar(value: Any) -> dict[str, Any]:
    """Encode SQLite scalar storage classes without JSON numeric coercion."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "integer", "decimal": "1" if value else "0"}
    if isinstance(value, int):
        return {"type": "integer", "decimal": str(value)}
    if isinstance(value, float):
        return {
            "type": "real",
            "ieee754_be_b64": base64.b64encode(struct.pack(">d", value)).decode("ascii"),
        }
    if isinstance(value, bytes):
        return {"type": "blob", "base64": base64.b64encode(value).decode("ascii")}
    if isinstance(value, str):
        raw = value.encode("utf-8", "surrogatepass")
        return {
            "type": "text",
            "text": value,
            "utf8_b64": base64.b64encode(raw).decode("ascii"),
        }
    raise CoreCaptureRefused(f"unsupported SQLite value type: {type(value).__name__}")


def _wire_field(field: CoreFieldValue) -> dict[str, Any]:
    value = _wire_scalar(field.value)
    if field.raw_bytes is not None:
        value["raw_b64"] = base64.b64encode(field.raw_bytes).decode("ascii")
    return {
        "column": field.column,
        "disposition": field.disposition.value,
        "value": value,
    }


def _json_line(payload: dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return encoded.encode("ascii") + _JSON_LINE_SUFFIX
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CoreCaptureFailed("canonical capture wire encoding failed") from exc


def _wire_record(record: CoreStateRecord) -> bytes:
    payload = {
        "kind": "record",
        "table": record.table,
        "evidence_class": record.evidence_class,
        "primary_key": [_wire_scalar(value) for value in record.primary_key],
        "fields": [_wire_field(field) for field in record.fields],
        "relations": [
            {
                "relation": relation.relation,
                "source_key": relation.source_key,
                "target_key": relation.target_key,
            }
            for relation in record.relations
        ],
    }
    return _json_line(payload)


def _manifest_line(manifest: CoreCaptureManifest) -> bytes:
    payload = {
        "kind": "manifest",
        "api_version": manifest.api_version,
        "field_policy_version": manifest.field_policy_version,
        "official_schema_source": manifest.official_schema_source,
        "policy_writer_provenance": manifest.policy_writer_provenance,
        "schema_version": manifest.schema_version,
        "core_tables": list(manifest.core_tables),
        "optional_tables": list(manifest.optional_tables),
        "extension_sources": list(manifest.extension_sources),
        "validated_extension_tables": list(manifest.validated_extension_tables),
        "declared_relations": list(manifest.declared_relations),
        "derived_relations": list(manifest.derived_relations),
    }
    # Legacy v30/v31 manifests retain their exact wire bytes.  Versioned provenance
    # is emitted only for the newly supported v32 wire contract.
    if manifest.schema_version >= SCHEMA32_SCHEMA_VERSION and manifest.schema_census_sha256:
        payload["schema_census_sha256"] = manifest.schema_census_sha256
    if manifest.schema_version >= SCHEMA32_SCHEMA_VERSION and manifest.field_policy_review_sha256:
        payload["field_policy_review_sha256"] = manifest.field_policy_review_sha256
    return _json_line(payload)


def _identity_text(value: Any) -> str:
    if isinstance(value, bytes):
        return "blob:" + base64.b64encode(value).decode("ascii")
    if value is None:
        return "null"
    return f"{type(value).__name__}:{value}"


def _stream_write(stream: BinaryIO, data: bytes) -> None:
    try:
        written = stream.write(data)
    except Exception as exc:
        raise CoreCaptureFailed("capture sink rejected the canonical stream") from exc
    if isinstance(written, bool) or not isinstance(written, int) or written != len(data):
        raise CoreCaptureFailed("capture sink did not report an exact byte count")


def _prepare_stream(stream: BinaryIO) -> int:
    """Validate rollback capability before the first byte reaches the sink."""
    try:
        seekable = getattr(stream, "seekable", None)
        if callable(seekable) and not seekable():
            raise CoreCaptureFailed("complete capture requires a seekable sink")
        tell = getattr(stream, "tell", None)
        seek = getattr(stream, "seek", None)
        truncate = getattr(stream, "truncate", None)
        flush = getattr(stream, "flush", None)
        if not all(callable(operation) for operation in (tell, seek, truncate, flush)):
            raise CoreCaptureFailed("complete capture sink cannot be rolled back")
        initial_position = tell()
        seek(initial_position)
        return initial_position
    except CoreCaptureFailed:
        raise
    except Exception as exc:
        raise CoreCaptureFailed("complete capture requires a seekable, truncatable sink") from exc


def _discard_stream(stream: BinaryIO, initial_position: int) -> None:
    with suppress(Exception):
        stream.seek(initial_position)
        stream.truncate(initial_position)
        stream.flush()


def _extension_attr(value: object, name: str) -> object:
    try:
        selected = getattr(value, name)
    except Exception as exc:
        raise CoreCaptureRefused(f"extension descriptor is missing {name}") from exc
    if callable(selected):
        raise CoreCaptureRefused(f"extension descriptor {name} must be a value")
    return selected


def _extension_tuple(value: object, name: str) -> tuple[object, ...]:
    selected = _extension_attr(value, name)
    if not isinstance(selected, tuple):
        raise CoreCaptureRefused(f"extension descriptor {name} must be a tuple")
    return selected


def _extension_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise CoreCaptureRefused(f"extension {label} must be a non-empty identifier")
    return value


def _extension_source_name(descriptor: object) -> str:
    return _extension_identifier(_extension_attr(descriptor, "source_name"), "source name")


def _extension_field_spec(field: object) -> _ExtensionFieldSpec:
    name = _extension_identifier(_extension_attr(field, "name"), "field name")
    try:
        disposition = extension_disposition(_extension_attr(field, "disposition"))
    except (TypeError, ValueError) as exc:
        raise CoreCaptureRefused("extension field disposition is not exact and closed") from exc
    preserve_raw_bytes = _extension_attr(field, "preserve_raw_bytes")
    if not isinstance(preserve_raw_bytes, bool):
        raise CoreCaptureRefused("extension field raw-byte policy is not boolean")
    return _ExtensionFieldSpec(name, FieldDisposition(disposition), preserve_raw_bytes)


def _extension_relation_spec(
    relation: object,
    source_table: str,
    source_columns: set[str],
    registered_columns: Mapping[str, tuple[str, ...]],
) -> _ExtensionRelationSpec:
    source_column = _extension_identifier(
        _extension_attr(relation, "source_column"), "relation source column"
    )
    target_table = _extension_identifier(
        _extension_attr(relation, "target_table"), "relation target table"
    )
    target_column = _extension_identifier(
        _extension_attr(relation, "target_column"), "relation target column"
    )
    if source_column not in source_columns or target_table not in registered_columns:
        raise CoreCaptureRefused("extension relation is outside the registered tables")
    if target_column not in registered_columns[target_table]:
        raise CoreCaptureRefused("extension relation target is outside the registered columns")
    declared_source = _extension_identifier(
        _extension_attr(relation, "source_table"), "relation source table"
    )
    if declared_source != source_table:
        raise CoreCaptureRefused("extension relation source table is malformed")
    label = f"{source_table}.{source_column}->{target_table}.{target_column}"
    if _extension_attr(relation, "label") != label:
        raise CoreCaptureRefused("extension relation label is malformed")
    return _ExtensionRelationSpec(
        source_table, source_column, target_table, target_column, label
    )


def _extension_table_spec(
    table: object,
    expected_name: str,
    expected_columns: tuple[str, ...],
    conn: sqlite3.Connection,
    deadline: float,
) -> tuple[_ExtensionTableSpec, tuple[object, ...]]:
    relation_name = _extension_identifier(
        _extension_attr(table, "relation_name"), "table name"
    )
    if relation_name != expected_name:
        raise CoreCaptureRefused("extension table membership changed after registration")
    fields = _extension_tuple(table, "fields")
    columns = _extension_tuple(table, "columns")
    field_specs = tuple(_extension_field_spec(field) for field in fields)
    field_names = tuple(field.name for field in field_specs)
    if not fields or field_names != expected_columns or columns != expected_columns:
        raise CoreCaptureRefused("extension table columns changed after registration")
    primary_key = tuple(
        _extension_identifier(column, "primary-key column")
        for column in _extension_tuple(table, "primary_key")
    )
    if not primary_key:
        raise CoreCaptureRefused("extension table must declare a primary key")
    _check_deadline(deadline)
    object_row = conn.execute(
        "SELECT type FROM sqlite_master WHERE name = ? LIMIT 1", (relation_name,)
    ).fetchone()
    _check_deadline(deadline)
    if object_row is None or object_row[0] != "table":
        raise CoreCaptureRefused("extension relation is not a live SQLite table")
    info = _table_info(conn, relation_name)
    _check_deadline(deadline)
    actual_columns = tuple(column for column, _ in info)
    actual_key = tuple(column for column, pk in sorted(info, key=lambda item: item[1]) if pk)
    if actual_columns != expected_columns or actual_key != primary_key:
        raise CoreCaptureRefused("extension table schema or primary key is not registered")
    return (
        _ExtensionTableSpec(relation_name, field_specs, primary_key, ()),
        _extension_tuple(table, "relations"),
    )


def _extension_descriptor_specs(
    descriptor: object,
    source_tables: Mapping[str, tuple[tuple[str, tuple[str, ...]], ...]],
    conn: sqlite3.Connection,
    deadline: float,
) -> tuple[_ExtensionTableSpec, ...]:
    source_name = _extension_source_name(descriptor)
    if source_name not in source_tables:
        raise CoreCaptureRefused("extension source is not registered for this capture")
    record_schema = _extension_attr(descriptor, "record_schema")
    if not isinstance(record_schema, str) or not record_schema:
        raise CoreCaptureRefused("extension record schema is missing")
    tables = _extension_tuple(descriptor, "tables")
    registered = source_tables[source_name]
    if not tables or tuple(_extension_attr(table, "relation_name") for table in tables) != tuple(
        name for name, _ in registered
    ):
        raise CoreCaptureRefused("extension table order or membership is not registered")
    registered_columns = dict(registered)
    table_specs: list[_ExtensionTableSpec] = []
    raw_relations: list[tuple[str, tuple[object, ...]]] = []
    for table, (name, columns) in zip(tables, registered, strict=True):
        spec, relations = _extension_table_spec(table, name, columns, conn, deadline)
        table_specs.append(spec)
        raw_relations.append((name, relations))
    seen_labels: set[str] = set()
    finalized: list[_ExtensionTableSpec] = []
    for spec, (source_table, relations) in zip(table_specs, raw_relations, strict=True):
        relation_specs = tuple(
            _extension_relation_spec(
                relation,
                source_table,
                {field.name for field in spec.fields},
                registered_columns,
            )
            for relation in relations
        )
        for relation in relation_specs:
            if relation.label in seen_labels:
                raise CoreCaptureRefused("extension logical relation label collides")
            seen_labels.add(relation.label)
        finalized.append(
            _ExtensionTableSpec(
                spec.relation_name, spec.fields, spec.primary_key, relation_specs
            )
        )
    declared_relations = _extension_tuple(descriptor, "relations")
    if declared_relations != tuple(relation.label for spec in finalized for relation in spec.relations):
        raise CoreCaptureRefused("extension relation summary is malformed")
    return tuple(finalized)


def _extension_paged_rows(
    conn: sqlite3.Connection,
    table: str,
    deadline: float,
) -> Iterator[tuple[Any, ...]]:
    """Read one registered table in bounded pages with a SQLite VM deadline."""
    def progress() -> int:
        return int(time.monotonic() > deadline)

    cursor = None
    try:
        while True:
            _check_deadline(deadline)
            conn.set_progress_handler(progress, 1_000)
            try:
                if cursor is None:
                    cursor = conn.execute(
                        f"SELECT rowid AS {_quote_identifier('__capture_rowid')}, * "
                        f"FROM {_quote_identifier(table)} ORDER BY rowid"
                    )
                page = cursor.fetchmany(_PAGE_SIZE)
            finally:
                conn.set_progress_handler(None, 0)
            if not page:
                return
            yield from page
    except sqlite3.Error as exc:
        if time.monotonic() > deadline:
            raise CoreCaptureRefused("complete capture exceeded its time budget") from exc
        raise CoreCaptureFailed(f"could not read extension table {table}") from exc


class _LiveCoreStateCapture:
    """The live reader held by ``open_complete_state_capture``."""

    def __init__(
        self,
        session_db: Any,
        conn: sqlite3.Connection,
        source_snapshot: tuple[Any, ...],
        manifest: CoreCaptureManifest,
        budget: CaptureBudget,
        deadline: float,
        table_policies: Mapping[str, Any],
        extension_registry: _ExtensionRegistry,
    ) -> None:
        self._session_db = session_db
        self._conn = conn
        self._source_snapshot = source_snapshot
        self._manifest = manifest
        self._budget = budget
        self._deadline = deadline
        self._table_policies = table_policies
        self._extension_registry = extension_registry
        self._extension_source_tables = extension_registry.source_tables
        self._header = _manifest_line(manifest)
        if len(self._header) > budget.max_bytes:
            raise CoreCaptureRefused("complete capture manifest exceeds its byte budget")
        self._rows = 0
        self._bytes = len(self._header)
        self._iterated = False
        self._core_complete = False
        self._extension_started = False
        self._extension_complete = not bool(extension_registry.tables)
        self._extension_required_sources = frozenset(
            source
            for source, tables in extension_registry.source_tables.items()
            if tables
        )
        self._extension_requested_sources: set[str] = set()
        self._extension_completed_sources: set[str] = set()
        self._complete = False
        self._failed = False
        self._written = False
        self._active = True
        self._current_encoded: bytes | None = None
        self._output_stream: BinaryIO | None = None
        self._output_initial_position: int | None = None

    @property
    def manifest(self) -> CoreCaptureManifest:
        return self._manifest

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def failed(self) -> bool:
        return self._failed

    def _check_live(self) -> None:
        if not self._active:
            self._failed = True
            raise CoreCaptureRefused("complete capture is no longer live")
        if time.monotonic() > self._deadline:
            self._failed = True
            raise CoreCaptureRefused("complete capture exceeded its time budget")
        _check_source_current(self._session_db, self._conn, self._source_snapshot)

    def _maybe_complete(self) -> None:
        if self._core_complete and self._extension_complete:
            self._check_live()
            self._complete = True

    def _refresh_extension_completion(self) -> None:
        if self._extension_started:
            self._extension_complete = (
                self._extension_completed_sources == self._extension_required_sources
            )

    def _deactivate(self) -> None:
        self._active = False

    def _observe(self, record: CoreStateRecord) -> bytes:
        if not self._active:
            self._failed = True
            raise CoreCaptureRefused("complete capture is no longer live")
        if time.monotonic() > self._deadline:
            self._failed = True
            raise CoreCaptureRefused("complete capture exceeded its time budget")
        if self._rows % _PAGE_SIZE == 0:
            _check_source_current(self._session_db, self._conn, self._source_snapshot)
        if self._rows >= self._budget.max_rows:
            self._failed = True
            raise CoreCaptureRefused("complete capture exceeded its row budget")
        encoded = _wire_record(record)
        if self._bytes + len(encoded) > self._budget.max_bytes:
            self._failed = True
            raise CoreCaptureRefused("complete capture exceeded its byte budget")
        self._rows += 1
        self._bytes += len(encoded)
        return encoded

    def _row_records(self, table: str) -> Iterator[CoreStateRecord]:
        policy = table_policy_for_mapping(table, self._table_policies)
        info = _table_info(self._conn, table)
        columns = tuple(column for column, _ in info)
        if columns != policy.column_names:
            self._failed = True
            raise CoreCaptureRefused(f"{table} changed after schema validation")
        pk_columns = tuple(column for column, pk in sorted(info, key=lambda item: item[1]) if pk)
        foreign_keys = _foreign_keys(self._conn, table)
        select = (
            f"SELECT rowid AS {_quote_identifier('__capture_rowid')}, * "
            f"FROM {_quote_identifier(table)} ORDER BY rowid"
        )
        try:
            cursor = self._conn.execute(select)
        except sqlite3.Error as exc:
            self._failed = True
            raise CoreCaptureFailed(f"could not read reviewed table {table}") from exc
        while True:
            self._check_live()
            rows = cursor.fetchmany(_PAGE_SIZE)
            if not rows:
                return
            for row in rows:
                values = {name: row[index + 1] for index, name in enumerate(columns)}
                key_columns = pk_columns or ("__capture_rowid",)
                key = tuple(row[0] if name == "__capture_rowid" else values[name] for name in key_columns)
                fields: list[CoreFieldValue] = []
                for column in policy.columns:
                    value = values[column.name]
                    raw_bytes: bytes | None = None
                    if column.disposition == MIXED_REQUIRES_PROJECTION:
                        if table == "sessions" and column.name == "model_config":
                            try:
                                projection = project_model_config(value)
                            except ModelConfigProjectionRefused as exc:
                                self._failed = True
                                raise CoreCaptureRefused(
                                    "sessions.model_config cannot be losslessly projected"
                                ) from exc
                            if projection is not None:
                                value = projection.value.decode("utf-8", "surrogatepass")
                                raw_bytes = projection.value
                        elif isinstance(value, str):
                            raw_bytes = value.encode("utf-8", "surrogatepass")
                        elif isinstance(value, bytes):
                            raw_bytes = value
                    elif isinstance(value, str):
                        raw_bytes = value.encode("utf-8", "surrogatepass")
                    elif isinstance(value, bytes):
                        raw_bytes = value
                    fields.append(
                        CoreFieldValue(
                            column=column.name,
                            disposition=FieldDisposition(column.disposition),
                            value=value,
                            raw_bytes=raw_bytes,
                        )
                    )
                relations = []
                for target, source_column, target_column in foreign_keys:
                    source_value = values.get(source_column)
                    if source_value is not None:
                        relations.append(
                            CoreRelation(
                                relation=f"{table}.{source_column}->{target}.{target_column}",
                                source_key=_identity_text(source_value),
                                target_key=_identity_text(source_value),
                            )
                        )
                evidence_class = (
                    "derived_evidence"
                    if policy.table_disposition == DERIVED_INTERNAL
                    else "core"
                )
                yield CoreStateRecord(
                    table,
                    key,
                    tuple(fields),
                    tuple(relations),
                    evidence_class=evidence_class,
                )

    def _derived_evidence_records(self) -> Iterator[CoreStateRecord]:
        for relation in sorted(self._manifest.derived_relations):
            if relation == "sqlite_sequence":
                if relation in self._manifest.core_tables:
                    continue
                cursor = self._conn.execute(
                    "SELECT rowid, name, seq FROM sqlite_sequence ORDER BY rowid"
                )
                while True:
                    self._check_live()
                    rows = cursor.fetchmany(_PAGE_SIZE)
                    if not rows:
                        break
                    for row in rows:
                        yield CoreStateRecord(
                            table=relation,
                            primary_key=(row[0],),
                            fields=(
                                CoreFieldValue(
                                    "name", FieldDisposition.DERIVED_INTERNAL, row[1],
                                    row[1].encode("utf-8", "surrogatepass")
                                    if isinstance(row[1], str) else row[1],
                                ),
                                CoreFieldValue(
                                    "seq", FieldDisposition.DERIVED_INTERNAL, row[2],
                                    row[2] if isinstance(row[2], bytes) else None,
                                ),
                            ),
                            evidence_class="derived_evidence",
                        )
                continue
            object_row = self._conn.execute(
                "SELECT type FROM sqlite_master WHERE name = ? LIMIT 1", (relation,)
            ).fetchone()
            if object_row is None:
                continue
            try:
                count = _count_relation_with_deadline(
                    self._conn, relation, lambda: self._deadline
                )
            except CoreCaptureError:
                self._failed = True
                raise
            yield CoreStateRecord(
                table=relation,
                primary_key=(relation,),
                fields=(
                    CoreFieldValue("relation_name", FieldDisposition.DERIVED_INTERNAL, relation),
                    CoreFieldValue("object_type", FieldDisposition.DERIVED_INTERNAL, str(object_row[0])),
                    CoreFieldValue("row_count", FieldDisposition.DERIVED_INTERNAL, int(count)),
                ),
                evidence_class="derived_evidence",
            )

    def _extension_target_exists(
        self,
        relation: _ExtensionRelationSpec,
        value: Any,
    ) -> bool:
        self._check_live()

        def progress() -> int:
            return int(time.monotonic() > self._deadline)

        try:
            self._conn.set_progress_handler(progress, 1_000)
            row = self._conn.execute(
                f"SELECT 1 FROM {_quote_identifier(relation.target_table)} "
                f"WHERE {_quote_identifier(relation.target_column)} = ? LIMIT 1",
                (value,),
            ).fetchone()
            self._check_live()
            return row is not None
        except sqlite3.Error as exc:
            if time.monotonic() > self._deadline:
                raise CoreCaptureRefused("complete capture exceeded its time budget") from exc
            raise CoreCaptureFailed("could not validate an extension relation target") from exc
        finally:
            try:
                self._conn.set_progress_handler(None, 0)
            except sqlite3.Error as exc:
                raise CoreCaptureFailed("could not clear the extension target deadline") from exc

    def _extension_record(
        self,
        spec: _ExtensionTableSpec,
        values: Mapping[str, Any],
    ) -> CoreStateRecord:
        fields: list[CoreFieldValue] = []
        for field in spec.fields:
            value = values[field.name]
            _wire_scalar(value)
            raw_bytes = None
            if field.preserve_raw_bytes:
                if isinstance(value, str):
                    raw_bytes = value.encode("utf-8", "surrogatepass")
                elif isinstance(value, bytes):
                    raw_bytes = value
            fields.append(CoreFieldValue(field.name, field.disposition, value, raw_bytes))
        relations: list[CoreRelation] = []
        for relation in spec.relations:
            source_value = values[relation.source_column]
            if source_value is not None and self._extension_target_exists(relation, source_value):
                identity = _identity_text(source_value)
                relations.append(CoreRelation(relation.label, identity, identity))
        return CoreStateRecord(
            table=spec.relation_name,
            primary_key=tuple(values[column] for column in spec.primary_key),
            fields=tuple(fields),
            relations=tuple(relations),
            evidence_class="core_extension",
        )

    def _extension_records(
        self,
        specs: tuple[_ExtensionTableSpec, ...],
        source_name: str,
    ) -> Iterator[CoreStateRecord]:
        try:
            for spec in specs:
                columns = tuple(field.name for field in spec.fields)
                for row in _extension_paged_rows(self._conn, spec.relation_name, self._deadline):
                    self._check_live()
                    values = {name: row[index + 1] for index, name in enumerate(columns)}
                    record = self._extension_record(spec, values)
                    self._current_encoded = self._observe(record)
                    yield record
            self._check_live()
            self._extension_completed_sources.add(source_name)
            self._refresh_extension_completion()
            self._maybe_complete()
        except BaseException:
            self._failed = True
            raise

    def iter_extension_records(
        self,
        *,
        descriptor: object,
        budget: object,
    ) -> Iterable[CoreStateRecord]:
        del budget
        self._check_live()
        self._extension_started = True
        self._extension_complete = False
        try:
            source_name = _extension_source_name(descriptor)
        except BaseException:
            self._failed = True
            raise
        if source_name in self._extension_completed_sources or source_name in self._extension_requested_sources:
            self._failed = True
            raise CoreCaptureRefused("complete extension source may be consumed only once")
        if source_name not in self._extension_source_tables:
            self._failed = True
            raise CoreCaptureRefused("extension source is not registered for this capture")
        if not self._extension_source_tables[source_name]:
            self._failed = True
            raise CoreCaptureRefused("registered extension has no row tables")
        self._extension_requested_sources.add(source_name)
        try:
            specs = _extension_descriptor_specs(
                descriptor,
                self._extension_source_tables,
                self._conn,
                self._deadline,
            )
        except BaseException:
            self._failed = True
            raise
        return self._extension_records(specs, source_name)

    def iter_records(self) -> Iterable[CoreStateRecord]:
        self._check_live()
        if self._iterated:
            self._failed = True
            raise CoreCaptureRefused("complete capture records may be consumed only once")
        self._iterated = True
        try:
            for table in self._manifest.core_tables + self._manifest.optional_tables:
                for record in self._row_records(table):
                    self._current_encoded = self._observe(record)
                    yield record
            for record in self._derived_evidence_records():
                self._current_encoded = self._observe(record)
                yield record
            self._check_live()
            self._core_complete = True
            self._maybe_complete()
        except BaseException:
            self._failed = True
            raise

    def discard_output(self) -> None:
        if self._output_stream is not None and self._output_initial_position is not None:
            _discard_stream(self._output_stream, self._output_initial_position)
            self._output_stream = None
            self._output_initial_position = None

    def write_to(self, stream: BinaryIO) -> CoreCaptureReceipt:
        if self._extension_registry.tables:
            self._failed = True
            raise CoreCaptureRefused(
                "core sink requires the registered extension composite publisher"
            )
        if self._written:
            self._failed = True
            raise CoreCaptureRefused("complete capture stream may be written only once")
        self._written = True
        try:
            self._check_live()
            initial_position = _prepare_stream(stream)
            self._output_stream = stream
            self._output_initial_position = initial_position
            self._check_live()
            _stream_write(stream, self._header)
            for _record in self.iter_records():
                if self._current_encoded is None:
                    raise CoreCaptureFailed("capture record was not encoded before sink write")
                _stream_write(stream, self._current_encoded)
            try:
                stream.flush()
            except Exception as exc:
                raise CoreCaptureFailed("capture sink could not flush the canonical stream") from exc
            # The sink may block or the source generation may drift during flush.
            # Perform the final deadline/currentness check while this method still
            # owns the rollback boundary, before returning a successful receipt.
            self._check_live()
            return CoreCaptureReceipt(self._manifest, self._rows, self._bytes)
        except BaseException:
            self._failed = True
            if "initial_position" in locals():
                _discard_stream(stream, initial_position)
                self._output_stream = None
                self._output_initial_position = None
            raise


def _build_manifest(
    conn: sqlite3.Connection,
    validation: Any,
    schema_policy: VersionedSchemaPolicy,
    extension_tables: dict[str, tuple[str, ...]],
    extension_relations: tuple[str, ...],
    extension_sources: tuple[str, ...],
) -> CoreCaptureManifest:
    core_tables = tuple(
        policy.name
        for policy in schema_policy.table_policies.values()
        if not policy.optional and policy.table_disposition != DERIVED_INTERNAL
    )
    optional_tables = validation.optional_tables
    present_derived = tuple(
        sorted((set(validation.present_tables) & set(DERIVED_SQLITE_TABLES)) | {"sqlite_sequence"})
    )
    declared_relations = _relation_labels(
        conn,
        core_tables + optional_tables,
        allowed_tables=tuple(schema_policy.table_policies),
    ) + extension_relations
    return CoreCaptureManifest(
        official_schema_source=schema_policy.official_schema_source,
        policy_writer_provenance=schema_policy.writer_source,
        schema_version=schema_policy.schema_version,
        schema_census_sha256=schema_policy.schema_census_sha256,
        field_policy_review_sha256=schema_policy.field_policy_review_sha256,
        core_tables=core_tables,
        optional_tables=optional_tables,
        extension_sources=extension_sources,
        validated_extension_tables=tuple(sorted(extension_tables)),
        declared_relations=tuple(sorted(set(declared_relations))),
        derived_relations=present_derived,
    )


@contextmanager
def open_complete_state_capture(
    session_db: Any,
    *,
    trusted_extensions: Iterable[TrustedCoreExtension] = (),
    budget: CaptureBudget = CaptureBudget(),
) -> Iterator[CoreStateCapture]:
    """Open a complete core capture under the real SessionDB writer fence."""
    deadline = time.monotonic() + budget.timeout_seconds
    registry_result = _extension_registry(trusted_extensions)
    if isinstance(registry_result, _ExtensionRegistry):
        registry = registry_result
    else:
        extension_tables, extension_relations, extension_sources = registry_result
        registry = _ExtensionRegistry(
            tables=extension_tables,
            relations=extension_relations,
            sources=extension_sources,
            source_tables={source: tuple(extension_tables.items()) for source in extension_sources},
        )
    extension_tables = registry.tables
    extension_relations = registry.relations
    extension_sources = registry.sources
    _check_deadline(deadline)
    lock = getattr(session_db, "_lock", None)
    if not isinstance(lock, type(threading.Lock())):
        raise CoreCaptureRefused("complete capture requires the canonical SessionDB writer mutex")
    if getattr(session_db, "_complete_export_active", False):
        raise CoreCaptureRefused("nested complete captures on one SessionDB are refused")
    session_db._complete_export_active = True
    try:
        acquired = lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
    except BaseException:
        session_db._complete_export_active = False
        raise
    if not acquired:
        session_db._complete_export_active = False
        raise CoreCaptureRefused("SessionDB writer mutex did not become available within the capture budget")
    conn: sqlite3.Connection | None = None
    capture: _LiveCoreStateCapture | None = None
    try:
        conn = getattr(session_db, "_conn", None)
        if conn is None or not isinstance(conn, sqlite3.Connection):
            raise CoreCaptureRefused("complete capture cannot reopen a closed SessionDB")
        _check_deadline(deadline)
        checker = getattr(session_db, "_raise_if_db_replaced", None)
        if not callable(checker):
            raise CoreCaptureRefused("complete capture requires the canonical SessionDB generation guard")
        try:
            checker()
        except Exception as exc:
            raise CoreCaptureRefused("SessionDB source generation is not current") from exc
        source_snapshot = _source_snapshot(session_db)
        observed_version = _read_schema_version_bounded(conn, deadline)
        if observed_version is None:
            raise CoreCaptureRefused("SessionDB schema_version is missing, duplicated, or invalid")
        try:
            schema_policy = schema_policy_for_version(observed_version)
        except ValueError as exc:
            raise CoreCaptureRefused(str(exc)) from exc
        _begin_immediate(conn, deadline)
        _check_deadline(deadline)
        fenced_version = _read_schema_version(conn)
        if fenced_version != observed_version:
            raise CoreCaptureRefused("SessionDB schema version changed before the capture fence")
        validation = validate_schema(
            conn,
            schema_policy=schema_policy,
            registered_extension_tables=extension_tables,
            registered_extension_relations=extension_relations,
        )
        if not validation.valid:
            raise CoreCaptureRefused(f"reviewed SessionDB schema refused: {validation.refusal_reason()}")
        _check_deadline(deadline)
        manifest = _build_manifest(
            conn,
            validation,
            schema_policy,
            extension_tables,
            extension_relations,
            extension_sources,
        )
        _check_deadline(deadline)
        capture = _LiveCoreStateCapture(
            session_db,
            conn,
            source_snapshot,
            manifest,
            budget,
            deadline,
            schema_policy.table_policies,
            registry,
        )
        yield capture
        if capture.failed or not capture.complete:
            raise CoreCaptureFailed("complete capture exited before every reviewed row was consumed")
        _check_source_current(session_db, conn, source_snapshot)
    except BaseException:
        if capture is not None:
            capture.discard_output()
        if conn is not None:
            with suppress(sqlite3.Error):
                conn.rollback()
        raise
    finally:
        if capture is not None:
            capture._deactivate()
        if conn is not None:
            with suppress(sqlite3.Error):
                if conn.in_transaction:
                    conn.rollback()
        lock.release()
        session_db._complete_export_active = False


def _capture_profile_path() -> Path:
    """Resolve the active profile through Hermes' official path resolver."""
    from hermes_state import _default_db_path

    path = Path(_default_db_path()).expanduser()
    try:
        return path.resolve()
    except OSError:
        return path


def _refuse_live_profile_owners(path: Path) -> None:
    """Refuse an in-process owner before attaching a second writer handle."""
    from hermes_cli.sqlite_safe_read import has_live_connection
    from hermes_state_registry import other_generations_for_path

    if has_live_connection(path):
        raise CoreCaptureRefused("active profile still has a live SQLite owner")
    holders = other_generations_for_path(path)
    if holders:
        raise CoreCaptureRefused("active profile still has a live SessionDB owner")


def _existing_rw_uri(path: Path) -> str:
    """Return SQLite's existing-only URI without opening another file descriptor."""
    return path.resolve().as_uri() + "?mode=rw"


def _set_no_close_checkpoint(conn: sqlite3.Connection) -> bool:
    flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
    setter = getattr(conn, "setconfig", None)
    if flag is None or not callable(setter):
        return False
    try:
        setter(flag, True)
    except (AttributeError, TypeError, ValueError, sqlite3.Error):
        return False
    return True


def _new_existing_profile_session_db(path: Path) -> Any:
    """Attach a pre-existing profile with the private capture-only SessionDB lifecycle."""
    session_db_class = _official_session_db_class()
    from hermes_state import _close_time_checkpoint_configurable
    import hermes_state_lockguard as lockguard
    from hermes_state_dbfile import (
        _connect_tracked_db,
        _stat_db_file_identity,
        _stat_sqlite_sidecar_identity,
        has_invalid_sqlite_header_preopen,
        refuse_deleted_wal_generation,
    )
    from hermes_state_wal import _on_disk_journal_mode

    if not _close_time_checkpoint_configurable():
        raise CoreCaptureRefused(
            "existing-only capture requires SQLite close-time checkpoint suppression"
        )

    class _CaptureSessionDB(session_db_class):
        def __init__(self, db_path: Path) -> None:
            self._capture_expected_identity = _stat_db_file_identity(db_path)
            if self._capture_expected_identity is None:
                raise CoreCaptureRefused("active profile state.db does not exist")
            try:
                super().__init__(db_path=db_path, read_only=False)
            except BaseException:
                self._release_capture_guard()
                raise

        def _record_db_file_identity(self) -> None:
            current = _stat_db_file_identity(self.db_path)
            if current != self._capture_expected_identity:
                raise CoreCaptureRefused("active profile state.db changed during attach")
            super()._record_db_file_identity()

        def _release_capture_guard(self) -> None:
            guard = getattr(self, "_wal_lock_guard", {})
            if guard:
                lockguard.release(guard)
                self._wal_lock_guard = {}

        def _close_connection_quietly(self, conn: Any) -> None:
            # SessionDB.__init__ invokes this dynamically from its failure-finally
            # block.  Release the OFD guard before the base close can drop SQLite's
            # own descriptors; the official guard contract requires this ordering.
            self._release_capture_guard()
            super()._close_connection_quietly(conn)

        def _open_writer(self) -> None:
            if not self.db_path.is_file():
                raise CoreCaptureRefused("active profile state.db does not exist")
            if has_invalid_sqlite_header_preopen(self.db_path):
                raise CoreCaptureRefused("active profile state.db has an invalid SQLite header")
            try:
                refuse_deleted_wal_generation(self.db_path)
                conn = _connect_tracked_db(
                    _existing_rw_uri(self.db_path),
                    tracking_path=self.db_path,
                    uri=True,
                    check_same_thread=False,
                    timeout=1.0,
                    isolation_level=None,
                )
            except CoreCaptureRefused:
                raise
            except Exception as exc:
                raise CoreCaptureRefused("active profile state.db cannot be safely attached") from exc
            try:
                if _stat_db_file_identity(self.db_path) != self._capture_expected_identity:
                    raise CoreCaptureRefused("active profile state.db changed during attach")
                self._conn = conn
                if not _set_no_close_checkpoint(conn):
                    raise CoreCaptureRefused(
                        "existing-only capture cannot suppress SQLite close-time checkpoint"
                    )
                initial_sidecars = _stat_sqlite_sidecar_identity(self.db_path)
                if initial_sidecars:
                    self._wal_lock_guard = lockguard.hold(self.db_path, self._wal_lock_guard)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON")
                mode = _on_disk_journal_mode(conn)
                if mode not in {"wal", "delete", "truncate", "persist", "memory", "off"}:
                    raise CoreCaptureRefused("active profile journal mode could not be observed")
                self._wal_active = mode == "wal"
                if self._wal_active:
                    self._wal_lock_guard = lockguard.hold(self.db_path, self._wal_lock_guard)
                    if lockguard.supported() and not self._wal_lock_guard:
                        raise CoreCaptureRefused("active profile WAL generation could not be guarded")
            except BaseException:
                self._close_connection_quietly(conn)
                self._conn = None
                raise

        def close(self) -> None:
            """Close without an explicit checkpoint, retaining official loss handling."""
            if self._shared_registry_owned:
                from hermes_state_registry import release
                release(self)
                return
            self._stop_token_writer()
            hook, self._token_atexit_hook = self._token_atexit_hook, None
            if hook is not None:
                atexit.unregister(hook)
            with self._read_conns_lock:
                self._read_conns_closed = True
            while self._evict_one_idle_read_conn():
                pass
            with self._lock:
                if self._conn is not None:
                    generation_lost = not self.read_only and (
                        self._db_wal_generation_lost
                        or (bool(self._db_sidecar_identity) and self._wal_generation_was_lost())
                    )
                    # Match SessionDB.close()'s retired-generation handling.  The
                    # normal close below never runs an explicit WAL checkpoint;
                    # NO_CKPT_ON_CLOSE was enabled before any SQLite probes.
                    retire_without_close = (
                        generation_lost and self._settle_lost_generation_locked()
                    )
                    if retire_without_close:
                        self._pin_connection(self._conn)
                        self._conn = None
                    else:
                        self._release_capture_guard()
                        conn, self._conn = self._conn, None
                        self._close_connection_quietly(conn)
                        self._db_sidecar_identity = {}
            self._read_budget.unregister(self)

    try:
        return _CaptureSessionDB(path)
    except CoreCaptureError:
        raise
    except Exception as exc:
        raise CoreCaptureRefused("active profile state.db could not be attached for capture") from exc


@contextmanager
def open_existing_complete_state_capture(
    *,
    trusted_extensions: Iterable[TrustedCoreExtension] = (),
    budget: CaptureBudget = CaptureBudget(),
) -> Iterator[CoreStateCapture]:
    """Capture the already-materialized active profile without schema side effects."""
    path = _capture_profile_path()
    _refuse_live_profile_owners(path)
    session_db = _new_existing_profile_session_db(path)
    try:
        with open_complete_state_capture(
            session_db,
            trusted_extensions=trusted_extensions,
            budget=budget,
        ) as capture:
            yield capture
    finally:
        session_db.close()


def capture_complete_state(
    session_db: Any,
    stream: BinaryIO,
    *,
    trusted_extensions: Iterable[TrustedCoreExtension] = (),
    budget: CaptureBudget = CaptureBudget(),
) -> CoreCaptureReceipt:
    """Capture the core component into a caller-owned temporary stream."""
    with open_complete_state_capture(
        session_db, trusted_extensions=trusted_extensions, budget=budget
    ) as capture:
        return capture.write_to(stream)


__all__ = [
    "CORE_CAPTURE_API_VERSION", "CaptureBudget", "CoreCaptureError", "CoreCaptureFailed",
    "CoreCaptureManifest", "CoreCaptureReceipt", "CoreCaptureRefused", "CoreCaptureUnavailable",
    "CoreExtensionDescriptor", "CoreFieldValue", "CoreRelation", "CoreStateCapture",
    "CoreStateRecord", "CoreTableExtension", "FieldDisposition", "TrustedCoreExtension",
    "capture_complete_state", "open_complete_state_capture", "open_existing_complete_state_capture",
]
