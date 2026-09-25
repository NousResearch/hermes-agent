"""Fail-closed, offline SQLite-to-PostgreSQL sandbox importer for supported StateStore slices.

This is deliberately not runtime routing.  It accepts an explicitly supplied SQLite
path, snapshots it with SQLite's backup API, and imports only the bounded StateStore
objects represented by PostgreSQLStateStore.  Any populated canonical object outside
that map rejects the rehearsal before PostgreSQL rows are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from state_store import PostgreSQLStateStoreConfig


_MANIFEST_TABLE = "sqlite_import_manifests"
_MANIFEST_VERSION = 1
_IDENTIFIER_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_ISOLATED_SCHEMA_RE = re.compile(r"^hermes_state_store_tenant_[0-9a-f]{32}$")
_SUPPORTED_OBJECTS = (
    "system_prompts",
    "sessions",
    "messages",
    "session_model_usage",
    "conversation_generations",
    "session_runtime_owners",
    "session_runtime_turns",
)
_DERIVED_OBJECT_PREFIXES = ("messages_fts",)
_NON_MIGRATED_TABLES = frozenset({
    "schema_version",
    "state_meta",
    "gateway_routing",
    "gateway_hygiene_state",
    "gateway_heartbeats",
    "compression_locks",
    "session_turn_leases",
    "async_delegations",
})
# Fields without a PostgreSQL StateStore contract.  A non-default value means the source is not
# in the bounded migration slice and therefore must not be silently truncated.
_UNSUPPORTED_SESSION_COLUMNS = {
    "expiry_finalized": 0,
    "system_prompt": None,
    "message_count": 0,
    "tool_call_count": 0,
    "last_activity_description": None,
    "last_activity_provenance": None,
    "handoff_state": None,
    "handoff_platform": None,
    "handoff_error": None,
    "compression_failure_cooldown_until": None,
    "compression_failure_error": None,
    "compression_fallback_streak": 0,
    "compression_ineffective_count": 0,
    "compression_recovery_deadline": None,
    "rewind_count": 0,
    "last_read_at": None,
    "tool_names": None,
}
_UNSUPPORTED_MESSAGE_COLUMNS = {"display_order": None, "display_identity": None}
_SUPPORTED_SOURCE_COLUMNS = {
    "system_prompts": frozenset({"hash", "prompt"}),
    "sessions": frozenset({
        "id", "source", "user_id", "session_key", "chat_id", "chat_type", "thread_id",
        "display_name", "origin_json", "model", "model_config", "system_prompt_hash",
        "parent_session_id", "started_at", "ended_at", "end_reason", "input_tokens",
        "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens", "cwd",
        "git_branch", "git_repo_root", "git_metadata_generation", "billing_provider",
        "billing_base_url", "billing_mode", "estimated_cost_usd", "actual_cost_usd", "cost_status",
        "cost_source", "pricing_version", "title", "title_source", "last_activity_at",
        "api_call_count", "profile_name", "archived", "pinned", "hidden",
        *_UNSUPPORTED_SESSION_COLUMNS,
    }),
    "messages": frozenset({
        "id", "session_id", "role", "content", "tool_call_id", "tool_calls", "tool_name",
        "effect_disposition", "timestamp", "token_count", "finish_reason", "reasoning",
        "reasoning_content", "reasoning_details", "codex_reasoning_items", "codex_message_items",
        "platform_message_id", "observed", "_compressed_summary", "active", "compacted",
        "api_content", "display_kind", "display_metadata", *_UNSUPPORTED_MESSAGE_COLUMNS,
    }),
    "session_model_usage": frozenset({
        "session_id", "model", "billing_provider", "billing_base_url", "billing_mode", "task",
        "api_call_count", "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
        "reasoning_tokens", "estimated_cost_usd", "actual_cost_usd", "cost_status", "cost_source",
        "first_seen", "last_seen",
    }),
    "conversation_generations": frozenset({"source", "session_key", "generation"}),
    "session_runtime_owners": frozenset({
        "namespace", "session_id", "installation_id", "host", "process_generation", "fence",
        "expires_at", "updated_at",
    }),
    "session_runtime_turns": frozenset({
        "namespace", "session_id", "turn_id", "state", "owner_fence", "receipt_json", "created_at",
        "updated_at",
    }),
}


@dataclass(frozen=True)
class SQLiteImportTargetCleanup:
    """Sanitized reconciliation evidence for a CLI-owned import target."""

    status: str
    target_schema: str
    detail: str | None = None

    def as_dict(self) -> dict[str, str]:
        result = {"status": self.status, "target_schema": self.target_schema}
        if self.detail:
            result["detail"] = self.detail
        return result


class SQLitePostgreSQLImportError(RuntimeError):
    """The offline importer rejected a source, target, or recovery attempt."""

    def __init__(
        self,
        message: str,
        *,
        cleanup: SQLiteImportTargetCleanup | None = None,
        stage: str | None = None,
    ) -> None:
        super().__init__(message)
        self.cleanup = cleanup
        self.stage = stage


@dataclass(frozen=True)
class SQLitePostgreSQLImportResult:
    import_id: str
    status: str
    source_fingerprint: str
    object_counts: Mapping[str, int]
    manifest: Mapping[str, Any]


_SQLITE_IMPORT_OWNERSHIP_SCHEMA_PREFIX = "hermes_sqlite_import_owner_"
_SQLITE_IMPORT_OWNERSHIP_MARKER_TABLE = "__hermes_owned_sqlite_import_target"


@dataclass(frozen=True)
class OwnedSQLiteImportTarget:
    """A CLI-allocated tenant plus a separate, invocation-owned marker namespace.

    The marker is deliberately not in the tenant (Alembic must see an empty
    tenant) and never in ``public``.  Both namespaces are created in one
    transaction so a failed allocation cannot leave a target that the caller
    has no capability to reconcile.
    """

    dsn: str
    schema: Any
    token: str

    @property
    def identity(self) -> str:
        return self.schema.name.removeprefix("hermes_state_store_tenant_")

    @property
    def ownership_schema(self) -> str:
        return f"{_SQLITE_IMPORT_OWNERSHIP_SCHEMA_PREFIX}{self.identity}"

    @property
    def _creator_scope(self) -> str:
        return f"sqlite-import:{self.identity}"

    def _verify_cursor(self, cursor: Any) -> None:
        cursor.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s), "
            "EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname=%s)",
            (self.schema.name, self.ownership_schema),
        )
        target_exists, ownership_exists = cursor.fetchone()
        if not target_exists or not ownership_exists:
            raise SQLitePostgreSQLImportError("SQLite import target or ownership marker namespace is absent")
        try:
            cursor.execute(
                f"SELECT token, creator_scope FROM {_quote(self.ownership_schema)}."
                f"{_quote(_SQLITE_IMPORT_OWNERSHIP_MARKER_TABLE)} WHERE schema_name=%s",
                (self.schema.name,),
            )
            rows = cursor.fetchall()
        except Exception as exc:
            raise SQLitePostgreSQLImportError("SQLite import target ownership marker is absent") from exc
        if rows != [(self.token, self._creator_scope)]:
            raise SQLitePostgreSQLImportError(
                "SQLite import target ownership marker is absent, changed, or non-singleton"
            )

    def verify(self) -> None:
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise SQLitePostgreSQLImportError("SQLite import requires psycopg") from exc
        with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
            self._verify_cursor(cursor)

    def drop(self) -> SQLiteImportTargetCleanup:
        """Atomically tear down only the exact marker-owned target and marker."""
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise SQLitePostgreSQLImportError("SQLite import requires psycopg") from exc
        try:
            with psycopg.connect(self.dsn) as connection, connection.cursor() as cursor:
                # Verify in the destructive transaction to avoid a check/drop gap.
                self._verify_cursor(cursor)
                cursor.execute(f"DROP SCHEMA {_quote(self.schema.name)} CASCADE")
                cursor.execute(f"DROP SCHEMA {_quote(self.ownership_schema)} CASCADE")
                connection.commit()
        except Exception as exc:
            raise SQLitePostgreSQLImportError(
                "SQLite import target cleanup was not committed; target remains marker-protected for reconciliation"
            ) from exc
        return SQLiteImportTargetCleanup("dropped", self.schema.name)


def allocate_owned_sqlite_import_target(dsn: str) -> OwnedSQLiteImportTarget:
    """Allocate a fresh tenant and durable proof in a private UUID companion schema."""
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover
        raise SQLitePostgreSQLImportError("SQLite import requires psycopg") from exc
    from state_store_alembic.runner import _owned_target_state_store_schema

    schema = _owned_target_state_store_schema(
        f"hermes_state_store_tenant_{uuid.uuid4().hex}"
    )
    target = OwnedSQLiteImportTarget(dsn, schema, uuid.uuid4().hex)
    try:
        # PostgreSQL schema DDL is transactional.  A failed allocation rolls back
        # both namespaces and the marker together rather than relying on a global
        # registry or best-effort cleanup of an unproven schema.
        with psycopg.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(f"CREATE SCHEMA {_quote(target.schema.name)}")
            cursor.execute(f"CREATE SCHEMA {_quote(target.ownership_schema)}")
            cursor.execute(
                f"CREATE TABLE {_quote(target.ownership_schema)}."
                f"{_quote(_SQLITE_IMPORT_OWNERSHIP_MARKER_TABLE)} "
                "(schema_name text PRIMARY KEY, token text NOT NULL, creator_scope text NOT NULL)"
            )
            cursor.execute(
                f"INSERT INTO {_quote(target.ownership_schema)}."
                f"{_quote(_SQLITE_IMPORT_OWNERSHIP_MARKER_TABLE)} "
                "(schema_name, token, creator_scope) VALUES (%s, %s, %s)",
                (target.schema.name, target.token, target._creator_scope),
            )
            connection.commit()
        target.verify()
        return target
    except Exception as exc:
        # A commit acknowledgement may be lost after the server committed.  Re-read
        # the durable marker: if it exists, return the capability instead of making
        # the target inaccessible to the caller's error/reconciliation path.
        try:
            target.verify()
        except SQLitePostgreSQLImportError:
            raise SQLitePostgreSQLImportError(
                "SQLite import allocation outcome requires reconciliation",
                cleanup=SQLiteImportTargetCleanup(
                    "reconciliation-required", target.schema.name
                ),
                stage="allocation",
            ) from exc
        return target


def source_object_mapping_manifest() -> dict[str, Any]:
    """Machine-readable inventory; callers may persist it alongside rehearsal evidence."""
    return {
        "version": 1,
        "objects": [
            *(
                {
                    "source": name,
                    "target": name,
                    "classification": "canonical-supported",
                }
                for name in _SUPPORTED_OBJECTS
            ),
            {
                "source": "messages_fts*",
                "target": "messages.search_document",
                "classification": "derived-rebuildable",
                "action": "never-import; generated PostgreSQL search document and GIN rebuild",
            },
            *(
                {
                    "source": name,
                    "target": None,
                    "classification": "intentionally-process-local/non-migrated",
                }
                for name in sorted(_NON_MIGRATED_TABLES)
            ),
            {
                "source": "sessions unsupported columns",
                "target": None,
                "classification": "canonical-unimplemented",
                "action": "reject non-default values",
            },
            {
                "source": "messages.display_order/display_identity",
                "target": None,
                "classification": "canonical-unimplemented",
                "action": "reject non-null values",
            },
        ],
    }


def _quote(identifier: str) -> str:
    if (
        not identifier
        or len(identifier) > 63
        or set(identifier) - _IDENTIFIER_CHARS
        or identifier[0].isdigit()
    ):
        raise SQLitePostgreSQLImportError(
            "SQLite import received an invalid isolated PostgreSQL schema"
        )
    return f'"{identifier}"'


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    Path(temporary).replace(path)


def _write_failure_evidence(path: Path, receipt: Mapping[str, Any]) -> Path:
    """Append failure evidence without overwriting any prior receipt."""
    destination = path
    if destination.exists():
        destination = destination.with_name(
            f"{destination.stem}.failure-{receipt['import_id']}{destination.suffix}"
        )
    _json_atomic(destination, receipt)
    return destination


def _source_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }


def _source_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')}


def _snapshot_sqlite(source: Path, snapshot_root: Path) -> Path:
    """Use SQLite backup API so a WAL-backed explicit source produces one consistent snapshot."""
    if not source.is_file():
        raise SQLitePostgreSQLImportError(
            "SQLite import source must be an existing regular file"
        )
    destination_dir = snapshot_root.resolve() / f"sqlite-import-{uuid.uuid4().hex}"
    destination_dir.mkdir(parents=True, mode=0o700)
    snapshot = destination_dir / "source.snapshot.db"
    read_connection = write_connection = None
    try:
        read_connection = sqlite3.connect(
            f"file:{source.resolve()}?mode=ro", uri=True, timeout=0.0
        )
        write_connection = sqlite3.connect(snapshot)
        read_connection.backup(write_connection, pages=256, sleep=0.05)
        write_connection.commit()
        integrity = [
            row[0] for row in write_connection.execute("PRAGMA integrity_check")
        ]
        if integrity != ["ok"]:
            raise SQLitePostgreSQLImportError(
                "SQLite import snapshot failed integrity_check"
            )
    except sqlite3.Error as exc:
        snapshot.unlink(missing_ok=True)
        raise SQLitePostgreSQLImportError(
            "SQLite import could not capture a consistent offline snapshot"
        ) from exc
    finally:
        if write_connection is not None:
            write_connection.close()
        if read_connection is not None:
            read_connection.close()
    return snapshot


def _normalise_json(value: Any, object_name: str, column: str) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise SQLitePostgreSQLImportError(
                f"SQLite import rejected invalid JSON in {object_name}.{column}"
            ) from exc
    return value


def _assert_default_columns(
    connection: sqlite3.Connection, table: str, expected: Mapping[str, Any]
) -> None:
    columns = _source_columns(connection, table)
    missing = set(expected) - columns
    if missing:
        raise SQLitePostgreSQLImportError(
            f"SQLite import source lacks required {table} columns: {sorted(missing)}"
        )
    for column, default in expected.items():
        if default is None:
            row = connection.execute(
                f'SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 1'
            ).fetchone()
        else:
            row = connection.execute(
                f'SELECT 1 FROM "{table}" WHERE COALESCE("{column}", ?) != ? LIMIT 1',
                (default, default),
            ).fetchone()
        if row is not None:
            raise SQLitePostgreSQLImportError(
                f"SQLite import rejects unimplemented canonical field {table}.{column}"
            )


def _assert_unknown_columns_unpopulated(connection: sqlite3.Connection, table: str) -> None:
    unknown = sorted(_source_columns(connection, table) - _SUPPORTED_SOURCE_COLUMNS[table])
    for column in unknown:
        if connection.execute(
            f'SELECT 1 FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT 1'
        ).fetchone() is not None:
            raise SQLitePostgreSQLImportError(
                f"SQLite import rejects populated unknown canonical field {table}.{column}"
            )


def _source_inventory(snapshot: Path) -> tuple[dict[str, int], dict[str, Any]]:
    with sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True) as connection:
        tables = _source_tables(connection)
        missing_supported = set(_SUPPORTED_OBJECTS) - tables
        if missing_supported:
            raise SQLitePostgreSQLImportError(
                f"SQLite import source lacks approved canonical maps: {sorted(missing_supported)}"
            )
        # sqlite_sequence is SQLite implementation metadata for AUTOINCREMENT, never a
        # source domain object.  It is deliberately excluded from mapping enforcement.
        unknown = (
            tables
            - set(_SUPPORTED_OBJECTS)
            - _NON_MIGRATED_TABLES
            - {"sqlite_sequence"}
        )
        non_fts_unknown = sorted(
            name for name in unknown if not name.startswith(_DERIVED_OBJECT_PREFIXES)
        )
        if non_fts_unknown:
            raise SQLitePostgreSQLImportError(
                f"SQLite import source has objects without an approved map: {non_fts_unknown}"
            )
        populated_non_migrated = []
        for table in sorted(_NON_MIGRATED_TABLES & tables):
            if (
                connection.execute(f'SELECT 1 FROM "{table}" LIMIT 1').fetchone()
                is not None
            ):
                populated_non_migrated.append(table)
        if populated_non_migrated:
            raise SQLitePostgreSQLImportError(
                f"SQLite import rejects populated non-migrated objects: {populated_non_migrated}"
            )
        _assert_default_columns(connection, "sessions", _UNSUPPORTED_SESSION_COLUMNS)
        _assert_default_columns(connection, "messages", _UNSUPPORTED_MESSAGE_COLUMNS)
        for table in _SUPPORTED_OBJECTS:
            _assert_unknown_columns_unpopulated(connection, table)
        counts = {
            table: int(
                connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            )
            for table in _SUPPORTED_OBJECTS
        }
        schema = {
            table: sorted(_source_columns(connection, table))
            for table in _SUPPORTED_OBJECTS
        }
        fts = sorted(
            name for name in tables if name.startswith(_DERIVED_OBJECT_PREFIXES)
        )
    return counts, {
        "tables": schema,
        "derived_fts_excluded": fts,
        "mapping": source_object_mapping_manifest(),
    }


def _target_counts(cursor: Any, schema: str) -> dict[str, int]:
    return {
        table: int(
            cursor.execute(
                f"SELECT count(*) FROM {_quote(schema)}.{_quote(table)}"
            ).fetchone()[0]
        )
        for table in _SUPPORTED_OBJECTS
    }


class SQLitePostgreSQLSandboxImporter:
    """Resumable importer for a newly isolated PostgreSQL StateStore schema only."""

    def __init__(
        self,
        settings: PostgreSQLStateStoreConfig,
        dsn: str,
        *,
        schema: Any,
        owned_target: Any | None = None,
    ) -> None:
        if not _ISOLATED_SCHEMA_RE.fullmatch(str(schema)):
            raise SQLitePostgreSQLImportError(
                "SQLite import target must be a newly generated isolated tenant schema"
            )
        from state_store_alembic.migration_helpers import require_trusted_tenant_schema

        try:
            trusted_schema = require_trusted_tenant_schema(schema)
        except Exception as exc:
            raise SQLitePostgreSQLImportError(
                "SQLite import target requires a resolver-issued TrustedTenantSchema"
            ) from exc
        self._settings = settings
        self._dsn = dsn
        self._tenant_schema = trusted_schema
        self._schema = trusted_schema.name
        self._owned_target = owned_target
        if (
            owned_target is None
            or getattr(owned_target, "dsn", None) != dsn
            or getattr(owned_target, "schema", None) != trusted_schema
            or not callable(getattr(owned_target, "verify", None))
        ):
            raise SQLitePostgreSQLImportError(
                "SQLite import requires an ownership-validated target matching its explicit PostgreSQL routing"
            )
        assert owned_target is not None
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise SQLitePostgreSQLImportError("SQLite import requires psycopg") from exc
        self._psycopg = psycopg
        from psycopg.types.json import Jsonb

        self._jsonb = Jsonb

    def _connect(self) -> Any:
        assert self._owned_target is not None
        self._owned_target.verify()
        return self._psycopg.connect(
            self._dsn, connect_timeout=self._settings.connect_timeout_seconds
        )

    def _prepare_target(self) -> None:
        """Bootstrap through Alembic, never by creating importer tables lazily."""
        from state_store_alembic.runner import upgrade_new_tenant_to_v25

        with self._connect() as connection:
            upgrade_new_tenant_to_v25(connection, self._tenant_schema)

    def _manifest_row(self, cursor: Any) -> Mapping[str, Any] | None:
        cursor.execute(
            f"SELECT import_id, source_fingerprint, source_counts, source_schema, pre_import_target, destination_counts, status, error "
            f"FROM {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
            "WHERE status='complete' ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        if row is None:
            return None
        fields = (
            "import_id",
            "source_fingerprint",
            "source_counts",
            "source_schema",
            "pre_import_target",
            "destination_counts",
            "status",
            "error",
        )
        return dict(zip(fields, row))

    def _failure_detail(self, *, stage: str, error: BaseException) -> str:
        return json.dumps(
            {
                "receipt_version": _MANIFEST_VERSION,
                "receipt_kind": "sqlite-import-failure",
                "stage": stage,
                "error_type": type(error).__name__,
                "message": str(error)[:800],
                "recorded_at": time.time(),
            },
            sort_keys=True,
        )

    def _record_failure(self, import_id: str, detail: str) -> None:
        """Finalize only a running primary manifest; complete rows are immutable."""
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
                "SET status='failed', error=%s, updated_at=%s "
                "WHERE import_id=%s AND status='running'",
                (detail[:1000], time.time(), import_id),
            )

    def _record_preflight_failure(
        self,
        *,
        fingerprint: str,
        source_counts: Mapping[str, int],
        source_schema: Mapping[str, Any],
        current_counts: Mapping[str, int],
        error: BaseException,
        stage: str,
    ) -> Mapping[str, Any]:
        """Append, rather than overwrite, a failure receipt in the v26 catalog."""
        receipt_id = uuid.uuid4().hex
        receipt = {
            "import_id": receipt_id,
            "source_fingerprint": fingerprint,
            "source_counts": dict(source_counts),
            "source_schema": dict(source_schema),
            "pre_import_target": dict(current_counts),
            "destination_counts": None,
            "status": "failed",
            "error": self._failure_detail(stage=stage, error=error),
        }
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"INSERT INTO {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
                "(import_id, source_fingerprint, source_counts, source_schema, pre_import_target, destination_counts, status, error, created_at, updated_at) "
                "VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, NULL, 'failed', %s, %s, %s)",
                (
                    receipt_id,
                    fingerprint,
                    json.dumps(receipt["source_counts"], sort_keys=True),
                    json.dumps(receipt["source_schema"], sort_keys=True),
                    json.dumps(receipt["pre_import_target"], sort_keys=True),
                    receipt["error"],
                    time.time(),
                    time.time(),
                ),
            )
        return receipt

    def _reconcile_running_manifests(self, cursor: Any) -> None:
        cursor.execute(
            f"SELECT import_id FROM {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
            "WHERE status='running'"
        )
        stale_ids = [str(row[0]) for row in cursor.fetchall()]
        for stale_id in stale_ids:
            cursor.execute(
                f"UPDATE {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
                "SET status='failed', error=%s, updated_at=%s "
                "WHERE import_id=%s AND status='running'",
                (
                    json.dumps(
                        {
                            "receipt_version": _MANIFEST_VERSION,
                            "receipt_kind": "sqlite-import-reconciled",
                            "stage": "recovery",
                            "message": "previous running manifest reconciled before a new attempt",
                            "recorded_at": time.time(),
                        },
                        sort_keys=True,
                    ),
                    time.time(),
                    stale_id,
                ),
            )

    def _record_catalog_failure_if_available(
        self,
        *,
        fingerprint: str,
        source_counts: Mapping[str, int],
        source_schema: Mapping[str, Any],
        error: BaseException,
        stage: str,
    ) -> Mapping[str, Any] | None:
        """Record only when an existing v26 manifest catalog is provably usable."""
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    "SELECT to_regclass(%s)",
                    (f"{self._schema}.{_MANIFEST_TABLE}",),
                )
                if cursor.fetchone()[0] is None:
                    return None
                current_counts = _target_counts(cursor, self._schema)
            return self._record_preflight_failure(
                fingerprint=fingerprint,
                source_counts=source_counts,
                source_schema=source_schema,
                current_counts=current_counts,
                error=error,
                stage=stage,
            )
        except Exception:
            # An unprovable catalog is unsafe to mutate. The caller still writes
            # its filesystem receipt and fails closed.
            return None

    def _assert_target_ready(
        self,
        cursor: Any,
        fingerprint: str,
        source_counts: Mapping[str, int],
        source_schema: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any] | None]:
        self._reconcile_running_manifests(cursor)
        manifest = self._manifest_row(cursor)
        current_counts = _target_counts(cursor, self._schema)
        if manifest is not None:
            if manifest["source_fingerprint"] != fingerprint:
                error = SQLitePostgreSQLImportError(
                    "SQLite import resume rejected source snapshot fingerprint mismatch"
                )
                self._record_preflight_failure(
                    fingerprint=fingerprint, source_counts=source_counts,
                    source_schema=source_schema, current_counts=current_counts,
                    error=error, stage="preflight-source-fingerprint",
                )
                raise error
            if dict(manifest["destination_counts"] or {}) != current_counts:
                error = SQLitePostgreSQLImportError(
                    "SQLite import target drifted after completed import"
                )
                self._record_preflight_failure(
                    fingerprint=fingerprint, source_counts=source_counts,
                    source_schema=source_schema, current_counts=current_counts,
                    error=error, stage="preflight-target-drift",
                )
                raise error
            return str(manifest["import_id"]), manifest
        if any(current_counts.values()):
            error = SQLitePostgreSQLImportError(
                "SQLite import target is not a new isolated schema"
            )
            self._record_preflight_failure(
                fingerprint=fingerprint, source_counts=source_counts,
                source_schema=source_schema, current_counts=current_counts,
                error=error, stage="preflight-populated-target",
            )
            raise error
        import_id = uuid.uuid4().hex
        now = time.time()
        cursor.execute(
            f"INSERT INTO {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
            "(import_id, source_fingerprint, source_counts, source_schema, pre_import_target, status, created_at, updated_at) "
            "VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, 'running', %s, %s)",
            (
                import_id,
                fingerprint,
                json.dumps(source_counts, sort_keys=True),
                json.dumps(source_schema, sort_keys=True),
                json.dumps(current_counts, sort_keys=True),
                now,
                now,
            ),
        )
        return import_id, None

    def _rows(self, source: sqlite3.Connection, table: str) -> Iterable[sqlite3.Row]:
        return source.execute(f'SELECT * FROM "{table}"')

    def _sessions_in_foreign_key_order(
        self, source: sqlite3.Connection
    ) -> list[sqlite3.Row]:
        """Return source sessions parent-first; reject malformed self-reference cycles."""
        rows = list(self._rows(source, "sessions"))
        by_id = {str(row["id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise SQLitePostgreSQLImportError(
                "SQLite import source has duplicate session IDs"
            )
        ordered: list[sqlite3.Row] = []
        pending = dict(by_id)
        while pending:
            ready = [
                session_id
                for session_id, row in pending.items()
                if row["parent_session_id"] is None
                or str(row["parent_session_id"]) not in pending
            ]
            if not ready:
                raise SQLitePostgreSQLImportError(
                    "SQLite import source sessions contain a parent-session cycle"
                )
            for session_id in sorted(ready):
                ordered.append(pending.pop(session_id))
        return ordered

    def _import_objects(
        self, cursor: Any, source: sqlite3.Connection, *, fail_after: str | None = None
    ) -> None:
        qschema = _quote(self._schema)
        for row in self._rows(source, "system_prompts"):
            cursor.execute(
                f"INSERT INTO {qschema}.system_prompts (hash, prompt) VALUES (%s, %s)",
                (row["hash"], row["prompt"]),
            )
        if fail_after == "system_prompts":
            raise RuntimeError("injected interruption")
        session_columns = (
            "id",
            "source",
            "user_id",
            "session_key",
            "chat_id",
            "chat_type",
            "thread_id",
            "display_name",
            "origin_json",
            "model",
            "model_config",
            "system_prompt_hash",
            "parent_session_id",
            "started_at",
            "ended_at",
            "end_reason",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "cwd",
            "git_branch",
            "git_repo_root",
            "git_metadata_generation",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
            "title_source",
            "last_activity_at",
            "api_call_count",
            "profile_name",
            "archived",
            "pinned",
            "hidden",
        )
        for row in self._sessions_in_foreign_key_order(source):
            values = [row[column] for column in session_columns]
            model_config_index = session_columns.index("model_config")
            model_config = _normalise_json(
                values[model_config_index], "sessions", "model_config"
            )
            values[model_config_index] = (
                None if model_config is None else self._jsonb(model_config)
            )
            for column in ("archived", "pinned", "hidden"):
                index = session_columns.index(column)
                values[index] = bool(values[index])
            cursor.execute(
                f"INSERT INTO {qschema}.sessions ({', '.join(session_columns)}) VALUES ({', '.join(['%s'] * len(session_columns))})",
                values,
            )
        if fail_after == "sessions":
            raise RuntimeError("injected interruption")
        message_columns = (
            "id",
            "session_id",
            "role",
            "content",
            "tool_call_id",
            "tool_calls",
            "tool_name",
            "effect_disposition",
            "created_at",
            "token_count",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "platform_message_id",
            "observed",
            "_compressed_summary",
            "active",
            "compacted",
            "api_content",
            "display_kind",
            "display_metadata",
        )
        for row in self._rows(source, "messages"):
            values = [
                row["timestamp"] if column == "created_at" else row[column]
                for column in message_columns
            ]
            for column in ("tool_calls", "display_metadata"):
                index = message_columns.index(column)
                parsed = _normalise_json(values[index], "messages", column)
                values[index] = None if parsed is None else self._jsonb(parsed)
            for column in ("observed", "_compressed_summary", "active", "compacted"):
                index = message_columns.index(column)
                values[index] = bool(values[index])
            cursor.execute(
                f"INSERT INTO {qschema}.messages ({', '.join(message_columns)}) OVERRIDING SYSTEM VALUE VALUES ({', '.join(['%s'] * len(message_columns))})",
                values,
            )
        if fail_after == "messages":
            raise RuntimeError("injected interruption")
        usage_columns = (
            "session_id",
            "model",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "task",
            "api_call_count",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "estimated_cost_usd",
            "actual_cost_usd",
            "cost_status",
            "cost_source",
            "first_seen",
            "last_seen",
        )
        for row in self._rows(source, "session_model_usage"):
            cursor.execute(
                f"INSERT INTO {qschema}.session_model_usage ({', '.join(usage_columns)}) VALUES ({', '.join(['%s'] * len(usage_columns))})",
                [row[column] for column in usage_columns],
            )
        if fail_after == "session_model_usage":
            raise RuntimeError("injected interruption")
        for row in self._rows(source, "conversation_generations"):
            cursor.execute(
                f"INSERT INTO {qschema}.conversation_generations (source, session_key, generation) VALUES (%s, %s, %s)",
                tuple(row),
            )
        if fail_after == "conversation_generations":
            raise RuntimeError("injected interruption")
        for row in self._rows(source, "session_runtime_owners"):
            cursor.execute(
                f"INSERT INTO {qschema}.session_runtime_owners (namespace, session_id, installation_id, host, process_generation, fence, expires_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                tuple(row),
            )
        for row in self._rows(source, "session_runtime_turns"):
            values = list(row)
            receipt = _normalise_json(
                values[5], "session_runtime_turns", "receipt_json"
            )
            values[5] = None if receipt is None else self._jsonb(receipt)
            cursor.execute(
                f"INSERT INTO {qschema}.session_runtime_turns (namespace, session_id, turn_id, state, owner_fence, receipt_json, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)",
                values,
            )
        if fail_after == "session_runtime_turns":
            raise RuntimeError("injected interruption")
        cursor.execute(
            f"SELECT pg_get_serial_sequence(%s, 'id')", (f"{self._schema}.messages",)
        )
        sequence = cursor.fetchone()[0]
        cursor.execute(
            f"SELECT setval(%s::regclass, COALESCE((SELECT max(id) FROM {qschema}.messages), 1), (SELECT count(*) > 0 FROM {qschema}.messages))",
            (sequence,),
        )

    def import_source(
        self,
        source_path: Path,
        *,
        snapshot_root: Path,
        evidence_path: Path | None = None,
        fail_after: str | None = None,
    ) -> SQLitePostgreSQLImportResult:
        source = source_path.resolve()
        # Explicitly prevent accidental use of the active default store.  Profile/custom sources are
        # intentionally not inferred; all callers must pass their disposable path directly.
        try:
            from hermes_constants import get_hermes_home

            if source == (get_hermes_home() / "state.db").resolve():
                raise SQLitePostgreSQLImportError(
                    "SQLite import refuses the active default state.db; supply a disposable explicit source"
                )
        except SQLitePostgreSQLImportError:
            raise
        except Exception:
            pass
        snapshot = _snapshot_sqlite(source, snapshot_root)
        fingerprint = _hash_file(snapshot)
        counts: Mapping[str, int] = {}
        source_schema: Mapping[str, Any] = {}
        import_id = ""
        preflight_started = False
        try:
            counts, source_schema = _source_inventory(snapshot)
            self._prepare_target()
            with (
                sqlite3.connect(
                    f"file:{snapshot}?mode=ro", uri=True
                ) as source_connection,
                self._connect() as connection,
            ):
                source_connection.row_factory = sqlite3.Row
                with connection.cursor() as cursor:
                    # Hold a target-specific session lock from admission through
                    # completion.  The durable running receipt commits before row
                    # writes so an interrupted import has a primary record to mark
                    # failed, while the lock still prevents a second caller from
                    # reconciling or admitting alongside this import.
                    cursor.execute(
                        "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                        (f"{self._schema}:sqlite-import",),
                    )
                    preflight_started = True
                    import_id, completed = self._assert_target_ready(
                        cursor, fingerprint, counts, source_schema
                    )
                    if completed is not None:
                        result = SQLitePostgreSQLImportResult(
                            import_id, "complete", fingerprint, counts, completed
                        )
                        if evidence_path:
                            _json_atomic(evidence_path, dict(completed))
                        return result
                    connection.commit()
                    self._import_objects(
                        cursor, source_connection, fail_after=fail_after
                    )
                    destination_counts = _target_counts(cursor, self._schema)
                    if destination_counts != counts:
                        raise SQLitePostgreSQLImportError(
                            "SQLite import destination counts do not match source manifest"
                        )
                    cursor.execute(
                        f"SELECT count(*) FROM {_quote(self._schema)}.messages m LEFT JOIN {_quote(self._schema)}.sessions s ON s.id=m.session_id WHERE s.id IS NULL"
                    )
                    if int(cursor.fetchone()[0]):
                        raise SQLitePostgreSQLImportError(
                            "SQLite import invariant failed: orphan messages"
                        )
                    cursor.execute(
                        f"SELECT count(*) FROM {_quote(self._schema)}.session_model_usage u LEFT JOIN {_quote(self._schema)}.sessions s ON s.id=u.session_id WHERE s.id IS NULL"
                    )
                    if int(cursor.fetchone()[0]):
                        raise SQLitePostgreSQLImportError(
                            "SQLite import invariant failed: orphan usage"
                        )
                    cursor.execute(
                        f"UPDATE {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} "
                        "SET status='complete', destination_counts=%s::jsonb, error=NULL, updated_at=%s "
                        "WHERE import_id=%s AND status='running'",
                        (
                            json.dumps(destination_counts, sort_keys=True),
                            time.time(),
                            import_id,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise SQLitePostgreSQLImportError(
                            "SQLite import primary manifest was not safely completable"
                        )
            with self._connect() as connection, connection.cursor() as cursor:
                manifest = self._manifest_row(cursor)
                assert manifest is not None and manifest["import_id"] == import_id
            if evidence_path:
                _json_atomic(evidence_path, dict(manifest))
            return SQLitePostgreSQLImportResult(
                import_id, "complete", fingerprint, counts, manifest
            )
        except Exception as exc:
            receipt: Mapping[str, Any] | None = None
            if import_id:
                self._record_failure(
                    import_id,
                    self._failure_detail(stage="import", error=exc),
                )
            elif counts and not preflight_started:
                receipt = self._record_catalog_failure_if_available(
                    fingerprint=fingerprint,
                    source_counts=counts,
                    source_schema=source_schema,
                    error=exc,
                    stage="preflight-before-primary-manifest",
                )
            if evidence_path:
                _write_failure_evidence(
                    evidence_path,
                    receipt or {
                        "import_id": uuid.uuid4().hex,
                        "source_fingerprint": fingerprint,
                        "source_counts": dict(counts),
                        "source_schema": dict(source_schema),
                        "pre_import_target": None,
                        "destination_counts": None,
                        "status": "failed",
                        "error": self._failure_detail(
                            stage="preflight-before-primary-manifest", error=exc
                        ),
                    },
                )
            raise SQLitePostgreSQLImportError(
                f"SQLite import failed; target remains isolated and must not be selected for runtime: {exc}"
            ) from exc
        finally:
            snapshot.unlink(missing_ok=True)
            snapshot.parent.rmdir() if snapshot.parent.exists() and not any(
                snapshot.parent.iterdir()
            ) else None


def import_into_allocated_target(
    settings: PostgreSQLStateStoreConfig,
    dsn: str,
    source_path: Path,
    *,
    snapshot_root: Path,
    evidence_path: Path | None = None,
) -> tuple[SQLitePostgreSQLImportResult, OwnedSQLiteImportTarget]:
    """Import into a fresh target, or reconcile it before returning a failure.

    A successful rehearsal intentionally retains its marker-owned target for the
    caller to inspect.  Every import failure attempts a marker-validated atomic
    drop and carries sanitized cleanup evidence to the CLI; the DSN is never
    included in that error path.
    """
    try:
        target = allocate_owned_sqlite_import_target(dsn)
    except SQLitePostgreSQLImportError as exc:
        raise SQLitePostgreSQLImportError(
            "SQLite import could not allocate an owned isolated PostgreSQL target",
            cleanup=exc.cleanup,
            stage="allocation",
        ) from exc
    except Exception as exc:
        raise SQLitePostgreSQLImportError(
            "SQLite import could not allocate an owned isolated PostgreSQL target",
            stage="allocation",
        ) from exc
    try:
        result = SQLitePostgreSQLSandboxImporter(
            settings, dsn, schema=target.schema, owned_target=target
        ).import_source(
            source_path,
            snapshot_root=snapshot_root,
            evidence_path=evidence_path,
        )
    except Exception as exc:
        try:
            cleanup = target.drop()
        except SQLitePostgreSQLImportError:
            cleanup = SQLiteImportTargetCleanup(
                "reconciliation-required", target.schema.name
            )
        raise SQLitePostgreSQLImportError(
            "SQLite import failed; owned target reconciliation result is available",
            cleanup=cleanup,
            stage="import",
        ) from exc
    return result, target


def _safe_cli_target_schema(value: Any) -> str | None:
    """Return only a generated tenant identifier that is safe to publish in JSON."""
    candidate = str(value) if value is not None else ""
    return candidate if _ISOLATED_SCHEMA_RE.fullmatch(candidate) else None


def _cli_cleanup_evidence(
    cleanup: SQLiteImportTargetCleanup | None, *, target: Any | None = None
) -> dict[str, str]:
    """Render reconciliation state without DSNs, exception text, or ownership tokens."""
    target_schema = _safe_cli_target_schema(
        getattr(cleanup, "target_schema", None)
        if cleanup is not None
        else getattr(getattr(target, "schema", None), "name", None)
    )
    if target_schema is None:
        return {"status": "not-allocated"}
    status = getattr(cleanup, "status", None) if cleanup is not None else None
    return {
        "status": "dropped" if status == "dropped" else "reconciliation-required",
        "target_schema": target_schema,
    }


def _cli_failure_evidence(
    stage: str, *, cleanup: SQLiteImportTargetCleanup | None = None, target: Any | None = None
) -> dict[str, Any]:
    return {
        "action": "sqlite-import",
        "status": "failed",
        "stage": stage,
        "error": f"sqlite-import-{stage}-failed",
        "cleanup": _cli_cleanup_evidence(cleanup, target=target),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline SQLite-to-PostgreSQL StateStore sandbox importer (no runtime cutover)"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)

    parser.add_argument(
        "--dsn",
        required=True,
        help="Explicit PostgreSQL server; the importer allocates a new owned tenant",
    )
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args(argv)
    settings = PostgreSQLStateStoreConfig(
        dsn_env="SQLITE_IMPORT_EXPLICIT_DSN", connect_timeout_seconds=5, pool_max_size=1
    )
    target: Any | None = None
    try:
        result, target = import_into_allocated_target(
            settings,
            arguments.dsn,
            arguments.source,
            snapshot_root=arguments.snapshot_root,
            evidence_path=arguments.evidence,
        )
        # The module CLI cannot hand its ownership capability to a later process.
        # Do not report a completed rehearsal while leaving its tenant behind.
        try:
            cleanup = target.drop()
        except Exception as exc:
            raise SQLitePostgreSQLImportError(
                "SQLite import completed but CLI target cleanup was not committed",
                cleanup=SQLiteImportTargetCleanup(
                    "reconciliation-required", target.schema.name
                ),
                stage="final-cleanup",
            ) from exc
        print(
            json.dumps(
                {
                    "import_id": result.import_id,
                    "status": result.status,
                    "source_fingerprint": result.source_fingerprint,
                    "object_counts": result.object_counts,
                    "target_schema": target.schema.name,
                    "cleanup": _cli_cleanup_evidence(cleanup, target=target),
                },
                sort_keys=True,
            )
        )
        return 0
    except SQLitePostgreSQLImportError as exc:
        stage = exc.stage if exc.stage in {"allocation", "import", "final-cleanup"} else "import"
        print(json.dumps(
            _cli_failure_evidence(stage, cleanup=exc.cleanup, target=target),
            sort_keys=True,
        ))
        return 2
    except Exception:
        print(json.dumps(_cli_failure_evidence("import", target=target), sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
