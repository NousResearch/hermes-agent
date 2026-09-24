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
from state_store_postgresql import PostgreSQLStateStore

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


class SQLitePostgreSQLImportError(RuntimeError):
    """The offline importer rejected a source, target, or recovery attempt."""


@dataclass(frozen=True)
class SQLitePostgreSQLImportResult:
    import_id: str
    status: str
    source_fingerprint: str
    object_counts: Mapping[str, int]
    manifest: Mapping[str, Any]


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
        schema: str,
        owned_target: Any | None = None,
    ) -> None:
        if not _ISOLATED_SCHEMA_RE.fullmatch(schema):
            raise SQLitePostgreSQLImportError(
                "SQLite import target must be a newly generated isolated tenant schema"
            )
        self._settings = settings
        self._dsn = dsn
        self._schema = schema
        self._owned_target = owned_target
        if (
            owned_target is None
            or getattr(owned_target, "dsn", None) != dsn
            or getattr(owned_target, "schema", None) != schema
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
        # Constructor creates and validates PG18 catalog; it never selects runtime routing.
        store = PostgreSQLStateStore(self._settings, self._dsn, schema=self._schema)
        store.close()
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} ("
                "import_id text PRIMARY KEY, source_fingerprint text NOT NULL, source_counts jsonb NOT NULL, "
                "source_schema jsonb NOT NULL, pre_import_target jsonb NOT NULL, destination_counts jsonb, "
                "status text NOT NULL CHECK(status IN ('running', 'failed', 'complete')), error text, "
                "created_at double precision NOT NULL, updated_at double precision NOT NULL)"
            )

    def _manifest_row(self, cursor: Any) -> Mapping[str, Any] | None:
        cursor.execute(
            f"SELECT import_id, source_fingerprint, source_counts, source_schema, pre_import_target, destination_counts, status, error "
            f"FROM {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} ORDER BY created_at DESC LIMIT 1"
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

    def _record_failure(self, import_id: str, detail: str) -> None:
        with self._connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                f"UPDATE {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} SET status='failed', error=%s, updated_at=%s WHERE import_id=%s",
                (detail[:1000], time.time(), import_id),
            )

    def _assert_target_ready(
        self,
        cursor: Any,
        fingerprint: str,
        source_counts: Mapping[str, int],
        source_schema: Mapping[str, Any],
    ) -> tuple[str, Mapping[str, Any] | None]:
        manifest = self._manifest_row(cursor)
        current_counts = _target_counts(cursor, self._schema)
        if manifest is None:
            if any(current_counts.values()):
                raise SQLitePostgreSQLImportError(
                    "SQLite import target is not a new isolated schema"
                )
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
        if manifest["source_fingerprint"] != fingerprint:
            raise SQLitePostgreSQLImportError(
                "SQLite import resume rejected source snapshot fingerprint mismatch"
            )
        if manifest["status"] == "complete":
            if dict(manifest["destination_counts"] or {}) != current_counts:
                raise SQLitePostgreSQLImportError(
                    "SQLite import target drifted after completed import"
                )
            return str(manifest["import_id"]), manifest
        if any(current_counts.values()):
            raise SQLitePostgreSQLImportError(
                "SQLite import target is marked unusable after failure; restore its pre-import sandbox snapshot before retry"
            )
        cursor.execute(
            f"UPDATE {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} SET status='running', error=NULL, updated_at=%s WHERE import_id=%s",
            (time.time(), manifest["import_id"]),
        )
        return str(manifest["import_id"]), manifest

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
        counts, source_schema = _source_inventory(snapshot)
        self._prepare_target()
        import_id = ""
        try:
            with self._connect() as connection:
                with connection.cursor() as cursor:
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
            with (
                sqlite3.connect(
                    f"file:{snapshot}?mode=ro", uri=True
                ) as source_connection,
                self._connect() as connection,
            ):
                source_connection.row_factory = sqlite3.Row
                with connection.cursor() as cursor:
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
                        f"UPDATE {_quote(self._schema)}.{_quote(_MANIFEST_TABLE)} SET status='complete', destination_counts=%s::jsonb, error=NULL, updated_at=%s WHERE import_id=%s",
                        (
                            json.dumps(destination_counts, sort_keys=True),
                            time.time(),
                            import_id,
                        ),
                    )
            with self._connect() as connection, connection.cursor() as cursor:
                manifest = self._manifest_row(cursor)
                assert manifest is not None
            if evidence_path:
                _json_atomic(evidence_path, dict(manifest))
            return SQLitePostgreSQLImportResult(
                import_id, "complete", fingerprint, counts, manifest
            )
        except Exception as exc:
            if import_id:
                self._record_failure(import_id, str(exc))
            raise SQLitePostgreSQLImportError(
                f"SQLite import failed; target remains isolated and must not be selected for runtime: {exc}"
            ) from exc
        finally:
            snapshot.unlink(missing_ok=True)
            snapshot.parent.rmdir() if snapshot.parent.exists() and not any(
                snapshot.parent.iterdir()
            ) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline SQLite-to-PostgreSQL StateStore sandbox importer (no runtime cutover)"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--dsn",
        required=True,
        help="Explicit sandbox DSN; do not use a runtime configuration",
    )
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args(argv)
    settings = PostgreSQLStateStoreConfig(
        dsn_env="SQLITE_IMPORT_EXPLICIT_DSN", connect_timeout_seconds=5, pool_max_size=1
    )
    result = SQLitePostgreSQLSandboxImporter(
        settings, arguments.dsn, schema=arguments.schema
    ).import_source(
        arguments.source,
        snapshot_root=arguments.snapshot_root,
        evidence_path=arguments.evidence,
    )
    print(
        json.dumps(
            {
                "import_id": result.import_id,
                "status": result.status,
                "source_fingerprint": result.source_fingerprint,
                "object_counts": result.object_counts,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
