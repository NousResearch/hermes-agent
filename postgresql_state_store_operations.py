"""PostgreSQL-native state-store doctor and logical-backup primitives.

The caller must resolve the active profile and its trusted tenant schema before
constructing these operations. SQLite file/WAL recovery semantics never apply here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets

import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from state_store import PostgreSQLStateStoreConfig

_MANIFEST_VERSION = 1
_TARGET_DATABASE_RE = re.compile(r"^hermes_state_restore_[0-9a-f]{32}$")
_RESTORE_MARKER_TABLE = "__hermes_owned_restore_target"
_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_REQUIRED_TABLES = (
    "schema_migrations", "sessions", "messages", "system_prompts", "session_model_usage",
    "conversation_generations", "session_runtime_owners", "session_runtime_turns",
    "compression_locks", "session_turn_leases",
)
# Tracks the state_store_postgresql schema ledger through _GATEWAY_TRANSCRIPT_SCHEMA_VERSION (25).
_REQUIRED_MIGRATIONS = tuple(range(1, 26))
_OPTIMIZE_TABLES = _REQUIRED_TABLES
_OPTIMIZE_STATEMENT_TIMEOUT_MS = 30_000
_OPTIMIZE_LOCK_TIMEOUT_MS = 2_000


class PostgreSQLSandboxOperationsError(RuntimeError):
    """An isolated PostgreSQL operational action could not prove its contract."""


@dataclass(frozen=True)
class PostgreSQLLogicalBackup:
    """Sanitized logical-backup location and manifest summary."""

    backup_directory: Path
    manifest_path: Path
    archive_path: Path
    manifest: Mapping[str, Any]


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise PostgreSQLSandboxOperationsError("PostgreSQL operational target received an invalid identifier")
    return f'"{identifier}"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


class PostgreSQLSandboxOperations:
    """Native operations for one already-resolved trusted state-store tenant.

    ``schema`` must come from ``postgresql_tenant_schema()``, never a CLI value.
    ``pg_dump`` and ``pg_restore`` are invoked without passwords in argv; any password
    parsed from the secret DSN is kept in the child environment only.
    """

    def __init__(
        self,
        settings: PostgreSQLStateStoreConfig,
        dsn: str,
        *,
        schema: str,
        delivery_schema: str | None = None,
        profile_identity: Mapping[str, str] | None = None,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not _IDENTIFIER_RE.fullmatch(schema) or (delivery_schema is not None and not _IDENTIFIER_RE.fullmatch(delivery_schema)):
            raise PostgreSQLSandboxOperationsError("PostgreSQL operational target received an invalid trusted schema")
        self._settings = settings
        self._dsn = dsn
        self._schema = schema
        self._delivery_schema = delivery_schema
        self._profile_identity = dict(profile_identity or {})
        self._command_runner = command_runner
        self._restore_target_tokens: dict[str, str] = {}
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - the state-store constructor proves this in integration
            raise PostgreSQLSandboxOperationsError("PostgreSQL State Store requires psycopg") from exc
        self._psycopg = psycopg
        self._conninfo_module = __import__("psycopg.conninfo", fromlist=["conninfo_to_dict", "make_conninfo"])
        self._conninfo = self._conninfo_module.conninfo_to_dict(dsn)

    def _connect(self, *, database: str | None = None) -> Any:
        conninfo = dict(self._conninfo)
        if database is not None:
            conninfo["dbname"] = database
        return self._psycopg.connect(**conninfo, connect_timeout=self._settings.connect_timeout_seconds)

    def _tool_connection(self, *, database: str | None = None) -> tuple[list[str], dict[str, str]]:
        conninfo = dict(self._conninfo)
        if database is not None:
            conninfo["dbname"] = database
        arguments: list[str] = []
        for key, flag in (("host", "--host"), ("port", "--port"), ("user", "--username"), ("dbname", "--dbname")):
            value = conninfo.get(key)
            if value:
                arguments.extend((flag, str(value)))
        environment = dict(os.environ)
        password = conninfo.get("password")
        if password:
            environment["PGPASSWORD"] = str(password)
        return arguments, environment

    def _run_tool(self, executable: str, arguments: Iterable[str], *, environment: Mapping[str, str]) -> None:
        try:
            result = self._command_runner(
                [executable, *arguments], env=dict(environment), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as exc:
            raise PostgreSQLSandboxOperationsError(f"PostgreSQL logical operation could not start {executable}") from exc
        if result.returncode:
            detail = (result.stderr or result.stdout or "unknown PostgreSQL tool failure").strip()
            raise PostgreSQLSandboxOperationsError(f"PostgreSQL logical operation failed: {detail[:500]}")

    def _tool_version(self, executable: str) -> str:
        try:
            result = self._command_runner(
                [executable, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
        except OSError as exc:
            raise PostgreSQLSandboxOperationsError(f"PostgreSQL logical operation could not start {executable}") from exc
        if result.returncode:
            raise PostgreSQLSandboxOperationsError(f"PostgreSQL logical operation could not determine {executable} version")
        return (result.stdout or result.stderr or executable).strip()[:200]

    def _schema_exists(self, cursor: Any, schema: str) -> bool:
        cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=%s)", (schema,))
        return bool(cursor.fetchone()[0])

    def _schema_snapshot(self, connection: Any, *, required_extensions: Iterable[str]) -> dict[str, Any]:
        required_extensions = tuple(sorted(set(required_extensions)))
        with connection.cursor() as cursor:
            cursor.execute("SHOW server_version_num")
            server_version_num = int(cursor.fetchone()[0])
            cursor.execute("SHOW server_version")
            server_version = str(cursor.fetchone()[0])
            if not self._schema_exists(cursor, self._schema):
                raise PostgreSQLSandboxOperationsError("PostgreSQL tenant schema is absent")
            cursor.execute("SELECT extname, extversion FROM pg_extension ORDER BY extname")
            extensions = {str(name): str(version) for name, version in cursor.fetchall()}
            missing_extensions = [name for name in required_extensions if name not in extensions]
            cursor.execute(f"SELECT version FROM {_quote_identifier(self._schema)}.schema_migrations ORDER BY version")
            migrations = [int(row[0]) for row in cursor.fetchall()]
            table_counts: dict[str, int] = {}
            for table in _REQUIRED_TABLES:
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s)",
                    (self._schema, table),
                )
                if not cursor.fetchone()[0]:
                    raise PostgreSQLSandboxOperationsError(f"PostgreSQL tenant catalog is missing required table {table}")
                cursor.execute(f"SELECT count(*) FROM {_quote_identifier(self._schema)}.{_quote_identifier(table)}")
                table_counts[table] = int(cursor.fetchone()[0])
            cursor.execute(
                f"SELECT count(*) FROM {_quote_identifier(self._schema)}.messages m "
                f"LEFT JOIN {_quote_identifier(self._schema)}.sessions s ON s.id=m.session_id WHERE s.id IS NULL"
            )
            message_orphans = int(cursor.fetchone()[0])
            cursor.execute(
                f"SELECT count(*) FROM {_quote_identifier(self._schema)}.session_model_usage u "
                f"LEFT JOIN {_quote_identifier(self._schema)}.sessions s ON s.id=u.session_id WHERE s.id IS NULL"
            )
            usage_orphans = int(cursor.fetchone()[0])
            cursor.execute(
                f"SELECT count(*) FROM {_quote_identifier(self._schema)}.session_runtime_owners "
                "WHERE expires_at > EXTRACT(EPOCH FROM clock_timestamp())"
            )
            active_leases = int(cursor.fetchone()[0])
        if migrations != list(_REQUIRED_MIGRATIONS):
            raise PostgreSQLSandboxOperationsError("PostgreSQL tenant migration catalog is unhealthy")
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT attgenerated='s' FROM pg_attribute "
                "WHERE attrelid=(%s || '.messages')::regclass AND attname='search_document' AND NOT attisdropped",
                (self._schema,),
            )
            generated_document = bool((cursor.fetchone() or (False,))[0])
            cursor.execute(
                "SELECT i.indisvalid AND i.indisready AND i.indislive AND am.amname='gin' "
                "AND array_agg(a.attname ORDER BY key.ordinality)=ARRAY['search_document']::name[] "
                "FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_am am ON am.oid=c.relam "
                "JOIN unnest(i.indkey) WITH ORDINALITY key(attnum, ordinality) ON true "
                "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=key.attnum "
                "WHERE c.relnamespace=%s::regnamespace AND c.relname='messages_search_document_gin' "
                "GROUP BY i.indisvalid, i.indisready, i.indislive, am.amname",
                (self._schema,),
            )
            gin_index = bool((cursor.fetchone() or (False,))[0])
        if not generated_document or not gin_index:
            raise PostgreSQLSandboxOperationsError("PostgreSQL tenant search catalog is unhealthy")
        if missing_extensions or message_orphans or usage_orphans:
            raise PostgreSQLSandboxOperationsError("PostgreSQL tenant invariant check failed")
        return {
            "backend": "postgresql",
            "schema": self._schema,
            "server_version": server_version,
            "server_version_num": server_version_num,
            "extensions": extensions,
            "required_extensions": list(required_extensions),
            "migration_versions": migrations,
            "table_counts": table_counts,
            "search": {"available": True, "generated_document": "valid", "gin_index": "valid"},
            "ownership": {"active_leases": active_leases, "catalog_present": True},
            "invariants": {"message_orphans": message_orphans, "usage_orphans": usage_orphans},
        }

    def _delivery_snapshot(self, connection: Any) -> dict[str, Any] | None:
        if self._delivery_schema is None:
            return None
        with connection.cursor() as cursor:
            if not self._schema_exists(cursor, self._delivery_schema):
                return {"schema": self._delivery_schema, "present": False}
            cursor.execute(f"SELECT version FROM {_quote_identifier(self._delivery_schema)}.delivery_schema_migrations ORDER BY version")
            versions = [int(row[0]) for row in cursor.fetchall()]
            cursor.execute(f"SELECT count(*) FROM {_quote_identifier(self._delivery_schema)}.delivery_obligations")
            obligations = int(cursor.fetchone()[0])
            cursor.execute("SELECT 1 FROM pg_indexes WHERE schemaname=%s AND indexname='delivery_claim_idx'", (self._delivery_schema,))
            claim_index = cursor.fetchone() is not None
        if versions != [1] or not claim_index:
            raise PostgreSQLSandboxOperationsError("PostgreSQL delivery ledger migration catalog is unhealthy")
        return {"schema": self._delivery_schema, "present": True, "migration_versions": versions, "obligation_count": obligations}

    def doctor(self, *, required_extensions: Iterable[str] = ()) -> dict[str, Any]:
        """Return a sanitized reachability/catalog/search/lease snapshot or fail closed."""
        try:
            with self._connect() as connection:
                snapshot = self._schema_snapshot(connection, required_extensions=required_extensions)
                snapshot["delivery_ledger"] = self._delivery_snapshot(connection)
                return snapshot
        except PostgreSQLSandboxOperationsError:
            raise
        except Exception as exc:
            raise PostgreSQLSandboxOperationsError("PostgreSQL sandbox doctor could not validate tenant health") from exc

    def _optimize_diagnostic(self, cursor: Any) -> dict[str, Any]:
        """Return a non-mutating, fail-closed readiness record for bounded VACUUM.

        Unlike ``doctor()``, this deliberately reports index drift instead of
        raising immediately so an operator can see why optimize refused.  It
        never repairs catalog drift or creates missing objects.
        """
        issues: list[str] = []
        if not self._schema_exists(cursor, self._schema):
            return {
                "backend": "postgresql", "schema": self._schema,
                "catalog": {"healthy": False, "issues": ["tenant_schema_missing"]},
                "search": {"generated_document": "unknown", "gin_index": "unknown", "healthy": False},
                "leases": {"session_turn_leases": 0, "compression_locks": 0, "runtime_turns": 0, "active_total": 0},
            }
        cursor.execute(f"SELECT version FROM {_quote_identifier(self._schema)}.schema_migrations ORDER BY version")
        migrations = [int(row[0]) for row in cursor.fetchall()]
        if migrations != list(_REQUIRED_MIGRATIONS):
            issues.append("migration_catalog_unhealthy")
        present_tables: set[str] = set()
        for table in _OPTIMIZE_TABLES:
            cursor.execute(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema=%s AND table_name=%s)",
                (self._schema, table),
            )
            if cursor.fetchone()[0]:
                present_tables.add(table)
            else:
                issues.append(f"missing_table:{table}")
        generated_document = "unknown"
        gin_index = "unknown"
        if "messages" in present_tables:
            cursor.execute(
                "SELECT attgenerated='s' FROM pg_attribute "
                "WHERE attrelid=(%s || '.messages')::regclass AND attname='search_document' AND NOT attisdropped",
                (self._schema,),
            )
            generated_document = "valid" if bool((cursor.fetchone() or (False,))[0]) else "invalid"
            cursor.execute(
                "SELECT i.indisvalid AND i.indisready AND i.indislive AND am.amname='gin' "
                "AND array_agg(a.attname ORDER BY key.ordinality)=ARRAY['search_document']::name[] "
                "FROM pg_class c JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_am am ON am.oid=c.relam "
                "JOIN unnest(i.indkey) WITH ORDINALITY key(attnum, ordinality) ON true "
                "JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=key.attnum "
                "WHERE c.relnamespace=%s::regnamespace AND c.relname='messages_search_document_gin' "
                "GROUP BY i.indisvalid, i.indisready, i.indislive, am.amname",
                (self._schema,),
            )
            gin_index = "valid" if bool((cursor.fetchone() or (False,))[0]) else "invalid"
        search_healthy = generated_document == "valid" and gin_index == "valid"
        if not search_healthy:
            issues.append("search_catalog_unhealthy")

        leases = {"session_turn_leases": 0, "compression_locks": 0, "runtime_turns": 0}
        if "session_turn_leases" in present_tables:
            cursor.execute(
                f"SELECT count(*) FROM {_quote_identifier(self._schema)}.session_turn_leases "
                "WHERE expires_at > EXTRACT(EPOCH FROM clock_timestamp())"
            )
            leases["session_turn_leases"] = int(cursor.fetchone()[0])
        if "compression_locks" in present_tables:
            cursor.execute(
                f"SELECT count(*) FROM {_quote_identifier(self._schema)}.compression_locks "
                "WHERE expires_at > EXTRACT(EPOCH FROM clock_timestamp())"
            )
            leases["compression_locks"] = int(cursor.fetchone()[0])
        if {"session_runtime_turns", "session_runtime_owners"} <= present_tables:
            cursor.execute(
                f"SELECT count(*) FROM {_quote_identifier(self._schema)}.session_runtime_turns AS turn "
                f"JOIN {_quote_identifier(self._schema)}.session_runtime_owners AS owner "
                "ON (owner.namespace, owner.session_id, owner.fence) = (turn.namespace, turn.session_id, turn.owner_fence) "
                "WHERE turn.state IN ('running', 'indeterminate') "
                "AND owner.expires_at > EXTRACT(EPOCH FROM clock_timestamp())"
            )
            leases["runtime_turns"] = int(cursor.fetchone()[0])
        leases["active_total"] = sum(leases.values())
        return {
            "backend": "postgresql", "schema": self._schema,
            "catalog": {"healthy": not issues, "issues": issues, "migration_versions": migrations},
            "search": {"generated_document": generated_document, "gin_index": gin_index, "healthy": search_healthy},
            "leases": leases,
        }

    def optimize(self) -> dict[str, Any]:
        """Run tenant-scoped ``VACUUM (ANALYZE)`` only after a bounded native gate.

        This is intentionally not SQLite's FTS merge/file-size operation: it
        neither opens ``state.db`` nor reports filesystem reclamation.  It
        refuses catalog/search drift and live turn/compression leases instead of
        attempting repair or waiting indefinitely for a lock.
        """
        try:
            connection = self._connect()
            connection.autocommit = True
            try:
                with connection.cursor() as cursor:
                    before = self._optimize_diagnostic(cursor)
                    if not before["catalog"]["healthy"]:
                        raise PostgreSQLSandboxOperationsError(
                            "PostgreSQL optimize refused: catalog or search index is unhealthy; repair it before retrying"
                        )
                    if before["leases"]["active_total"]:
                        raise PostgreSQLSandboxOperationsError(
                            "PostgreSQL optimize refused: active session-turn or compression lease exists"
                        )
                    missing_privileges: list[str] = []
                    for table in _OPTIMIZE_TABLES:
                        cursor.execute("SELECT has_table_privilege(%s, 'MAINTAIN')", (f"{self._schema}.{table}",))
                        if not cursor.fetchone()[0]:
                            missing_privileges.append(table)
                    if missing_privileges:
                        raise PostgreSQLSandboxOperationsError(
                            "PostgreSQL optimize refused: missing MAINTAIN privilege on " + ", ".join(missing_privileges)
                        )
                    cursor.execute(f"SET lock_timeout = '{_OPTIMIZE_LOCK_TIMEOUT_MS}ms'")
                    cursor.execute(f"SET statement_timeout = '{_OPTIMIZE_STATEMENT_TIMEOUT_MS}ms'")
                    targets = ", ".join(
                        f"{_quote_identifier(self._schema)}.{_quote_identifier(table)}" for table in _OPTIMIZE_TABLES
                    )
                    cursor.execute(f"VACUUM (ANALYZE) {targets}")
                    after = self._optimize_diagnostic(cursor)
                    if not after["catalog"]["healthy"]:
                        raise PostgreSQLSandboxOperationsError(
                            "PostgreSQL optimize completed but post-operation catalog validation failed"
                        )
                    return {
                        "backend": "postgresql", "operation": "vacuum_analyze", "schema": self._schema,
                        "tables": list(_OPTIMIZE_TABLES),
                        "timeouts": {"statement_timeout_ms": _OPTIMIZE_STATEMENT_TIMEOUT_MS, "lock_timeout_ms": _OPTIMIZE_LOCK_TIMEOUT_MS},
                        "before": before, "after": after,
                        "file_size_equivalence": False,
                    }
            finally:
                connection.close()
        except PostgreSQLSandboxOperationsError:
            raise
        except Exception as exc:
            sqlstate = getattr(exc, "sqlstate", None)
            if sqlstate == "55P03":
                raise PostgreSQLSandboxOperationsError("PostgreSQL optimize failed: lock timeout; no SQLite fallback is permitted") from exc
            if sqlstate == "57014":
                raise PostgreSQLSandboxOperationsError("PostgreSQL optimize failed: statement timeout; no SQLite fallback is permitted") from exc
            if sqlstate == "42501":
                raise PostgreSQLSandboxOperationsError("PostgreSQL optimize failed: insufficient privilege; no SQLite fallback is permitted") from exc
            raise PostgreSQLSandboxOperationsError("PostgreSQL optimize failed without a SQLite fallback") from exc

    def backup(self, backup_root: Path, *, required_extensions: Iterable[str] = (), quiesced: bool = False) -> PostgreSQLLogicalBackup:
        """Create a custom-format schema dump and checksum manifest after an explicit quiesce gate."""
        if not quiesced:
            raise PostgreSQLSandboxOperationsError("PostgreSQL logical backup requires explicit --quiesced confirmation")
        required_extensions = tuple(required_extensions)
        with self._connect() as snapshot_connection:
            with snapshot_connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            snapshot = self._schema_snapshot(snapshot_connection, required_extensions=required_extensions)
            snapshot["delivery_ledger"] = self._delivery_snapshot(snapshot_connection)
            with snapshot_connection.cursor() as cursor:
                cursor.execute("SELECT pg_export_snapshot()")
                exported_snapshot = str(cursor.fetchone()[0])
            root = backup_root.expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            backup_id = f"pg18-{int(time.time())}-{uuid.uuid4().hex}"
            directory = root / backup_id
            directory.mkdir(mode=0o700)
            archive = directory / "tenant.dump"
            temporary = directory / "tenant.dump.partial"
            tool_connection, environment = self._tool_connection()
            schema_arguments = [f"--schema={self._schema}"]
            if snapshot["delivery_ledger"] and snapshot["delivery_ledger"]["present"]:
                schema_arguments.append(f"--schema={self._delivery_schema}")
            self._run_tool(
                "pg_dump",
                [*tool_connection, "--format=custom", "--no-owner", "--no-privileges", f"--snapshot={exported_snapshot}", *schema_arguments, f"--file={temporary}"],
                environment=environment,
            )
        if not temporary.is_file() or not temporary.stat().st_size:
            raise PostgreSQLSandboxOperationsError("PostgreSQL logical backup produced no archive")
        temporary.replace(archive)
        manifest = {
            "manifest_version": _MANIFEST_VERSION,
            "backup_id": backup_id,
            "backend": "postgresql",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": self._profile_identity,
            "archive": {"name": archive.name, "bytes": archive.stat().st_size, "sha256": _sha256(archive), "format": "pg_dump_custom"},
            "tenant": snapshot,
            "tool_versions": {"pg_dump": self._tool_version("pg_dump"), "pg_restore": self._tool_version("pg_restore")},
            "restore_contract": "new disposable database only; no --clean and no --create",
        }
        manifest_path = directory / "manifest.json"
        _json_atomic(manifest_path, manifest)
        return PostgreSQLLogicalBackup(directory, manifest_path, archive, manifest)

    def _read_manifest(self, backup_directory: Path) -> tuple[Path, dict[str, Any]]:
        directory = backup_directory.resolve()
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            archive = manifest["archive"]
            if (manifest["manifest_version"] != _MANIFEST_VERSION or manifest["backend"] != "postgresql"
                    or archive["format"] != "pg_dump_custom" or not isinstance(manifest["profile"], dict)):
                raise ValueError("unsupported manifest")
            if manifest["tenant"]["schema"] != self._schema:
                raise ValueError("tenant schema mismatch")
            archive_path = directory / str(archive["name"])
            if archive_path.parent != directory or archive_path.name != "tenant.dump":
                raise ValueError("unsafe archive path")
            if archive_path.stat().st_size != int(archive["bytes"]) or _sha256(archive_path) != archive["sha256"]:
                raise ValueError("archive checksum mismatch")
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise PostgreSQLSandboxOperationsError("PostgreSQL logical restore rejected its manifest") from exc
        return archive_path, manifest

    def _drop_database(self, database: str) -> None:
        token = self._restore_target_tokens.get(database)
        if token is None:
            raise PostgreSQLSandboxOperationsError("PostgreSQL restore teardown lacks an owned target marker")
        with self._connect(database=database) as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT token FROM public.{_quote_identifier(_RESTORE_MARKER_TABLE)}")
            if cursor.fetchall() != [(token,)]:
                raise PostgreSQLSandboxOperationsError("PostgreSQL restore teardown marker changed or is absent")
        with self._connect(database="postgres") as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid <> pg_backend_pid()", (database,))
                cursor.execute(f"DROP DATABASE IF EXISTS {_quote_identifier(database)}")

    def restore_and_verify(
        self, backup_directory: Path, *, target_database: str | None = None, keep_restored_target: bool = False,
    ) -> dict[str, Any]:
        """Restore only into a newly-created disposable database, verify, then remove it by default."""
        archive, manifest = self._read_manifest(backup_directory)
        database = target_database or f"hermes_state_restore_{uuid.uuid4().hex}"
        if not _TARGET_DATABASE_RE.fullmatch(database):
            raise PostgreSQLSandboxOperationsError("PostgreSQL restore target must be a generated isolated database name")
        created = False
        try:
            with self._connect(database="postgres") as connection:
                connection.autocommit = True
                with connection.cursor() as cursor:
                    cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname=%s)", (database,))
                    if cursor.fetchone()[0]:
                        raise PostgreSQLSandboxOperationsError("PostgreSQL restore target already exists")
                    cursor.execute(f"CREATE DATABASE {_quote_identifier(database)}")
                    created = True
            marker_token = secrets.token_urlsafe(32)
            self._restore_target_tokens[database] = marker_token
            with self._connect(database=database) as target_connection, target_connection.cursor() as cursor:
                cursor.execute(
                    f"CREATE TABLE public.{_quote_identifier(_RESTORE_MARKER_TABLE)} "
                    "(token text PRIMARY KEY, creator_scope text NOT NULL)"
                )
                cursor.execute(
                    f"INSERT INTO public.{_quote_identifier(_RESTORE_MARKER_TABLE)} (token, creator_scope) VALUES (%s, %s)",
                    (marker_token, f"restore:{database}"),
                )
                cursor.execute(f"SELECT token FROM public.{_quote_identifier(_RESTORE_MARKER_TABLE)}")
                if cursor.fetchall() != [(marker_token,)]:
                    raise PostgreSQLSandboxOperationsError("PostgreSQL restore target marker could not be validated")
                for extension in manifest["tenant"]["required_extensions"]:
                    cursor.execute(f"CREATE EXTENSION IF NOT EXISTS {_quote_identifier(str(extension))}")
                cursor.execute(f"CREATE SCHEMA {_quote_identifier(self._schema)}")
                if manifest["tenant"]["delivery_ledger"] and manifest["tenant"]["delivery_ledger"]["present"]:
                    assert self._delivery_schema is not None
                    cursor.execute(f"CREATE SCHEMA {_quote_identifier(self._delivery_schema)}")
            tool_connection, environment = self._tool_connection(database=database)
            schema_arguments = [f"--schema={self._schema}"]
            if manifest["tenant"]["delivery_ledger"] and manifest["tenant"]["delivery_ledger"]["present"]:
                schema_arguments.append(f"--schema={self._delivery_schema}")
            self._run_tool(
                "pg_restore",
                [*tool_connection, "--no-owner", "--no-privileges", "--exit-on-error", "--single-transaction", *schema_arguments, str(archive)],
                environment=environment,
            )
            restored_dsn = self._conninfo_module.make_conninfo(**{**self._conninfo, "dbname": database})
            restored = PostgreSQLSandboxOperations(
                self._settings, restored_dsn, schema=self._schema, delivery_schema=self._delivery_schema,
                profile_identity=self._profile_identity, command_runner=self._command_runner,
            )
            snapshot = restored.doctor(required_extensions=manifest["tenant"]["required_extensions"])
            expected = manifest["tenant"]
            for key in ("migration_versions", "table_counts", "search", "invariants", "delivery_ledger"):
                if snapshot[key] != expected[key]:
                    raise PostgreSQLSandboxOperationsError(f"PostgreSQL restore verification mismatch for {key}")
            return {"verified": True, "restored_database": database if keep_restored_target else None, "snapshot": snapshot}
        finally:
            if created and not keep_restored_target:
                self._drop_database(database)
            self._restore_target_tokens.pop(database, None)
