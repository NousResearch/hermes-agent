"""PostgreSQL target adapter for the state-postgres migration engine.

The adapter accepts only a DSN environment-variable name.  It delegates
connection construction to :class:`PostgresStateStore`, whose diagnostics are
already designed not to expose that environment value.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterator, Optional

from hermes_cli.state_postgres_migration import TableSpec
from state_store.postgres.core import PostgresStateStore
from state_store.postgres.ddl import (
    CORE_TABLES,
    SCHEMA_VERSION,
    TELEGRAM_TABLES,
    schema_statements,
)
from state_store.spec import StateStoreSpec


_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_CONTROL_TABLE = "state_postgres_migrations"
_KNOWN_TABLES = {table.name: table for table in CORE_TABLES + TELEGRAM_TABLES}
_CONTROL_COLUMN_CONTRACT = (
    ("run_id", "text", True, "", "", None),
    ("status", "text", True, "", "", None),
    ("target_schema", "text", True, "", "", None),
    ("staging_schema", "text", False, "", "", None),
    ("verified_source", "jsonb", False, "", "", None),
    ("report", "jsonb", False, "", "", None),
    ("last_failure", "jsonb", False, "", "", None),
    ("published_at", "timestamp with time zone", False, "", "", None),
    ("completed_at", "timestamp with time zone", False, "", "", None),
)


def _quote_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError("PostgreSQL migration identifier is invalid")
    return f'"{value}"'


def _generated_identifier(prefix: str, value: str) -> str:
    digest = sha256(value.encode("utf-8")).hexdigest()[:24]
    candidate = f"{prefix}_{digest}"
    if not _IDENTIFIER_RE.fullmatch(candidate):
        raise AssertionError("generated PostgreSQL identifier is invalid")
    return candidate


def _advisory_key(value: str) -> int:
    return int.from_bytes(
        sha256(value.encode("utf-8")).digest()[:8], "big", signed=False
    ) & ((1 << 63) - 1)


def _lexicographic_predicate(
    columns: Sequence[str], after_key: tuple[Any, ...]
) -> tuple[str, tuple[Any, ...]]:
    if len(columns) != len(after_key):
        raise ValueError("PostgreSQL keyset cursor does not match key columns")
    clauses: list[str] = []
    parameters: list[Any] = []
    for index, column in enumerate(columns):
        prefix = [
            f"{_quote_identifier(previous)} = %s" for previous in columns[:index]
        ]
        clauses.append(
            "(" + " AND ".join(prefix + [f"{_quote_identifier(column)} > %s"]) + ")"
        )
        parameters.extend(after_key[:index])
        parameters.append(after_key[index])
    return "(" + " OR ".join(clauses) + ")", tuple(parameters)


class PostgresMigrationTargetAdapter:
    """Production target adapter using the bounded ``PostgresStateStore``."""

    def __init__(
        self,
        *,
        dsn_env: str,
        schema: str,
        home: Path,
        store: Optional[PostgresStateStore] = None,
        environ: Optional[Mapping[str, str]] = None,
    ) -> None:
        if not _IDENTIFIER_RE.fullmatch(schema):
            raise ValueError("PostgreSQL migration schema must be a lowercase identifier")
        if not dsn_env or not (dsn_env[0].isalpha() or dsn_env[0] == "_") or not all(
            character.isalnum() or character == "_" for character in dsn_env
        ):
            raise ValueError("PostgreSQL migration DSN must be an environment-variable name")
        self.schema = schema
        self.dsn_env = dsn_env
        self._control_schema = _generated_identifier("hermes_state_mig", schema)
        self._store = store or PostgresStateStore(
            StateStoreSpec(
                home=Path(home),
                profile="default",
                backend="postgres",
                sqlite_path=Path(home) / "state.db",
                postgres_dsn_env=dsn_env,
                postgres_schema=schema,
                read_only=False,
            ),
            environ=environ,
        )
        self._lock_connection_cm: Any = None
        self._lock_connection: Any = None
        self._locked_run_id: Optional[str] = None
        self._staging_manifests: dict[str, tuple[str, ...]] = {}

    @property
    def redacted_identity(self) -> str:
        return f"postgres schema={self.schema} dsn_env={self.dsn_env}"

    def publish_schema_is_empty(self) -> bool:
        with self._transaction() as connection:
            # An empty pre-existing schema can still contain routines, types,
            # or ACLs that change how unqualified migration SQL resolves.
            return not self._namespace_exists(connection, self.schema)

    def acquire_advisory_lock(self, run_id: str) -> None:
        if self._lock_connection is not None:
            if self._locked_run_id != run_id:
                raise RuntimeError("another state-postgres migration lock is already held")
            return
        self._store.open()
        pool = getattr(self._store, "_pool", None)
        if pool is None:
            raise RuntimeError("PostgreSQL migration pool is unavailable")
        lease = pool.connection()
        connection = lease.__enter__()
        try:
            connection.execute(
                "SELECT pg_advisory_lock(%s)",
                (_advisory_key(f"hermes-state-postgres:{self.schema}"),),
            )
            connection.commit()
        except Exception:
            lease.__exit__(*__import__("sys").exc_info())
            raise
        self._lock_connection_cm = lease
        self._lock_connection = connection
        self._locked_run_id = run_id

    def release_advisory_lock(self, run_id: str) -> None:
        if self._lock_connection is None:
            return
        if self._locked_run_id != run_id:
            raise RuntimeError("state-postgres migration lock run ID does not match")
        connection = self._lock_connection
        lease = self._lock_connection_cm
        self._lock_connection = None
        self._lock_connection_cm = None
        self._locked_run_id = None
        try:
            connection.execute(
                "SELECT pg_advisory_unlock(%s)",
                (_advisory_key(f"hermes-state-postgres:{self.schema}"),),
            )
            connection.commit()
        finally:
            lease.__exit__(None, None, None)

    def create_or_resume_staging(
        self, run_id: str, manifest: Sequence[TableSpec]
    ) -> str:
        self._require_advisory_lock(run_id)
        staging_schema = _generated_identifier(
            f"{self.schema[:30]}_stage", run_id
        )
        include_telegram = any(
            table.name in {item.name for item in TELEGRAM_TABLES} for table in manifest
        )
        with self._transaction() as connection:
            self._ensure_managed_namespace(
                connection,
                staging_schema,
                purpose="staging",
                require_absent=True,
            )
            self._set_search_path(connection, staging_schema)
            for statement in schema_statements(include_telegram=include_telegram):
                connection.execute(statement)
            connection.execute("DELETE FROM schema_version")
            connection.execute("INSERT INTO schema_version (version) VALUES (%s)", (SCHEMA_VERSION,))
        self._staging_manifests[staging_schema] = tuple(table.name for table in manifest)
        return staging_schema

    def begin_session_parent_link_copy(self, staging_schema: str) -> None:
        with self._transaction() as connection:
            self._set_search_path(connection, staging_schema)
            # A source keyset can yield a child before its parent.  Recreate and
            # validate the FK only after every sessions batch is staged.
            connection.execute(
                "ALTER TABLE sessions DROP CONSTRAINT IF EXISTS "
                "sessions_parent_session_id_fkey"
            )

    def copy_rows(
        self, staging_schema: str, table: TableSpec, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        if not rows:
            return
        contract = _KNOWN_TABLES.get(table.name)
        if contract is None:
            raise ValueError(f"unknown PostgreSQL migration table {table.name!r}")
        columns = tuple(rows[0].keys())
        if not columns or set(columns) != set(contract.columns):
            raise ValueError(
                f"{table.name} source columns do not match the PostgreSQL durable contract"
            )
        if not set(table.key_columns).issubset(columns):
            raise ValueError(f"{table.name} migration key columns are missing")
        for row in rows:
            if set(row.keys()) != set(columns):
                raise ValueError(f"{table.name} batch has inconsistent row columns")
        quoted_columns = ", ".join(_quote_identifier(column) for column in columns)
        placeholders = ", ".join("%s" for _ in columns)
        conflict_columns = ", ".join(_quote_identifier(column) for column in table.key_columns)
        update_columns = [column for column in columns if column not in table.key_columns]
        if update_columns:
            updates = ", ".join(
                f"{_quote_identifier(column)} = EXCLUDED.{_quote_identifier(column)}"
                for column in update_columns
            )
            conflict_action = f"DO UPDATE SET {updates}"
        else:
            conflict_action = "DO NOTHING"
        statement = (
            f"INSERT INTO {_quote_identifier(staging_schema)}.{_quote_identifier(table.name)} "
            f"({quoted_columns}) VALUES ({placeholders}) "
            f"ON CONFLICT ({conflict_columns}) {conflict_action}"
        )
        with self._transaction() as connection:
            for row in rows:
                connection.execute(statement, tuple(row[column] for column in columns))

    def finalize_session_parent_links(self, staging_schema: str) -> None:
        with self._transaction() as connection:
            self._set_search_path(connection, staging_schema)
            connection.execute(
                "ALTER TABLE sessions ADD CONSTRAINT sessions_parent_session_id_fkey "
                "FOREIGN KEY (parent_session_id) REFERENCES sessions(id) NOT VALID"
            )
            connection.execute(
                "ALTER TABLE sessions VALIDATE CONSTRAINT sessions_parent_session_id_fkey"
            )

    def reset_identity(
        self, staging_schema: str, table: TableSpec, identity_column: str
    ) -> None:
        if table.name != "messages" or identity_column != "id":
            return
        with self._transaction() as connection:
            connection.execute(
                f"""
                SELECT setval(
                    pg_get_serial_sequence(%s, %s),
                    COALESCE(MAX({_quote_identifier(identity_column)}), 1),
                    MAX({_quote_identifier(identity_column)}) IS NOT NULL
                )
                FROM {_quote_identifier(staging_schema)}.{_quote_identifier(table.name)}
                """,
                (f"{staging_schema}.{table.name}", identity_column),
            )

    def fetchmany_keyset(
        self,
        staging_schema: str,
        table: TableSpec,
        columns: Sequence[str],
        after_key: Optional[tuple[Any, ...]],
        batch_size: int,
    ) -> list[Mapping[str, Any]]:
        if batch_size < 1:
            raise ValueError("PostgreSQL migration batch_size must be positive")
        selected = ", ".join(_quote_identifier(column) for column in columns)
        order_by = ", ".join(_quote_identifier(column) for column in table.key_columns)
        parameters: tuple[Any, ...] = ()
        where = ""
        if after_key is not None:
            predicate, parameters = _lexicographic_predicate(table.key_columns, after_key)
            where = f" WHERE {predicate}"
        with self._transaction() as connection:
            cursor = connection.execute(
                f"SELECT {selected} FROM {_quote_identifier(staging_schema)}."
                f"{_quote_identifier(table.name)}{where} ORDER BY {order_by} LIMIT %s",
                (*parameters, batch_size),
            )
            return self._cursor_mappings(cursor, columns, batch_size)

    def read_row(
        self,
        staging_schema: str,
        table: TableSpec,
        key: tuple[Any, ...],
        columns: Sequence[str],
    ) -> Optional[Mapping[str, Any]]:
        if len(key) != len(table.key_columns):
            raise ValueError("PostgreSQL migration probe key does not match key columns")
        selected = ", ".join(_quote_identifier(column) for column in columns)
        predicate = " AND ".join(
            f"{_quote_identifier(column)} = %s" for column in table.key_columns
        )
        with self._transaction() as connection:
            cursor = connection.execute(
                f"SELECT {selected} FROM {_quote_identifier(staging_schema)}."
                f"{_quote_identifier(table.name)} WHERE {predicate}",
                key,
            )
            row = cursor.fetchone()
        return self._row_mapping(row, columns) if row is not None else None

    def validate_foreign_keys(self, staging_schema: str) -> list[str]:
        checks = (
            (
                "sessions.parent_session_id",
                """
                SELECT child.id FROM sessions AS child
                LEFT JOIN sessions AS parent ON parent.id = child.parent_session_id
                WHERE child.parent_session_id IS NOT NULL AND parent.id IS NULL LIMIT 20
                """,
            ),
            (
                "messages.session_id",
                """
                SELECT child.id FROM messages AS child
                LEFT JOIN sessions AS parent ON parent.id = child.session_id
                WHERE parent.id IS NULL LIMIT 20
                """,
            ),
            (
                "session_model_usage.session_id",
                """
                SELECT child.session_id FROM session_model_usage AS child
                LEFT JOIN sessions AS parent ON parent.id = child.session_id
                WHERE parent.id IS NULL LIMIT 20
                """,
            ),
        )
        if "telegram_dm_topic_bindings" in self._staging_manifests.get(staging_schema, ()):
            checks += (
                (
                    "telegram_dm_topic_bindings.session_id",
                    """
                    SELECT child.session_id FROM telegram_dm_topic_bindings AS child
                    LEFT JOIN sessions AS parent ON parent.id = child.session_id
                    WHERE parent.id IS NULL LIMIT 20
                    """,
                ),
            )
        violations: list[str] = []
        with self._transaction() as connection:
            self._set_search_path(connection, staging_schema)
            for name, statement in checks:
                rows = self._cursor_values(connection.execute(statement), 20)
                violations.extend(f"{name}={value}" for value in rows)
        return violations

    def validate_session_lineage(self, staging_schema: str) -> list[str]:
        with self._transaction() as connection:
            self._set_search_path(connection, staging_schema)
            cursor = connection.execute(
                """
                WITH RECURSIVE lineage(start_id, current_id, parent_id, path, cycle) AS (
                    SELECT id, id, parent_session_id, ARRAY[id], false
                    FROM sessions
                    WHERE parent_session_id IS NOT NULL
                    UNION ALL
                    SELECT lineage.start_id, parent.id, parent.parent_session_id,
                           lineage.path || parent.id,
                           parent.id = ANY(lineage.path)
                    FROM lineage
                    JOIN sessions AS parent ON parent.id = lineage.parent_id
                    WHERE NOT lineage.cycle AND lineage.parent_id IS NOT NULL
                )
                SELECT DISTINCT start_id FROM lineage WHERE cycle LIMIT 20
                """
            )
            return [str(value) for value in self._cursor_values(cursor, 20)]

    def atomic_publish(
        self,
        staging_schema: str,
        run_id: str,
        verified_source: Mapping[str, Any],
    ) -> None:
        self._require_advisory_lock(run_id)
        manifest = self._staging_manifests.get(staging_schema)
        if manifest is None:
            raise RuntimeError("unknown state-postgres staging schema")
        publish_tables = ("schema_version", *manifest)
        with self._transaction() as connection:
            self._ensure_control_table(connection)
            cursor = connection.execute(
                f"SELECT status, verified_source FROM {_quote_identifier(self._control_schema)}."
                f"{_quote_identifier(_CONTROL_TABLE)} WHERE run_id = %s FOR UPDATE",
                (run_id,),
            )
            existing = cursor.fetchone()
            if existing is not None:
                status = self._row_value(existing, "status", 0)
                evidence = self._decode_json(self._row_value(existing, "verified_source", 1))
                if status in {"published", "completed"}:
                    if evidence != dict(verified_source):
                        raise RuntimeError(
                            "published state-postgres run has conflicting verified evidence"
                        )
                    self._assert_namespace_policy(
                        connection,
                        self.schema,
                        purpose="published target",
                    )
                    self._reject_namespace_nonrelation_objects(
                        connection,
                        self.schema,
                        purpose="published target",
                    )
                    return
            self._ensure_managed_namespace(
                connection,
                self.schema,
                purpose="target publish",
                require_absent=True,
            )
            for table_name in publish_tables:
                connection.execute(
                    f"ALTER TABLE {_quote_identifier(staging_schema)}."
                    f"{_quote_identifier(table_name)} SET SCHEMA {_quote_identifier(self.schema)}"
                )
            evidence = json.dumps(dict(verified_source), sort_keys=True, separators=(",", ":"))
            connection.execute(
                f"""
                INSERT INTO {_quote_identifier(self._control_schema)}.{_quote_identifier(_CONTROL_TABLE)}
                    (run_id, status, target_schema, staging_schema, verified_source, published_at)
                VALUES (%s, 'published', %s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (run_id) DO UPDATE SET
                    status = 'published',
                    target_schema = EXCLUDED.target_schema,
                    staging_schema = EXCLUDED.staging_schema,
                    verified_source = EXCLUDED.verified_source,
                    published_at = EXCLUDED.published_at,
                    last_failure = NULL
                """,
                (run_id, self.schema, staging_schema, evidence),
            )

    def published_run_report(self, run_id: str) -> Optional[Mapping[str, Any]]:
        with self._transaction() as connection:
            self._ensure_control_table(connection)
            cursor = connection.execute(
                f"""
                SELECT run_id, status, verified_source, report
                FROM {_quote_identifier(self._control_schema)}.{_quote_identifier(_CONTROL_TABLE)}
                WHERE run_id = %s AND status IN ('published', 'completed')
                """,
                (run_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            self._assert_namespace_policy(
                connection,
                self.schema,
                purpose="published target",
            )
            self._reject_namespace_nonrelation_objects(
                connection,
                self.schema,
                purpose="published target",
            )
        return {
            "run_id": self._row_value(row, "run_id", 0),
            "status": self._row_value(row, "status", 1),
            "verified_source": self._decode_json(
                self._row_value(row, "verified_source", 2)
            ),
            "report": self._decode_json(self._row_value(row, "report", 3)),
        }

    def record_success(self, report: Mapping[str, Any]) -> None:
        run_id = self._report_run_id(report)
        report_json = json.dumps(dict(report), sort_keys=True, separators=(",", ":"))
        with self._transaction() as connection:
            self._ensure_control_table(connection)
            cursor = connection.execute(
                f"""
                UPDATE {_quote_identifier(self._control_schema)}.{_quote_identifier(_CONTROL_TABLE)}
                SET status = 'completed', report = %s::jsonb, completed_at = CURRENT_TIMESTAMP,
                    last_failure = NULL
                WHERE run_id = %s AND status IN ('published', 'completed')
                """,
                (report_json, run_id),
            )
            if getattr(cursor, "rowcount", 1) == 0:
                raise RuntimeError("cannot mark an unpublished state-postgres run successful")

    def record_failure(self, report: Mapping[str, Any]) -> None:
        run_id = self._report_run_id(report)
        report_json = json.dumps(dict(report), sort_keys=True, separators=(",", ":"))
        with self._transaction() as connection:
            self._ensure_control_table(connection)
            connection.execute(
                f"""
                INSERT INTO {_quote_identifier(self._control_schema)}.{_quote_identifier(_CONTROL_TABLE)}
                    (run_id, status, target_schema, last_failure)
                VALUES (%s, 'failed', %s, %s::jsonb)
                ON CONFLICT (run_id) DO UPDATE SET
                    last_failure = EXCLUDED.last_failure
                """,
                (run_id, self.schema, report_json),
            )

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        with self._store.transaction(
            read_only=False,
            configure_search_path=False,
        ) as connection:
            yield connection

    def _ensure_control_table(self, connection: Any) -> None:
        created = self._ensure_managed_namespace(
            connection,
            self._control_schema,
            purpose="migration control",
            require_absent=False,
        )
        if created:
            connection.execute(
                f"""
                CREATE TABLE {_quote_identifier(self._control_schema)}.
                    {_quote_identifier(_CONTROL_TABLE)} (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    target_schema TEXT NOT NULL,
                    staging_schema TEXT,
                    verified_source JSONB,
                    report JSONB,
                    last_failure JSONB,
                    published_at TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ
                )
                """
            )
        self._assert_control_table_contract(connection)

    @staticmethod
    def _set_search_path(connection: Any, schema: str) -> None:
        connection.execute(
            f"SET LOCAL search_path TO {_quote_identifier(schema)}, pg_catalog"
        )

    def _require_advisory_lock(self, run_id: str) -> None:
        if self._lock_connection is None or self._locked_run_id != run_id:
            raise RuntimeError(
                "state-postgres migration namespace admission requires its advisory lock"
            )
        try:
            self._lock_connection.execute("SELECT 1")
            self._lock_connection.commit()
        except Exception:
            raise RuntimeError(
                "state-postgres migration advisory lock connection was lost"
            ) from None

    @staticmethod
    def _namespace_exists(connection: Any, schema: str) -> bool:
        cursor = connection.execute(
            """
            SELECT 1
            FROM pg_catalog.pg_namespace AS namespace
            WHERE namespace.nspname = %s
            LIMIT 1
            """,
            (schema,),
        )
        return cursor.fetchone() is not None

    def _ensure_managed_namespace(
        self,
        connection: Any,
        schema: str,
        *,
        purpose: str,
        require_absent: bool,
    ) -> bool:
        created = False
        if self._namespace_exists(connection, schema):
            if require_absent:
                raise RuntimeError(
                    f"{purpose} schema already exists; a fresh namespace is required"
                )
        else:
            connection.execute(
                f"CREATE SCHEMA {_quote_identifier(schema)} AUTHORIZATION CURRENT_USER"
            )
            created = True
        self._assert_namespace_policy(connection, schema, purpose=purpose)
        self._reject_namespace_nonrelation_objects(connection, schema, purpose=purpose)
        return created

    def _assert_control_table_contract(self, connection: Any) -> None:
        cursor = connection.execute(
            """
            WITH namespace AS (
                SELECT oid
                FROM pg_catalog.pg_namespace
                WHERE nspname = %s
            ),
            relation AS (
                SELECT relation.*
                FROM pg_catalog.pg_class AS relation
                WHERE relation.relnamespace = (SELECT oid FROM namespace)
                  AND relation.relname = %s
            ),
            columns AS (
                SELECT pg_catalog.jsonb_agg(
                    pg_catalog.jsonb_build_array(
                        attribute.attname,
                        pg_catalog.format_type(
                            attribute.atttypid, attribute.atttypmod
                        ),
                        attribute.attnotnull,
                        attribute.attidentity,
                        attribute.attgenerated,
                        pg_catalog.pg_get_expr(
                            default_expr.adbin, default_expr.adrelid
                        )
                    )
                    ORDER BY attribute.attnum
                ) AS signature
                FROM relation
                JOIN pg_catalog.pg_attribute AS attribute
                  ON attribute.attrelid = relation.oid
                LEFT JOIN pg_catalog.pg_attrdef AS default_expr
                  ON default_expr.adrelid = attribute.attrelid
                 AND default_expr.adnum = attribute.attnum
                WHERE attribute.attnum > 0
                  AND NOT attribute.attisdropped
            )
            SELECT (
                (SELECT count(*)
                 FROM pg_catalog.pg_class AS item
                 WHERE item.relnamespace = (SELECT oid FROM namespace)
                   AND item.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')) = 1
                AND relation.relkind = 'r'
                AND relation.relpersistence = 'p'
                AND relation.relacl IS NULL
                AND relation.reloptions IS NULL
                AND NOT relation.relrowsecurity
                AND NOT relation.relforcerowsecurity
                AND NOT relation.relispartition
                AND NOT relation.relhassubclass
                AND columns.signature = %s::jsonb
                AND (
                    SELECT count(*)
                    FROM pg_catalog.pg_constraint AS constraint_meta
                    WHERE constraint_meta.conrelid = relation.oid
                ) = 1
                AND EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_constraint AS constraint_meta
                    JOIN pg_catalog.pg_attribute AS attribute
                      ON attribute.attrelid = relation.oid
                     AND attribute.attname = 'run_id'
                    WHERE constraint_meta.conrelid = relation.oid
                      AND constraint_meta.contype = 'p'
                      AND constraint_meta.conkey = ARRAY[attribute.attnum]::smallint[]
                )
                AND NOT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_inherits AS inheritance
                    WHERE inheritance.inhrelid = relation.oid
                       OR inheritance.inhparent = relation.oid
                )
            ) AS contract_valid
            FROM relation
            CROSS JOIN columns
            """,
            (
                self._control_schema,
                _CONTROL_TABLE,
                json.dumps(_CONTROL_COLUMN_CONTRACT),
            ),
        )
        row = cursor.fetchone()
        if row is None or not bool(self._row_value(row, "contract_valid", 0)):
            raise RuntimeError(
                "migration control table does not match the authored contract"
            )

    @staticmethod
    def _assert_namespace_policy(connection: Any, schema: str, *, purpose: str) -> None:
        cursor = connection.execute(
            """
            SELECT
                namespace.nspowner = (
                    SELECT role.oid
                    FROM pg_catalog.pg_roles AS role
                    WHERE role.rolname = CURRENT_USER
                ) AS owned_by_current_user,
                has_schema_privilege(namespace.oid, 'CREATE') AS current_user_can_create,
                namespace.nspacl IS NULL AS has_default_acl
            FROM pg_catalog.pg_namespace AS namespace
            WHERE namespace.nspname = %s
            """,
            (schema,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"{purpose} schema is missing after admission")
        owned_by_current_user = bool(
            PostgresMigrationTargetAdapter._row_value(row, "owned_by_current_user", 0)
        )
        current_user_can_create = bool(
            PostgresMigrationTargetAdapter._row_value(row, "current_user_can_create", 1)
        )
        has_default_acl = bool(
            PostgresMigrationTargetAdapter._row_value(row, "has_default_acl", 2)
        )
        if not owned_by_current_user:
            raise RuntimeError(
                f"{purpose} schema must be owned by the current database user"
            )
        if not current_user_can_create:
            raise RuntimeError(
                f"{purpose} schema must grant CREATE to the current database user"
            )
        if not has_default_acl:
            raise RuntimeError(
                f"{purpose} schema has explicit ACLs; namespace reuse is refused"
            )

    @staticmethod
    def _reject_namespace_nonrelation_objects(
        connection: Any,
        schema: str,
        *,
        purpose: str,
    ) -> None:
        cursor = connection.execute(
            """
            WITH namespace AS (
                SELECT oid
                FROM pg_catalog.pg_namespace
                WHERE nspname = %s
            )
            SELECT object_kind
            FROM (
                SELECT 'relation with unsafe owner or ACL' AS object_kind
                FROM pg_catalog.pg_class AS relation
                WHERE relation.relnamespace = (SELECT oid FROM namespace)
                  AND (
                    relation.relowner <> (
                        SELECT role.oid
                        FROM pg_catalog.pg_roles AS role
                        WHERE role.rolname = CURRENT_USER
                    )
                    OR relation.relacl IS NOT NULL
                  )
                UNION ALL
                SELECT 'function or procedure' AS object_kind
                FROM pg_catalog.pg_proc
                WHERE pronamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'type with unsafe owner or ACL'
                FROM pg_catalog.pg_type AS type_
                WHERE type_.typnamespace = (SELECT oid FROM namespace)
                  AND (
                    type_.typowner <> (
                        SELECT role.oid
                        FROM pg_catalog.pg_roles AS role
                        WHERE role.rolname = CURRENT_USER
                    )
                    OR type_.typacl IS NOT NULL
                  )
                UNION ALL
                SELECT CASE WHEN type_.typtype = 'd' THEN 'domain' ELSE 'type' END
                FROM pg_catalog.pg_type AS type_
                LEFT JOIN pg_catalog.pg_type AS element
                  ON element.oid = type_.typelem
                WHERE type_.typnamespace = (SELECT oid FROM namespace)
                  AND type_.typrelid = 0
                  AND (type_.typelem = 0 OR element.typrelid = 0)
                UNION ALL
                SELECT 'collation'
                FROM pg_catalog.pg_collation
                WHERE collnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'conversion'
                FROM pg_catalog.pg_conversion
                WHERE connamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'operator'
                FROM pg_catalog.pg_operator
                WHERE oprnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'operator class'
                FROM pg_catalog.pg_opclass
                WHERE opcnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'operator family'
                FROM pg_catalog.pg_opfamily
                WHERE opfnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'text search dictionary'
                FROM pg_catalog.pg_ts_dict
                WHERE dictnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'text search configuration'
                FROM pg_catalog.pg_ts_config
                WHERE cfgnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'text search parser'
                FROM pg_catalog.pg_ts_parser
                WHERE prsnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'text search template'
                FROM pg_catalog.pg_ts_template
                WHERE tmplnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'extension'
                FROM pg_catalog.pg_extension
                WHERE extnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'trigger'
                FROM pg_catalog.pg_trigger AS trigger
                JOIN pg_catalog.pg_class AS relation
                  ON relation.oid = trigger.tgrelid
                WHERE relation.relnamespace = (SELECT oid FROM namespace)
                  AND NOT trigger.tgisinternal
                UNION ALL
                SELECT 'row security policy'
                FROM pg_catalog.pg_policy AS policy
                JOIN pg_catalog.pg_class AS relation
                  ON relation.oid = policy.polrelid
                WHERE relation.relnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'rewrite rule'
                FROM pg_catalog.pg_rewrite AS rewrite
                JOIN pg_catalog.pg_class AS relation
                  ON relation.oid = rewrite.ev_class
                WHERE relation.relnamespace = (SELECT oid FROM namespace)
                UNION ALL
                SELECT 'default privilege'
                FROM pg_catalog.pg_default_acl
                WHERE defaclrole = (
                    SELECT role.oid
                    FROM pg_catalog.pg_roles AS role
                    WHERE role.rolname = CURRENT_USER
                )
                  AND defaclnamespace IN ((SELECT oid FROM namespace), 0)
            ) AS namespace_objects
            LIMIT 1
            """,
            (schema,),
        )
        row = cursor.fetchone()
        if row is not None:
            object_kind = PostgresMigrationTargetAdapter._row_value(
                row, "object_kind", 0
            )
            raise RuntimeError(
                f"{purpose} schema contains a {object_kind}; namespace reuse is refused"
            )

    @staticmethod
    def _cursor_mappings(
        cursor: Any, columns: Sequence[str], limit: int
    ) -> list[Mapping[str, Any]]:
        fetchmany = getattr(cursor, "fetchmany", None)
        rows = fetchmany(limit) if fetchmany is not None else []
        return [
            PostgresMigrationTargetAdapter._row_mapping(row, columns) for row in rows
        ]

    @staticmethod
    def _cursor_values(cursor: Any, limit: int) -> list[Any]:
        fetchmany = getattr(cursor, "fetchmany", None)
        rows = fetchmany(limit) if fetchmany is not None else []
        return [PostgresMigrationTargetAdapter._row_value(row, "", 0) for row in rows]

    @staticmethod
    def _row_mapping(row: Any, columns: Sequence[str]) -> Mapping[str, Any]:
        if isinstance(row, Mapping):
            return dict(row)
        return {column: row[index] for index, column in enumerate(columns)}

    @staticmethod
    def _row_value(row: Any, name: str, index: int) -> Any:
        if isinstance(row, Mapping):
            return row.get(name)
        return row[index]

    @staticmethod
    def _decode_json(value: Any) -> Any:
        if value is None or isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            return json.loads(value)
        return value

    @staticmethod
    def _report_run_id(report: Mapping[str, Any]) -> str:
        run_id = report.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("state-postgres migration report has no run ID")
        return run_id
