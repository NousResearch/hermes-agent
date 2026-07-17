"""Synchronous PostgreSQL state-store lifecycle with lazy psycopg imports."""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import os
import threading
import time
from typing import Any, Optional, Protocol

from state_store.schema import SCHEMA_V22_MANIFEST
from state_store.spec import StateStoreSpec

from state_store.postgres.ddl import (
    CORE_INDEXES,
    CORE_INDEX_CONTRACTS,
    CORE_TABLES,
    POSTGRES_COLUMN_CONTRACTS,
    SCHEMA_VERSION,
    TELEGRAM_INDEXES,
    TELEGRAM_INDEX_CONTRACTS,
    TELEGRAM_TABLES,
    V16_TO_V22_COLUMN_UPGRADES,
    schema_table_statements,
)


class PostgresStateStoreError(RuntimeError):
    """Base error whose text is safe for user-facing diagnostics."""


class PostgresDriverUnavailableError(PostgresStateStoreError):
    """Raised only when PostgreSQL is selected without psycopg installed."""


class PostgresConfigurationError(PostgresStateStoreError):
    """Raised for invalid non-secret PostgreSQL backend configuration."""


class PostgresConnectionError(PostgresStateStoreError):
    """Raised when a connection operation fails without exposing the DSN."""


class PostgresPoolTimeoutError(PostgresStateStoreError):
    """Raised when the bounded connection pool cannot lease a connection."""


class PostgresReadOnlyError(PostgresStateStoreError):
    """Raised when schema mutation is requested from a read-only store."""


class PostgresSchemaVersionError(PostgresStateStoreError):
    """Raised when an existing schema is newer than this implementation."""


class PostgresSchemaValidationError(PostgresStateStoreError):
    """Raised when a durable schema cannot satisfy the v22 contract."""


class _CursorLike(Protocol):
    def fetchone(self) -> Any:
        """Return one result row."""

    def fetchall(self) -> list[Any]:
        """Return every result row."""


class _ConnectionLike(Protocol):
    def close(self) -> None:
        """Close the connection."""

    def execute(self, query: str, params: Any = None) -> _CursorLike:
        """Execute one statement."""

    def transaction(self) -> Any:
        """Return a transaction context manager."""


class _PoolLike(Protocol):
    def close(self) -> None:
        """Close all pool resources."""

    def connection(self) -> Any:
        """Return a leased connection context manager."""


Connector = Callable[..., _ConnectionLike]
DriverLoader = Callable[[], Any]
PoolFactory = Callable[..., _PoolLike]

_REQUIRED_PRIMARY_KEYS = {
    "sessions": ("id",),
    "messages": ("id",),
    "session_model_usage": (
        "session_id",
        "model",
        "billing_provider",
        "billing_base_url",
        "billing_mode",
        "task",
    ),
    "state_meta": ("key",),
    "gateway_routing": ("scope", "session_key"),
    "compression_locks": ("session_id",),
    "async_delegations": ("delegation_id",),
}
_REQUIRED_TELEGRAM_PRIMARY_KEYS = {
    "telegram_dm_topic_mode": ("chat_id",),
    "telegram_dm_topic_bindings": ("chat_id", "thread_id"),
}
_REQUIRED_FOREIGN_KEYS = {
    ("sessions", "parent_session_id", "sessions", "id"): ("a", False, False),
    ("messages", "session_id", "sessions", "id"): ("a", False, False),
    ("session_model_usage", "session_id", "sessions", "id"): ("c", False, False),
}
_REQUIRED_TELEGRAM_FOREIGN_KEYS = {
    ("telegram_dm_topic_bindings", "session_id", "sessions", "id"): (
        "c",
        False,
        False,
    ),
}
_REQUIRED_NOT_NULL_COLUMNS = {
    "schema_version": {"version"},
    "sessions": {
        "id",
        "source",
        "started_at",
        "compression_fallback_streak",
        "rewind_count",
        "archived",
    },
    "messages": {"id", "session_id", "role", "timestamp", "active", "compacted"},
    "session_model_usage": {
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
    },
    "state_meta": {"key"},
    "gateway_routing": {"scope", "session_key", "entry_json", "updated_at"},
    "compression_locks": {"session_id", "holder", "acquired_at", "expires_at"},
    "async_delegations": {
        "delegation_id",
        "origin_session",
        "origin_ui_session_id",
        "state",
        "dispatched_at",
        "updated_at",
        "delivery_state",
        "delivery_attempts",
    },
}
_REQUIRED_TELEGRAM_NOT_NULL_COLUMNS = {
    "telegram_dm_topic_mode": {
        "chat_id",
        "user_id",
        "enabled",
        "activated_at",
        "updated_at",
    },
    "telegram_dm_topic_bindings": {
        "chat_id",
        "thread_id",
        "user_id",
        "session_key",
        "session_id",
        "managed_mode",
        "linked_at",
        "updated_at",
    },
}
_REQUIRED_CORE_INDEXES = {
    "idx_sessions_source",
    "idx_sessions_source_id",
    "idx_sessions_parent",
    "idx_sessions_started",
    "idx_messages_session",
    "idx_compression_locks_expires",
    "idx_session_model_usage_session",
    "idx_session_model_usage_model",
    "idx_async_delegations_delivery",
    "idx_messages_session_active",
    "idx_messages_active_null",
    "idx_sessions_session_key",
    "idx_sessions_gateway_peer",
    "idx_sessions_handoff_state",
    "idx_messages_platform_msg_id",
    "idx_sessions_title_unique",
}
_REQUIRED_TELEGRAM_INDEXES = {
    "idx_telegram_dm_topic_bindings_session",
    "idx_telegram_dm_topic_bindings_user",
}


@dataclass(frozen=True)
class MigrationReport:
    """Outcome of one idempotent schema lifecycle run."""

    previous_version: Optional[int]
    version: int
    telegram_enabled: bool


@dataclass(frozen=True)
class PostgresHealthReport:
    """Secret-safe health and backend-capability result."""

    available: bool
    schema: str
    schema_version: Optional[int]
    target_version: int
    read_only: bool
    capabilities: Mapping[str, bool]
    detail: Optional[str] = None


class BoundedConnectionPool:
    """Small synchronous pool that never opens more than ``max_size`` sockets."""

    def __init__(
        self,
        connector: Callable[[], _ConnectionLike],
        *,
        max_size: int,
        acquire_timeout_s: float,
    ) -> None:
        self._connector = connector
        self._max_size = max_size
        self._acquire_timeout_s = acquire_timeout_s
        self._idle: list[_ConnectionLike] = []
        self._created = 0
        self._closed = False
        self._condition = threading.Condition()

    @contextmanager
    def connection(self) -> Iterator[_ConnectionLike]:
        connection = self._acquire()
        discard = False
        try:
            yield connection
        except BaseException:
            # A failed transaction is aborted in PostgreSQL and cannot be reused.
            discard = True
            raise
        finally:
            self._release(connection, discard=discard)

    def _acquire(self) -> _ConnectionLike:
        deadline = time.monotonic() + self._acquire_timeout_s
        while True:
            stale_connection: Optional[_ConnectionLike] = None
            should_connect = False
            with self._condition:
                if self._closed:
                    raise PostgresConnectionError("PostgreSQL state store is closed")
                if self._idle:
                    connection = self._idle.pop()
                    if not self._connection_is_closed(connection):
                        return connection
                    self._created -= 1
                    stale_connection = connection
                    self._condition.notify()
                elif self._created < self._max_size:
                    self._created += 1
                    should_connect = True
                    break
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PostgresPoolTimeoutError(
                            "PostgreSQL state connection pool is exhausted"
                        )
                    self._condition.wait(remaining)
            if stale_connection is not None:
                self._close_connection(stale_connection)

        if should_connect:
            try:
                return self._connector()
            except Exception:
                with self._condition:
                    self._created -= 1
                    self._condition.notify()
                raise
        raise AssertionError("connection acquisition reached an invalid state")

    def _release(self, connection: _ConnectionLike, *, discard: bool) -> None:
        close_connection = False
        with self._condition:
            if self._closed or discard or self._connection_is_closed(connection):
                self._created -= 1
                close_connection = True
            else:
                self._idle.append(connection)
            self._condition.notify()
        if close_connection:
            self._close_connection(connection)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            idle = self._idle
            self._idle = []
            self._created -= len(idle)
            self._condition.notify_all()
        for connection in idle:
            self._close_connection(connection)

    @staticmethod
    def _connection_is_closed(connection: _ConnectionLike) -> bool:
        try:
            return bool(getattr(connection, "closed", False))
        except Exception:
            return True

    @staticmethod
    def _close_connection(connection: _ConnectionLike) -> None:
        try:
            connection.close()
        except Exception:
            pass


class PostgresStateStore:
    """Own a bounded synchronous PostgreSQL pool for one explicit state spec."""

    def __init__(
        self,
        spec: StateStoreSpec,
        *,
        environ: Optional[Mapping[str, str]] = None,
        connector: Optional[Connector] = None,
        driver_loader: Optional[DriverLoader] = None,
        pool_factory: Optional[PoolFactory] = None,
        max_pool_size: int = 4,
        acquire_timeout_s: float = 5.0,
        connect_timeout_s: int = 5,
        statement_timeout_ms: int = 30_000,
        lock_timeout_ms: int = 5_000,
        idle_in_transaction_timeout_ms: int = 30_000,
    ) -> None:
        self.spec = spec
        self._schema = self._validated_schema(spec)
        self._environ = os.environ if environ is None else environ
        self._connector = connector
        self._driver_loader = driver_loader or self._load_psycopg
        self._pool_factory = pool_factory or BoundedConnectionPool
        self._pool: Optional[_PoolLike] = None
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._max_pool_size = self._positive_int(max_pool_size, "max_pool_size")
        self._acquire_timeout_s = self._positive_float(
            acquire_timeout_s, "acquire_timeout_s"
        )
        self._connect_timeout_s = self._positive_int(
            connect_timeout_s, "connect_timeout_s"
        )
        self._statement_timeout_ms = self._positive_int(
            statement_timeout_ms, "statement_timeout_ms"
        )
        self._lock_timeout_ms = self._positive_int(lock_timeout_ms, "lock_timeout_ms")
        self._idle_in_transaction_timeout_ms = self._positive_int(
            idle_in_transaction_timeout_ms,
            "idle_in_transaction_timeout_ms",
        )

    @property
    def application_name(self) -> str:
        """Stable non-secret application identity passed to psycopg."""

        fingerprint = sha256(self.spec.store_key.encode("utf-8")).hexdigest()[:16]
        return f"hermes-state-{fingerprint}"

    @property
    def closed(self) -> bool:
        """Whether this store has released its pool and cannot be reused."""

        return self._closed

    def open(self) -> None:
        """Create the bounded pool lazily without opening a database socket."""

        with self._lifecycle_lock:
            if self._closed:
                raise PostgresConnectionError("PostgreSQL state store is closed")
            if self._pool is not None:
                return
            dsn = self._environ.get(self.spec.postgres_dsn_env)
            if not dsn:
                raise PostgresConfigurationError(
                    "PostgreSQL state DSN is not available from its configured environment"
                )

            def connect() -> _ConnectionLike:
                try:
                    if self._connector is not None:
                        return self._connector(
                            dsn,
                            connect_timeout=self._connect_timeout_s,
                            application_name=self.application_name,
                        )
                    driver = self._driver_loader()
                    return driver.connect(
                        dsn,
                        connect_timeout=self._connect_timeout_s,
                        application_name=self.application_name,
                    )
                except PostgresStateStoreError:
                    raise
                except Exception:
                    raise PostgresConnectionError(
                        "Unable to connect to the PostgreSQL state store"
                    ) from None

            try:
                self._pool = self._pool_factory(
                    connector=connect,
                    max_size=self._max_pool_size,
                    acquire_timeout_s=self._acquire_timeout_s,
                )
            except TypeError:
                # A simple injected fake may accept positional factory arguments.
                self._pool = self._pool_factory(
                    connect, self._max_pool_size, self._acquire_timeout_s
                )

    @contextmanager
    def transaction(
        self,
        *,
        read_only: Optional[bool] = None,
        configure_search_path: bool = True,
    ) -> Iterator[_ConnectionLike]:
        """Lease a connection and apply bounded session/transaction settings."""

        if self.spec.read_only and read_only is False:
            raise PostgresReadOnlyError(
                "Cannot request a writable PostgreSQL state transaction"
            )
        self.open()
        effective_read_only = self.spec.read_only if read_only is None else read_only
        with self._lifecycle_lock:
            if self._closed:
                raise PostgresConnectionError("PostgreSQL state store is closed")
            if self._pool is None:
                raise AssertionError("PostgreSQL pool was not created")
            pool = self._pool
        operation_started = False
        try:
            with pool.connection() as connection:
                with connection.transaction():
                    self._configure_transaction(
                        connection,
                        read_only=bool(effective_read_only),
                        configure_search_path=configure_search_path,
                    )
                    operation_started = True
                    yield connection
        except PostgresStateStoreError:
            raise
        except Exception as exc:
            if operation_started and not self._is_driver_exception(exc):
                raise
            raise PostgresConnectionError(
                "PostgreSQL state operation failed"
            ) from None

    def migrate(self, *, include_telegram: bool = False) -> MigrationReport:
        """Create or advance this PostgreSQL schema to durable version 22."""

        if self.spec.read_only:
            raise PostgresReadOnlyError(
                "Cannot migrate a read-only PostgreSQL state store"
            )
        with self.transaction(
            read_only=False,
            configure_search_path=False,
        ) as connection:
            connection.execute(
                f"CREATE SCHEMA IF NOT EXISTS {self._quoted_schema}"
            )
            self._set_search_path(connection)
            connection.execute(
                "SELECT pg_advisory_xact_lock(%s)",
                (self._migration_lock_key(),),
            )
            # Establish the version table before deciding whether later DDL is safe.
            connection.execute(CORE_TABLES[0].create_sql)
            row = connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1 "
                "FOR UPDATE"
            ).fetchone()
            previous_version = self._row_value(row)
            if previous_version is not None:
                try:
                    previous_version = int(previous_version)
                except (TypeError, ValueError):
                    raise PostgresSchemaValidationError(
                        "PostgreSQL state schema version is invalid"
                    ) from None
                if previous_version > SCHEMA_VERSION:
                    raise PostgresSchemaVersionError(
                        "PostgreSQL state schema is newer than this Hermes build"
                    )
            self._apply_schema_upgrade(connection, include_telegram=include_telegram)
            self._validate_schema(connection, include_telegram=include_telegram)
            if previous_version is not None:
                if previous_version < SCHEMA_VERSION:
                    connection.execute(
                        "UPDATE schema_version SET version = %s",
                        (SCHEMA_VERSION,),
                    )
            else:
                connection.execute(
                    "INSERT INTO schema_version (version) VALUES (%s)",
                    (SCHEMA_VERSION,),
                )
            return MigrationReport(
                previous_version=previous_version,
                version=SCHEMA_VERSION,
                telegram_enabled=include_telegram,
            )

    def validate_schema(self, *, include_telegram: bool = False) -> None:
        """Validate an existing durable schema without applying any DDL."""

        with self.transaction(
            read_only=True,
            configure_search_path=False,
        ) as connection:
            schema_exists, version_table_exists = self._row_fields(
                connection.execute(
                    """
                    SELECT
                        to_regnamespace(%s) IS NOT NULL AS schema_exists,
                        to_regclass(%s) IS NOT NULL AS version_table_exists
                    """,
                    (self._schema, f"{self._schema}.schema_version"),
                ).fetchone(),
                ("schema_exists", "version_table_exists"),
            )
            if not schema_exists or not version_table_exists:
                raise PostgresSchemaValidationError(
                    "PostgreSQL state schema is missing or not initialized"
                )
            self._set_search_path(connection)
            row = connection.execute(
                "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
            ).fetchone()
            version = self._row_value(row)
            try:
                version = int(version) if version is not None else None
            except (TypeError, ValueError):
                raise PostgresSchemaValidationError(
                    "PostgreSQL state schema version is invalid"
                ) from None
            if version != SCHEMA_VERSION:
                if version is not None and version > SCHEMA_VERSION:
                    raise PostgresSchemaVersionError(
                        "PostgreSQL state schema is newer than this Hermes build"
                    )
                raise PostgresSchemaValidationError(
                    "PostgreSQL state schema version is not current"
                )
            self._validate_schema(connection, include_telegram=include_telegram)

    def _apply_schema_upgrade(
        self,
        connection: _ConnectionLike,
        *,
        include_telegram: bool,
    ) -> None:
        """Apply the ordered, idempotent v16-to-v22 structural upgrade."""

        for statement in schema_table_statements(include_telegram=include_telegram):
            connection.execute(statement)
        for upgrade in V16_TO_V22_COLUMN_UPGRADES:
            connection.execute(upgrade.add_sql)
        # The experimental v16 backend used INTEGER for the message identity.
        # Widening is lossless and makes the v22 identity contract explicit.
        connection.execute("ALTER TABLE messages ALTER COLUMN id TYPE BIGINT")
        self._ensure_foreign_key_contract(
            connection,
            include_telegram=include_telegram,
        )
        for statement in CORE_INDEXES:
            connection.execute(statement)
        if include_telegram:
            for statement in TELEGRAM_INDEXES:
                connection.execute(statement)

    def _validate_schema(
        self,
        connection: _ConnectionLike,
        *,
        include_telegram: bool,
    ) -> None:
        """Verify every durable v22 structural requirement before versioning."""

        tables = CORE_TABLES + (TELEGRAM_TABLES if include_telegram else ())
        required_columns = {
            table.name: POSTGRES_COLUMN_CONTRACTS[table.name] for table in tables
        }
        column_rows = connection.execute(
            """
            SELECT
                relation.relname AS table_name,
                attribute.attname AS column_name,
                format_type(attribute.atttypid, attribute.atttypmod) AS type_name,
                attribute.attnotnull AS not_null,
                attribute.attidentity AS identity,
                pg_get_expr(default_expr.adbin, default_expr.adrelid) AS default_expr
            FROM pg_attribute AS attribute
            JOIN pg_class AS relation ON relation.oid = attribute.attrelid
            JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            LEFT JOIN pg_attrdef AS default_expr
              ON default_expr.adrelid = attribute.attrelid
             AND default_expr.adnum = attribute.attnum
            WHERE namespace.nspname = %s
              AND relation.relkind IN ('r', 'p')
              AND attribute.attnum > 0
              AND NOT attribute.attisdropped
            """,
            (self._schema,),
        ).fetchall()
        actual_columns: dict[str, dict[str, tuple[str, bool, str, Optional[str]]]] = {}
        for row in column_rows:
            table_name, column_name, type_name, not_null, identity, default_expr = (
                self._row_fields(
                row,
                (
                    "table_name",
                    "column_name",
                    "type_name",
                    "not_null",
                    "identity",
                    "default_expr",
                ),
                )
            )
            actual_columns.setdefault(str(table_name), {})[str(column_name)] = (
                str(type_name).lower(),
                bool(not_null),
                str(identity),
                self._normalize_default(default_expr),
            )
        for table_name, contracts in required_columns.items():
            for contract in contracts:
                actual = actual_columns.get(table_name, {}).get(contract.name)
                expected_identity = "d" if contract.identity == "BY DEFAULT" else ""
                if actual != (
                    contract.type_name,
                    contract.not_null,
                    expected_identity,
                    contract.default,
                ):
                    raise PostgresSchemaValidationError(
                        "PostgreSQL state schema has invalid durable column metadata"
                    )

        primary_rows = connection.execute(
            """
            SELECT tc.table_name, kcu.column_name, kcu.ordinal_position
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON tc.constraint_catalog = kcu.constraint_catalog
             AND tc.constraint_schema = kcu.constraint_schema
             AND tc.constraint_name = kcu.constraint_name
            WHERE tc.table_schema = %s
              AND tc.constraint_type = 'PRIMARY KEY'
            ORDER BY tc.table_name, kcu.ordinal_position
            """,
            (self._schema,),
        ).fetchall()
        actual_primary_keys: dict[str, list[tuple[int, str]]] = {}
        for row in primary_rows:
            table_name, column_name, ordinal = self._row_fields(
                row,
                ("table_name", "column_name", "ordinal_position"),
            )
            actual_primary_keys.setdefault(str(table_name), []).append(
                (int(ordinal), str(column_name))
            )
        required_primary_keys = dict(_REQUIRED_PRIMARY_KEYS)
        if include_telegram:
            required_primary_keys.update(_REQUIRED_TELEGRAM_PRIMARY_KEYS)
        if any(
            tuple(
                column
                for _, column in sorted(actual_primary_keys.get(table_name, []))
            )
            != expected_columns
            for table_name, expected_columns in required_primary_keys.items()
        ):
            raise PostgresSchemaValidationError(
                "PostgreSQL state schema is missing required primary keys"
            )

        required_foreign_keys = dict(_REQUIRED_FOREIGN_KEYS)
        if include_telegram:
            required_foreign_keys.update(_REQUIRED_TELEGRAM_FOREIGN_KEYS)
        actual_foreign_keys = {
            foreign_key[1:5]: foreign_key[5:] for foreign_key in self._foreign_key_rows(connection)
        }
        if any(
            actual_foreign_keys.get(key) != expected
            for key, expected in required_foreign_keys.items()
        ):
            raise PostgresSchemaValidationError(
                "PostgreSQL state schema has invalid foreign key metadata"
            )

        index_rows = connection.execute(
            """
            SELECT
                table_relation.relname AS table_name,
                index_relation.relname AS index_name,
                index_meta.indisunique AS is_unique,
                key_column.ordinality,
                attribute.attname AS column_name,
                (index_meta.indoption[key_column.ordinality - 1] & 1) = 1 AS is_desc,
                pg_get_expr(index_meta.indpred, index_meta.indrelid) AS predicate
            FROM pg_index AS index_meta
            JOIN pg_class AS index_relation ON index_relation.oid = index_meta.indexrelid
            JOIN pg_class AS table_relation ON table_relation.oid = index_meta.indrelid
            JOIN pg_namespace AS namespace ON namespace.oid = table_relation.relnamespace
            JOIN LATERAL unnest(index_meta.indkey) WITH ORDINALITY
              AS key_column(attnum, ordinality)
              ON key_column.ordinality <= index_meta.indnkeyatts
            JOIN pg_attribute AS attribute
              ON attribute.attrelid = table_relation.oid
             AND attribute.attnum = key_column.attnum
            WHERE namespace.nspname = %s
            ORDER BY index_relation.relname, key_column.ordinality
            """,
            (self._schema,),
        ).fetchall()
        actual_indexes: dict[str, dict[str, Any]] = {}
        for row in index_rows:
            table_name, index_name, unique, ordinal, column_name, descending, predicate = (
                self._row_fields(
                    row,
                    (
                        "table_name",
                        "index_name",
                        "is_unique",
                        "ordinality",
                        "column_name",
                        "is_desc",
                        "predicate",
                    ),
                )
            )
            index = actual_indexes.setdefault(
                str(index_name),
                {
                    "table": str(table_name),
                    "unique": bool(unique),
                    "columns": [],
                    "predicate": self._normalize_predicate(predicate),
                },
            )
            index["columns"].append((int(ordinal), str(column_name), bool(descending)))
        required_indexes = CORE_INDEX_CONTRACTS
        if include_telegram:
            required_indexes = required_indexes + TELEGRAM_INDEX_CONTRACTS
        for contract in required_indexes:
            actual = actual_indexes.get(contract.name)
            expected_predicate = self._normalize_predicate(contract.predicate)
            expected_columns = contract.columns
            if actual is None or (
                actual["table"],
                actual["unique"],
                tuple(
                    (column_name, descending)
                    for _, column_name, descending in sorted(actual["columns"])
                ),
                actual["predicate"],
            ) != (
                contract.table,
                contract.unique,
                expected_columns,
                expected_predicate,
            ):
                raise PostgresSchemaValidationError(
                    "PostgreSQL state schema has invalid durable index metadata"
                )

    def _ensure_foreign_key_contract(
        self,
        connection: _ConnectionLike,
        *,
        include_telegram: bool,
    ) -> None:
        required_foreign_keys = dict(_REQUIRED_FOREIGN_KEYS)
        if include_telegram:
            required_foreign_keys.update(_REQUIRED_TELEGRAM_FOREIGN_KEYS)
        for key, expected in required_foreign_keys.items():
            table_name, column_name, foreign_table, foreign_column = key
            current = [
                row
                for row in self._foreign_key_rows(connection)
                if row[1] == table_name and row[2] == column_name
            ]
            if len(current) == 1 and current[0][3:] == (
                foreign_table,
                foreign_column,
                *expected,
            ):
                continue
            for constraint_name, *_ in current:
                connection.execute(
                    "ALTER TABLE "
                    f"{self._quote_identifier(table_name)} DROP CONSTRAINT "
                    f"{self._quote_identifier(constraint_name)}"
                )
            delete_action = "CASCADE" if expected[0] == "c" else "NO ACTION"
            connection.execute(
                "ALTER TABLE "
                f"{self._quote_identifier(table_name)} ADD FOREIGN KEY "
                f"({self._quote_identifier(column_name)}) REFERENCES "
                f"{self._quote_identifier(foreign_table)} "
                f"({self._quote_identifier(foreign_column)}) "
                f"ON DELETE {delete_action} NOT DEFERRABLE"
            )

    def _foreign_key_rows(
        self,
        connection: _ConnectionLike,
    ) -> list[tuple[str, str, str, str, str, str, bool, bool]]:
        rows = connection.execute(
            """
            SELECT
                constraint_meta.conname AS constraint_name,
                table_relation.relname AS table_name,
                attribute.attname AS column_name,
                foreign_relation.relname AS foreign_table_name,
                foreign_attribute.attname AS foreign_column_name,
                constraint_meta.confdeltype AS delete_action,
                constraint_meta.condeferrable AS deferrable,
                constraint_meta.condeferred AS initially_deferred
            FROM pg_constraint AS constraint_meta
            JOIN pg_class AS table_relation ON table_relation.oid = constraint_meta.conrelid
            JOIN pg_namespace AS namespace ON namespace.oid = table_relation.relnamespace
            JOIN pg_class AS foreign_relation ON foreign_relation.oid = constraint_meta.confrelid
            JOIN LATERAL unnest(constraint_meta.conkey) WITH ORDINALITY
              AS local_key(attnum, ordinality) ON true
            JOIN LATERAL unnest(constraint_meta.confkey) WITH ORDINALITY
              AS referenced_key(attnum, ordinality)
              ON referenced_key.ordinality = local_key.ordinality
            JOIN pg_attribute AS attribute
              ON attribute.attrelid = table_relation.oid
             AND attribute.attnum = local_key.attnum
            JOIN pg_attribute AS foreign_attribute
              ON foreign_attribute.attrelid = foreign_relation.oid
             AND foreign_attribute.attnum = referenced_key.attnum
            WHERE namespace.nspname = %s
              AND constraint_meta.contype = 'f'
            """,
            (self._schema,),
        ).fetchall()
        return [
            tuple(
                self._row_fields(
                    row,
                    (
                        "constraint_name",
                        "table_name",
                        "column_name",
                        "foreign_table_name",
                        "foreign_column_name",
                        "delete_action",
                        "deferrable",
                        "initially_deferred",
                    ),
                )
            )
            for row in rows
        ]

    def ensure_telegram_schema(self) -> MigrationReport:
        """Explicitly create the lazy Telegram durable tables and indexes."""

        return self.migrate(include_telegram=True)

    def ensure_search_schema(self) -> None:
        """Create the derived PostgreSQL full-text and trigram search schema."""

        if self.spec.read_only:
            raise PostgresReadOnlyError(
                "Cannot create search indexes in a read-only PostgreSQL state store"
            )
        from state_store.postgres.search_ddl import search_capability_setup_sql

        with self.transaction(read_only=False) as connection:
            for statement in search_capability_setup_sql():
                connection.execute(statement)

    def health_report(self) -> PostgresHealthReport:
        """Return connection/schema capability data without leaking DSN details."""

        capabilities = {
            "core_schema": False,
            "telegram_schema": False,
            "migrations": not self.spec.read_only,
            "read_only": self.spec.read_only,
            "full_text_search": False,
        }
        try:
            with self.transaction(read_only=True) as connection:
                connection.execute("SELECT 1").fetchone()
                self._set_search_path(connection)
                row = connection.execute(
                    "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
                ).fetchone()
                version = self._row_value(row)
                version = int(version) if version is not None else None
                capabilities["core_schema"] = version == SCHEMA_VERSION
                telegram_rows = connection.execute(
                    "SELECT to_regclass(%s)",
                    (f"{self._schema}.telegram_dm_topic_bindings",),
                ).fetchone()
                capabilities["telegram_schema"] = self._row_value(telegram_rows) is not None
                search_rows = connection.execute(
                    "SELECT to_regclass(%s), to_regclass(%s), to_regclass(%s)",
                    (
                        f"{self._schema}.idx_messages_search_vector_gin",
                        f"{self._schema}.idx_messages_search_document_trgm",
                        f"{self._schema}.idx_sessions_id_trgm",
                    ),
                ).fetchone()
                capabilities["full_text_search"] = bool(
                    search_rows and all(value is not None for value in search_rows)
                )
                return PostgresHealthReport(
                    available=True,
                    schema=self._schema,
                    schema_version=version,
                    target_version=SCHEMA_VERSION,
                    read_only=self.spec.read_only,
                    capabilities=capabilities,
                )
        except PostgresStateStoreError:
            return PostgresHealthReport(
                available=False,
                schema=self._schema,
                schema_version=None,
                target_version=SCHEMA_VERSION,
                read_only=self.spec.read_only,
                capabilities=capabilities,
                detail="PostgreSQL state store is unavailable",
            )

    def close(self) -> None:
        """Release idle resources and make this lifecycle instance unusable."""

        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            pool = self._pool
            self._pool = None
        if pool is not None:
            pool.close()

    def _configure_transaction(
        self,
        connection: _ConnectionLike,
        *,
        read_only: bool,
        configure_search_path: bool,
    ) -> None:
        if read_only:
            # PostgreSQL must see this before any transaction-scoped setup query.
            connection.execute("SET TRANSACTION READ ONLY")
        connection.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (f"{self._statement_timeout_ms}ms",),
        )
        connection.execute(
            "SELECT set_config('lock_timeout', %s, false)",
            (f"{self._lock_timeout_ms}ms",),
        )
        connection.execute(
            "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
            (f"{self._idle_in_transaction_timeout_ms}ms",),
        )
        if configure_search_path:
            self._set_search_path(connection)

    def _set_search_path(self, connection: _ConnectionLike) -> None:
        connection.execute(
            "SELECT set_config('search_path', %s, true)",
            (f"{self._quoted_schema}, pg_catalog",),
        )

    @classmethod
    def _validated_schema(cls, spec: StateStoreSpec) -> str:
        if spec.backend != "postgres":
            raise PostgresConfigurationError(
                "PostgresStateStore requires a PostgreSQL state-store specification"
            )
        schema = spec.postgres_schema
        if not isinstance(schema, str) or not cls._is_safe_schema(schema):
            raise PostgresConfigurationError(
                "PostgreSQL state schema must be an explicit safe identifier"
            )
        return schema

    @staticmethod
    def _is_safe_schema(schema: str) -> bool:
        return (
            1 <= len(schema) <= 63
            and (schema[0] == "_" or "a" <= schema[0] <= "z")
            and all(
                char == "_" or "a" <= char <= "z" or "0" <= char <= "9"
                for char in schema
            )
        )

    @staticmethod
    def _positive_int(value: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise PostgresConfigurationError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _positive_float(value: float, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise PostgresConfigurationError(f"{name} must be positive")
        return float(value)

    @property
    def _quoted_schema(self) -> str:
        return self._quote_identifier(self._schema)

    def _migration_lock_key(self) -> int:
        digest = sha256(
            f"hermes-state-postgres:{self._schema}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)

    @staticmethod
    def _row_value(row: Any) -> Any:
        if row is None:
            return None
        if isinstance(row, Mapping):
            return next(iter(row.values()), None)
        try:
            return row[0]
        except (IndexError, KeyError, TypeError):
            return row

    @staticmethod
    def _row_fields(row: Any, names: tuple[str, ...]) -> tuple[Any, ...]:
        if isinstance(row, Mapping):
            return tuple(row[name] for name in names)
        return tuple(row[index] for index in range(len(names)))

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return '"' + value.replace('"', '""') + '"'

    @staticmethod
    def _normalize_default(value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(str(value).split())
        while normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1].strip()
        for cast in ("::text", "::integer", "::bigint", "::double precision"):
            if normalized.endswith(cast):
                normalized = normalized[: -len(cast)]
        return normalized

    @staticmethod
    def _normalize_predicate(value: Any) -> Optional[str]:
        if value is None:
            return None
        normalized = " ".join(str(value).split()).lower()
        while normalized.startswith("(") and normalized.endswith(")"):
            normalized = normalized[1:-1].strip()
        return normalized

    @staticmethod
    def _is_driver_exception(exc: BaseException) -> bool:
        module = type(exc).__module__
        return module == "psycopg" or module.startswith(
            ("psycopg.", "psycopg_pool.")
        )

    @staticmethod
    def _load_psycopg() -> Any:
        try:
            import psycopg
        except ModuleNotFoundError:
            raise PostgresDriverUnavailableError(
                "PostgreSQL state support requires the optional psycopg dependency"
            ) from None
        return psycopg


def ddl_table_names(*, include_telegram: bool = False) -> frozenset[str]:
    """Return manifest-relevant PostgreSQL table names for static parity tests."""

    tables = CORE_TABLES + (TELEGRAM_TABLES if include_telegram else ())
    return frozenset(table.name for table in tables)


def schema_manifest_matches_postgres_ddl() -> bool:
    """Ensure the authored PostgreSQL manifest covers every portable table."""

    return (
        ddl_table_names() == SCHEMA_V22_MANIFEST.core_tables
        and ddl_table_names(include_telegram=True)
        == SCHEMA_V22_MANIFEST.core_tables | SCHEMA_V22_MANIFEST.telegram_tables
    )
