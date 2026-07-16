from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading
from typing import Any, Iterator

import pytest

from state_store.postgres import (
    BoundedConnectionPool,
    CORE_INDEXES,
    CORE_INDEX_CONTRACTS,
    CORE_TABLES,
    POSTGRES_COLUMN_CONTRACTS,
    SCHEMA_VERSION,
    TELEGRAM_INDEXES,
    TELEGRAM_INDEX_CONTRACTS,
    TELEGRAM_TABLES,
    V16_TO_V22_COLUMN_UPGRADES,
    PostgresConfigurationError,
    PostgresConnectionError,
    PostgresPoolTimeoutError,
    PostgresReadOnlyError,
    PostgresSchemaValidationError,
    PostgresSchemaVersionError,
    PostgresStateStore,
    schema_statements,
)
from state_store.spec import StateStoreSpec


def _index_name(statement: str) -> str:
    return statement.split("INDEX IF NOT EXISTS ", 1)[1].split(" ", 1)[0]


class FakeCursor:
    def __init__(self, row: Any = None, rows: tuple[Any, ...] = ()) -> None:
        self._row = row
        self._rows = rows

    def fetchone(self) -> Any:
        return self._row

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class FakeTransaction:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection

    def __enter__(self) -> "FakeConnection":
        self.connection.transaction_entries += 1
        return self.connection

    def __exit__(self, *unused: Any) -> None:
        self.connection.transaction_exits += 1


class FakeConnection:
    def __init__(
        self,
        *,
        schema_version: int | None = None,
        telegram_exists: bool = False,
        shape: str = "current",
        apply_upgrades: bool = True,
        apply_indexes: bool = True,
    ) -> None:
        self.schema_version = schema_version
        self.telegram_exists = telegram_exists
        self.apply_upgrades = apply_upgrades
        self.apply_indexes = apply_indexes
        self.queries: list[tuple[str, Any]] = []
        self.closed = False
        self.transaction_entries = 0
        self.transaction_exits = 0
        self.columns = {
            table.name: set(table.columns) for table in CORE_TABLES + TELEGRAM_TABLES
        }
        self.column_metadata = {
            (table_name, contract.name): (
                contract.type_name,
                contract.not_null,
                "d" if contract.identity == "BY DEFAULT" else "",
                contract.default,
            )
            for table_name, contracts in POSTGRES_COLUMN_CONTRACTS.items()
            for contract in contracts
        }
        self.primary_keys = {
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
            "telegram_dm_topic_mode": ("chat_id",),
            "telegram_dm_topic_bindings": ("chat_id", "thread_id"),
        }
        self.foreign_keys = {
            ("sessions", "parent_session_id", "sessions", "id"),
            ("messages", "session_id", "sessions", "id"),
            ("session_model_usage", "session_id", "sessions", "id"),
            ("telegram_dm_topic_bindings", "session_id", "sessions", "id"),
        }
        self.foreign_key_metadata = {
            key: (f"{key[0]}_{key[1]}_fkey", "a", False, False)
            for key in self.foreign_keys
        }
        self.indexes = {
            _index_name(statement) for statement in CORE_INDEXES + TELEGRAM_INDEXES
        }
        self.index_overrides: dict[
            str,
            tuple[str, bool, tuple[tuple[str, bool], ...], str | None],
        ] = {}
        if shape == "v16":
            for upgrade in V16_TO_V22_COLUMN_UPGRADES:
                self.columns[upgrade.table].discard(upgrade.column)
            for table_name in (
                "session_model_usage",
                "gateway_routing",
                "async_delegations",
                "telegram_dm_topic_mode",
                "telegram_dm_topic_bindings",
            ):
                self.columns.pop(table_name, None)
                self.primary_keys.pop(table_name, None)
            self.foreign_keys = {
                foreign_key
                for foreign_key in self.foreign_keys
                if foreign_key[0] in self.columns
            }
            self.foreign_key_metadata = {
                key: self.foreign_key_metadata[key] for key in self.foreign_keys
            }
            self.foreign_key_metadata[
                ("sessions", "parent_session_id", "sessions", "id")
            ] = ("sessions_parent_session_id_fkey", "n", False, False)
            self.foreign_key_metadata[
                ("messages", "session_id", "sessions", "id")
            ] = ("messages_session_id_fkey", "c", False, False)
            type_name, not_null, identity, default = self.column_metadata[
                ("messages", "id")
            ]
            self.column_metadata[("messages", "id")] = (
                "integer",
                not_null,
                identity,
                default,
            )
            self.indexes = {
                "idx_sessions_source",
                "idx_sessions_source_id",
                "idx_sessions_parent",
                "idx_sessions_started",
                "idx_messages_session",
                "idx_messages_session_active",
                "idx_compression_locks_expires",
            }
        elif shape != "current":
            raise ValueError(f"unsupported fake schema shape: {shape}")

    def close(self) -> None:
        self.closed = True

    def execute(self, query: str, params: Any = None) -> FakeCursor:
        self.queries.append((query, params))
        normalized = " ".join(query.split())
        if normalized.startswith("SELECT version FROM schema_version"):
            return FakeCursor(
                None if self.schema_version is None else (self.schema_version,)
            )
        if "FROM pg_attribute AS attribute" in normalized:
            return FakeCursor(
                rows=tuple(
                    (
                        table_name,
                        column_name,
                        *self.column_metadata[(table_name, column_name)],
                    )
                    for table_name, columns in self.columns.items()
                    for column_name in sorted(columns)
                )
            )
        if "constraint_type = 'PRIMARY KEY'" in normalized:
            return FakeCursor(
                rows=tuple(
                    (table_name, column_name, ordinal)
                    for table_name, columns in self.primary_keys.items()
                    for ordinal, column_name in enumerate(columns, start=1)
                )
            )
        if "FROM pg_constraint AS constraint_meta" in normalized:
            return FakeCursor(
                rows=tuple(
                    (
                        self.foreign_key_metadata[foreign_key][0],
                        *foreign_key,
                        *self.foreign_key_metadata[foreign_key][1:],
                    )
                    for foreign_key in sorted(self.foreign_keys)
                )
            )
        if "FROM pg_index AS index_meta" in normalized:
            contracts = CORE_INDEX_CONTRACTS + TELEGRAM_INDEX_CONTRACTS
            return FakeCursor(
                rows=tuple(
                    (
                        metadata[0],
                        contract.name,
                        metadata[1],
                        ordinal,
                        column_name,
                        descending,
                        metadata[3],
                    )
                    for contract in contracts
                    if contract.name in self.indexes
                    for metadata in (
                        self.index_overrides.get(
                            contract.name,
                            (
                                contract.table,
                                contract.unique,
                                contract.columns,
                                contract.predicate,
                            ),
                        ),
                    )
                    for ordinal, (column_name, descending) in enumerate(
                        metadata[2],
                        start=1,
                    )
                )
            )
        if normalized.startswith("SELECT to_regclass"):
            return FakeCursor(
                ("tenant_state.telegram_dm_topic_bindings",)
                if self.telegram_exists
                else (None,)
            )
        if normalized.startswith("INSERT INTO schema_version"):
            self.schema_version = params[0]
        if normalized.startswith("UPDATE schema_version"):
            self.schema_version = params[0]
        for table in CORE_TABLES + TELEGRAM_TABLES:
            if query == table.create_sql and table.name not in self.columns:
                self.columns[table.name] = set(table.columns)
                self._add_table_constraints(table.name)
        for upgrade in V16_TO_V22_COLUMN_UPGRADES:
            if query == upgrade.add_sql and self.apply_upgrades:
                self.columns[upgrade.table].add(upgrade.column)
        if query == "ALTER TABLE messages ALTER COLUMN id TYPE BIGINT":
            _, not_null, identity, default = self.column_metadata[("messages", "id")]
            self.column_metadata[("messages", "id")] = (
                "bigint",
                not_null,
                identity,
                default,
            )
        if "DROP CONSTRAINT" in normalized:
            for key, metadata in tuple(self.foreign_key_metadata.items()):
                if f'"{metadata[0]}"' in query:
                    self.foreign_keys.discard(key)
                    self.foreign_key_metadata.pop(key)
        if "ADD FOREIGN KEY" in normalized:
            for key, delete_action in (
                (("sessions", "parent_session_id", "sessions", "id"), "a"),
                (("messages", "session_id", "sessions", "id"), "a"),
                (("session_model_usage", "session_id", "sessions", "id"), "c"),
                (
                    (
                        "telegram_dm_topic_bindings",
                        "session_id",
                        "sessions",
                        "id",
                    ),
                    "c",
                ),
            ):
                if (
                    f'"{key[0]}" ADD FOREIGN KEY ("{key[1]}")' in query
                ):
                    self.foreign_keys.add(key)
                    self.foreign_key_metadata[key] = (
                        f"{key[0]}_{key[1]}_fkey",
                        delete_action,
                        False,
                        False,
                    )
        for statement in CORE_INDEXES + TELEGRAM_INDEXES:
            if query == statement and self.apply_indexes:
                self.indexes.add(_index_name(statement))
        return FakeCursor()

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    def _add_table_constraints(self, table_name: str) -> None:
        primary_keys = {
            "session_model_usage": (
                "session_id",
                "model",
                "billing_provider",
                "billing_base_url",
                "billing_mode",
                "task",
            ),
            "gateway_routing": ("scope", "session_key"),
            "async_delegations": ("delegation_id",),
            "telegram_dm_topic_mode": ("chat_id",),
            "telegram_dm_topic_bindings": ("chat_id", "thread_id"),
        }
        foreign_keys = {
            "session_model_usage": ("session_id", "sessions", "id"),
            "telegram_dm_topic_bindings": ("session_id", "sessions", "id"),
        }
        if table_name in primary_keys:
            self.primary_keys[table_name] = primary_keys[table_name]
        if table_name in foreign_keys:
            column, foreign_table, foreign_column = foreign_keys[table_name]
            key = (table_name, column, foreign_table, foreign_column)
            self.foreign_keys.add(key)
            self.foreign_key_metadata[key] = (
                f"{table_name}_{column}_fkey",
                "c",
                False,
                False,
            )


class FakePool:
    def __init__(self, connector: Any) -> None:
        self._connector = connector
        self._connection: FakeConnection | None = None
        self.closed = False

    @contextmanager
    def connection(self) -> Iterator[FakeConnection]:
        if self.closed:
            raise RuntimeError("fake pool is closed")
        if self._connection is None:
            self._connection = self._connector()
        yield self._connection

    def close(self) -> None:
        self.closed = True
        if self._connection is not None:
            self._connection.close()


def postgres_spec(*, read_only: bool = False, schema: str = "tenant_state") -> StateStoreSpec:
    return StateStoreSpec(
        home=Path("/tmp/hermes"),
        profile="test",
        backend="postgres",
        sqlite_path=Path("/tmp/hermes/state.db"),
        postgres_dsn_env="HERMES_STATE_POSTGRES_DSN",
        postgres_schema=schema,
        read_only=read_only,
    )


def make_store(
    connection: FakeConnection,
    *,
    read_only: bool = False,
    connector: Any = None,
) -> tuple[PostgresStateStore, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def default_connector(dsn: str, **kwargs: Any) -> FakeConnection:
        captured["dsn"] = dsn
        captured["connect_kwargs"] = kwargs
        return connection

    def pool_factory(
        *,
        connector: Any,
        max_size: int,
        acquire_timeout_s: float,
    ) -> FakePool:
        captured["pool_max_size"] = max_size
        captured["pool_acquire_timeout_s"] = acquire_timeout_s
        pool = FakePool(connector)
        captured["pool"] = pool
        return pool

    return (
        PostgresStateStore(
            postgres_spec(read_only=read_only),
            environ={
                "HERMES_STATE_POSTGRES_DSN": (
                    "postgresql://secret-user:secret-pass@db/private"
                )
            },
            connector=connector or default_connector,
            pool_factory=pool_factory,
            max_pool_size=3,
            acquire_timeout_s=0.1,
            connect_timeout_s=7,
            statement_timeout_ms=11_000,
            lock_timeout_ms=2_000,
            idle_in_transaction_timeout_ms=13_000,
        ),
        captured,
    )


def query_texts(connection: FakeConnection) -> list[str]:
    return [query for query, _ in connection.queries]


def test_driver_loading_and_connection_are_lazy() -> None:
    connection = FakeConnection()
    driver_loads: list[bool] = []
    connects: list[tuple[str, dict[str, Any]]] = []

    class Driver:
        def connect(self, dsn: str, **kwargs: Any) -> FakeConnection:
            connects.append((dsn, kwargs))
            return connection

    captured: dict[str, Any] = {}

    def pool_factory(*, connector: Any, **kwargs: Any) -> FakePool:
        captured["pool"] = FakePool(connector)
        return captured["pool"]

    def load_driver() -> Driver:
        driver_loads.append(True)
        return Driver()

    store = PostgresStateStore(
        postgres_spec(),
        environ={"HERMES_STATE_POSTGRES_DSN": "postgresql://user:pass@db/state"},
        driver_loader=load_driver,
        pool_factory=pool_factory,
    )
    store.open()
    assert driver_loads == []

    with store.transaction() as leased:
        assert leased is connection

    assert driver_loads == [True]
    assert connects[0][1]["connect_timeout"] == 5
    assert connects[0][1]["application_name"] == store.application_name


def test_transaction_configures_bounded_timeouts_safe_path_and_read_only() -> None:
    connection = FakeConnection()
    store, captured = make_store(connection, read_only=True)

    with store.transaction() as leased:
        assert leased is connection

    assert captured["pool_max_size"] == 3
    assert captured["pool_acquire_timeout_s"] == 0.1
    assert captured["dsn"] == "postgresql://secret-user:secret-pass@db/private"
    assert captured["connect_kwargs"] == {
        "connect_timeout": 7,
        "application_name": store.application_name,
    }
    assert "secret" not in store.application_name
    assert connection.queries[:5] == [
        ("SET TRANSACTION READ ONLY", None),
        ("SELECT set_config('statement_timeout', %s, false)", ("11000ms",)),
        ("SELECT set_config('lock_timeout', %s, false)", ("2000ms",)),
        (
            "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
            ("13000ms",),
        ),
        (
            "SELECT set_config('search_path', %s, true)",
            ('"tenant_state", pg_catalog',),
        ),
    ]


def test_connection_and_health_errors_do_not_expose_the_dsn() -> None:
    secret_dsn = "postgresql://secret-user:secret-pass@db/private"

    def failing_connector(dsn: str, **kwargs: Any) -> FakeConnection:
        raise RuntimeError(f"cannot connect to {dsn}")

    store, _ = make_store(FakeConnection(), connector=failing_connector)
    with pytest.raises(PostgresConnectionError) as exc_info:
        with store.transaction():
            pass
    assert secret_dsn not in str(exc_info.value)
    assert "secret-pass" not in str(exc_info.value)

    report = store.health_report()
    assert not report.available
    assert report.detail == "PostgreSQL state store is unavailable"
    assert secret_dsn not in report.detail


def test_migration_is_locked_idempotent_and_telegram_is_explicit() -> None:
    connection = FakeConnection()
    store, _ = make_store(connection)

    first = store.migrate()
    first_queries = query_texts(connection)
    assert first.previous_version is None
    assert first.version == SCHEMA_VERSION
    assert not first.telegram_enabled
    assert connection.schema_version == SCHEMA_VERSION
    assert first_queries.index('CREATE SCHEMA IF NOT EXISTS "tenant_state"') < first_queries.index(
        "SELECT set_config('search_path', %s, true)"
    )
    assert "SELECT pg_advisory_xact_lock(%s)" in first_queries
    assert all(statement in first_queries for statement in schema_statements())
    assert all(table.create_sql not in first_queries for table in TELEGRAM_TABLES)
    assert all(index not in first_queries for index in TELEGRAM_INDEXES)

    insert_count = sum(
        "INSERT INTO schema_version" in query for query in query_texts(connection)
    )
    second = store.migrate()
    assert second.previous_version == SCHEMA_VERSION
    assert sum(
        "INSERT INTO schema_version" in query for query in query_texts(connection)
    ) == insert_count

    telegram = store.ensure_telegram_schema()
    all_queries = query_texts(connection)
    assert telegram.telegram_enabled
    assert all(table.create_sql in all_queries for table in TELEGRAM_TABLES)
    assert all(index in all_queries for index in TELEGRAM_INDEXES)


def test_migration_rejects_newer_version_before_non_version_ddl() -> None:
    connection = FakeConnection(schema_version=SCHEMA_VERSION + 1)
    store, _ = make_store(connection)

    with pytest.raises(PostgresSchemaVersionError):
        store.migrate(include_telegram=True)

    queries = query_texts(connection)
    assert CORE_TABLES[0].create_sql in queries
    assert CORE_TABLES[1].create_sql not in queries
    assert all(table.create_sql not in queries for table in TELEGRAM_TABLES)


def test_v16_experimental_schema_upgrades_and_validates_before_v22_advance() -> None:
    connection = FakeConnection(schema_version=16, shape="v16")
    store, _ = make_store(connection)

    report = store.migrate()

    assert report.previous_version == 16
    assert connection.schema_version == SCHEMA_VERSION
    assert all(
        upgrade.add_sql in query_texts(connection)
        for upgrade in V16_TO_V22_COLUMN_UPGRADES
    )
    assert "session_model_usage" in connection.columns
    assert "gateway_routing" in connection.columns
    assert "async_delegations" in connection.columns
    assert connection.foreign_key_metadata[
        ("sessions", "parent_session_id", "sessions", "id")
    ][1:] == ("a", False, False)
    assert connection.foreign_key_metadata[
        ("messages", "session_id", "sessions", "id")
    ][1:] == ("a", False, False)


def test_incomplete_schema_never_advances_the_version() -> None:
    connection = FakeConnection(
        schema_version=16,
        shape="v16",
        apply_upgrades=False,
    )
    store, _ = make_store(connection)

    with pytest.raises(PostgresSchemaValidationError):
        store.migrate()

    assert connection.schema_version == 16
    assert not any(
        query.startswith("UPDATE schema_version") for query in query_texts(connection)
    )


@pytest.mark.parametrize(
    "missing_kind",
    [
        "column_type",
        "column_default",
        "column_identity",
        "primary_key",
        "index_columns",
        "index_unique",
        "index_predicate",
    ],
)
def test_schema_validation_rejects_missing_required_structure(
    missing_kind: str,
) -> None:
    connection = FakeConnection(schema_version=SCHEMA_VERSION)
    if missing_kind == "column_type":
        _, not_null, identity, default = connection.column_metadata[("sessions", "source")]
        connection.column_metadata[("sessions", "source")] = (
            "integer",
            not_null,
            identity,
            default,
        )
    elif missing_kind == "column_default":
        type_name, not_null, identity, _ = connection.column_metadata[
            ("messages", "active")
        ]
        connection.column_metadata[("messages", "active")] = (
            type_name,
            not_null,
            identity,
            "0",
        )
    elif missing_kind == "column_identity":
        type_name, not_null, _, default = connection.column_metadata[
            ("messages", "id")
        ]
        connection.column_metadata[("messages", "id")] = (
            type_name,
            not_null,
            "",
            default,
        )
    elif missing_kind == "primary_key":
        connection.primary_keys["sessions"] = ()
    elif missing_kind == "index_columns":
        connection.index_overrides["idx_messages_session"] = (
            "messages",
            False,
            (("timestamp", False), ("session_id", False)),
            None,
        )
    elif missing_kind == "index_unique":
        connection.index_overrides["idx_sessions_title_unique"] = (
            "sessions",
            False,
            (("title", False),),
            "title IS NOT NULL",
        )
    else:
        connection.index_overrides["idx_messages_active_null"] = (
            "messages",
            False,
            (("active", False),),
            None,
        )
    store, _ = make_store(connection)

    with pytest.raises(PostgresSchemaValidationError):
        store.migrate()


def test_foreign_key_action_and_deferrability_are_repaired_before_validation() -> None:
    connection = FakeConnection(schema_version=16, shape="v16")
    key = ("messages", "session_id", "sessions", "id")
    connection.foreign_key_metadata[key] = (
        "messages_session_id_fkey",
        "c",
        True,
        True,
    )
    store, _ = make_store(connection)

    store.migrate()

    assert connection.foreign_key_metadata[key][1:] == ("a", False, False)


def test_read_only_migrations_fail_and_health_reports_capabilities() -> None:
    store, _ = make_store(FakeConnection(), read_only=True)
    with pytest.raises(PostgresReadOnlyError):
        store.migrate()
    with pytest.raises(PostgresReadOnlyError):
        with store.transaction(read_only=False):
            pass

    connection = FakeConnection(
        schema_version=SCHEMA_VERSION,
        telegram_exists=True,
    )
    healthy, _ = make_store(connection, read_only=True)
    report = healthy.health_report()
    assert report.available
    assert report.schema_version == SCHEMA_VERSION
    assert report.capabilities == {
        "core_schema": True,
        "telegram_schema": True,
        "migrations": False,
        "read_only": True,
        "full_text_search": False,
    }


@pytest.mark.parametrize("schema", ["_tenant_state", "a", "x" * 63])
def test_safe_postgres_identifiers_are_accepted(schema: str) -> None:
    store = PostgresStateStore(
        postgres_spec(schema=schema),
        environ={"HERMES_STATE_POSTGRES_DSN": "postgresql://example"},
    )
    assert store.spec.postgres_schema == schema


@pytest.mark.parametrize(
    "schema",
    ["", "Tenant", "1tenant", "tenant-name", "tenant;drop", "x" * 64, "x٢"],
)
def test_unsafe_postgres_identifiers_are_rejected_without_echoing_them(
    schema: str,
) -> None:
    with pytest.raises(PostgresConfigurationError) as exc_info:
        PostgresStateStore(
            postgres_spec(schema=schema),
            environ={"HERMES_STATE_POSTGRES_DSN": "postgresql://example"},
        )
    if schema:
        assert schema not in str(exc_info.value)


def test_close_is_idempotent_and_bounded_pool_enforces_its_limit() -> None:
    connection = FakeConnection()
    store, captured = make_store(connection)
    with store.transaction():
        pass
    store.close()
    store.close()
    assert captured["pool"].closed
    assert connection.closed
    with pytest.raises(PostgresConnectionError):
        with store.transaction():
            pass

    created: list[FakeConnection] = []

    def connector() -> FakeConnection:
        connection = FakeConnection()
        created.append(connection)
        return connection

    pool = BoundedConnectionPool(connector, max_size=1, acquire_timeout_s=0.01)
    with pool.connection() as leased:
        assert leased is created[0]
        with pytest.raises(PostgresPoolTimeoutError):
            with pool.connection():
                pass
    leased.close()
    with pool.connection() as replacement:
        assert replacement is created[1]
    with pytest.raises(RuntimeError):
        with pool.connection() as broken:
            assert broken is replacement
            raise RuntimeError("transaction failed")
    assert replacement.closed
    with pool.connection() as recovered:
        assert recovered is created[2]
    pool.close()
    assert recovered.closed
    with pytest.raises(PostgresConnectionError):
        with pool.connection():
            pass


def test_concurrent_open_initializes_exactly_one_pool() -> None:
    connection = FakeConnection()
    factory_calls = 0
    factory_lock = threading.Lock()
    start = threading.Barrier(3)

    def pool_factory(*, connector: Any, **kwargs: Any) -> FakePool:
        del kwargs
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
        return FakePool(connector)

    store = PostgresStateStore(
        postgres_spec(),
        environ={"HERMES_STATE_POSTGRES_DSN": "postgresql://example"},
        connector=lambda *args, **kwargs: connection,
        pool_factory=pool_factory,
    )
    errors: list[BaseException] = []

    def open_store() -> None:
        try:
            start.wait()
            store.open()
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=open_store)
    second = threading.Thread(target=open_store)
    first.start()
    second.start()
    start.wait()
    first.join()
    second.join()

    assert errors == []
    assert factory_calls == 1
