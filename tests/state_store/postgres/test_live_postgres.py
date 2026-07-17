"""Live PostgreSQL contract tests for the durable Hermes state backend.

These tests require an isolated disposable database selected with
``HERMES_TEST_POSTGRES_DSN``. Each test gets a unique schema and drops it when
finished; no test uses a profile's SQLite database as a fallback target.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Any, Callable
from uuid import uuid4

import pytest

from hermes_cli.state_postgres_migration import (
    MigrationPhase,
    MigrationRequest,
    StatePostgresMigration,
)
from hermes_state import SessionDB
from state_store.postgres import (
    PostgresReadOnlyError,
    PostgresSchemaValidationError,
)
from state_store.postgres.migration_adapter import PostgresMigrationTargetAdapter
from state_store.postgres.session_db import PostgresSessionDB
from state_store.spec import StateStoreSpec
from state_store.sqlite.migration_adapter import SQLiteMigrationSourceAdapter


pytestmark = pytest.mark.integration

_DSN_ENV = "HERMES_TEST_POSTGRES_DSN"


class LivePostgresState:
    """Own disposable schemas and SessionDB instances for one test."""

    def __init__(self, dsn: str, tmp_path: Path, psycopg: Any) -> None:
        self.dsn = dsn
        self.tmp_path = tmp_path
        self.psycopg = psycopg
        self.schemas: set[str] = set()
        self.databases: list[PostgresSessionDB] = []
        self.stores: list[Any] = []

    def schema(self, prefix: str = "hermes_it") -> str:
        value = f"{prefix}_{uuid4().hex[:20]}"
        self.schemas.add(value)
        return value

    def open(self, schema: str, *, read_only: bool = False) -> PostgresSessionDB:
        database = PostgresSessionDB.from_spec(
            self._spec(schema, read_only=read_only),
            environ={_DSN_ENV: self.dsn},
        )
        self.databases.append(database)
        return database

    def migration_target(self, schema: str) -> PostgresMigrationTargetAdapter:
        target = PostgresMigrationTargetAdapter(
            dsn_env=_DSN_ENV,
            schema=schema,
            home=self.tmp_path,
            environ={_DSN_ENV: self.dsn},
        )
        self.schemas.add(target._control_schema)  # noqa: SLF001 - fixture cleanup
        self.stores.append(target._store)  # noqa: SLF001 - fixture cleanup
        return target

    def schema_exists(self, schema: str) -> bool:
        with self.psycopg.connect(self.dsn) as connection:
            row = connection.execute(
                "SELECT to_regnamespace(%s)", (schema,)
            ).fetchone()
        return bool(row and row[0])

    def close(self) -> None:
        for database in reversed(self.databases):
            database.close()
        for store in reversed(self.stores):
            store.close()

        with self.psycopg.connect(self.dsn, autocommit=True) as connection:
            # Migration staging schemas are deterministic descendants of the
            # test schema, but are not otherwise retained by the adapter.
            for schema in sorted(self.schemas):
                rows = connection.execute(
                    "SELECT nspname FROM pg_namespace "
                    "WHERE nspname = %s OR nspname LIKE %s",
                    (schema, f"{schema[:30]}_stage_%"),
                ).fetchall()
                for (name,) in rows:
                    connection.execute(
                        self.psycopg.sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                            self.psycopg.sql.Identifier(name)
                        )
                    )

    def _spec(self, schema: str, *, read_only: bool) -> StateStoreSpec:
        return StateStoreSpec(
            home=self.tmp_path,
            profile="postgres-integration",
            backend="postgres",
            sqlite_path=self.tmp_path / "state.db",
            postgres_dsn_env=_DSN_ENV,
            postgres_schema=schema,
            read_only=read_only,
        )


@pytest.fixture
def postgres_state(tmp_path: Path) -> LivePostgresState:
    dsn = os.environ.get(_DSN_ENV)
    if not dsn:
        pytest.skip(f"{_DSN_ENV} is required for live PostgreSQL tests")
    psycopg = pytest.importorskip("psycopg")
    state = LivePostgresState(dsn, tmp_path, psycopg)
    try:
        yield state
    finally:
        state.close()


def _race(
    operations: list[Callable[[], Any]],
) -> tuple[list[Any], list[BaseException]]:
    barrier = threading.Barrier(len(operations))
    values: list[Any] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def run(operation: Callable[[], Any]) -> None:
        try:
            barrier.wait(timeout=10)
            value = operation()
        except BaseException as exc:  # Thread failures must fail the contract.
            with lock:
                failures.append(exc)
        else:
            with lock:
                values.append(value)

    threads = [threading.Thread(target=run, args=(operation,)) for operation in operations]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert not [thread for thread in threads if thread.is_alive()]
    return values, failures


def _seed_sqlite_source(home: Path) -> Path:
    source = SessionDB.for_home(home)
    try:
        source.create_session("parent", "cli", model="test-model")
        source.create_session(
            "child",
            "cli",
            model="test-model",
            parent_session_id="parent",
        )
        source.append_message("parent", "user", "migrate this durable message")
        source.append_message("parent", "assistant", "migration acknowledgement")
        source.append_message("child", "user", "child message")
        source.save_gateway_routing_entry(
            "telegram:123",
            '{"session_id":"parent"}',
            scope="telegram",
        )
        source.set_meta("migration-test", "present")
    finally:
        source.close()
    return home / "state.db"


def test_schema_creation_happens_before_read_only_access(postgres_state: LivePostgresState):
    schema = postgres_state.schema()

    assert not postgres_state.schema_exists(schema)
    with pytest.raises(PostgresSchemaValidationError):
        postgres_state.open(schema, read_only=True)
    assert not postgres_state.schema_exists(schema)

    writable = postgres_state.open(schema)
    writable.create_session("ordered", "cli")
    readonly = postgres_state.open(schema, read_only=True)

    assert readonly.get_session("ordered")["id"] == "ordered"
    with pytest.raises(PostgresReadOnlyError):
        readonly.create_session("must-not-write", "cli")


def test_lifecycle_messages_and_search_use_live_postgres(
    postgres_state: LivePostgresState,
):
    database = postgres_state.open(postgres_state.schema())
    database.create_session("lifecycle", "cli", model="test-model")
    assert database.set_session_title("lifecycle", "PostgreSQL lifecycle")

    first_id = database.append_message(
        "lifecycle",
        "user",
        "Need durable PostgreSQL state search",
        timestamp=100.0,
    )
    second_id = database.append_message(
        "lifecycle",
        "assistant",
        "The state is durable and searchable.",
        timestamp=101.0,
    )

    assert first_id < second_id
    assert [row["content"] for row in database.get_messages("lifecycle")] == [
        "Need durable PostgreSQL state search",
        "The state is durable and searchable.",
    ]
    assert database.get_session("lifecycle")["message_count"] == 2
    assert database.search_sessions_by_id("life")[0]["id"] == "lifecycle"
    assert [row["id"] for row in database.search_sessions(source="cli")] == [
        "lifecycle"
    ]
    search_results = database.search_messages("durable", limit=10)
    assert [row["session_id"] for row in search_results] == [
        "lifecycle",
        "lifecycle",
    ]
    assert {row["role"] for row in search_results} == {"user", "assistant"}

    database.end_session("lifecycle", "operator-test")
    assert database.get_session("lifecycle")["end_reason"] == "operator-test"
    database.reopen_session("lifecycle")
    assert database.get_session("lifecycle")["ended_at"] is None


def test_concurrent_title_assignment_keeps_sqlite_contract(
    postgres_state: LivePostgresState,
):
    database = postgres_state.open(postgres_state.schema())
    database.create_session("title-one", "cli")
    database.create_session("title-two", "cli")

    values, failures = _race(
        [
            lambda: database.set_session_title("title-one", "shared title"),
            lambda: database.set_session_title("title-two", "shared title"),
        ]
    )

    assert values == [True]
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert database.get_session_by_title("shared title")["id"] in {
        "title-one",
        "title-two",
    }


def test_concurrent_handoff_claim_allows_one_winner(postgres_state: LivePostgresState):
    database = postgres_state.open(postgres_state.schema())
    database.create_session("handoff", "cli")
    assert database.request_handoff("handoff", "telegram")

    values, failures = _race(
        [
            lambda: database.claim_handoff("handoff"),
            lambda: database.claim_handoff("handoff"),
        ]
    )

    assert failures == []
    assert sorted(values) == [False, True]
    assert database.get_handoff_state("handoff")["state"] == "running"


def test_concurrent_compression_lock_allows_one_holder(
    postgres_state: LivePostgresState,
):
    database = postgres_state.open(postgres_state.schema())
    database.create_session("compression", "cli")

    values, failures = _race(
        [
            lambda: database.try_acquire_compression_lock(
                "compression", "worker-one", ttl_seconds=60
            ),
            lambda: database.try_acquire_compression_lock(
                "compression", "worker-two", ttl_seconds=60
            ),
        ]
    )

    assert failures == []
    assert sorted(values) == [False, True]
    assert database.get_compression_lock_holder("compression") in {
        "worker-one",
        "worker-two",
    }


def test_sqlite_to_postgres_migration_matches_digests_and_recovers_cutover(
    postgres_state: LivePostgresState,
):
    sqlite_path = _seed_sqlite_source(postgres_state.tmp_path)
    schema = postgres_state.schema()
    target = postgres_state.migration_target(schema)

    def fail_cutover(_: Any) -> None:
        raise RuntimeError("intentional cutover interruption")

    first = StatePostgresMigration(
        SQLiteMigrationSourceAdapter(sqlite_path),
        target,
        cutover=fail_cutover,
    ).run(MigrationRequest(apply=True, run_id="live-recovery", batch_size=2))

    assert first.phase is MigrationPhase.FAILED
    assert first.published
    assert not first.cutover_complete
    assert all(
        table.source_count == table.target_count
        and table.source_digest == table.target_digest
        for table in first.tables.values()
    )

    cutovers: list[str] = []
    recovered = StatePostgresMigration(
        SQLiteMigrationSourceAdapter(sqlite_path),
        target,
        cutover=lambda report: cutovers.append(report.run_id),
    ).run(MigrationRequest(apply=True, run_id="live-recovery", batch_size=2))

    assert recovered.phase is MigrationPhase.COMPLETE
    assert recovered.recovered_published_run
    assert recovered.cutover_complete
    assert cutovers == ["live-recovery"]
    assert all(
        table.source_count == table.target_count
        and table.source_digest == table.target_digest
        for table in recovered.tables.values()
    )

    database = postgres_state.open(schema)
    assert database.get_session("parent")["id"] == "parent"
    assert [row["content"] for row in database.get_messages("parent")] == [
        "migrate this durable message",
        "migration acknowledgement",
    ]


def test_recovery_refuses_executable_objects_in_published_target_schema(
    postgres_state: LivePostgresState,
):
    sqlite_path = _seed_sqlite_source(postgres_state.tmp_path)
    schema = postgres_state.schema()
    target = postgres_state.migration_target(schema)

    def fail_cutover(_: Any) -> None:
        raise RuntimeError("stop after publish")

    first = StatePostgresMigration(
        SQLiteMigrationSourceAdapter(sqlite_path),
        target,
        cutover=fail_cutover,
    ).run(MigrationRequest(apply=True, run_id="live-namespace-safety", batch_size=2))

    assert first.phase is MigrationPhase.FAILED
    assert first.published
    with postgres_state.psycopg.connect(postgres_state.dsn, autocommit=True) as connection:
        connection.execute(
            postgres_state.psycopg.sql.SQL(
                "CREATE FUNCTION {}.migration_shadow() RETURNS integer "
                "LANGUAGE sql AS 'SELECT 1'"
            ).format(postgres_state.psycopg.sql.Identifier(schema))
        )

    cutovers: list[str] = []
    recovered = StatePostgresMigration(
        SQLiteMigrationSourceAdapter(sqlite_path),
        target,
        cutover=lambda report: cutovers.append(report.run_id),
    ).run(MigrationRequest(apply=True, run_id="live-namespace-safety", batch_size=2))

    assert recovered.phase is MigrationPhase.FAILED
    assert not recovered.cutover_complete
    assert "function or procedure" in (recovered.failure or "")
    assert cutovers == []


def test_missing_postgres_schema_never_falls_back_to_sqlite(
    postgres_state: LivePostgresState,
):
    legacy = SessionDB.for_home(postgres_state.tmp_path)
    try:
        legacy.create_session("sqlite-only", "cli")
    finally:
        legacy.close()

    schema = postgres_state.schema()
    config = {
        "sessions": {
            "state": {
                "backend": "postgres",
                "sqlite_path": "state.db",
                "postgres": {"dsn_env": _DSN_ENV, "schema": schema},
            }
        }
    }

    with pytest.raises(PostgresSchemaValidationError):
        SessionDB.for_home(
            postgres_state.tmp_path,
            read_only=True,
            config=config,
            environ={_DSN_ENV: postgres_state.dsn},
        )
    assert not postgres_state.schema_exists(schema)
