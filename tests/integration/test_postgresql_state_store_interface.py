"""PostgreSQL StateStoreInterface conformance and clear_stored_system_prompts evidence.

The PG backend proves the same backend-neutral contract the SQLite reference pins
in tests/test_state_store_interface.py: structural isinstance conformance of both
PostgreSQLStateStore and its CLI facade, plus the out-of-line prompt-invalidation
behavior — references NULLed before snapshot rows are deleted, session rows kept,
idempotent on a re-run.
"""

from __future__ import annotations

import uuid

import pytest

from cli_session_store import PostgreSQLCLISessionStore
from state_store import PostgreSQLStateStoreConfig
from state_store_interface import StateStoreInterface
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"


@pytest.fixture
def store(postgresql_test_target: OwnedPostgreSQLTestTarget):
    settings = PostgreSQLStateStoreConfig(dsn_env="TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)
    handle = PostgreSQLStateStore(settings, _DSN, schema=postgresql_test_target.schema)
    try:
        yield handle
    finally:
        handle.close()


def test_postgresql_state_store_satisfies_the_protocol(store):
    # Structural typing against the @runtime_checkable protocol: no inheritance,
    # the isinstance pass IS the plugin-compatibility check.
    assert isinstance(store, StateStoreInterface)


def test_cli_facade_satisfies_the_protocol(store):
    assert isinstance(PostgreSQLCLISessionStore(store), StateStoreInterface)


def test_clear_stored_system_prompts_nulls_references_and_deletes_snapshots(
    store, postgresql_test_target: OwnedPostgreSQLTestTarget
):
    first = f"interface-clear-{uuid.uuid4()}"
    second = f"interface-clear-{uuid.uuid4()}"
    store.ensure_session(first, source="interface-test")
    store.ensure_session(second, source="interface-test")
    store.set_system_prompt(first, "shared snapshot")
    store.set_system_prompt(second, "shared snapshot")  # same hash: one snapshot row, two references

    result = store.clear_stored_system_prompts()

    assert result == {"cleared": 2, "storage_mode": "out-of-line"}
    assert store.get_system_prompt(first) is None
    assert store.get_system_prompt(second) is None
    assert store.get_session(first)["id"] == first  # session rows survive the clear
    with postgresql_test_target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT count(*) FROM {postgresql_test_target.schema}.sessions "
            "WHERE system_prompt_hash IS NOT NULL"
        )
        assert cursor.fetchone()[0] == 0
        cursor.execute(f"SELECT count(*) FROM {postgresql_test_target.schema}.system_prompts")
        assert cursor.fetchone()[0] == 0

    # Idempotent: nothing stored anymore, the second run clears zero.
    assert store.clear_stored_system_prompts() == {"cleared": 0, "storage_mode": "out-of-line"}


def test_clear_stored_system_prompts_on_empty_schema_reports_zero(store):
    assert store.clear_stored_system_prompts() == {"cleared": 0, "storage_mode": "out-of-line"}


def test_cli_facade_clear_passthrough_smoke(store):
    session_id = f"facade-clear-{uuid.uuid4()}"
    facade = PostgreSQLCLISessionStore(store)
    facade.create_session(session_id, "interface-test", system_prompt="facade snapshot")

    result = facade.clear_stored_system_prompts()

    assert result == {"cleared": 1, "storage_mode": "out-of-line"}
    assert store.get_system_prompt(session_id) is None
