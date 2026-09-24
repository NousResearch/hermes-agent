"""Real isolated PostgreSQL evidence for ``hermes sessions optimize``."""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from postgresql_state_store_operations import PostgreSQLSandboxOperations, PostgreSQLSandboxOperationsError
from state_store import PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from state_store_runtime_readiness import trap_state_db_opens

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="HERMES_STATE_STORE_TEST_DSN", connect_timeout_seconds=5, pool_max_size=2)


@pytest.fixture
def pg_optimize(tmp_path, monkeypatch, postgresql_test_target):
    home = tmp_path / ".hermes-pg-optimize"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n"
        "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 5\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    (home / ".env").write_text(f"HERMES_STATE_STORE_TEST_DSN={_DSN}\n", encoding="utf-8")
    import state_store

    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: postgresql_test_target.schema)
    token = set_hermes_home_override(str(home))
    store = PostgreSQLStateStore(_SETTINGS, _DSN, schema=postgresql_test_target.schema)
    operations = PostgreSQLSandboxOperations(_SETTINGS, _DSN, schema=postgresql_test_target.schema)
    try:
        yield home, store, operations, postgresql_test_target
    finally:
        store.close()
        reset_hermes_home_override(token)


def test_selected_pg_sessions_optimize_is_native_bounded_and_never_opens_sqlite(pg_optimize, capsys):
    home, _store, _operations, _target = pg_optimize
    from hermes_cli.sessions_cmd import cmd_sessions

    with trap_state_db_opens(home) as opens:
        assert cmd_sessions(Namespace(sessions_action="optimize")) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["operation"] == "vacuum_analyze"
    assert report["schema"] == _target.schema
    assert report["before"]["catalog"]["healthy"] is True
    assert report["after"]["catalog"]["healthy"] is True
    assert report["before"]["search"]["gin_index"] == "valid"
    assert report["timeouts"] == {"statement_timeout_ms": 30000, "lock_timeout_ms": 2000}
    assert report["file_size_equivalence"] is False
    assert opens == []
    assert not (home / "state.db").exists()


def test_pg_optimize_refuses_live_turn_or_compression_leases_without_vacuum(pg_optimize):
    _home, store, operations, target = pg_optimize
    store.ensure_session("leased", source="integration")
    target.execute(
        f"INSERT INTO {target.schema}.compression_locks (session_id, holder, fence, expires_at, updated_at) "
        "VALUES ('leased', 'test', 1, EXTRACT(EPOCH FROM clock_timestamp()) + 60, EXTRACT(EPOCH FROM clock_timestamp()))"
    )

    with pytest.raises(PostgreSQLSandboxOperationsError, match="active session-turn or compression lease"):
        operations.optimize()

    with operations._connect() as connection, connection.cursor() as cursor:
        diagnostic = operations._optimize_diagnostic(cursor)
    assert diagnostic["leases"]["compression_locks"] == 1


def test_pg_optimize_refuses_search_catalog_drift_and_existing_repair_restores_health(pg_optimize):
    _home, store, operations, target = pg_optimize
    store.ensure_session("drift", source="integration")
    target.execute(f"DROP INDEX {target.schema}.messages_search_document_gin")

    with pytest.raises(PostgreSQLSandboxOperationsError, match="catalog or search index is unhealthy"):
        operations.optimize()
    assert store.search_index_status()["gin_index"] == "missing"
    assert store.rebuild_search_index()["rebuild"]["operation"] == "create"
    assert operations.optimize()["after"]["search"]["healthy"] is True


def test_pg_optimize_reports_a_real_tenant_lock_timeout(pg_optimize):
    _home, _store, operations, target = pg_optimize
    blocker = target.connect()
    try:
        with blocker.cursor() as cursor:
            cursor.execute(f"LOCK TABLE {target.schema}.sessions IN ACCESS EXCLUSIVE MODE")
            with pytest.raises(PostgreSQLSandboxOperationsError, match="lock timeout"):
                operations.optimize()
    finally:
        blocker.rollback()
        blocker.close()
