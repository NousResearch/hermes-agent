"""Selected PostgreSQL run-idempotency store must fail closed without SQLite artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.platforms import api_server_run_idempotency_postgresql as pg_run_idempotency
from gateway.platforms.api_server_run_idempotency_adapter import (
    open_run_idempotency_store,
    selected_run_idempotency_store_factory,
)
from state_store import StateStoreConfigurationError


class _StaleCursor:
    def __init__(self, rows):
        self.rows = rows
        self.statements = []

    def execute(self, sql, params):
        self.statements.append((sql, params))

    def fetchall(self):
        return self.rows


@pytest.mark.parametrize("corrupt_status", [
    "{", None, b"\xff", "[]", "null", "42", "{}", '{"status": "running"}',
    '{"status": ["completed"]}', '{"status": {"value": "completed"}}',
])
def test_pg_prune_keeps_corrupt_and_nonterminal_stale_rows(corrupt_status):
    store = object.__new__(pg_run_idempotency.PostgreSQLRunIdempotencyStore)
    cur = _StaleCursor([
        ("scope", "completed", '{"status": "completed"}'),
        ("scope", "retained", corrupt_status),
        ("scope", "failed", '{"status": "failed"}'),
    ])

    store._prune_stale_terminal(cur, 100.0)

    deletes = [(sql, params) for sql, params in cur.statements if sql.startswith("DELETE")]
    assert [params for _, params in deletes] == [
        ("scope", "completed"), ("scope", "failed"),
    ]


def test_pg_prune_propagates_unexpected_parser_error_before_deletion(monkeypatch):
    store = object.__new__(pg_run_idempotency.PostgreSQLRunIdempotencyStore)
    cur = _StaleCursor([("scope", "completed", '{"status": "completed"}')])
    failure = RuntimeError("parser failed unexpectedly")

    def fail_parse(_status):
        raise failure

    monkeypatch.setattr(pg_run_idempotency.json, "loads", fail_parse)
    with pytest.raises(RuntimeError) as caught:
        store._prune_stale_terminal(cur, 100.0)
    assert caught.value is failure
    assert len(cur.statements) == 1
    assert cur.statements[0][0].startswith("SELECT")


_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _selected_pg_home(tmp_path: Path, monkeypatch, dsn: str | None = "postgresql://fixture/only") -> Path:
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    if dsn is None:
        monkeypatch.delenv("HERMES_STATE_STORE_TEST_DSN", raising=False)
    else:
        monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", dsn)
    return home


def test_sqlite_default_factory_returns_none(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    assert selected_run_idempotency_store_factory() is None


def test_selected_pg_without_dsn_raises_never_none(tmp_path, monkeypatch):
    _selected_pg_home(tmp_path, monkeypatch, dsn=None)

    with pytest.raises((ValueError, StateStoreConfigurationError)):
        selected_run_idempotency_store_factory()


def test_selected_pg_config_load_failure_never_routes_to_sqlite(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch)
    import hermes_cli.config

    failure = RuntimeError("config load failed")

    def fail_load_config():
        raise failure

    monkeypatch.setattr(hermes_cli.config, "load_config", fail_load_config)
    with pytest.raises(RuntimeError) as caught:
        selected_run_idempotency_store_factory()
    assert caught.value is failure
    assert not (home / "state.db").exists()
    assert not (home / "runs_idempotency.db").exists()


def test_selected_pg_unreachable_dsn_returns_factory_that_raises_typed(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch, dsn="postgresql://127.0.0.1:1/does_not_exist")

    factory = selected_run_idempotency_store_factory()
    assert factory is not None

    import psycopg
    with pytest.raises(psycopg.OperationalError):
        factory()

    assert not (home / "runs_idempotency.db").exists()
    assert not (home / "state.db").exists()


def test_open_run_idempotency_store_rejects_missing_or_unknown_backend():
    with pytest.raises(ValueError):
        open_run_idempotency_store(backend="postgresql")
    with pytest.raises(ValueError):
        open_run_idempotency_store(backend="bogus")


def test_sqlite_run_idempotency_store_regression():
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore

    store = RunIdempotencyStore(db_path=":memory:")
    try:
        outcome, _ = store.reserve("s", "k", "fp", "run-1", {"status": "running"})
        assert outcome == "created"
        outcome, _ = store.reserve("s", "k", "fp", "run-2", {"status": "running"})
        assert outcome == "reused"
        outcome, _ = store.reserve("s", "k", "fp-other", "run-3", {"status": "running"})
        assert outcome == "conflict"
    finally:
        store.close()
