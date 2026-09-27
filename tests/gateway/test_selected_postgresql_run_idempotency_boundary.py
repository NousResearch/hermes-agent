"""Selected PostgreSQL run-idempotency store must fail closed without SQLite artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest

from gateway.platforms.api_server_run_idempotency_adapter import (
    open_run_idempotency_store,
    selected_run_idempotency_store_factory,
)
from state_store import StateStoreConfigurationError

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
