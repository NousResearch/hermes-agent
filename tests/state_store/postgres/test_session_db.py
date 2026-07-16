from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hermes_state import APISessionMutationResult, SessionDB
from state_store.postgres import (
    PostgresConfigurationError,
    PostgresReadOnlyError,
    PostgresSchemaValidationError,
)
from state_store.postgres.session_db import PostgresSessionDB
from state_store.postgres.session_db_base import PostgresSessionDBBase
from state_store.session_api import (
    APISessionMutationAbort,
    session_api_signature_manifest,
)
from state_store.spec import StateStoreSpec


def _spec(*, read_only: bool = False) -> StateStoreSpec:
    return StateStoreSpec(
        home=Path("/tmp/hermes"),
        profile="test",
        backend="postgres",
        sqlite_path=Path("/tmp/hermes/state.db"),
        postgres_dsn_env="HERMES_TEST_POSTGRES_DSN",
        postgres_schema="test_state",
        read_only=read_only,
    )


class FakeCursor:
    def __init__(
        self,
        *,
        row: Any = None,
        rows: list[Any] | None = None,
        columns: tuple[str, ...] = (),
        rowcount: int = 0,
    ) -> None:
        self._row = row
        self._rows = list(rows or [])
        self.description = tuple((column,) for column in columns)
        self.rowcount = rowcount
        self.fetchmany_sizes: list[int] = []

    def fetchone(self) -> Any:
        return self._row

    def fetchmany(self, size: int) -> list[Any]:
        self.fetchmany_sizes.append(size)
        batch, self._rows = self._rows[:size], self._rows[size:]
        return batch

    def fetchall(self) -> list[Any]:
        raise AssertionError("SessionDB base must not call fetchall")


class FakeConnection:
    def __init__(self, cursors: list[FakeCursor]) -> None:
        self._cursors = cursors
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        self.executed.append((query, params))
        return self._cursors.pop(0)


class FakeBaseStore:
    def __init__(
        self,
        *,
        spec: StateStoreSpec,
        cursors: list[FakeCursor] | None = None,
    ) -> None:
        self.spec = spec
        self.connection = FakeConnection(cursors or [])
        self.transaction_modes: list[bool | None] = []
        self.closed = False

    @contextmanager
    def transaction(self, *, read_only: bool | None = None) -> Iterator[FakeConnection]:
        self.transaction_modes.append(read_only)
        yield self.connection

    def close(self) -> None:
        self.closed = True


class FakeLifecycleStore:
    def __init__(
        self,
        spec: StateStoreSpec,
        *,
        available: bool = True,
        core_schema: bool = True,
        migrate_error: BaseException | None = None,
    ) -> None:
        self.spec = spec
        self.available = available
        self.core_schema = core_schema
        self.migrate_error = migrate_error
        self.calls: list[str] = []
        self.closed = False

    def migrate(self) -> None:
        self.calls.append("migrate")
        if self.migrate_error is not None:
            raise self.migrate_error

    def health_report(self) -> Any:
        self.calls.append("health_report")
        return SimpleNamespace(
            available=self.available,
            capabilities={"core_schema": self.core_schema, "future": True},
        )

    def close(self) -> None:
        self.calls.append("close")
        self.closed = True


def test_shared_api_contract_is_immutable_and_used_by_sqlite() -> None:
    manifest = session_api_signature_manifest(SessionDB)

    assert "create_session" in manifest.signatures
    assert "for_home" in manifest.signatures
    assert manifest.signatures["create_session"].startswith(
        "(self, session_id: str, source: str"
    )
    assert APISessionMutationResult.__module__ == "state_store.session_api"
    assert APISessionMutationAbort.__module__ == "state_store.session_api"
    with pytest.raises(TypeError):
        manifest.signatures["create_session"] = "changed"  # type: ignore[index]


def test_postgres_base_normalizes_mapping_and_tuple_rows_without_cursor_escape() -> None:
    store = FakeBaseStore(
        spec=_spec(),
        cursors=[
            FakeCursor(row={"id": "mapping"}),
            FakeCursor(
                rows=[("one", 1), ("two", 2)],
                columns=("name", "count"),
            ),
            FakeCursor(row=("created",), columns=("status",)),
            FakeCursor(rowcount=3),
        ],
    )
    db = PostgresSessionDBBase(store, capabilities={"core_schema": True})

    assert db._read_one("SELECT mapping") == {"id": "mapping"}
    assert db._read_many("SELECT tuple", limit=2) == [
        {"name": "one", "count": 1},
        {"name": "two", "count": 2},
    ]
    assert db._write_returning("INSERT returning") == {"status": "created"}
    assert db._write("UPDATE rows") == 3
    assert store.transaction_modes == [True, True, False, False]
    assert db.db_path is None


@pytest.mark.parametrize("limit", (0, -1, True, PostgresSessionDBBase._MAX_READ_ROWS + 1))
def test_postgres_base_rejects_invalid_read_limits(limit: int) -> None:
    store = FakeBaseStore(spec=_spec())
    db = PostgresSessionDBBase(store, capabilities={})

    with pytest.raises(ValueError):
        db._read_many("SELECT blocked", limit=limit)

    assert store.transaction_modes == []


def test_postgres_base_read_many_streams_only_the_requested_bound() -> None:
    cursor = FakeCursor(
        rows=[("one",), ("two",), ("three",)],
        columns=("id",),
    )
    store = FakeBaseStore(spec=_spec(), cursors=[cursor])
    db = PostgresSessionDBBase(store, capabilities={})

    assert db._read_many("SELECT ids", limit=2) == [{"id": "one"}, {"id": "two"}]
    assert cursor.fetchmany_sizes == [2]
    assert cursor._rows == [("three",)]


def test_postgres_base_cannot_bypass_read_only_state() -> None:
    store = FakeBaseStore(spec=_spec(read_only=True))
    db = PostgresSessionDBBase(store, capabilities={})

    with pytest.raises(PostgresReadOnlyError):
        db._write("DELETE FROM sessions")

    assert store.transaction_modes == []


def test_postgres_base_keeps_sql_and_cursors_inside_bounded_helpers() -> None:
    source = inspect.getsource(PostgresSessionDBBase)

    assert ".fetchall(" not in source
    assert "import re" not in source
    assert "re.sub(" not in source
    assert "re.compile(" not in source


def test_postgres_session_db_requires_configured_dsn_without_leaking_value() -> None:
    secret = "postgresql://user:super-secret@example.invalid/db"
    factory_calls: list[tuple[Any, ...]] = []

    def factory(*args: Any, **kwargs: Any) -> FakeLifecycleStore:
        factory_calls.append((args, kwargs))
        raise AssertionError("factory must not be called without a configured DSN")

    with pytest.raises(PostgresConfigurationError) as exc_info:
        PostgresSessionDB.from_spec(
            _spec(),
            environ={},
            state_store_factory=factory,
        )

    assert factory_calls == []
    assert secret not in str(exc_info.value)


def test_postgres_session_db_writable_factory_migrates_then_validates() -> None:
    lifecycle = FakeLifecycleStore(_spec())
    factory_calls: list[tuple[StateStoreSpec, Mapping[str, str]]] = []

    def factory(
        spec: StateStoreSpec,
        *,
        environ: Mapping[str, str],
    ) -> FakeLifecycleStore:
        factory_calls.append((spec, environ))
        return lifecycle

    db = PostgresSessionDB.from_spec(
        _spec(),
        environ={"HERMES_TEST_POSTGRES_DSN": "postgresql://ignored"},
        state_store_factory=factory,
    )

    assert lifecycle.calls == ["migrate", "health_report"]
    assert factory_calls == [(
        _spec(),
        {"HERMES_TEST_POSTGRES_DSN": "postgresql://ignored"},
    )]
    assert db.capabilities == {"core_schema": True, "future": True}
    db.close()
    assert lifecycle.closed is True


def test_postgres_session_db_read_only_validates_without_migration() -> None:
    spec = _spec(read_only=True)
    lifecycle = FakeLifecycleStore(spec)

    db = PostgresSessionDB.from_spec(
        spec,
        environ={"HERMES_TEST_POSTGRES_DSN": "postgresql://ignored"},
        state_store_factory=lambda *_args, **_kwargs: lifecycle,
    )

    assert lifecycle.calls == ["health_report"]
    assert db.read_only is True


@pytest.mark.parametrize(
    ("available", "core_schema"),
    ((False, True), (True, False)),
)
def test_postgres_session_db_fails_closed_for_invalid_read_only_schema(
    available: bool,
    core_schema: bool,
) -> None:
    lifecycle = FakeLifecycleStore(
        _spec(read_only=True),
        available=available,
        core_schema=core_schema,
    )

    with pytest.raises(PostgresSchemaValidationError):
        PostgresSessionDB.from_spec(
            _spec(read_only=True),
            environ={"HERMES_TEST_POSTGRES_DSN": "postgresql://ignored"},
            state_store_factory=lambda *_args, **_kwargs: lifecycle,
        )

    assert lifecycle.calls == ["health_report", "close"]


def test_postgres_session_db_closes_store_when_writable_initialization_fails() -> None:
    lifecycle = FakeLifecycleStore(
        _spec(),
        migrate_error=PostgresSchemaValidationError("safe failure"),
    )

    with pytest.raises(PostgresSchemaValidationError):
        PostgresSessionDB.from_spec(
            _spec(),
            environ={"HERMES_TEST_POSTGRES_DSN": "postgresql://ignored"},
            state_store_factory=lambda *_args, **_kwargs: lifecycle,
        )

    assert lifecycle.calls == ["migrate", "close"]


def test_session_db_for_home_preserves_sqlite_and_dispatches_postgres_lazily(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sqlite = SessionDB.for_home(tmp_path, config={})
    try:
        assert type(sqlite) is SessionDB
        assert sqlite.db_path == tmp_path / "state.db"
    finally:
        sqlite.close()

    expected = object()
    calls: list[tuple[StateStoreSpec, Mapping[str, str] | None]] = []

    def from_spec(
        cls: type[PostgresSessionDB],
        spec: StateStoreSpec,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> object:
        calls.append((spec, environ))
        return expected

    monkeypatch.setattr(PostgresSessionDB, "from_spec", classmethod(from_spec))
    environ = {"HERMES_TEST_POSTGRES_DSN": "postgresql://ignored"}
    config = {
        "sessions": {
            "state": {
                "backend": "postgres",
                "postgres": {
                    "dsn_env": "HERMES_TEST_POSTGRES_DSN",
                    "schema": "test_state",
                },
            },
        },
    }

    assert SessionDB.for_home(tmp_path, config=config, environ=environ) is expected
    assert calls == [(
        StateStoreSpec(
            home=tmp_path,
            profile="default",
            backend="postgres",
            sqlite_path=tmp_path / "state.db",
            postgres_dsn_env="HERMES_TEST_POSTGRES_DSN",
            postgres_schema="test_state",
            read_only=False,
        ),
        environ,
    )]


def test_explicit_session_db_constructor_remains_sqlite_only(tmp_path: Path) -> None:
    db = SessionDB(db_path=tmp_path / "explicit.db")
    try:
        assert type(db) is SessionDB
        assert db.db_path == tmp_path / "explicit.db"
    finally:
        db.close()
