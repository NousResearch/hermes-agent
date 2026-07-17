from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import inspect
from pathlib import Path
from typing import Any

import pytest

from hermes_state import SessionDB
from state_store.postgres.ops_messages import PostgresSessionDBMessageOperations
from state_store.spec import StateStoreSpec


def _spec() -> StateStoreSpec:
    return StateStoreSpec(
        home=Path("/tmp/hermes-postgres-ops"),
        profile="test",
        backend="postgres",
        sqlite_path=Path("/tmp/hermes-postgres-ops/state.db"),
        postgres_dsn_env="HERMES_TEST_POSTGRES_DSN",
        postgres_schema="test_state",
        read_only=False,
    )


class Cursor:
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
        raise AssertionError("PostgreSQL operations must not use fetchall")


class Connection:
    def __init__(self, cursors: list[Cursor]) -> None:
        self.cursors = cursors
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Cursor:
        self.executed.append((query, tuple(params)))
        if not self.cursors:
            raise AssertionError(f"unexpected SQL: {query}")
        return self.cursors.pop(0)


class Store:
    def __init__(self, cursors: list[Cursor]) -> None:
        self.spec = _spec()
        self.connection = Connection(cursors)
        self.transaction_modes: list[bool | None] = []

    @contextmanager
    def transaction(self, *, read_only: bool | None = None) -> Iterator[Connection]:
        self.transaction_modes.append(read_only)
        yield self.connection

    def close(self) -> None:
        pass


def _db(*cursors: Cursor) -> tuple[PostgresSessionDBMessageOperations, Store]:
    store = Store(list(cursors))
    return PostgresSessionDBMessageOperations(store, capabilities={"core_schema": True}), store


def test_public_message_operation_signatures_match_sqlite_contract() -> None:
    names = (
        "append_message",
        "replace_messages",
        "has_archived_messages",
        "archive_and_compact",
        "get_messages",
        "get_messages_around",
        "get_anchored_view",
        "resolve_resume_session_id",
        "get_messages_as_conversation",
        "get_conversation_root",
        "rewind_to_message",
        "restore_rewound",
        "list_recent_user_messages",
        "has_platform_message_id",
        "get_compression_lineage",
        "export_session",
        "export_session_lineage",
        "export_all",
        "import_sessions",
        "clear_messages",
        "delete_session",
        "delete_session_if_empty",
        "delete_sessions",
        "count_empty_sessions",
        "delete_empty_sessions",
        "prune_empty_ghost_sessions",
        "finalize_orphaned_compression_sessions",
        "list_prune_candidates",
        "archive_sessions",
        "prune_sessions",
        "vacuum",
        "maybe_auto_prune_and_vacuum",
    )
    for name in names:
        expected = inspect.signature(getattr(SessionDB, name))
        actual = inspect.signature(getattr(PostgresSessionDBMessageOperations, name))
        assert tuple(expected.parameters) == tuple(actual.parameters), name
        assert [
            parameter.default for parameter in expected.parameters.values()
        ] == [parameter.default for parameter in actual.parameters.values()], name


def test_source_uses_native_postgres_sql_and_never_fetchall() -> None:
    source = inspect.getsource(PostgresSessionDBMessageOperations)

    assert ".fetchall(" not in source
    assert "?" not in source
    assert "RETURNING id" in source
    assert "FOR UPDATE" in source


@pytest.mark.parametrize("reader", ("messages", "conversation", "export"))
def test_unbounded_transcript_reads_include_rows_beyond_max_read_limit(
    reader: str,
) -> None:
    rows = [
        {
            "id": index,
            "role": "assistant",
            "content": f"message-{index}",
            "active": 1,
        }
        for index in range(1, 10_002)
    ]
    message_cursor = Cursor(
        rows=rows,
        columns=("id", "role", "content", "active"),
    )
    if reader == "export":
        db, store = _db(
            Cursor(row={"id": "session-1"}, columns=("id",)),
            message_cursor,
        )
        messages = db.export_session("session-1")["messages"]
    elif reader == "conversation":
        db, store = _db(message_cursor)
        messages = db.get_messages_as_conversation("session-1")
    else:
        db, store = _db(message_cursor)
        messages = db.get_messages("session-1")

    assert len(messages) == 10_001
    assert messages[-1]["content"] == "message-10001"
    assert message_cursor.fetchmany_sizes == [
        db._READ_BATCH_SIZE
    ] * 41
    assert db._MAX_READ_ROWS not in store.connection.executed[-1][1]


def test_append_uses_one_write_transaction_identity_and_atomic_counters() -> None:
    db, store = _db(
        Cursor(row={"id": 901}, columns=("id",)),
        Cursor(rowcount=1),
    )

    assert db.append_message(
        "session-1",
        "assistant",
        tool_calls=[{"id": "call-1"}],
        observed=True,
    ) == 901

    assert store.transaction_modes == [False]
    insert, update = store.connection.executed
    assert "INSERT INTO messages" in insert[0]
    assert "RETURNING id" in insert[0]
    assert insert[1][-1] == 1
    assert "message_count = COALESCE(message_count, 0) + 1" in update[0]
    assert update[1] == (1, "session-1")


def test_replace_is_one_transaction_and_failure_does_not_open_a_second_transaction() -> None:
    class FailingConnection(Connection):
        def execute(self, query: str, params: tuple[Any, ...] = ()) -> Cursor:
            if "INSERT INTO messages" in query:
                raise RuntimeError("driver write failed")
            return super().execute(query, params)

    store = Store([Cursor(rowcount=2)])
    store.connection = FailingConnection(store.connection.cursors)
    db = PostgresSessionDBMessageOperations(store, capabilities={})

    with pytest.raises(RuntimeError, match="driver write failed"):
        db.replace_messages("session-1", [{"role": "user", "content": "hello"}])

    assert store.transaction_modes == [False]
    assert "DELETE FROM messages" in store.connection.executed[0][0]


def test_compaction_and_rewind_keep_active_and_compacted_semantics() -> None:
    db, store = _db(
        Cursor(rowcount=3),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
        Cursor(
            row={"id": 20, "session_id": "s", "role": "user", "content": "retry"},
            columns=("id", "session_id", "role", "content"),
        ),
        Cursor(rows=[{"id": 20}, {"id": 21}], columns=("id",)),
        Cursor(rowcount=1),
        Cursor(row={"id": 19}, columns=("id",)),
    )

    assert db.archive_and_compact("s", [{"role": "assistant", "content": "summary"}]) == 1
    rewind = db.rewind_to_message("s", 20)

    assert rewind["rewound_count"] == 2
    assert rewind["target_message"]["content"] == "retry"
    assert rewind["new_head_id"] == 19
    sql = "\n".join(query for query, _ in store.connection.executed)
    assert "SET active = 0, compacted = 1" in sql
    assert "UPDATE messages SET active = 0" in sql
    assert "UPDATE sessions\n                SET rewind_count" in sql
    assert store.transaction_modes == [False, False]


def test_import_attaches_child_before_parent_after_all_rows_exist() -> None:
    db, store = _db(
        Cursor(row=None),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
        Cursor(row=None),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
        Cursor(row={"found": 1}, columns=("found",)),
        Cursor(rowcount=1),
    )

    result = db.import_sessions(
        [
            {
                "id": "child",
                "source": "import",
                "parent_session_id": "parent",
                "messages": [{"role": "user", "content": "child turn"}],
            },
            {
                "id": "parent",
                "source": "import",
                "messages": [{"role": "user", "content": "parent turn"}],
            },
        ]
    )

    assert result == {
        "ok": True,
        "imported": 2,
        "skipped": 0,
        "detached": 0,
        "imported_ids": ["child", "parent"],
        "skipped_ids": [],
        "errors": [],
    }
    inserts = [
        params for query, params in store.connection.executed if "INSERT INTO sessions" in query
    ]
    assert [params[0] for params in inserts] == ["child", "parent"]
    assert store.connection.executed[-1][1] == ("parent", "child")


def test_delete_cascades_only_delegate_children_and_orphans_other_children() -> None:
    db, store = _db(
        Cursor(rows=[{"id": "parent"}], columns=("id",)),
        Cursor(rows=[{"id": "delegate"}], columns=("id",)),
        Cursor(rows=[], columns=("id",)),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
        Cursor(rowcount=1),
    )

    assert db.delete_session("parent") is True

    sql = "\n".join(query for query, _ in store.connection.executed)
    assert 'LIKE \'%%"_delegate_from"%%\'' in sql
    assert "SET parent_session_id = NULL" in sql
    assert store.connection.executed[-3][1] == (["parent", "delegate"],)
    assert store.connection.executed[-2][1] == (["parent", "delegate"],)
    assert store.connection.executed[-1][1] == (["parent", "delegate"],)


def test_prune_candidates_keep_filters_in_sql_and_bound_reads() -> None:
    cursor = Cursor(
        rows=[{"id": "old", "archived": 0}],
        columns=("id", "archived"),
    )
    db, store = _db(cursor)

    rows = db.list_prune_candidates(
        older_than_days=None,
        source="tui",
        title_like="bug",
        provider="OPENAI",
        min_messages=2,
        archived=False,
    )

    assert rows == [{"id": "old", "archived": 0}]
    query, params = store.connection.executed[0]
    assert "s.source = %s" in query
    assert "LOWER(COALESCE(s.title, '')) LIKE %s" in query
    assert "LOWER(COALESCE(s.billing_provider, '')) = %s" in query
    assert "s.message_count >= %s" in query
    assert "s.archived = %s" in query
    assert params[-1] == db._MAX_READ_ROWS
    assert cursor.fetchmany_sizes == [db._READ_BATCH_SIZE, db._READ_BATCH_SIZE]


def test_vacuum_is_explicit_postgres_noop_and_auto_prune_marks_vacuumed_only_after_prune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db, _store = _db(Cursor(row=None), Cursor(rowcount=1))
    assert db.vacuum() == 0
    monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 2)

    result = db.maybe_auto_prune_and_vacuum(vacuum=True)

    assert result == {"skipped": False, "pruned": 2, "vacuumed": True}
