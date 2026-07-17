from collections.abc import Iterator
from contextlib import contextmanager
import inspect
from pathlib import Path
from typing import Any

from hermes_state import SessionDB
from state_store.postgres.ops_search import PostgresSearchOperations
from state_store.postgres.search_ddl import (
    POSTGRES_SEARCH_CAPABILITY,
    SEARCH_CAPABILITY_SETUP_SQL,
    SEARCH_DOCUMENT_EXPRESSION,
    SEARCH_REINDEX_STATEMENTS,
    search_capability_setup_sql,
)
from state_store.postgres.session_db_base import PostgresSessionDBBase
from state_store.spec import StateStoreSpec


class FakeCursor:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self._rows = list(rows or [])
        self.description = ()

    def fetchmany(self, size: int) -> list[Any]:
        batch, self._rows = self._rows[:size], self._rows[size:]
        return batch

    def fetchone(self) -> Any:
        return None

    def fetchall(self) -> list[Any]:
        raise AssertionError("PostgreSQL search operations must not use fetchall")


class FakeConnection:
    def __init__(self, cursors: list[FakeCursor]) -> None:
        self._cursors = list(cursors)
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        self.queries.append((query, tuple(params)))
        if self._cursors:
            return self._cursors.pop(0)
        return FakeCursor()


class FakeStore:
    def __init__(
        self,
        *,
        read_only: bool = False,
        cursors: list[FakeCursor] | None = None,
    ) -> None:
        self.spec = StateStoreSpec(
            home=Path("/tmp/hermes"),
            profile="test",
            backend="postgres",
            sqlite_path=Path("/tmp/hermes/state.db"),
            postgres_dsn_env="HERMES_TEST_POSTGRES_DSN",
            postgres_schema="test_state",
            read_only=read_only,
        )
        self.connection = FakeConnection(cursors or [])
        self.transaction_modes: list[bool | None] = []

    @contextmanager
    def transaction(self, *, read_only: bool | None = None) -> Iterator[FakeConnection]:
        self.transaction_modes.append(read_only)
        yield self.connection

    def close(self) -> None:
        pass


class SearchDB(PostgresSearchOperations, PostgresSessionDBBase):
    """Concrete test-only composition for the isolated search mixin."""


def _db(
    *,
    rows: list[dict[str, Any]] | None = None,
    read_only: bool = False,
    capabilities: dict[str, bool] | None = None,
) -> tuple[SearchDB, FakeStore]:
    store = FakeStore(read_only=read_only, cursors=[FakeCursor(rows)])
    db = SearchDB(
        store,
        capabilities=(
            {POSTGRES_SEARCH_CAPABILITY: True}
            if capabilities is None
            else capabilities
        ),
    )
    return db, store


def test_search_method_signatures_match_sqlite_session_db() -> None:
    for name in (
        "search_messages",
        "search_sessions_by_id",
        "search_sessions",
        "optimize_fts",
        "rebuild_fts",
    ):
        assert inspect.signature(getattr(PostgresSearchOperations, name)) == inspect.signature(
            getattr(SessionDB, name)
        )


def test_search_ddl_is_postgresql_native_and_covers_vector_and_trigram_paths() -> None:
    statements = search_capability_setup_sql()
    assert statements == SEARCH_CAPABILITY_SETUP_SQL
    assert SEARCH_DOCUMENT_EXPRESSION in statements[1]
    assert SEARCH_DOCUMENT_EXPRESSION in statements[3]

    ddl = "\n".join(statements).lower()
    assert "create extension if not exists pg_trgm" in ddl
    assert "generated always as" in ddl
    assert "to_tsvector" in ddl
    assert "using gin" in ddl
    assert "gin_trgm_ops" in ddl
    for forbidden in (
        "sqlite",
        "pragma",
        "fts5",
        "virtual table",
        "rowid",
        "autoincrement",
    ):
        assert forbidden not in ddl


def test_lexical_search_uses_rank_headline_filters_and_preview_only_results() -> None:
    secret = "do not return this complete message"
    db, store = _db(
        rows=[
            {
                "id": 7,
                "session_id": "session-7",
                "role": "assistant",
                "snippet": "x" * 800,
                "content": secret,
                "timestamp": 123.0,
                "tool_name": "docker",
                "source": "cli",
                "model": "test-model",
                "session_started": 100.0,
                "context": [
                    {"role": "user", "content": "a" * 300},
                    {"role": "assistant", "content": "answer"},
                    {"role": "tool", "content": "tail"},
                    {"role": "ignored", "content": "not returned"},
                ],
            }
        ]
    )
    query = "docker compose --build"

    results = db.search_messages(
        query,
        source_filter=["cli", "telegram"],
        exclude_sources=["cron"],
        role_filter=["assistant"],
        limit=20,
        offset=4,
    )

    assert len(results) == 1
    result = results[0]
    assert "content" not in result
    assert len(result["snippet"]) == 512
    assert result["context"] == [
        {"role": "user", "content": "a" * 200},
        {"role": "assistant", "content": "answer"},
        {"role": "tool", "content": "tail"},
    ]

    sql, params = store.connection.queries[0]
    assert query not in sql
    assert "websearch_to_tsquery" in sql
    assert "plainto_tsquery" in sql
    assert "ts_rank(m.search_vector, parsed.query)" in sql
    assert "ts_headline" in sql
    assert "LEFT(" in sql
    assert "(m.active = 1 OR m.compacted = 1)" in sql
    assert "s.source IN (%s, %s)" in sql
    assert "s.source NOT IN (%s)" in sql
    assert "m.role IN (%s)" in sql
    assert "ORDER BY rank DESC, m.timestamp DESC, m.id DESC" in sql
    assert "jsonb_agg" in sql
    assert "context_message.content" in sql
    assert secret not in sql
    assert params == (
        query,
        query,
        "cli",
        "telegram",
        "cron",
        "assistant",
        20,
        4,
    )
    assert store.transaction_modes == [True]


def test_cjk_search_uses_escaped_ilike_and_trigram_similarity() -> None:
    db, store = _db(rows=[])
    query = "大别山项目"

    assert db.search_messages(query, limit=3, sort="oldest") == []

    sql, params = store.connection.queries[0]
    assert query not in sql
    assert "ILIKE %s ESCAPE E'\\\\'" in sql
    assert sql.count("public.similarity(") == 3
    assert "websearch_to_tsquery" not in sql
    assert "ORDER BY m.timestamp ASC, rank DESC, m.id ASC" in sql
    assert params == (
        query,
        query,
        query,
        f"%{query}%",
        f"%{query}%",
        f"%{query}%",
        3,
        0,
    )


def test_search_messages_escapes_like_metacharacters_and_fails_closed() -> None:
    db, store = _db(rows=[])
    unsafe = "100%_\\needle"

    assert db.search_messages(unsafe + "大", limit=1) == []
    _, params = store.connection.queries[0]
    assert params[3:6] == (
        "%100\\%\\_\\\\needle大%",
        "%100\\%\\_\\\\needle大%",
        "%100\\%\\_\\\\needle大%",
    )

    blocked, blocked_store = _db(rows=[])
    assert blocked.search_messages("\x00bad") == []
    assert blocked.search_messages("   ") == []
    assert blocked.search_messages("valid", source_filter=[]) == []
    assert blocked_store.connection.queries == []

    unavailable, unavailable_store = _db(rows=[], capabilities={})
    assert unavailable.search_messages("valid") == []
    assert unavailable_store.connection.queries == []


def test_search_messages_uses_fixed_sort_clauses_and_accepts_compacted_rows() -> None:
    db, store = _db(rows=[])
    assert db.search_messages("first", sort="newest") == []
    store.connection._cursors.append(FakeCursor())
    assert db.search_messages("second", sort="oldest") == []
    store.connection._cursors.append(FakeCursor())
    assert db.search_messages("third", sort="unexpected") == []

    newest_sql = store.connection.queries[0][0]
    oldest_sql = store.connection.queries[1][0]
    default_sql = store.connection.queries[2][0]
    assert "ORDER BY m.timestamp DESC, rank DESC, m.id DESC" in newest_sql
    assert "ORDER BY m.timestamp ASC, rank DESC, m.id ASC" in oldest_sql
    assert "ORDER BY rank DESC, m.timestamp DESC, m.id DESC" in default_sql
    assert all("(m.active = 1 OR m.compacted = 1)" in sql for sql in (
        newest_sql,
        oldest_sql,
        default_sql,
    ))


def test_session_id_search_uses_escaped_bound_patterns_and_hides_rank() -> None:
    db, store = _db(rows=[{"id": "abc", "_id_match_rank": 0, "last_active": 3.0}])
    query = "ab%_\\c"

    assert db.search_sessions_by_id(query, limit=2, include_archived=False) == [
        {"id": "abc", "last_active": 3.0}
    ]

    sql, params = store.connection.queries[0]
    assert query not in sql
    assert "ILIKE %s ESCAPE E'\\\\'" in sql
    assert "AND s.archived = 0" in sql
    assert "ORDER BY _id_match_rank, last_active DESC" in sql
    assert params == (
        query,
        "ab\\%\\_\\\\c%",
        "%ab\\%\\_\\\\c%",
        2,
    )


def test_search_sessions_preserves_source_filter_and_bounded_pagination() -> None:
    db, store = _db(rows=[{"id": "one", "source": "cli", "last_active": 2.0}])

    assert db.search_sessions(source="cli", limit=5, offset=2) == [
        {"id": "one", "source": "cli", "last_active": 2.0}
    ]

    sql, params = store.connection.queries[0]
    assert "WHERE s.source = %s" in sql
    assert "MAX(message.timestamp)" in sql
    assert params == ("cli", 5, 2)

    invalid, invalid_store = _db(rows=[])
    assert invalid.search_sessions(limit=0) == []
    assert invalid.search_sessions(limit=1, offset=-1) == []
    assert invalid_store.connection.queries == []


def test_optimize_fts_reindexes_only_ready_writable_stores() -> None:
    db, store = _db(rows=[])
    assert db.optimize_fts() == len(SEARCH_REINDEX_STATEMENTS)
    assert [query for query, _ in store.connection.queries] == list(
        SEARCH_REINDEX_STATEMENTS
    )
    assert store.transaction_modes == [False]

    read_only, read_only_store = _db(rows=[], read_only=True)
    assert read_only.optimize_fts() == 0
    assert read_only_store.connection.queries == []

    unavailable, unavailable_store = _db(rows=[], capabilities={})
    assert unavailable.optimize_fts() == 0
    assert unavailable_store.connection.queries == []


def test_rebuild_fts_uses_postgres_reindex_contract() -> None:
    db, store = _db(rows=[])

    assert db.rebuild_fts() == len(SEARCH_REINDEX_STATEMENTS)
    assert [query for query, _ in store.connection.queries] == list(
        SEARCH_REINDEX_STATEMENTS
    )


def test_search_source_has_no_cursor_escape_or_sqlite_translation() -> None:
    source = inspect.getsource(PostgresSearchOperations)

    assert ".fetchall(" not in source
    assert "sqlite" not in source.lower()
    assert "re.sub(" not in source
    assert "re.compile(" not in source
