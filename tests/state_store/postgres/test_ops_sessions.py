from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import inspect
from pathlib import Path
from typing import Any

import pytest

from hermes_state import SessionDB
from state_store.postgres.ops_sessions import PostgresSessionOperations
from state_store.postgres.session_db_base import PostgresSessionDBBase
from state_store.spec import StateStoreSpec


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
        raise AssertionError("PostgreSQL operations must use bounded reads")


class FakeConnection:
    def __init__(self, cursors: list[FakeCursor]) -> None:
        self._cursors = cursors
        self.executed: list[tuple[str, tuple[Any, ...]]] = []

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> FakeCursor:
        self.executed.append((query, params))
        if not self._cursors:
            raise AssertionError(f"unexpected query: {query}")
        return self._cursors.pop(0)


class FakeStore:
    def __init__(self, cursors: list[FakeCursor]) -> None:
        self.spec = StateStoreSpec(
            home=Path("/tmp/hermes"),
            profile="test",
            backend="postgres",
            sqlite_path=Path("/tmp/hermes/state.db"),
            postgres_dsn_env="HERMES_TEST_POSTGRES_DSN",
            postgres_schema="test_state",
            read_only=False,
        )
        self.connection = FakeConnection(cursors)
        self.transaction_modes: list[bool | None] = []
        self.commits = 0
        self.rollbacks = 0
        self.telegram_schema_calls = 0

    @contextmanager
    def transaction(self, *, read_only: bool | None = None) -> Iterator[FakeConnection]:
        self.transaction_modes.append(read_only)
        try:
            yield self.connection
        except BaseException:
            self.rollbacks += 1
            raise
        else:
            self.commits += 1

    def ensure_telegram_schema(self) -> None:
        self.telegram_schema_calls += 1

    def close(self) -> None:
        return None


class OpsDB(PostgresSessionOperations, PostgresSessionDBBase):
    pass


def _db(*cursors: FakeCursor) -> tuple[OpsDB, FakeStore]:
    store = FakeStore(list(cursors))
    return OpsDB(store, capabilities={"core_schema": True}), store


def _queries(store: FakeStore) -> str:
    return "\n".join(query for query, _ in store.connection.executed)


OWNED_PUBLIC_METHODS = (
    "create_session",
    "ensure_session",
    "get_session",
    "resolve_session_id",
    "resolve_session_by_title",
    "end_session",
    "reopen_session",
    "create_api_session_with_title",
    "fork_api_session",
    "record_gateway_session_peer",
    "set_expiry_finalized",
    "save_gateway_routing_entry",
    "replace_gateway_routing_entries",
    "load_gateway_routing_entries",
    "delete_gateway_routing_entries",
    "list_gateway_sessions",
    "find_session_by_origin",
    "find_latest_gateway_session_for_peer",
    "update_session_cwd",
    "backfill_repo_roots",
    "update_session_meta",
    "update_system_prompt",
    "update_session_model",
    "update_session_billing_route",
    "set_session_title",
    "get_session_title",
    "get_session_by_title",
    "get_next_title_in_lineage",
    "get_compression_tip",
    "set_session_archived",
    "distinct_session_cwds",
    "list_sessions_rich",
    "list_cron_job_runs",
    "session_count",
    "finalize_orphaned_compression_sessions",
    "record_compression_failure_cooldown",
    "get_compression_failure_cooldown",
    "clear_compression_failure_cooldown",
    "get_compression_fallback_streak",
    "set_compression_fallback_streak",
    "try_acquire_compression_lock",
    "refresh_compression_lock",
    "release_compression_lock",
    "get_compression_lock_holder",
    "update_token_counts",
    "record_auxiliary_usage",
    "get_insights_sessions",
    "get_insights_tool_name_counts",
    "get_insights_assistant_tool_calls_page",
    "get_insights_message_stats",
    "get_insights_model_usage",
    "persist_async_delegation",
    "delete_async_delegation",
    "prune_async_delegations",
    "complete_async_delegation",
    "note_async_delegation_delivery_attempt",
    "list_recoverable_async_delegations",
    "mark_async_delegation_unknown",
    "list_pending_async_delegation_events",
    "mark_async_delegation_delivered",
    "claim_async_delegation_delivery",
    "release_async_delegation_delivery",
    "complete_async_delegation_delivery",
    "get_async_delegation",
    "get_meta",
    "set_meta",
    "apply_telegram_topic_migration",
    "enable_telegram_topic_mode",
    "disable_telegram_topic_mode",
    "is_telegram_topic_mode_enabled",
    "get_telegram_topic_binding",
    "list_telegram_topic_bindings_for_chat",
    "get_telegram_topic_binding_by_session",
    "delete_telegram_topic_binding",
    "bind_telegram_topic",
    "is_telegram_session_linked_to_topic",
    "list_unlinked_telegram_sessions_for_user",
    "request_handoff",
    "get_handoff_state",
    "list_pending_handoffs",
    "claim_handoff",
    "complete_handoff",
    "fail_handoff",
)


def _parameter_contract(method: Any) -> tuple[tuple[str, Any, Any], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(method).parameters.values()
    )


def test_owned_public_signatures_match_sqlite_session_db() -> None:
    for name in OWNED_PUBLIC_METHODS:
        assert hasattr(SessionDB, name), name
        assert _parameter_contract(getattr(OpsDB, name)) == _parameter_contract(
            getattr(SessionDB, name)
        ), name


def test_operations_keep_cursors_bounded_and_private() -> None:
    source = inspect.getsource(PostgresSessionOperations)

    assert ".fetchall(" not in source
    assert "self._conn" not in source
    assert "self._run(" in source


def test_api_create_uses_one_locked_transaction_and_returns_existing_outcome() -> None:
    db, store = _db(FakeCursor(row={"id": "already-there"}))

    result = db.create_api_session_with_title(
        "already-there",
        model="model",
        system_prompt="prompt",
        title="unused",
    )

    assert result.outcome == "destination_exists"
    assert store.transaction_modes == [False]
    assert store.commits == 1
    assert "FOR UPDATE" in _queries(store)


def test_api_create_title_conflict_rolls_back_without_leaking_partial_session() -> None:
    db, store = _db(
        FakeCursor(),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(row={"id": "other"}),
        FakeCursor(),
    )

    result = db.create_api_session_with_title(
        "new",
        model="model",
        system_prompt="prompt",
        title="taken",
    )

    assert result.outcome == "invalid_title"
    assert result.error == "Title 'taken' is already in use by session other"
    assert store.rollbacks == 1
    assert store.commits == 0
    assert "INSERT INTO sessions" in _queries(store)


def test_api_fork_is_one_transaction_with_parent_end_copy_and_title_lock() -> None:
    db, store = _db(
        FakeCursor(row={"id": "parent", "model": "m", "system_prompt": "p", "title": "fork"}),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(row={"title_count": 1, "max_suffix": 1}),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(),
        FakeCursor(row={"id": "child", "title": "fork #2"}),
    )

    result = db.fork_api_session("parent", "child")

    assert result.outcome == "created"
    assert result.session == {"id": "child", "title": "fork #2"}
    assert store.transaction_modes == [False]
    assert store.commits == 1
    queries = _queries(store)
    assert "end_reason = 'branched'" in queries
    assert "INSERT INTO messages" in queries
    assert "FROM messages WHERE session_id = %s AND active = 1 ORDER BY id" in queries
    assert "WITH locked AS MATERIALIZED" in queries
    assert "FOR UPDATE" in queries
    assert "pg_advisory_xact_lock" in queries


@pytest.mark.parametrize(
    ("cursors", "expected"),
    (
        ((FakeCursor(),), "source_missing"),
        (
            (
                FakeCursor(row={"id": "parent"}),
                FakeCursor(row={"id": "child"}),
            ),
            "destination_exists",
        ),
    ),
)
def test_api_fork_preserves_sqlite_abort_outcomes(
    cursors: tuple[FakeCursor, ...], expected: str
) -> None:
    db, store = _db(*cursors)

    assert db.fork_api_session("parent", "child").outcome == expected
    assert store.transaction_modes == [False]
    assert store.commits == 1


def test_title_generation_locks_in_writable_transaction() -> None:
    db, store = _db(FakeCursor(), FakeCursor(row={"title_count": 2, "max_suffix": 4}))

    assert db.get_next_title_in_lineage("report #2") == "report #5"
    assert store.transaction_modes == [False]
    assert "FOR UPDATE" in _queries(store)
    assert "pg_advisory_xact_lock" in _queries(store)


def test_compression_lock_uses_server_time_atomic_conflict_claim() -> None:
    db, store = _db(FakeCursor(row={"holder": "first"}), FakeCursor())

    assert db.try_acquire_compression_lock("session", "first", ttl_seconds=30) is True
    assert db.try_acquire_compression_lock("session", "second", ttl_seconds=30) is False

    queries = _queries(store)
    assert "clock_timestamp()" in queries
    assert "ON CONFLICT(session_id) DO UPDATE" in queries
    assert "WHERE compression_locks.expires_at <" in queries
    assert "RETURNING holder" in queries


def test_async_delivery_claim_allows_one_winner_and_locks_row() -> None:
    db, store = _db(
        FakeCursor(row={"delivery_state": "pending"}),
        FakeCursor(row={"delegation_id": "d"}),
        FakeCursor(row={"delivery_state": "pending"}),
        FakeCursor(),
    )

    assert db.claim_async_delegation_delivery("d", "first", updated_at=100) is True
    assert db.claim_async_delegation_delivery("d", "second", updated_at=100) is False

    queries = _queries(store)
    assert "SELECT delivery_state FROM async_delegations" in queries
    assert "FOR UPDATE" in queries
    assert "delivery_claim IS NULL OR delivery_claimed_at < %s" in queries
    assert "RETURNING delegation_id" in queries


def test_async_pruning_never_targets_running_or_finalizing_records() -> None:
    db, store = _db(FakeCursor(), FakeCursor(), FakeCursor())

    db.prune_async_delegations(
        retention_seconds=60,
        max_retained_completed=10,
        max_pending=5,
        now=100,
    )

    queries = _queries(store)
    assert queries.count("state NOT IN ('running', 'finalizing')") == 2
    assert "delivery_state = 'delivered'" in queries


def test_insights_page_is_keyset_bounded_and_aggregates_in_database() -> None:
    page_cursor = FakeCursor(
        rows=[
            (11, "[]"),
            (12, "[]"),
            (13, "[]"),
        ],
        columns=("id", "tool_calls"),
    )
    db, store = _db(
        page_cursor,
        FakeCursor(
            row={
                "total_messages": 3,
                "user_messages": 1,
                "assistant_messages": 1,
                "tool_messages": 1,
            }
        ),
    )

    assert db.get_insights_assistant_tool_calls_page(
        1.0, after_message_id=10, limit=2
    ) == [{"id": 11, "tool_calls": "[]"}, {"id": 12, "tool_calls": "[]"}]
    assert db.get_insights_message_stats(1.0)["total_messages"] == 3

    queries = _queries(store)
    assert "m.id > %s" in queries
    assert "ORDER BY m.id ASC LIMIT %s" in queries
    assert "COUNT(*) FILTER" in queries
    assert page_cursor.fetchmany_sizes == [2]
    assert page_cursor._rows == [(13, "[]")]


def test_distinct_session_cwds_uses_bounded_postgres_aggregation() -> None:
    cursor = FakeCursor(
        rows=[
            {"cwd": "/repo", "sessions": "2", "last_active": "4.5"},
            {"cwd": "/other", "sessions": 1, "last_active": None},
        ]
    )
    db, store = _db(cursor, FakeCursor(rows=[]))

    assert db.distinct_session_cwds() == [
        {"cwd": "/repo", "sessions": 2, "last_active": 4.5},
        {"cwd": "/other", "sessions": 1, "last_active": 0.0},
    ]
    assert db.distinct_session_cwds(include_archived=True) == []

    first_query, first_params = store.connection.executed[0]
    second_query, _ = store.connection.executed[1]
    assert "COUNT(*)::bigint AS sessions" in first_query
    assert "MAX(COALESCE(ended_at, started_at, 0)) AS last_active" in first_query
    assert "TRIM(cwd) != ''" in first_query
    assert "archived = 0" in first_query
    assert "archived = 0" not in second_query
    assert first_params == (db._MAX_READ_ROWS,)
    assert cursor.fetchmany_sizes == [db._READ_BATCH_SIZE, db._READ_BATCH_SIZE]
    assert all(size <= db._READ_BATCH_SIZE for size in cursor.fetchmany_sizes)
    assert store.transaction_modes == [True, True]


def test_session_count_matches_sqlite_filters_with_native_jsonb_predicates() -> None:
    db, store = _db(FakeCursor(row={"count": "3"}))

    assert db.session_count(
        source="cli",
        cwd_prefix="/repo/",
        min_message_count=2,
        archived_only=True,
        exclude_children=True,
        exclude_sources=["cron"],
    ) == 3

    query, params = store.connection.executed[0]
    assert "COUNT(*)::bigint AS count" in query
    assert "COALESCE(s.model_config, '{}')::jsonb -> '_branched_from' IS NOT NULL" in query
    assert "COALESCE(s.model_config, '{}')::jsonb -> '_delegate_from' IS NULL" in query
    assert "NOT (s.source = ANY(%s))" in query
    assert "s.cwd = %s OR s.cwd LIKE %s OR s.cwd LIKE %s" in query
    assert "s.archived = 1" in query
    assert params == ("cli", ["cron"], "/repo", "/repo/%", "/repo\\%", 2)
    assert store.transaction_modes == [True]


def test_state_meta_telegram_lazy_schema_and_handoff_claim_are_atomic() -> None:
    db, store = _db(FakeCursor(row={"id": "session"}))

    db.apply_telegram_topic_migration()
    assert db.claim_handoff("session") is True

    assert store.telegram_schema_calls == 1
    assert "handoff_state = 'pending'" in _queries(store)
    assert "RETURNING id" in _queries(store)
