"""Live PG18 conformance for the dedicated PostgreSQL run-idempotency store."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from gateway.platforms.api_server_run_idempotency_postgresql import (
    PostgreSQLRunIdempotencyStore,
)
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

pytestmark = pytest.mark.integration

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"


@pytest.fixture
def store(postgresql_run_idempotency_target: OwnedPostgreSQLTestTarget):
    value = PostgreSQLRunIdempotencyStore(_DSN, schema=postgresql_run_idempotency_target.schema)
    try:
        yield value
    finally:
        value.close()


def test_fresh_catalog_is_dedicated_and_validated(store, postgresql_run_idempotency_target):
    with postgresql_run_idempotency_target.connect() as con, con.cursor() as cur:
        cur.execute(f"SELECT version FROM {postgresql_run_idempotency_target.schema}.run_idempotency_schema_migrations")
        assert cur.fetchall() == [(1,)]
    store.close()
    postgresql_run_idempotency_target.execute(
        f"DROP INDEX {postgresql_run_idempotency_target.schema}.run_idempotency_run_id")
    with pytest.raises(Exception, match="schema drift"):
        PostgreSQLRunIdempotencyStore(_DSN, schema=postgresql_run_idempotency_target.schema)


def test_reserve_created_then_reused_then_conflict(store):
    outcome, record = store.reserve("s", "k", "fp", "run-1", {"status": "running"})
    assert outcome == "created"
    assert record["run_id"] == "run-1"

    outcome, record = store.reserve("s", "k", "fp", "run-2", {"status": "running"})
    assert outcome == "reused"
    assert record["run_id"] == "run-1"

    outcome, _record = store.reserve("s", "k", "fp-other", "run-3", {"status": "running"})
    assert outcome == "conflict"


def test_lookup_missing_reused_conflict(store):
    assert store.lookup("s", "absent", "fp") == ("missing", None)

    store.reserve("s", "k", "fp", "run-1", {"status": "running"})

    outcome, record = store.lookup("s", "k", "fp")
    assert outcome == "reused"
    assert record["run_id"] == "run-1"

    outcome, _record = store.lookup("s", "k", "other-fp")
    assert outcome == "conflict"


def test_status_and_retention_round_trip(store):
    outcome, created = store.reserve(
        "s", "k", "fp", "run-1", {"status": "running"}, owner_pid=123, owner_started=456)
    assert outcome == "created"

    assert store.owns_run("s", "run-1") is True
    assert store.owns_run("s", "run-missing") is False

    status = store.status_for_run("s", "run-1")
    assert status == {
        "status": {"status": "running"},
        "owner_pid": 123,
        "owner_started": 456,
        "updated_at": created["updated_at"],
    }
    assert store.status_for_run("s", "run-missing") is None

    assert store.extend_retention("s", "run-1", 1000.0) is True
    assert store.extend_retention("s", "run-missing", 1000.0) is False

    store.update_status("run-1", {"status": "completed"})
    assert store.status_for_run("s", "run-1")["status"] == {"status": "completed"}
    assert store.owns_run("s", "run-1") is True


def test_concurrent_check_and_claim_exactly_one_created(store):
    workers = 8
    barrier = Barrier(workers)

    def reserve(i: int) -> str:
        barrier.wait()
        outcome, _record = store.reserve("s", "hot-key", "fp", f"run-hot-{i}", {"status": "running"})
        return outcome

    with ThreadPoolExecutor(max_workers=workers) as pool:
        outcomes = list(pool.map(reserve, range(workers)))

    assert outcomes.count("created") == 1
    assert outcomes.count("reused") == workers - 1
    assert "conflict" not in outcomes


def test_terminal_prune_removes_only_terminal_rows(store, postgresql_run_idempotency_target):
    store.reserve("s", "terminal-key", "fp-t", "run-terminal", {"status": "running"})
    store.update_status("run-terminal", {"status": "completed"})
    store.reserve("s", "live-key", "fp-l", "run-live", {"status": "running"})

    past = time.time() - 100.0
    for key in ("terminal-key", "live-key"):
        postgresql_run_idempotency_target.execute(
            f"UPDATE {postgresql_run_idempotency_target.schema}.run_idempotency "
            f"SET retention_until=%s WHERE scope=%s AND idempotency_key=%s",
            (past, "s", key))

    # Any subsequent reserve triggers the aged-terminal sweep.
    store.reserve("s", "new-key", "fp-n", "run-new", {"status": "running"})

    assert store.lookup("s", "terminal-key", "fp-t") == ("missing", None)
    assert store.lookup("s", "live-key", "fp-l")[0] == "reused"
    assert store.lookup("s", "new-key", "fp-n")[0] == "reused"
