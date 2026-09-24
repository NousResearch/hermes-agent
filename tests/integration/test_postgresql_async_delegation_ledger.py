"""Live PG18 conformance for the async-delegation ledger adapter and its runtime seam.

Mirrors the delivery-ledger conformance suites: direct adapter contract tests
against a disposable owned PG schema, plus the runtime selector (``selected_async_delegation_ledger``)
exercised through the ``tools.async_delegation`` seam with a no-``state.db``-open guard.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

import tools.async_delegation as ad
from tools.async_delegation_ledger_adapter import (
    SqliteAsyncDelegationLedger,
    selected_async_delegation_ledger,
)
from tools.async_delegation_ledger_postgresql import (
    AsyncDelegationLedgerPostgreSQLConfig,
    PostgreSQLAsyncDelegationLedger,
)
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_DSN_ENV = "OWNED_ASYNC_DSN"


@pytest.fixture
def ledger(postgresql_async_delegation_target: OwnedPostgreSQLTestTarget):
    value = PostgreSQLAsyncDelegationLedger(
        _DSN, schema=postgresql_async_delegation_target.schema,
        settings=AsyncDelegationLedgerPostgreSQLConfig(connect_timeout_seconds=3))
    try:
        yield value, postgresql_async_delegation_target
    finally:
        value.close()


def _record(delegation_id: str = "d1", **extra) -> dict:
    record = {"delegation_id": delegation_id, "session_key": "s",
              "origin_ui_session_id": "", "parent_session_id": None,
              "dispatched_at": time.time(), "goal": "g", "origin_session_id": ""}
    record.update(extra)
    return record


def _completion(delegation_id: str):
    return ({"type": "async_delegation", "delegation_id": delegation_id,
             "status": "completed", "completed_at": time.time()},
            {"status": "completed", "summary": "ok"})


# ── Adapter contract ────────────────────────────────────────────────────────
def test_fresh_catalog_is_dedicated_and_validated(ledger):
    ledger, target = ledger
    assert ledger.debug_rows() == []
    with target.connect() as con, con.cursor() as cur:
        cur.execute(f"SELECT version FROM {target.schema}.async_delegation_schema_migrations")
        assert cur.fetchall() == [(1,)]
    target.execute(f"DROP INDEX {target.schema}.idx_async_delegations_delivery")
    with pytest.raises(Exception, match="schema drift"):
        PostgreSQLAsyncDelegationLedger(_DSN, schema=target.schema)


def test_dispatch_completion_claim_ack_roundtrip(ledger):
    ledger, _target = ledger
    ledger.persist_dispatch(_record("d1"))
    assert ledger.get_durable_delegation("d1")["state"] == "running"
    ledger.persist_completion(*_completion("d1"))
    row = ledger.get_durable_delegation("d1")
    assert row["state"] == "completed" and row["delivery_state"] == "pending"

    assert ledger.claim_completion_delivery("d1", "claim-1") is True
    assert ledger.claim_completion_delivery("d1", "claim-2") is False  # held by claim-1
    assert ledger.complete_completion_delivery("d1", "claim-1") is True
    assert ledger.get_durable_delegation("d1")["delivery_state"] == "delivered"
    # mark_delivered is idempotent-only once delivered
    assert ledger.mark_completion_delivered("d1") is False


def test_concurrent_claim_only_one_wins(ledger):
    ledger, _target = ledger
    ledger.persist_dispatch(_record("d1"))
    ledger.persist_completion(*_completion("d1"))
    barrier = Barrier(2)

    def claim():
        barrier.wait()
        return ledger.claim_completion_delivery("d1", f"claim-{threading.get_ident()}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(), range(2)))
    assert sum(results) == 1


def test_release_caps_attempts_to_dropped(ledger):
    from tools.async_delegation import _MAX_DELIVERY_ATTEMPTS

    ledger, target = ledger
    ledger.persist_dispatch(_record("d1"))
    ledger.persist_completion(*_completion("d1"))
    assert ledger.claim_completion_delivery("d1", "c1") is True
    target.execute(
        f"UPDATE {target.schema}.async_delegations SET delivery_attempts=%s "
        "WHERE delegation_id='d1'", (_MAX_DELIVERY_ATTEMPTS,))
    assert ledger.release_completion_delivery("d1", "c1") is True  # capped -> dropped
    assert ledger.get_durable_delegation("d1")["delivery_state"] == "dropped"


def test_record_unit_child_partial_and_recover(ledger):
    ledger, target = ledger
    ledger.persist_dispatch(_record("d2", is_batch=True, goals=["a", "b"]))
    ledger.record_unit_child("d2", {"task_index": 0, "status": "completed"})
    result = ledger.get_durable_delegation("d2")["result"]
    assert result == {"results": [{"task_index": 0, "status": "completed"}], "partial": True}

    # Dead owner -> recovered to unknown with partial results replayed.
    target.execute(
        f"UPDATE {target.schema}.async_delegations SET owner_pid=NULL WHERE delegation_id='d2'")
    assert ledger.recover_abandoned_delegations() == 1
    recovered = ledger.get_durable_delegation("d2")
    assert recovered["state"] == "unknown"
    assert recovered["result"]["last_known_status"] == "running"
    assert recovered["result"]["task_transcripts"] == {}
    assert recovered["result"]["results"] == [{"task_index": 0, "status": "completed"}, {
        "task_index": 1, "status": "unknown", "summary": None,
        "error": "Delegation owner exited before recording a terminal result; outcome unknown.",
    }]


def test_restore_undelivered_replays_pending(ledger):
    ledger, _target = ledger
    ledger.persist_dispatch(_record("d1"))
    ledger.persist_completion(*_completion("d1"))

    class Queue:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    q = Queue()
    assert ledger.restore_undelivered_completions(q) == 1
    assert q.items[0]["restored"] is True
    # After replay the row stays pending (delivery is claimed separately).
    assert ledger.get_durable_delegation("d1")["delivery_state"] == "pending"


def test_orphan_sweep_reoffers_pending_once(ledger):
    from tools.async_delegation import _ORPHAN_STALE_S

    ledger, target = ledger
    delegation_id = f"orphan-{time.time_ns()}"
    ledger.persist_dispatch(_record(delegation_id))
    ledger.persist_completion(*_completion(delegation_id))
    now = time.time()
    target.execute(
        f"UPDATE {target.schema}.async_delegations "
        "SET owner_pid=NULL, updated_at=%s WHERE delegation_id=%s",
        (now - _ORPHAN_STALE_S - 1, delegation_id),
    )
    q = queue.Queue()
    assert ledger.sweep_orphaned_completions(q, now=now) == 1
    assert q.get_nowait()["delegation_id"] == delegation_id
    assert ledger.sweep_orphaned_completions(q, now=now + _ORPHAN_STALE_S) == 0


def test_transaction_rollback_preserves_no_partial_dispatch(ledger, monkeypatch):
    ledger, _target = ledger
    original = ledger._validate
    monkeypatch.setattr(ledger, "_validate",
                        lambda cursor: (_ for _ in ()).throw(RuntimeError("inject")))
    with pytest.raises(RuntimeError):
        ledger._migrate(ledger._connect())
    monkeypatch.setattr(ledger, "_validate", original)
    assert ledger.debug_rows() == []


# ── Runtime seam (selector) ─────────────────────────────────────────────────
def _pg_home(tmp_path, monkeypatch, *, dsn_value: str):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n"
        "    dsn_env: OWNED_ASYNC_DSN\n    connect_timeout_seconds: 1\n    pool_max_size: 2\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(_DSN_ENV, dsn_value)
    from tools.async_delegation_ledger_adapter import _SELECTED_LEDGER_CACHE
    _SELECTED_LEDGER_CACHE.clear()
    return home


@pytest.mark.integration
def test_selected_pg_routes_async_delegation_seam_without_state_db(
    postgresql_async_delegation_target, tmp_path, monkeypatch,
):
    from state_store_runtime_readiness import trap_state_db_opens

    home = _pg_home(tmp_path, monkeypatch, dsn_value=postgresql_async_delegation_target.dsn)
    ledger = selected_async_delegation_ledger()
    assert isinstance(ledger, PostgreSQLAsyncDelegationLedger)
    try:
        with trap_state_db_opens(home) as events:
            ad._persist_dispatch(_record("seam-1"))
            ad._persist_completion(*_completion("seam-1"))
            assert ad.claim_completion_delivery("seam-1", "c1") is True
            assert ad.complete_completion_delivery("seam-1", "c1") is True
            assert ad.get_durable_delegation("seam-1")["delivery_state"] == "delivered"
            orphan_id = f"seam-orphan-{time.time_ns()}"
            ad._persist_dispatch(_record(orphan_id))
            ad._persist_completion(*_completion(orphan_id))
            now = time.time()
            with ledger._transaction() as cur:
                cur.execute(
                    "UPDATE async_delegations SET owner_pid=NULL, updated_at=%s "
                    "WHERE delegation_id=%s",
                    (now - ad._ORPHAN_STALE_S - 1, orphan_id),
                )
            q = queue.Queue()
            assert ad.sweep_orphaned_completions(q, now=now) == 1
            assert q.get_nowait()["delegation_id"] == orphan_id
        assert events == []
        assert not (home / "state.db").exists()
    finally:
        ledger.close()


@pytest.mark.integration
def test_selected_pg_unreachable_raises_without_state_db(tmp_path, monkeypatch):
    import psycopg
    from state_store import StateStoreConfigurationError
    from state_store_runtime_readiness import trap_state_db_opens

    home = _pg_home(tmp_path, monkeypatch, dsn_value="postgresql://fixture/only")
    with trap_state_db_opens(home) as events:
        with pytest.raises((StateStoreConfigurationError, psycopg.Error, ValueError)):
            selected_async_delegation_ledger()
    assert events == []
    assert not (home / "state.db").exists()


@pytest.mark.integration
def test_sqlite_home_selected_returns_none_and_sqlite_smoke(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("state_store:\n  backend: sqlite\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    from tools.async_delegation_ledger_adapter import _SELECTED_LEDGER_CACHE
    _SELECTED_LEDGER_CACHE.clear()

    assert selected_async_delegation_ledger() is None

    wrapper = SqliteAsyncDelegationLedger()
    wrapper.persist_dispatch(_record("sqlite-smoke"))
    wrapper.persist_completion(*_completion("sqlite-smoke"))
    assert wrapper.claim_completion_delivery("sqlite-smoke", "c1") is True
    assert wrapper.complete_completion_delivery("sqlite-smoke", "c1") is True
    assert wrapper.get_durable_delegation("sqlite-smoke")["delivery_state"] == "delivered"
    assert (home / "state.db").exists()
