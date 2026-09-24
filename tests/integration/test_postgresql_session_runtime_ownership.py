"""PG18 direct evidence for fenced SessionRuntimeOwnership; intentionally no runtime route."""
from __future__ import annotations


from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import Any

import pytest

from hermes_state_runtime_ownership import RuntimeOwner
from state_store import open_state_store
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SCHEMA = ""
_TARGET: OwnedPostgreSQLTestTarget | None = None


def _config() -> dict[str, object]:
    return {"state_store": {"backend": "postgresql", "postgresql": {
        "dsn_env": "HERMES_STATE_STORE_TEST_DSN", "connect_timeout_seconds": 5, "pool_max_size": 2,
    }}}


def _owner(name: str) -> RuntimeOwner:
    return RuntimeOwner(f"installation-{name}", f"host-{name}", f"generation-{name}")


@pytest.fixture(autouse=True)
def pg18_schema(monkeypatch, postgresql_test_target):
    global _SCHEMA, _TARGET
    _SCHEMA, _TARGET = postgresql_test_target.schema, postgresql_test_target
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: _SCHEMA)
    yield
    _TARGET = None


def _target() -> OwnedPostgreSQLTestTarget:
    assert _TARGET is not None
    return _TARGET


def _store() -> Any:
    return open_state_store(_config())


def _expire(store: Any, namespace: str, session_id: str) -> None:
    _target().execute(
            f"UPDATE {_SCHEMA}.session_runtime_owners SET expires_at = EXTRACT(EPOCH FROM clock_timestamp()) - 1 "
            "WHERE namespace=%s AND session_id=%s", (namespace, session_id),
        )


def _turn_state(store: Any, namespace: str, session_id: str, turn_id: str) -> tuple[str, int, object]:
    with store._connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT state, owner_fence, receipt_json FROM session_runtime_turns "
                       "WHERE namespace=%s AND session_id=%s AND turn_id=%s", (namespace, session_id, turn_id))
        row = cursor.fetchone()
    assert row is not None
    return str(row[0]), int(row[1]), row[2]


def test_pg18_two_owner_contention_same_owner_restart_and_pool_reset():
    first, second = _store(), _store()
    try:
        barrier = Barrier(2)
        def claim(store, owner):
            barrier.wait()
            return store.acquire_session_runtime_ownership("contention", owner, ttl_seconds=30)
        with ThreadPoolExecutor(max_workers=2) as executor:
            receipts = list(executor.map(lambda pair: claim(*pair), ((first, _owner("a")), (second, _owner("b")))))
        winner = next(receipt for receipt in receipts if receipt is not None)
        assert sum(receipt is not None for receipt in receipts) == 1
        assert winner.fence == 1
        restarted = first.acquire_session_runtime_ownership("contention", winner.owner, ttl_seconds=30)
        assert restarted is not None and restarted.fence == winner.fence
        with first._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SET search_path TO public")
        with first._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SHOW search_path")
            assert cursor.fetchone()[0].split(",")[0].strip(' \"') == first._schema
    finally:
        first.close(); second.close()


def test_pg18_crash_takeover_indeterminate_stale_rejection_and_explicit_settlement():
    machine_a, machine_b = _store(), _store()
    try:
        first = machine_a.acquire_session_runtime_ownership("crash", _owner("a"), ttl_seconds=30, namespace="team")
        assert first is not None and machine_a.begin_session_runtime_turn(first, "turn-1")
        # Deliberately abandon A without release: closing the pool models process loss, not a retry.
        machine_a.close()
        _expire(machine_b, "team", "crash")
        second = machine_b.acquire_session_runtime_ownership("crash", _owner("b"), ttl_seconds=30, namespace="team")
        assert second is not None and second.fence == first.fence + 1
        assert _turn_state(machine_b, "team", "crash", "turn-1")[:2] == ("indeterminate", first.fence)
        assert machine_b.renew_session_runtime_ownership(first) is None
        assert not machine_b.release_session_runtime_ownership(first)
        assert not machine_b.begin_session_runtime_turn(first, "stale-turn")
        assert not machine_b.resolve_session_runtime_turn(first, "turn-1", state="indeterminate")
        with pytest.raises(ValueError, match="verified receipt"):
            machine_b.resolve_session_runtime_turn(second, "turn-1", state="settled")
        assert machine_b.resolve_session_runtime_turn(second, "turn-1", state="settled", receipt_data={"effect_id": "verified-1"})
        assert _turn_state(machine_b, "team", "crash", "turn-1")[0] == "settled"
        assert not machine_b.resolve_session_runtime_turn(second, "turn-1", state="settled", receipt_data={"effect_id": "conflict"})
        assert machine_b.begin_session_runtime_turn(second, "turn-2")
    finally:
        machine_b.close()


def test_pg18_namespace_isolation_release_fence_and_catalog_rollback():
    store = _store()
    try:
        alpha = store.acquire_session_runtime_ownership("same", _owner("a"), ttl_seconds=30, namespace="alpha")
        beta = store.acquire_session_runtime_ownership("same", _owner("b"), ttl_seconds=30, namespace="beta")
        assert alpha is not None and beta is not None and alpha.fence == beta.fence == 1
        assert store.release_session_runtime_ownership(alpha)
        successor = store.acquire_session_runtime_ownership("same", _owner("c"), ttl_seconds=30, namespace="alpha")
        assert successor is not None and successor.fence == 2
        assert store.acquire_session_runtime_ownership("same", _owner("d"), ttl_seconds=30, namespace="beta") is None
        # An injected failing transaction rolls back; the adapter's next server-time CAS remains usable.
        with pytest.raises(Exception):
            _target().execute(f"INSERT INTO {_SCHEMA}.session_runtime_turns (namespace, session_id, turn_id, state, owner_fence, created_at, updated_at) "
                              "VALUES ('alpha', 'same', 'bad', 'invalid', 2, 0, 0)")
        assert store.begin_session_runtime_turn(successor, "after-rollback")
        with store._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT version FROM schema_migrations ORDER BY version")
            # The selected-PG compression facade adds the v20 atomic
            # parent/child publication receipt migration.  Keep this direct
            # ownership test coupled to the current catalog (the v25 gateway
            # transcript migration is the latest here), not the v19 precursor
            # that introduced the ownership tables.
            assert [int(row[0]) for row in cursor.fetchall()][-1] == 25
    finally:
        store.close()


def test_pg18_runtime_catalog_drift_fails_closed():
    store = _store(); store.close()
    _target().execute(f"DROP INDEX {_SCHEMA}.session_runtime_owners_expires")
    with pytest.raises(Exception, match="missing index"):
        _store()


def test_pg18_capabilities_remain_process_local_and_factory_is_directly_usable():
    store = _store()
    try:
        assert not store.supports_session_runtime_handoff_capability("browser")
        assert not store.supports_session_runtime_handoff_capability("computer_use")
        assert not store.supports_session_runtime_handoff_capability("approval_wait")
        assert store.supports_session_runtime_handoff_capability("durable_transcript")
        with pytest.raises(ValueError, match="finite"):
            store.acquire_session_runtime_ownership("finite", _owner("a"), ttl_seconds=float("inf"))
    finally:
        store.close()
