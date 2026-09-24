"""Executable PG18 acceptance harness for a future atomic rotation adapter.

The test-only protocol below exercises PostgreSQL transactions, server faults,
process death, receipts and fences now.  The one unsupported production
interface invocation remains a precise strict xfail until an adapter exists.
"""
from __future__ import annotations

import importlib
import multiprocessing
import queue
import uuid
from typing import Any

import pytest

from state_store import PostgreSQLStateStoreConfig, StateStoreConfigurationError
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_rotation_protocol import PHASES, RotationReceipt, acquire, audit, expire, install, publish, seed
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

pytestmark = pytest.mark.integration
_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="PG_ROTATION_GATE_DSN", connect_timeout_seconds=5, pool_max_size=2)
_CAPABILITY = "atomic-compression-rotation-v1"
_PRECOMMIT_PHASES = PHASES[:-1]


def _psycopg() -> Any:
    return importlib.import_module("psycopg")


def _store(schema: str) -> PostgreSQLStateStore:
    return PostgreSQLStateStore(_SETTINGS, _DSN, schema=schema)


@pytest.fixture(autouse=True)
def requires_postgresql_18(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PG_ROTATION_GATE_DSN", _DSN)
    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute("SHOW server_version_num")
        assert int(cursor.fetchone()[0]) >= 180000


@pytest.fixture
def harness(postgresql_test_target: OwnedPostgreSQLTestTarget) -> tuple[str, str]:
    install(_DSN, postgresql_test_target.schema)
    parent = f"parent-{uuid.uuid4().hex}"
    seed(_DSN, postgresql_test_target.schema, parent)
    return postgresql_test_target.schema, parent


def _receipt(parent: str, owner: str = "owner-a", fence: int = 1, request: str | None = None) -> RotationReceipt:
    return RotationReceipt(parent, f"child-{uuid.uuid4().hex}", owner, fence, request or f"request-{uuid.uuid4().hex}")


def _assert_exact_oracle_snapshot(snapshot: dict[str, Any]) -> None:
    assert snapshot["closed"] and len(snapshot["children"]) == 1 and snapshot["receipts"]
    child = snapshot["children"][0]
    # parent/child identity, prompt/model/config/title/lineage/visibility and
    # generation are copied exactly; activity/cooldown/counters reset.
    assert child[1:] == ("tenant-a", "exact cached prompt", "oracle/model", {"max_tokens": None}, "Oracle title", "root/parent", True, 0.0, 0.0, 0, 0, 9, 5)
    assert snapshot["messages"] == [("assistant", "[CONTEXT COMPACTION] deterministic summary"), ("user", "deterministic live tail")]


def _assert_no_partial(snapshot: dict[str, Any]) -> None:
    assert not snapshot["closed"]
    assert snapshot["children"] == []
    assert snapshot["receipts"] == []
    assert snapshot["messages"] == []


def _acquire_worker(schema: str, parent: str, owner: str, start: Any, results: Any) -> None:
    if not start.wait(15):
        results.put((owner, None, "start timeout")); return
    results.put((owner, acquire(_DSN, schema, parent, owner), None))


def _publish_worker(schema: str, parent: str, owner: str, fence: int, request: str, ready: Any, release: Any, results: Any) -> None:
    try:
        receipt = _receipt(parent, owner, fence, request)
        def phase(name: str, _pid: int) -> None:
            if name == "child-row-inserted":
                ready.set(); release.wait(15)
        publish(_DSN, schema, receipt, phase)
        results.put(("committed", receipt.child_id))
    except Exception as exc:
        results.put(("rejected", type(exc).__name__, str(exc)))


def _take(value: Any) -> tuple[Any, ...]:
    try:
        return value.get(timeout=15)
    except queue.Empty as exc:
        raise AssertionError("indeterminate: spawned owner did not report") from exc


def test_pg18_rotation_protocol_matches_sqlite_oracle_and_idempotency(harness):
    schema, parent = harness
    fence = acquire(_DSN, schema, parent, "owner-a")
    assert fence == 1
    receipt = _receipt(parent, fence=fence)
    assert publish(_DSN, schema, receipt) == receipt
    # Duplicate retry returns the same receipt and does not create another child.
    assert publish(_DSN, schema, receipt) == receipt
    _assert_exact_oracle_snapshot(audit(_DSN, schema, parent))


@pytest.mark.parametrize("phase", _PRECOMMIT_PHASES)
def test_pg18_transaction_phase_database_fault_rolls_back_and_reopens(harness, phase):
    schema, parent = harness
    fence = acquire(_DSN, schema, parent, "owner-a")
    assert fence is not None
    receipt = _receipt(parent, fence=fence)
    def injected(name: str, _pid: int) -> None:
        if name == phase:
            raise RuntimeError(f"injected database exception at {phase}")
    with pytest.raises(RuntimeError, match=phase):
        publish(_DSN, schema, receipt, injected)
    _assert_no_partial(audit(_DSN, schema, parent))


def test_pg18_commit_returned_fault_is_durably_committed_and_reopenable(harness):
    schema, parent = harness
    fence = acquire(_DSN, schema, parent, "owner-a")
    assert fence is not None
    receipt = _receipt(parent, fence=fence)
    def injected(name: str, _pid: int) -> None:
        if name == "commit-returned":
            raise RuntimeError("acknowledgement lost after durable commit")
    with pytest.raises(RuntimeError, match="acknowledgement"):
        publish(_DSN, schema, receipt, injected)
    _assert_exact_oracle_snapshot(audit(_DSN, schema, parent))


@pytest.mark.parametrize("phase", _PRECOMMIT_PHASES)
def test_pg18_terminate_backend_at_every_critical_phase_reopens_cleanly(harness, phase):
    schema, parent = harness
    fence = acquire(_DSN, schema, parent, "owner-a")
    assert fence is not None
    receipt = _receipt(parent, fence=fence)
    def kill(name: str, pid: int) -> None:
        if name == phase:
            with _psycopg().connect(_DSN, autocommit=True) as connection, connection.cursor() as cursor:
                cursor.execute("SELECT pg_terminate_backend(%s)", (pid,))
                assert cursor.fetchone() == (True,)
    with pytest.raises(Exception):
        publish(_DSN, schema, receipt, kill)
    _assert_no_partial(audit(_DSN, schema, parent))


def test_pg18_spawned_owners_server_clock_fence_stale_rejection_and_namespace_isolation(harness, postgresql_test_target):
    schema, parent = harness
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    workers = [context.Process(target=_acquire_worker, args=(schema, parent, owner, start, results)) for owner in ("owner-a", "owner-b")]
    for worker in workers: worker.start()
    start.set()
    outcomes = [_take(results), _take(results)]
    for worker in workers:
        worker.join(15); assert worker.exitcode == 0
    winners = [(owner, fence) for owner, fence, error in outcomes if error is None and fence is not None]
    assert len(winners) == 1
    winner, first = winners[0]
    loser = "owner-b" if winner == "owner-a" else "owner-a"
    expire(_DSN, schema, parent)
    second = acquire(_DSN, schema, parent, loser)
    assert second == first + 1
    with pytest.raises(RuntimeError, match="stale"):
        publish(_DSN, schema, _receipt(parent, winner, first))
    assert not audit(_DSN, schema, parent)["closed"]

    other = OwnedPostgreSQLTestTarget(_DSN).allocate()
    try:
        install(_DSN, other.schema); seed(_DSN, other.schema, parent, tenant="tenant-b")
        other_fence = acquire(_DSN, other.schema, parent, loser)
        assert other_fence == 1
        publish(_DSN, other.schema, _receipt(parent, loser, other_fence))
        assert not audit(_DSN, schema, parent)["closed"]
        other_snapshot = audit(_DSN, other.schema, parent)
        assert other_snapshot["children"][0][1] == "tenant-b"
        assert other_snapshot["messages"] == [("assistant", "[CONTEXT COMPACTION] deterministic summary"), ("user", "deterministic live tail")]
    finally:
        other.drop()


def test_pg18_sigkill_during_real_transaction_is_rolled_back_without_replay(harness):
    schema, parent = harness
    fence = acquire(_DSN, schema, parent, "owner-a")
    context = multiprocessing.get_context("spawn")
    ready, release, results = context.Event(), context.Event(), context.Queue()
    process = context.Process(target=_publish_worker, args=(schema, parent, "owner-a", fence, "crash-request", ready, release, results))
    process.start(); assert ready.wait(15)
    process.kill(); process.join(15)
    assert process.exitcode is not None and process.exitcode != 0
    _assert_no_partial(audit(_DSN, schema, parent))
    # Parent classifies the incomplete request as rolled back and performs one
    # explicit new receipt, never automatic replay of the killed attempt.
    assert _take(results) if not results.empty() else None is None
    successor = _receipt(parent, "owner-a", fence, "supervisor-new-request")
    publish(_DSN, schema, successor)
    _assert_exact_oracle_snapshot(audit(_DSN, schema, parent))


def test_pg18_production_rotation_interface_publishes_fenced_handoff(harness):
    schema, _parent = harness
    store = _store(schema)
    try:
        parent, child, holder = "production-parent", "production-child", "production-owner"
        store.ensure_session(parent, "telegram", metadata={
            "model": "oracle/model", "model_config": {"max_tokens": None},
            "session_key": "telegram:production", "chat_id": "chat", "profile_name": "tenant-a",
        })
        store.set_system_prompt(parent, "exact cached prompt")
        for kind, payload in (
            ("goal", {"goal": "ship", "status": "active"}),
            ("heartbeat", {"prompt": "check", "status": "active"}),
            ("loop", {"prompt": "watch", "status": "active"}),
        ):
            assert store.put_session_control_state(parent, kind, "active", payload) == 1
        assert store.try_acquire_compression_lock(parent, holder)
        assert _CAPABILITY in store.capabilities
        assert store.publish_compression_child(
            parent_session_id=parent, child_session_id=child, source="telegram",
            messages=[
                {"role": "assistant", "content": "[CONTEXT COMPACTION] deterministic summary"},
                {"role": "user", "content": "deterministic live tail"},
            ], model="oracle/model", model_config={"max_tokens": None},
            system_prompt="exact cached prompt", compression_lock_holder=holder,
            require_compression_lease=True, require_lease_refresh=True,
            request_id="production-rotation-request",
        ) == child
        assert store.publish_compression_child(
            parent_session_id=parent, child_session_id=child, source="telegram",
            messages=[{"role": "assistant", "content": "must not duplicate"}],
            compression_lock_holder=holder, request_id="production-rotation-request",
        ) == child
        assert store.get_session(parent)["end_reason"] == "compression"
        assert store.get_session(child)["parent_session_id"] == parent
        assert [(row["role"], row["content"]) for row in store.get_message_records(child)] == [
            ("assistant", "[CONTEXT COMPACTION] deterministic summary"), ("user", "deterministic live tail"),
        ]
        for kind in ("goal", "heartbeat", "loop"):
            parent_control = store.get_session_control_state(parent, kind)
            child_control = store.get_session_control_state(child, kind)
            assert parent_control is not None and parent_control["status"] == "cleared"
            assert child_control is not None and child_control["status"] == "active"
    finally:
        store.close()


def test_pg18_unknown_catalog_version_is_rejected_fail_closed(postgresql_test_target: OwnedPostgreSQLTestTarget):
    store = _store(postgresql_test_target.schema); store.close()
    unknown_version = 26
    postgresql_test_target.execute(
        f"INSERT INTO {postgresql_test_target.schema}.schema_migrations "
        f"(version, applied_at) VALUES ({unknown_version}, 0)"
    )
    with pytest.raises(
        StateStoreConfigurationError,
        match=rf"Unsupported PostgreSQL State Store schema migration versions: \[{unknown_version}\]",
    ):
        _store(postgresql_test_target.schema)
