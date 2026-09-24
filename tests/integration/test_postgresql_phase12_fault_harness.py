"""Phase 12: real PG18 fault injection and independent-process conformance.

This is direct-adapter evidence only.  It deliberately does not select PostgreSQL
for the CLI, gateway, cron, TUI, ACP, hosted, browser, or async-delegation paths.
"""
from __future__ import annotations

import importlib
import multiprocessing
import queue
import sqlite3
import threading

from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from gateway.delivery_ledger_postgresql import DeliveryLedgerPostgreSQLConfig, PostgreSQLDeliveryLedger
from hermes_state_common import SCHEMA_SQL
from hermes_state_runtime_ownership import RuntimeOwner
from postgresql_state_store_operations import PostgreSQLSandboxOperations, PostgreSQLSandboxOperationsError
from postgresql_state_store_sqlite_import import SQLitePostgreSQLImportError, SQLitePostgreSQLSandboxImporter
from state_store import MessageRecord, PostgreSQLStateStoreConfig
from state_store_postgresql import PostgreSQLStateStore
from tests.integration.postgresql_test_target import OwnedPostgreSQLTestTarget

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_SETTINGS = PostgreSQLStateStoreConfig(dsn_env="PHASE12_DSN", connect_timeout_seconds=5, pool_max_size=2)


def _psycopg():
    return importlib.import_module("psycopg")


def _state_store(schema: str, *, pool_size: int = 2) -> PostgreSQLStateStore:
    return PostgreSQLStateStore(
        PostgreSQLStateStoreConfig(dsn_env="PHASE12_DSN", connect_timeout_seconds=5, pool_max_size=pool_size),
        _DSN,
        schema=schema,
    )


def _owner(label: str) -> RuntimeOwner:
    return RuntimeOwner(f"phase12-installation-{label}", f"phase12-host-{label}", f"phase12-generation-{label}")



@pytest.fixture(autouse=True)
def requires_postgresql_18() -> None:
    """Reject a lookalike fixture: every Phase 12 result is PG18-specific."""
    with _psycopg().connect(_DSN) as connection, connection.cursor() as cursor:
        cursor.execute("SHOW server_version_num")
        assert int(cursor.fetchone()[0]) >= 180000


@pytest.fixture
def phase12_targets(
    postgresql_test_target: OwnedPostgreSQLTestTarget,
    postgresql_delivery_target: OwnedPostgreSQLTestTarget,
):
    return postgresql_test_target, postgresql_delivery_target


def _ownership_worker(schema: str, label: str, start: Any, results: Any) -> None:
    store = _state_store(schema)
    try:
        results.put(("ready", label))
        if not start.wait(15):
            results.put(("rejected", label, "start timeout"))
            return
        receipt = store.acquire_session_runtime_ownership("phase12-session", _owner(label), ttl_seconds=60, namespace="phase12")
        results.put(("committed" if receipt else "rejected", label, receipt))
    finally:
        store.close()


def _crash_turn_worker(schema: str, results: Any) -> None:
    store = _state_store(schema)
    receipt = store.acquire_session_runtime_ownership("crash-session", _owner("crashed"), ttl_seconds=60, namespace="phase12")
    assert receipt is not None
    assert store.begin_session_runtime_turn(receipt, "turn-killed")
    results.put(("committed", receipt))
    # Parent sends SIGKILL after the durable receipt has crossed the process boundary.
    multiprocessing.Event().wait()


def _delivery_claim_worker(schema: str, results: Any) -> None:
    ledger = PostgreSQLDeliveryLedger(
        _DSN,
        schema=schema,
        settings=DeliveryLedgerPostgreSQLConfig(lease_seconds=60),
        owner_identity=("phase12-child", "phase12-host", "phase12-child-generation"),
    )
    try:
        initial = ledger.record_obligation(
            obligation_id="phase12-obligation", session_key="session", platform="test", chat_id="chat", thread_id=None, content="payload",
        )
        receipt = ledger.mark_attempting(initial)
        results.put(("committed" if receipt else "rejected", receipt))
        multiprocessing.Event().wait()
    finally:
        ledger.close()


def _search_lock_worker(schema: str, locked: Any, release: Any) -> None:
    connection = _psycopg().connect(_DSN, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))", (f"{schema}:search-index-maintenance",))
        locked.set()
        release.wait(15)
    finally:
        connection.close()


def _namespace_writer(schema: str, label: str, start: Any, results: Any) -> None:
    store = _state_store(schema)
    try:
        results.put(("ready", label, store._schema))
        if not start.wait(15):
            results.put(("rejected", label, "start timeout"))
            return
        store.ensure_session("same-session", source="phase12", metadata={"profile_name": label})
        store.append_message("same-session", role="user", content=f"payload-{label}")
        results.put(("committed", label, store._schema, store.get_messages("same-session")[0]["content"]))
    finally:
        store.close()


def _take(queue_value: Any, *, timeout: float = 15) -> tuple[Any, ...]:
    try:
        return queue_value.get(timeout=timeout)
    except queue.Empty as exc:
        raise AssertionError("indeterminate: child did not report a durable outcome") from exc


def test_pg18_process_ownership_contention_sigkill_takeover_and_no_turn_replay(phase12_targets):
    state_target, _delivery_target = phase12_targets
    schema = state_target.schema
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    processes = [context.Process(target=_ownership_worker, args=(schema, label, start, results)) for label in ("a", "b")]
    for process in processes:
        process.start()
    assert {_take(results)[:2], _take(results)[:2]} == {("ready", "a"), ("ready", "b")}
    start.set()
    outcomes = [_take(results), _take(results)]
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    assert sum(item[0] == "committed" for item in outcomes) == 1
    assert sum(item[0] == "rejected" for item in outcomes) == 1

    crashed_results = context.Queue()
    crashed = context.Process(target=_crash_turn_worker, args=(schema, crashed_results))
    crashed.start()
    outcome, first = _take(crashed_results)
    assert outcome == "committed" and first.fence == 1
    crashed.kill()
    crashed.join(15)
    assert crashed.exitcode is not None and crashed.exitcode != 0

    supervisor = _state_store(schema)
    try:
        state_target.execute(
            f"UPDATE \"{schema}\".session_runtime_owners SET expires_at=EXTRACT(EPOCH FROM clock_timestamp())-1 WHERE namespace=%s AND session_id=%s",
            ("phase12", "crash-session"),
        )
        successor = supervisor.acquire_session_runtime_ownership("crash-session", _owner("successor"), ttl_seconds=60, namespace="phase12")
        assert successor is not None and successor.fence == first.fence + 1
        with supervisor._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT state, owner_fence FROM session_runtime_turns WHERE namespace=%s AND session_id=%s AND turn_id=%s", ("phase12", "crash-session", "turn-killed"))
            assert cursor.fetchone() == ("indeterminate", first.fence)
        assert supervisor.renew_session_runtime_ownership(first) is None
        assert not supervisor.resolve_session_runtime_turn(first, "turn-killed", state="settled", receipt_data={"effect": "stale"})
        assert supervisor.resolve_session_runtime_turn(successor, "turn-killed", state="settled", receipt_data={"effect": "verified"})
        assert not supervisor.resolve_session_runtime_turn(successor, "turn-killed", state="settled", receipt_data={"effect": "replay"})
    finally:
        supervisor.close()


def test_pg18_server_fault_rolls_back_message_usage_and_pool_waiter_recovers(phase12_targets):
    state_target, _delivery_target = phase12_targets
    schema = state_target.schema
    store = _state_store(schema, pool_size=1)
    session_id = "atomic-session"
    try:
        store.ensure_session(session_id, source="phase12")
        state_target.execute(f"""CREATE FUNCTION \"{schema}\".phase12_message_fault() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN IF NEW.content='fault-message' THEN RAISE EXCEPTION 'phase12 message fault'; END IF; RETURN NEW; END $$""")
        state_target.execute(f"CREATE TRIGGER phase12_message_fault BEFORE INSERT ON \"{schema}\".messages FOR EACH ROW EXECUTE FUNCTION \"{schema}\".phase12_message_fault()")
        with pytest.raises(Exception, match="phase12 message fault"):
            store.append_message_records(session_id, [MessageRecord(role="user", content="before"), MessageRecord(role="assistant", content="fault-message")])
        assert store.get_messages(session_id) == []
        state_target.execute(f"DROP TRIGGER phase12_message_fault ON \"{schema}\".messages")
        state_target.execute(f"CREATE FUNCTION \"{schema}\".phase12_usage_fault() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'phase12 usage fault'; END $$")
        state_target.execute(f"CREATE TRIGGER phase12_usage_fault BEFORE INSERT ON \"{schema}\".session_model_usage FOR EACH ROW EXECUTE FUNCTION \"{schema}\".phase12_usage_fault()")
        with pytest.raises(Exception, match="phase12 usage fault"):
            store.update_token_counts(session_id, input_tokens=7, output_tokens=3, model="phase12", api_call_count=1)
        with store._connection() as connection, connection.cursor() as cursor:
            cursor.execute("SELECT input_tokens, output_tokens, api_call_count FROM sessions WHERE id=%s", (session_id,))
            assert cursor.fetchone() == (0, 0, 0)
        state_target.execute(f"DROP TRIGGER phase12_usage_fault ON \"{schema}\".session_model_usage")

        held, release, completed = threading.Event(), threading.Event(), threading.Event()
        def hold_connection() -> None:
            with store._connection():
                held.set()
                assert release.wait(15)
        def append_after_wait() -> None:
            assert held.wait(15)
            store.append_message(session_id, role="user", content="pool-recovered")
            completed.set()
        holder, waiter = threading.Thread(target=hold_connection), threading.Thread(target=append_after_wait)
        holder.start(); assert held.wait(15); waiter.start()
        assert not completed.wait(0.1)  # the only lease is held; no retry/fallback is possible.
        release.set(); holder.join(15); waiter.join(15)
        assert not holder.is_alive() and not waiter.is_alive() and completed.is_set()
        assert [row["content"] for row in store.get_messages(session_id)] == ["pool-recovered"]
    finally:
        store.close()


def test_pg18_delivery_receipt_fence_search_repair_lock_and_concurrent_namespaces(phase12_targets):
    state_target, delivery_target = phase12_targets
    schema, delivery_schema = state_target.schema, delivery_target.schema
    context = multiprocessing.get_context("spawn")
    delivery_results = context.Queue()
    delivery = context.Process(target=_delivery_claim_worker, args=(delivery_schema, delivery_results))
    delivery.start()
    outcome, stale_receipt = _take(delivery_results)
    assert outcome == "committed" and stale_receipt is not None
    delivery.kill(); delivery.join(15)
    assert delivery.exitcode is not None and delivery.exitcode != 0
    delivery_target.execute(f"UPDATE \"{delivery_schema}\".delivery_obligations SET lease_expires_at=EXTRACT(EPOCH FROM clock_timestamp())-1")
    successor_ledger = PostgreSQLDeliveryLedger(_DSN, schema=delivery_schema, settings=DeliveryLedgerPostgreSQLConfig(lease_seconds=60))
    try:
        assert not successor_ledger.mark_delivered(stale_receipt)
        reclaimed = successor_ledger.sweep_recoverable()
        assert len(reclaimed) == 1 and successor_ledger.mark_delivered(reclaimed[0]["receipt"])
    finally:
        successor_ledger.close()

    store = _state_store(schema)
    lock_ready, release_lock = context.Event(), context.Event()
    lock_process = context.Process(target=_search_lock_worker, args=(schema, lock_ready, release_lock))
    lock_process.start(); assert lock_ready.wait(15)
    try:
        blocked = store.rebuild_search_index()
        assert blocked["rebuild"]["operation"] == "already_running"
        assert store.search_index_status()["rebuild"]["in_progress"] is True
    finally:
        release_lock.set(); lock_process.join(15); store.close()
    assert lock_process.exitcode == 0

    targets = [OwnedPostgreSQLTestTarget(_DSN).allocate() for _ in range(3)]
    schemas = [target.schema for target in targets]
    start, results = context.Event(), context.Queue()
    workers = [context.Process(target=_namespace_writer, args=(candidate, label, start, results)) for candidate, label in zip(schemas, ("root", "alice", "bob"))]
    try:
        for worker in workers: worker.start()
        ready = [_take(results), _take(results), _take(results)]
        assert {item[1] for item in ready} == {"root", "alice", "bob"}
        start.set()
        committed = [_take(results), _take(results), _take(results)]
        assert {item[0] for item in committed} == {"committed"}
        assert {item[2] for item in committed} == set(schemas)
        for worker in workers:
            worker.join(15); assert worker.exitcode == 0
        for candidate, label in zip(schemas, ("root", "alice", "bob")):
            verifier = _state_store(candidate)
            try:
                assert [row["content"] for row in verifier.get_messages("same-session")] == [f"payload-{label}"]
            finally:
                verifier.close()
    finally:
        for worker in workers:
            if worker.is_alive(): worker.kill(); worker.join(15)
        for target in targets:
            target.drop()


def test_pg18_backup_failure_and_interrupted_import_are_isolated(tmp_path: Path, phase12_targets):
    state_target, _delivery_target = phase12_targets
    schema = state_target.schema
    store = _state_store(schema)
    source = tmp_path / "migration-only-source.db"
    with sqlite3.connect(source) as connection:
        connection.executescript(SCHEMA_SQL)
        connection.execute("INSERT INTO sessions (id, source, started_at) VALUES ('import-session', 'phase12', 1)")
        connection.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES ('import-session', 'user', 'import payload', 2)")
    importer = SQLitePostgreSQLSandboxImporter(
        _SETTINGS, _DSN, schema=schema, owned_target=state_target
    )
    try:
        with pytest.raises(SQLitePostgreSQLImportError, match="target remains isolated"):
            importer.import_source(source, snapshot_root=tmp_path, fail_after="messages")
        with state_target.connect() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT status, destination_counts FROM {schema}.sqlite_import_manifests")
            assert cursor.fetchone() == ("failed", None)
            cursor.execute(f"SELECT count(*) FROM {schema}.messages")
            assert cursor.fetchone()[0] == 0

        def interrupted_dump(arguments: list[str], **_kwargs: Any) -> CompletedProcess[str]:
            return CompletedProcess(arguments, 1, stdout="", stderr="simulated pg_dump interruption")
        operations = PostgreSQLSandboxOperations(_SETTINGS, _DSN, schema=schema, command_runner=interrupted_dump)
        with pytest.raises(PostgreSQLSandboxOperationsError, match="could not determine pg_dump version|simulated pg_dump interruption"):
            operations.backup(tmp_path / "backups", quiesced=True)
        assert not list((tmp_path / "backups").glob("*/manifest.json")) if (tmp_path / "backups").exists() else True
    finally:
        store.close()
