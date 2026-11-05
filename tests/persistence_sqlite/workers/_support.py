"""Shared helpers for the worker backend tests: a profile store seeded through the backend with one session, a live epoch and a registered execution; a frozen clock; the one-transaction and rollback checks of a verb; and a process killed inside a verb's receipt transaction."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from persistence import ExecutionGeneration, PersistenceError
from persistence.core.fences import AuthorityEpoch
from persistence.workers.executions import WorkerExecutionId
from tests.fixtures.storage_seed import seed_session
from tests.persistence_sqlite.authority._support import AUTHORITY_TABLES, counted_transactions, dump
from tests.persistence_sqlite.conftest import store_for

SESSION = "sess-worker"
EXECUTION = "exec-1"
SECRET = "adoption-secret-0123456789abcdef"
#: The clock :func:`frozen` pins.
NOW = 1_700_000_000.25
#: Every table a worker operation writes.
WORKER_TABLES = (*AUTHORITY_TABLES, "async_delegations", "compression_locks", "session_turn_leases", "system_prompts",
                 "session_model_usage", "conversation_generations")


class _UidMint:
    """Sequential stand-in for the random message and tool-call uid mint."""

    def __init__(self) -> None:
        self.next = 1

    def __call__(self) -> str:
        self.next += 1
        return f"{self.next - 1:032x}"


UIDS = _UidMint()


@pytest.fixture
def frozen(monkeypatch):
    """Pin ``time.time`` and the uid mint so the verbs stamp fixed values."""
    import time

    import agent.message_metadata
    from persistence.sqlite.conversation._store import messages, rules

    monkeypatch.setattr(time, "time", lambda: NOW)
    UIDS.next = 1
    for module in (agent.message_metadata, rules, messages):
        monkeypatch.setattr(module, "mint_uid", UIDS)
    return NOW


def seed_store(store: Any, *, source: str = "cli", extra: Callable[[Any, int], None] | None = None, **session: Any) -> int:
    """Seed the opened ``store`` through the backend: :data:`SESSION`, a live epoch, :data:`EXECUTION` registered on it; then ``extra(store, epoch)``. Returns the epoch."""
    seed_session(store, SESSION, source, **session)
    epoch = store.runtime.begin_epoch(instance_id="owner")
    store.worker_executions.register(epoch=epoch, execution_id=EXECUTION, session_id=SESSION, generation=0,
                                     kind="compute", adoption_secret=SECRET)
    if extra is not None:
        extra(store, epoch)
    return epoch


def ident(epoch: Any, sequence: Any = 1, *, generation: Any = 0, execution_id: Any = EXECUTION,
          session_id: Any = SESSION) -> dict[str, Any]:
    """The receipt identity of one worker write, as the verbs take it."""
    return {"epoch": AuthorityEpoch(epoch) if type(epoch) is int else epoch, "execution_id": execution_id,
            "session_id": session_id, "generation": ExecutionGeneration(generation) if type(generation) is int else generation,
            "sequence": sequence}


def worker_birth() -> float:
    import psutil

    from hermes_cli.process_identity import worker_birth as birth

    return birth(psutil.Process(os.getpid()))


def owner_started_at() -> Any:
    from gateway.status import get_process_start_time

    return get_process_start_time(os.getpid())


def one_transaction(store: Any, verb: Callable[[], Any], *, kind: str = "write") -> Any:
    """Run ``verb`` and assert it opened exactly one transaction of ``kind`` on the store's holder."""
    holder = store.holder()
    with counted_transactions(holder) as counts:
        result = verb()
    assert counts == {"write": 1 if kind == "write" else 0, "read": 1 if kind == "read" else 0}, counts
    return result


@contextmanager
def failing(store: Any, table: str, event: str = "INSERT") -> Iterator[None]:
    """Make the writer connection's next ``event`` on ``table`` fail inside the verb's transaction (a temporary
    trigger on the holder's writer connection)."""
    holder = store.holder()
    holder._execute_write(lambda conn: conn.execute(
        f"CREATE TEMP TRIGGER inject_failure BEFORE {event} ON main.{table} "
        "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"))
    try:
        yield
    finally:
        holder._execute_write(lambda conn: conn.execute("DROP TRIGGER IF EXISTS temp.inject_failure"))


def rolls_back(store: Any, path: Path, verb: Callable[[], Any], table: str = "worker_receipts", event: str = "INSERT") -> None:
    """A failure injected after the verb's first statement (at ``event`` on ``table``) leaves every table unchanged."""
    before = dump(path, WORKER_TABLES)
    with failing(store, table, event), pytest.raises(PersistenceError) as caught:
        verb()
    assert isinstance(caught.value.__cause__, sqlite3.IntegrityError), caught.value
    assert dump(path, WORKER_TABLES) == before


#: The receipted writes the crash cases drive, keyed by name.
CRASH_OPERATIONS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "fallback": lambda store, idn: store.worker_executions.set_compression_fallback_streak(**idn, streak=7),
    "append": lambda store, idn: store.worker_executions.append_transcript(
        **idn, messages=[{"role": "assistant", "content": "crash-window"}]),
    "usage": lambda store, idn: store.worker_executions.record_usage(**idn, input_tokens=11, api_call_count=1, model="model-x"),
}


def crash_child(path: str, scope_name: str, epoch: int, operation: str, signal: str) -> None:
    """Child process body: open the store, then run ``operation`` and stop dead inside its receipt transaction, at the
    receipt insert, after signalling the parent."""
    import threading

    from persistence.sqlite import SQLiteBackend
    from tests.contracts._scopes import profile_scope

    scope = profile_scope(scope_name)
    store = SQLiteBackend({scope.hermes_home_key: Path(path)}).open(scope)
    holder = store.holder()
    holder._execute_write(lambda conn: None)

    def trace(statement: str) -> None:
        if "INSERT INTO worker_receipts" in statement:
            Path(signal).write_text("in-transaction")
            threading.Event().wait(600)  # health: allow HX013 -- held open until the parent kills this process

    holder._conn.set_trace_callback(trace)
    CRASH_OPERATIONS[operation](store, ident(epoch))


def kill_inside_receipt(path: Path, scope: Any, epoch: int, operation: str) -> None:
    """Run :func:`crash_child` in a spawned process and kill it while it holds the receipt transaction open."""
    import multiprocessing
    import time

    signal = path.with_name(f"crash-signal-{operation}")
    child = multiprocessing.get_context("spawn").Process(target=crash_child, args=(str(path), scope.name, epoch, operation, str(signal)))
    child.start()
    try:
        deadline = time.monotonic() + 120
        while not signal.exists():
            assert child.is_alive(), "the child exited before the receipt insert"
            assert time.monotonic() < deadline, "the child never reached the receipt insert"
            time.sleep(0.05)
    finally:
        child.kill()
        child.join(timeout=30)
    assert child.exitcode is not None and child.exitcode < 0


def store_path(scope: Any) -> Path:
    return store_for(scope)


def execution(store: Any) -> Any:
    """The execution row through ``verify_claim``."""
    return store.worker_executions.verify_claim(execution_id=WorkerExecutionId(EXECUTION), session_id=SESSION,
                                                generation=ExecutionGeneration(0), adoption_secret=SECRET)


def revision(path: Path, session_id: str = SESSION) -> int:
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT runtime_revision FROM sessions WHERE id=?", (session_id,)).fetchone()[0]
