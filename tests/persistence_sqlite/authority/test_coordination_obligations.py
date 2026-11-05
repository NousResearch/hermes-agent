"""The SQLite delivery-obligation verbs' transaction shape: the sweeps take owners and deadlines as data and claim only
rows unchanged since the caller's read, opening no connection and probing no process inside the transaction; every
ledger verb reaches the writer with the ledger's 10 s patience; and the table step runs once per writer connection
outside any transaction.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable

import pytest

import gateway.delivery_ledger as dl
from persistence.authority.coordination import OwnerStamp, RecordedOwner, RetryDue
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.gateway import _delivery_ledger
from tests.contracts.authority.test_coordination import dead_pid
from tests.persistence_sqlite.authority.conftest import NOW
from tests.persistence_sqlite.authority.test_coordination_delegations import _forbid_probes_and_connections, dump
from tests.persistence_sqlite.conftest import store_for

pytestmark = pytest.mark.usefixtures("clock")

TABLES = ("delivery_obligations",)
LIVE = OwnerStamp(pid=os.getpid(), started_at=dl._start_time(os.getpid()))
DEAD_PID = dead_pid()
_COLUMNS = ("obligation_id", "session_key", "platform", "chat_id", "thread_id", "content", "state", "attempts",
            "created_at", "updated_at", "owner_pid", "owner_started_at", "last_error", "adapter_profile")


def ob(obligation_id: str, **values: Any) -> dict[str, Any]:
    """A pending row of a dead owner on telegram, unless ``values`` say otherwise."""
    base = {"obligation_id": obligation_id, "session_key": f"agent:main:telegram:dm:{obligation_id}",
            "platform": "telegram", "chat_id": "42", "thread_id": None, "content": f"reply {obligation_id}",
            "state": "pending", "attempts": 0, "created_at": NOW - 600, "updated_at": NOW - 600,
            "owner_pid": DEAD_PID, "owner_started_at": 11, "last_error": None, "adapter_profile": "default"}
    base.update(values)
    return base


def mine(obligation_id: str, **values: Any) -> dict[str, Any]:
    return ob(obligation_id, owner_pid=LIVE.pid, owner_started_at=LIVE.started_at, **values)


def seed_rows(path, rows: list[dict[str, Any]], *, table: bool = True) -> None:
    StoreHolder(path).close()
    if not table:
        return
    with sqlite3.connect(path) as conn:
        _delivery_ledger._ensure_obligations_table(conn)
        conn.executemany(f"INSERT INTO delivery_obligations ({', '.join(_COLUMNS)}) VALUES "
                         f"({', '.join('?' for _ in _COLUMNS)})", [tuple(r[c] for c in _COLUMNS) for r in rows])


def dead_owners(coord) -> frozenset[RecordedOwner]:
    """The caller's probe over the recorded owners, outside any transaction."""
    return frozenset(o for o in coord.list_obligation_owners() if not dl._owner_alive(o.pid, o.started_at))


def due_rows(coord, platform: str, profile=None) -> list[RetryDue]:
    """The runtime helper's deadlines over the candidate read, rows that are never due dropped."""
    due = []
    for c in coord.list_runtime_retry_candidates(platform, profile=profile, owner=LIVE):
        not_before = dl.retry_not_before(c.updated_at, c.last_error, c.attempts)
        if not_before is not None:
            due.append(RetryDue(obligation_id=c.obligation_id, updated_at=c.updated_at, last_error=c.last_error,
                                attempts=c.attempts, not_before=not_before))
    return due


# ── transaction shape ─────────────────────────────────────────────────────────

def _ready(backend, scope, rows):
    seed_rows(store_for(scope), rows)
    coord = backend.open(scope).coordination
    coord.list_obligation_owners()  # open the writer and a pooled reader before the forbidden window
    coord.mark_obligation_delivered(obligation_id="warm")
    return coord


def test_recoverable_sweep_takes_owners_as_data_and_skips_rows_changed_since_the_read(backend, scope_a, monkeypatch, clock):
    coord = _ready(backend, scope_a, [ob("a"), ob("b"), mine("live")])
    dead = dead_owners(coord)
    assert dead == {RecordedOwner(pid=DEAD_PID, started_at=11)}
    # A live gateway records ``b`` again after the probe: the row now names a live owner and is left alone.
    coord.record_delivery_obligation(obligation_id="b", session_key="k", platform="telegram", chat_id="42", thread_id=None,
                                     content="again", owner=LIVE)
    _forbid_probes_and_connections(monkeypatch)
    claimed = coord.sweep_recoverable_obligations(now=NOW, owner=LIVE, dead_owners=dead)
    monkeypatch.undo()
    assert [c.obligation_id for c in claimed] == ["a"]
    assert coord.list_owned_failed_obligations(owner=LIVE) == ()


def test_runtime_sweep_takes_deadlines_as_data_and_skips_rows_changed_since_the_read(backend, scope_a, monkeypatch, clock):
    coord = _ready(backend, scope_a, [mine("x", state="failed", last_error="send_path_degraded"),
                                      mine("y", state="failed", last_error="send_path_degraded")])
    due = due_rows(coord, "telegram")
    assert [d.obligation_id for d in due] == ["x", "y"]
    clock.now += 1
    coord.mark_obligation_failed(obligation_id="y", error="flood_control:30")  # rejected again after the read
    _forbid_probes_and_connections(monkeypatch)
    claimed = coord.sweep_failed_obligations_for_runtime("telegram", now=NOW + 1, owner=LIVE, due=due)
    monkeypatch.undo()
    assert [c.obligation_id for c in claimed] == ["x"]
    assert [(r.last_error, r.attempts) for r in coord.list_owned_failed_obligations(owner=LIVE)] == [("flood_control:30", 0)]


LEDGER_WRITES: dict[str, Callable[[Any], Any]] = {
    "record_delivery_obligation": lambda c: c.record_delivery_obligation(
        obligation_id="n", session_key="k", platform="telegram", chat_id="1", thread_id=None, content="x", owner=LIVE),
    "record_crash_left_reply": lambda c: c.record_crash_left_reply(
        obligation_id="n", session_key="k", platform="telegram", chat_id="1", thread_id=None, content="x", since=NOW),
    "mark_obligation_attempting": lambda c: c.mark_obligation_attempting(obligation_id="a"),
    "mark_obligation_delivered": lambda c: c.mark_obligation_delivered(obligation_id="a"),
    "mark_obligation_failed": lambda c: c.mark_obligation_failed(obligation_id="a", error="x"),
    "release_runtime_obligation_claim": lambda c: c.release_runtime_obligation_claim(obligation_id="a", owner=LIVE),
    "sweep_recoverable_obligations": lambda c: c.sweep_recoverable_obligations(owner=LIVE, dead_owners=frozenset()),
    "sweep_failed_obligations_for_runtime": lambda c: c.sweep_failed_obligations_for_runtime("telegram", owner=LIVE, due=[]),
}
LEDGER_READS: dict[str, Callable[[Any], Any]] = {
    "list_owned_failed_obligations": lambda c: c.list_owned_failed_obligations(owner=LIVE),
    "list_obligation_owners": lambda c: c.list_obligation_owners(),
    "list_runtime_retry_candidates": lambda c: c.list_runtime_retry_candidates("telegram", owner=LIVE),
}


def _record_patience(monkeypatch) -> tuple[list[Any], list[Any]]:
    writes: list[Any] = []
    reads: list[Any] = []
    execute_write, read_ctx = StoreHolder._execute_write, StoreHolder._read_ctx_with_timeout

    def spy_write(self, fn, patience_s=None):
        writes.append(patience_s)
        return execute_write(self, fn, patience_s=patience_s)

    def spy_read(self, busy_timeout_s):
        reads.append(busy_timeout_s)
        return read_ctx(self, busy_timeout_s)
    monkeypatch.setattr(StoreHolder, "_execute_write", spy_write)
    monkeypatch.setattr(StoreHolder, "_read_ctx_with_timeout", spy_read)
    return writes, reads


@pytest.mark.parametrize("verb", sorted(LEDGER_WRITES))
def test_ledger_writes_reach_the_writer_with_ten_second_patience(backend, scope_a, monkeypatch, verb):
    coord = _ready(backend, scope_a, [mine("a", state="attempting")])
    writes, _ = _record_patience(monkeypatch)
    LEDGER_WRITES[verb](coord)
    assert writes == [10.0]


@pytest.mark.parametrize("verb", sorted(LEDGER_READS))
def test_ledger_reads_wait_ten_seconds_for_the_lock(backend, scope_a, monkeypatch, verb):
    coord = _ready(backend, scope_a, [mine("a", state="failed")])
    writes, reads = _record_patience(monkeypatch)
    LEDGER_READS[verb](coord)
    assert (writes, reads) == ([], [10.0])


def test_a_read_on_a_store_without_the_table_creates_it_outside_any_write_then_reads(backend, scope_a, monkeypatch):
    seed_rows(store_for(scope_a), [], table=False)
    coord = backend.open(scope_a).coordination
    writes, reads = _record_patience(monkeypatch)
    assert coord.list_obligation_owners() == frozenset()
    assert (writes, reads) == ([], [10.0])
    assert "delivery_obligations" in dump(store_for(scope_a), TABLES)


def test_the_table_step_runs_once_per_writer_connection_outside_any_transaction(backend, scope_a, monkeypatch):
    """As main's gateway/delivery_ledger.py ran ``_initialize_schema`` on connect, before any transaction."""
    seed_rows(store_for(scope_a), [], table=False)
    step, runs = _delivery_ledger._ensure_obligations_table, []

    def spy(conn):
        runs.append(conn.in_transaction)
        step(conn)
    monkeypatch.setattr(_delivery_ledger, "_ensure_obligations_table", spy)
    coord = backend.open(scope_a).coordination
    coord.record_delivery_obligation(obligation_id="o", session_key="k", platform="telegram", chat_id="1", thread_id=None,
                                     content="x", owner=LIVE)
    coord.mark_obligation_failed(obligation_id="o", error="boom")
    assert [row.platform for row in coord.list_owned_failed_obligations(owner=LIVE)] == ["telegram"]
    assert backend.open(scope_a).coordination.list_obligation_owners() == {RecordedOwner(pid=LIVE.pid, started_at=LIVE.started_at)}
    assert runs == [False]


def test_a_new_writer_connection_runs_the_table_step_again(backend, scope_a):
    """A file restored in place without the table comes back on a new writer connection, which runs the step again,
    as each connection main's gateway/delivery_ledger.py opened did."""
    store = backend.open(scope_a)
    store.coordination.record_delivery_obligation(obligation_id="a", session_key="k", platform="telegram", chat_id="1",
                                                  thread_id=None, content="x", owner=LIVE)
    with sqlite3.connect(store_for(scope_a)) as conn:
        conn.execute("DROP TABLE delivery_obligations")
    conn.close()
    holder = store.holder()
    with holder._lock:
        holder._close_connection_quietly(holder._conn)
        holder._conn = None
    store.coordination.record_delivery_obligation(obligation_id="b", session_key="k", platform="telegram", chat_id="1",
                                                  thread_id=None, content="y", owner=LIVE)
    assert store.coordination.list_obligation_owners() == {RecordedOwner(pid=LIVE.pid, started_at=LIVE.started_at)}
