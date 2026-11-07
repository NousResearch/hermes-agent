"""The SQLite delivery-obligation verbs and this tree's gateway/delivery_ledger.py against main's ledger module, and
the verbs' transaction shape.

Each parity case writes the ledger rows into a schema-built ``state.db``, keeps a copy, runs main's ledger function
(``gateway/delivery_ledger.py`` of the pinned checkout, so the case skips without one) with ``HERMES_HOME`` at the
profile home (the ledger resolves its file there), restores the copy and runs this tree's ledger function the same way,
restores it again and runs the backend verb (composed with the owner probe and the retry deadlines the ledger
function computes, which stay with the caller), then compares the outcomes (claims projected to the legacy dicts) and
the ``delivery_obligations`` rows. The shape cases
prove the sweeps take owners and deadlines as data, claim only rows unchanged since the caller's read, open no
connection and probe no process inside the transaction, reach the writer with the ledger's 10 s patience, and run the
table step once per writer connection outside any transaction.
"""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from typing import Any, Callable

import pytest

import gateway.delivery_ledger as dl
from persistence.authority.coordination import OwnerStamp, RecordedOwner, RetryDue
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.gateway import _delivery_ledger
from tests.contracts.authority.test_coordination import dead_pid
from tests.persistence_sqlite.authority.test_coordination_delegations import _forbid_probes_and_connections
from tests.persistence_sqlite.authority.test_coordination_parity import (
    DAY, NOW, assert_same, copy_db, dump, outcome,
)
from tests.persistence_sqlite._legacy_store import oracle_tree
from tests.persistence_sqlite.conftest import store_for

pytestmark = pytest.mark.usefixtures("clock")

TABLES = ("delivery_obligations",)
LIVE = OwnerStamp(pid=os.getpid(), started_at=dl._start_time(os.getpid()))
DEAD_PID = dead_pid()
_MAIN_LEDGER = "_main_gateway_delivery_ledger"
_COLUMNS = ("obligation_id", "session_key", "platform", "chat_id", "thread_id", "content", "state", "attempts",
            "created_at", "updated_at", "owner_pid", "owner_started_at", "last_error", "adapter_profile")


def main_ledger():
    """main's ``gateway/delivery_ledger.py`` from the pinned checkout, loaded once beside this tree's module; skips the
    test without a checkout."""
    path = oracle_tree() / "gateway" / "delivery_ledger.py"
    module = sys.modules.get(_MAIN_LEDGER)
    if module is None:
        spec = importlib.util.spec_from_file_location(_MAIN_LEDGER, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MAIN_LEDGER] = module
        spec.loader.exec_module(module)
    return module


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


def run_ledger_both(backend, scope, monkeypatch, rows: list[dict[str, Any]], legacy: Callable[[Any], Any],
                    verb: Callable[[Any], Any], *, table: bool = True):
    """Seed, run ``legacy`` on main's ledger module on the profile home, restore the seed, run it on this tree's
    module, restore the seed, run the verb; assert equal rows and the same outcome from both ledger modules."""
    path = store_for(scope)
    seed_rows(path, rows, table=table)
    snapshot = path.with_name("seed.db")
    copy_db(path, snapshot)
    main = main_ledger()
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(path.parent))
        before = outcome(lambda: legacy(main))
    legacy_rows = dump(path, TABLES)
    copy_db(snapshot, path)
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(path.parent))
        assert_same(before, outcome(lambda: legacy(dl)))
    assert dump(path, TABLES) == legacy_rows
    copy_db(snapshot, path)
    after = outcome(lambda: verb(backend.open(scope).coordination))
    assert dump(path, TABLES) == legacy_rows
    return before, after


def as_legacy(claimed) -> list[dict[str, Any]]:
    """Claims in the legacy dict shape of main's gateway/delivery_ledger.py."""
    out = []
    for c in claimed:
        common = {"obligation_id": c.obligation_id, "session_key": c.session_key, "platform": c.platform,
                  "chat_id": c.chat_id, "thread_id": c.thread_id, "content": c.content, "profile": c.profile,
                  "attempts": c.attempts}
        if c.adopted:
            out.append({**common, "adopted": True, "not_before": c.not_before})
            continue
        out.append({**common, "needs_marker": c.needs_marker, **({"marker": c.marker} if c.marker else {}),
                    **({"runtime_recovery": True} if c.runtime_recovery else {}),
                    **({"last_error": c.last_error} if c.last_error else {})})
    return out


def pending_retries_from(rows) -> list[dict[str, Any]]:
    """The legacy reader's per-adapter earliest deadline (main's gateway/delivery_ledger.py) over owned failed rows."""
    earliest: dict[tuple, float] = {}
    for row in rows:
        if dl.is_reconnect_only(row.last_error):
            continue
        due = dl.retry_not_before(row.updated_at, row.last_error, row.attempts)
        if due is None or row.attempts >= dl.MAX_ATTEMPTS or (NOW - row.created_at) > dl.STALE_AFTER_SECONDS:
            continue
        key = (row.platform, row.adapter_profile or "default")
        if key not in earliest or due < earliest[key]:
            earliest[key] = due
    return [{"platform": platform, "profile": profile, "not_before": due}
            for (platform, profile), due in sorted(earliest.items())]


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


# ── writes ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("adapter_profile", [None, " work ", "default"])
def test_record_with_retention_prune(backend, scope_a, monkeypatch, adapter_profile):
    rows = ([ob(f"old-delivered-{i}", state="delivered", updated_at=NOW - 8 * DAY) for i in range(3)]
            + [ob(f"row-{i:03d}", state=("delivered", "abandoned", "failed")[i % 3], updated_at=NOW - 1000 + i)
               for i in range(500)])

    def verb(c):
        c.record_delivery_obligation(obligation_id="new", session_key="agent:main:telegram:dm:1", platform="telegram",
                                     chat_id=7, thread_id=99, content="hello", adapter_profile=adapter_profile, owner=LIVE)
    before, after = run_ledger_both(
        backend, scope_a, monkeypatch, rows,
        lambda m: m.record_obligation(obligation_id="new", session_key="agent:main:telegram:dm:1", platform="telegram",
                                     chat_id=7, thread_id=99, content="hello", adapter_profile=adapter_profile), verb)
    assert_same(before, after)


@pytest.mark.parametrize("since", [NOW - 700, NOW - 500])
def test_crash_left_reply(backend, scope_a, monkeypatch, since):
    rows = [ob("ledgered", session_key="k", content="same")]
    kwargs = dict(obligation_id="crash", session_key="k", platform="telegram", chat_id="42", thread_id="",
                  content="same", since=since, adapter_profile=None)
    before, after = run_ledger_both(backend, scope_a, monkeypatch, rows, lambda m: m.record_crash_left_reply(**kwargs),
                                    lambda c: c.record_crash_left_reply(**kwargs))
    assert_same(before, after)


def test_record_creates_the_table_and_adds_the_profile_column(backend, scope_a, monkeypatch):
    """A store without the ledger table gets it with the first write, as the legacy ledger connection created it."""
    kwargs = dict(obligation_id="o", session_key="k", platform="slack", chat_id="C1", thread_id=None, content="x")
    before, after = run_ledger_both(backend, scope_a, monkeypatch, [], lambda m: m.record_obligation(**kwargs),
                                    lambda c: c.record_delivery_obligation(**kwargs, owner=LIVE), table=False)
    assert_same(before, after)


def test_legacy_table_without_adapter_profile_gets_the_column(backend, scope_a, monkeypatch):
    path = store_for(scope_a)

    def seed_legacy_shape():
        StoreHolder(path).close()
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE delivery_obligations (obligation_id TEXT PRIMARY KEY, session_key TEXT NOT NULL, "
                         "platform TEXT NOT NULL, chat_id TEXT NOT NULL, thread_id TEXT, content TEXT NOT NULL, "
                         "state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL, "
                         "updated_at REAL NOT NULL, owner_pid INTEGER, owner_started_at INTEGER, last_error TEXT)")
            conn.execute("INSERT INTO delivery_obligations VALUES ('a', 'k', 'telegram', '1', NULL, 'x', 'failed', 1, "
                         "?, ?, ?, ?, 'boom')", (NOW - 60, NOW - 60, LIVE.pid, LIVE.started_at))
    seed_legacy_shape()
    snapshot = path.with_name("seed.db")
    copy_db(path, snapshot)
    main = main_ledger()
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(path.parent))
        before = outcome(lambda: main.pending_retries(now=NOW))
    legacy_rows = dump(path, TABLES)
    copy_db(snapshot, path)
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(path.parent))
        assert outcome(lambda: dl.pending_retries(now=NOW)) == before
    assert dump(path, TABLES) == legacy_rows
    copy_db(snapshot, path)
    after = outcome(lambda: backend.open(scope_a).coordination.list_owned_failed_obligations(owner=LIVE))
    assert dump(path, TABLES) == legacy_rows
    assert "adapter_profile" in legacy_rows["delivery_obligations.columns"]
    assert_same(before, after, pending_retries_from)
    assert before[1]


@pytest.mark.parametrize("action", ["attempting", "delivered", "failed"])
@pytest.mark.parametrize("obligation_id", ["a", "missing"])
def test_state_marks(backend, scope_a, monkeypatch, action, obligation_id):
    error = "e" * 700
    legacy = {"attempting": lambda m: m.mark_attempting(obligation_id),
              "delivered": lambda m: m.mark_delivered(obligation_id),
              "failed": lambda m: m.mark_failed(obligation_id, error)}[action]
    verb = {"attempting": lambda c: c.mark_obligation_attempting(obligation_id=obligation_id),
            "delivered": lambda c: c.mark_obligation_delivered(obligation_id=obligation_id),
            "failed": lambda c: c.mark_obligation_failed(obligation_id=obligation_id, error=error)}[action]
    before, after = run_ledger_both(backend, scope_a, monkeypatch, [ob("a", state="failed", last_error="old")], legacy, verb)
    assert_same(before, after)


@pytest.mark.parametrize("obligation_id", ["mine", "mine-zero", "mine-failed", "theirs", "missing"])
@pytest.mark.parametrize("error", ["", "send_path_degraded"])
def test_release_runtime_claim(backend, scope_a, monkeypatch, obligation_id, error):
    rows = [mine("mine", state="attempting", attempts=2), mine("mine-zero", state="attempting", attempts=0),
            mine("mine-failed", state="failed", attempts=1), ob("theirs", state="attempting", attempts=1)]
    before, after = run_ledger_both(
        backend, scope_a, monkeypatch, rows, lambda m: m.release_runtime_claim(obligation_id, error),
        lambda c: c.release_runtime_obligation_claim(obligation_id=obligation_id, error=error, owner=LIVE))
    assert_same(before, after)


SWEEP_ROWS = [
    ob("pending"),
    ob("pending-claimed-before", attempts=1),
    ob("attempting", state="attempting", attempts=1),
    ob("failed", state="failed", attempts=1, last_error="HTTP 500"),
    ob("flood-waiting", state="failed", attempts=1, last_error="flood_control:300", updated_at=NOW - 10),
    ob("flood-due", state="failed", attempts=1, last_error="Flood control exceeded. Retry in 5 seconds", updated_at=NOW - 10),
    ob("spent", attempts=3),
    ob("stale", created_at=NOW - DAY - 1),
    ob("legacy-profile", adapter_profile=None),
    ob("discord", platform="discord"),
    ob("work-bot", adapter_profile="work"),
    ob("crash-left", state="attempting", owner_pid=None, owner_started_at=None),
    ob("unstamped-dead", owner_started_at=None),
    mine("live"),
    ob("delivered", state="delivered"),
]
TARGETS = {
    "all": {},
    "telegram": {"deliverable_platforms": {"telegram"}},
    "default-bot": {"deliverable_targets": {("telegram", "default"), ("telegram", None)}},
}


@pytest.mark.parametrize("targets", sorted(TARGETS))
def test_sweep_recoverable(backend, scope_a, monkeypatch, targets):
    kwargs = TARGETS[targets]

    def verb(c):
        frozen = {key: frozenset(value) for key, value in kwargs.items()}
        return c.sweep_recoverable_obligations(now=NOW, owner=LIVE, dead_owners=dead_owners(c), **frozen)
    before, after = run_ledger_both(backend, scope_a, monkeypatch, SWEEP_ROWS,
                                    lambda m: m.sweep_recoverable(NOW, **kwargs), verb)
    assert_same(before, after, as_legacy)


RUNTIME_ROWS = [
    mine("degraded", state="failed", attempts=1, last_error="send_path_degraded", updated_at=NOW - 1),
    mine("flood-due", state="failed", attempts=1, last_error="flood_control:5", updated_at=NOW - 10),
    mine("flood-waiting", state="failed", attempts=1, last_error="flood_control:60", updated_at=NOW - 10),
    mine("backoff-passed", state="failed", attempts=0, last_error="HTTP 502", updated_at=NOW - 31),
    mine("backoff-waiting", state="failed", attempts=1, last_error="HTTP 502", updated_at=NOW - 31),
    mine("last-attempt", state="failed", attempts=2, last_error="HTTP 502", updated_at=NOW - 500),
    mine("chat-gone", state="failed", attempts=0, last_error="Forbidden: bot was blocked by the user"),
    mine("spent", state="failed", attempts=3, last_error="send_path_degraded"),
    mine("stale", state="failed", attempts=1, last_error="send_path_degraded", created_at=NOW - DAY - 1),
    mine("work-bot", state="failed", last_error="send_path_degraded", adapter_profile="work"),
    mine("discord", state="failed", last_error="send_path_degraded", platform="discord"),
    ob("theirs", state="failed", last_error="send_path_degraded"),
    mine("attempting", state="attempting", last_error="send_path_degraded"),
]


@pytest.mark.parametrize("profile", [None, "work"])
def test_sweep_failed_for_runtime(backend, scope_a, monkeypatch, profile):
    def verb(c):
        return c.sweep_failed_obligations_for_runtime("telegram", now=NOW, profile=profile, owner=LIVE,
                                                       due=due_rows(c, "telegram", profile))
    before, after = run_ledger_both(backend, scope_a, monkeypatch, RUNTIME_ROWS,
                                    lambda m: m.sweep_failed_for_runtime("telegram", NOW, profile=profile), verb)
    assert_same(before, after, as_legacy)


def test_runtime_sweep_without_a_start_stamp_reads_nothing(backend, scope_a, monkeypatch):
    stamp = OwnerStamp(pid=LIVE.pid, started_at=None)
    for module in (main_ledger(), dl):
        monkeypatch.setattr(module, "_owner_stamp", lambda: (stamp.pid, None))
    before, after = run_ledger_both(
        backend, scope_a, monkeypatch, RUNTIME_ROWS, lambda m: m.sweep_failed_for_runtime("telegram", NOW),
        lambda c: (c.list_runtime_retry_candidates("telegram", owner=stamp),
                   c.sweep_failed_obligations_for_runtime("telegram", now=NOW, owner=stamp, due=[]),
                   c.list_owned_failed_obligations(owner=stamp), c.release_runtime_obligation_claim(obligation_id="degraded", owner=stamp)))
    assert before == ("ok", []) and after == ("ok", ((), (), (), False))


def test_pending_retries(backend, scope_a, monkeypatch):
    """The per-adapter earliest deadline the legacy reader computed, over the backend's owned failed rows."""
    before, after = run_ledger_both(backend, scope_a, monkeypatch, RUNTIME_ROWS, lambda m: m.pending_retries(NOW),
                                    lambda c: c.list_owned_failed_obligations(owner=LIVE))
    assert_same(before, after, pending_retries_from)
    assert before[1]


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
