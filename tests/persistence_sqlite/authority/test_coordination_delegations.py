"""The SQLite delegation-ledger verbs against tools/async_delegation.py, and their transaction shape.

Each parity case writes the ledger rows into a schema-built ``state.db``, keeps a copy, runs the legacy function with
``HERMES_HOME`` at the profile home (the ledger resolves its file there), restores the copy and runs the backend verb
(composed with the owner probe and the redaction the legacy function ran, which stay with the caller), then compares
the outcomes and the ``async_delegations`` rows. The shape cases prove the sweeps take owners as data, touch only rows
unchanged since the caller's read, open no connection and probe no process inside the transaction, and reach the
writer with the ledger's 10 s patience.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
from typing import Any, Callable

import pytest

import tools.async_delegation as ad
import tools.async_delegation_recovery_hints as hints
from gateway.status import get_process_start_time
from persistence.authority.coordination import DELEGATION_CLAIM_STALE_SECONDS, DelegationRecovery, OwnerStamp, RecordedOwner
from persistence.sqlite._holder import StoreHolder
from tests.contracts.authority.test_coordination import dead_pid
from tests.persistence_sqlite.authority.test_coordination_parity import (
    DAY, NOW, assert_same, copy_db, dump, outcome,
)
from tests.persistence_sqlite.conftest import store_for
from tools.delegation_redaction import redact_child_result, redact_completion_event

pytestmark = pytest.mark.usefixtures("clock")

TABLES = ("async_delegations",)
LIVE_PID = os.getpid()
LIVE_STARTED = get_process_start_time(LIVE_PID)
DEAD_PID = dead_pid()

_COLUMNS = ("delegation_id", "origin_session", "origin_ui_session_id", "parent_session_id", "state", "dispatched_at",
            "completed_at", "updated_at", "event_json", "result_json", "delivery_state", "delivery_attempts",
            "delivered_at", "owner_pid", "owner_started_at", "task_json", "delivery_claim", "delivery_claimed_at",
            "origin_session_id")


def row(delegation_id: str, **values: Any) -> dict[str, Any]:
    """A terminal row with a pending delivery, owned by a dead process, unless ``values`` say otherwise."""
    event = {"type": "async_delegation", "delegation_id": delegation_id, "status": "completed", "summary": "done",
             "session_key": "agent:main:cli", "completed_at": NOW - 600}
    base = {"delegation_id": delegation_id, "origin_session": "agent:main:cli", "origin_ui_session_id": "ui-1",
            "parent_session_id": "parent-1", "state": "completed", "dispatched_at": NOW - 3600,
            "completed_at": NOW - 600, "updated_at": NOW - 600, "event_json": json.dumps(event),
            "result_json": json.dumps({"status": "completed", "summary": "done"}), "delivery_state": "pending",
            "delivery_attempts": 0, "delivered_at": None, "owner_pid": DEAD_PID, "owner_started_at": 11,
            "task_json": json.dumps({"goal": "g", "is_batch": False}), "delivery_claim": None,
            "delivery_claimed_at": None, "origin_session_id": ""}
    base.update(values)
    return base


def seed_rows(path, rows: list[dict[str, Any]]) -> None:
    StoreHolder(path).close()
    with sqlite3.connect(path) as conn:
        conn.executemany(f"INSERT INTO async_delegations ({', '.join(_COLUMNS)}) VALUES "
                         f"({', '.join('?' for _ in _COLUMNS)})", [tuple(r[c] for c in _COLUMNS) for r in rows])


def alive(pid: Any, started: Any) -> bool:
    return pid == LIVE_PID


@pytest.fixture(autouse=True)
def legacy_state(monkeypatch):
    """The legacy module's in-memory offer sets start empty; recovery hints read nothing from disk."""
    monkeypatch.setattr(ad, "_offered", set())
    monkeypatch.setattr(ad, "_startup_replayed", set())
    monkeypatch.setattr(ad, "_owner_liveness", lambda: alive)
    monkeypatch.setattr(hints, "git_state_hint", lambda cwd: None)
    monkeypatch.setattr(hints, "transcript_tails", lambda transcripts: None)


def run_ledger_both(backend, scope, monkeypatch, rows: list[dict[str, Any]], legacy: Callable[[], Any],
                    verb: Callable[[Any], Any]):
    """Seed, run the legacy function on the profile home, restore the seed, run the verb; assert equal rows."""
    path = store_for(scope)
    seed_rows(path, rows)
    snapshot = path.with_name("seed.db")
    copy_db(path, snapshot)
    with monkeypatch.context() as patch:
        patch.setenv("HERMES_HOME", str(path.parent))
        before = outcome(legacy)
    legacy_rows = dump(path, TABLES)
    copy_db(snapshot, path)
    after = outcome(lambda: verb(backend.open(scope).coordination))
    assert dump(path, TABLES) == legacy_rows
    return before, after


def drain(target: queue.Queue) -> list[Any]:
    items = []
    while not target.empty():
        items.append(target.get_nowait())
    return items


# ── writes ────────────────────────────────────────────────────────────────────

def test_dispatch_then_prune(backend, scope_a, monkeypatch):
    record = {"delegation_id": "d-new", "session_key": "agent:main:cli", "origin_ui_session_id": "ui-9",
              "parent_session_id": "p-9", "dispatched_at": NOW - 1, "origin_session_id": "o-9", "goal": "write",
              "context": "ctx", "toolsets": ["terminal"], "is_batch": False, "scope_id": "scope-1"}
    old = [row(f"t-{i:02d}", updated_at=NOW - 8 * DAY - i, delivery_state="delivered" if i % 3 else "pending")
           for i in range(60)]
    task = {k: record[k] for k in ("goal", "goals", "context", "toolsets", "role", "model", "is_batch",
                                   "task_indexes", "task_transcripts", "scope_id", "user_id", "user_name") if k in record}
    task["owner_cwd"] = os.getcwd()

    def verb(c):
        c.record_delegation_dispatch(
            delegation_id="d-new", origin_session="agent:main:cli", origin_ui_session_id="ui-9", parent_session_id="p-9",
            dispatched_at=NOW - 1, owner=OwnerStamp(pid=LIVE_PID, started_at=LIVE_STARTED), task=task,
            origin_session_id="o-9")
        c.prune_delegation_ledger()
    before, after = run_ledger_both(backend, scope_a, monkeypatch, old, lambda: ad._persist_dispatch(record), verb)
    assert_same(before, after)


@pytest.mark.parametrize("delegation_id", ["d-1", "missing"])
def test_completion(backend, scope_a, monkeypatch, delegation_id):
    event = {"type": "async_delegation", "delegation_id": delegation_id, "status": "error", "completed_at": NOW - 5,
             "summary": "token sk-ant-api03-" + "a" * 40, "error": "boom"}
    result = {"status": "error", "summary": "Bearer abcdefghijklmnopqrstuvwxyz0123", "error": "boom"}

    def verb(c):
        c.record_delegation_completion(delegation_id=delegation_id, state=event.get("status", "completed"),
                                       completed_at=event.get("completed_at", NOW), event=redact_completion_event(event),
                                       result=redact_child_result(result))
    before, after = run_ledger_both(backend, scope_a, monkeypatch, [row("d-1", state="running", event_json=None)],
                                    lambda: ad._persist_completion(event, result), verb)
    assert_same(before, after)


@pytest.mark.parametrize("delegation_id", ["running", "done", "missing"])
def test_unit_child(backend, scope_a, monkeypatch, delegation_id):
    stored = redact_child_result({"results": [{"task_index": 0, "status": "completed", "summary": "first"},
                                              {"task_index": 1, "status": "error", "summary": "old"}], "partial": True})
    rows = [row("running", state="running", result_json=json.dumps(stored)), row("done")]
    entry = {"task_index": 1, "status": "completed", "summary": "key sk-ant-api03-" + "b" * 40}
    before, after = run_ledger_both(
        backend, scope_a, monkeypatch, rows, lambda: ad.record_unit_child(delegation_id, entry),
        lambda c: c.record_unit_child(delegation_id=delegation_id, entry=redact_child_result(entry)))
    assert_same(before, after)


def test_redaction_is_idempotent_on_its_own_output():
    """``record_unit_child`` stores the caller-redacted entry and does not redact the stored partial again."""
    value = {"results": [{"task_index": 0, "summary": "sk-ant-api03-" + "c" * 40, "error": "Bearer abcdefghijklmnop0123456789"},
                         {"task_index": 1, "status": "completed", "summary": "password=hunter2hunter2"}],
             "partial": True, "summary": "ghp_" + "d" * 36}
    once = redact_child_result(value)
    assert redact_child_result(once) == once


def test_recover_abandoned(backend, scope_a, monkeypatch):
    task = {"goal": "unit", "goals": ["a", "b"], "is_batch": True, "task_indexes": [0, 1], "scope_id": "s1"}
    partial = {"results": [{"task_index": 0, "status": "completed", "summary": "a done"}], "partial": True}
    rows = [row("dead-running", state="running", completed_at=None, event_json=None, task_json=json.dumps(task),
                result_json=json.dumps(partial), updated_at=NOW - 50),
            row("dead-finalizing", state="finalizing", completed_at=None, event_json=None, result_json=None,
                owner_started_at=None),
            row("live-running", state="running", owner_pid=LIVE_PID, owner_started_at=LIVE_STARTED, event_json=None),
            row("terminal")]
    legacy_after: dict[str, Any] = {}
    path = store_for(scope_a)

    def legacy():
        count = ad.recover_abandoned_delegations()
        with sqlite3.connect(path) as conn:
            legacy_after.update({r[0]: (json.loads(r[1]), json.loads(r[2])) for r in conn.execute(
                "SELECT delegation_id, event_json, result_json FROM async_delegations WHERE state='unknown'")})
        return count

    def verb(c):
        recoveries = [DelegationRecovery(delegation_id=item.delegation_id, expected_state=item.state,
                                         expected_updated_at=item.updated_at, completed_at=NOW,
                                         event=legacy_after[item.delegation_id][0], result=legacy_after[item.delegation_id][1])
                      for item in c.list_inflight_delegations() if not alive(item.owner_pid, item.owner_started_at)]
        return c.recover_abandoned_delegations(recoveries=recoveries)
    before, after = run_ledger_both(backend, scope_a, monkeypatch, rows, legacy, verb)
    assert before == ("ok", 2)
    assert_same(before, after)


def _offered_events(completions, *, startup: bool) -> list[dict[str, Any]]:
    return [{**redact_completion_event(item.event), "restored": True, **({"startup_replay": True} if startup else {})}
            for item in completions]


REPLAY_ROWS = [
    row("fresh-b", completed_at=NOW - 30),
    row("fresh-a", completed_at=NOW - 30),
    row("no-completion", completed_at=None, dispatched_at=NOW - 100),
    row("too-old", completed_at=NOW - 49 * 3600, updated_at=NOW - 49 * 3600),
    row("running", state="running"),
    row("delivered", delivery_state="delivered"),
    row("no-event", event_json=None),
    row("secret", event_json=json.dumps({"type": "async_delegation", "delegation_id": "secret",
                                         "summary": "sk-ant-api03-" + "e" * 40})),
]


def test_take_replayable_completions(backend, scope_a, monkeypatch):
    monkeypatch.setattr(ad, "recover_abandoned_delegations", lambda: 0)
    target: queue.Queue = queue.Queue()

    def legacy():
        count = ad.restore_undelivered_completions(target)
        return count, drain(target)
    before, after = run_ledger_both(backend, scope_a, monkeypatch, REPLAY_ROWS, legacy,
                                    lambda c: c.take_replayable_completions())
    assert_same(before, after, lambda pending: (len(pending), _offered_events(pending, startup=True)))


ORPHAN_ROWS = [
    row("dead-idle"),
    row("dead-idle-unstamped", owner_started_at=None),
    row("live-idle", owner_pid=LIVE_PID, owner_started_at=LIVE_STARTED),
    row("dead-recent", updated_at=NOW - 30),
    row("dead-spent", delivery_attempts=8),
    row("dead-old", completed_at=NOW - 49 * 3600),
    row("dead-claimed", delivery_claim="c-1", delivery_claimed_at=NOW - 100),
    row("dead-stale-claim", delivery_claim="c-2", delivery_claimed_at=NOW - 400),
    row("dead-offered"),
    row("dead-running", state="running"),
]


def test_sweep_orphaned_completions(backend, scope_a, monkeypatch):
    monkeypatch.setattr(ad, "recover_abandoned_delegations", lambda: 0)
    target: queue.Queue = queue.Queue()

    def legacy():
        ad._offered.add((ad.hermes_home_key(ad.get_hermes_home()), "dead-offered"))
        count = ad.sweep_orphaned_completions(target, now=NOW)
        return count, drain(target)

    def verb(c):
        candidates = c.list_orphan_completions(now=NOW)
        dead = frozenset(item.owner for item in candidates if not alive(item.owner.pid, item.owner.started_at))
        # No copy is queued, held or re-offered here, so a copy this process made is lost once idle past a claim lease.
        lost = frozenset(item.delegation_id for item in candidates
                         if (item.delegation_id == "dead-offered" or item.owner.pid == LIVE_PID)
                         and item.updated_at < NOW - DELEGATION_CLAIM_STALE_SECONDS)
        return c.sweep_orphaned_completions(now=NOW, exclude_ids=frozenset({"dead-offered"}), dead_owners=dead, lost_ids=lost)
    before, after = run_ledger_both(backend, scope_a, monkeypatch, ORPHAN_ROWS, legacy, verb)
    assert before[1][0] == 5
    assert_same(before, after, lambda pending: (len(pending), _offered_events(pending, startup=False)))


CLAIM_ROWS = [
    row("free"),
    row("mine", delivery_claim="me", delivery_claimed_at=NOW - 10, delivery_attempts=3),
    row("mine-spent", delivery_claim="me", delivery_claimed_at=NOW - 10, delivery_attempts=8),
    row("mine-zero", delivery_claim="me", delivery_claimed_at=NOW - 10, delivery_attempts=0),
    row("theirs", delivery_claim="them", delivery_claimed_at=NOW - 10),
    row("theirs-stale", delivery_claim="them", delivery_claimed_at=NOW - 301),
    row("delivered", delivery_state="delivered", delivery_claim="me"),
]
CLAIM_IDS = [r["delegation_id"] for r in CLAIM_ROWS] + ["missing"]
SETTLE = {
    "claim": (ad.claim_completion_delivery, "claim_delivery"),
    "release": (ad.release_completion_delivery, "release_delivery"),
    "defer": (ad.defer_completion_delivery, "defer_delivery"),
    "drop": (ad.drop_completion_delivery, "drop_delivery"),
    "complete": (ad.complete_completion_delivery, "complete_delivery"),
}


@pytest.mark.parametrize("action", sorted(SETTLE))
@pytest.mark.parametrize("delegation_id", CLAIM_IDS)
def test_delivery_claims(backend, scope_a, monkeypatch, action, delegation_id):
    legacy_fn, verb_name = SETTLE[action]
    before, after = run_ledger_both(
        backend, scope_a, monkeypatch, CLAIM_ROWS, lambda: legacy_fn(delegation_id, "me"),
        lambda c: getattr(c, verb_name)(delegation_id=delegation_id, claim_id="me"))
    assert_same(before, after)


REOPEN_ROWS = [
    row("delivered", delivery_state="delivered", delivered_at=NOW - 5, delivery_attempts=2),
    row("delivered-spent", delivery_state="delivered", delivered_at=NOW - 5, delivery_attempts=8),
    row("delivered-no-event", delivery_state="delivered", event_json=None),
    row("delivered-old", delivery_state="delivered", completed_at=NOW - 49 * 3600),
    row("pending"),
]


@pytest.mark.parametrize("delegation_id", [r["delegation_id"] for r in REOPEN_ROWS] + ["missing"])
def test_reopen_admitted_completion(backend, scope_a, monkeypatch, delegation_id):
    target: queue.Queue = queue.Queue()

    def legacy():
        requeued = ad.requeue_admitted_completion(delegation_id, target)
        return requeued, drain(target)

    def project(pending):
        offered = _offered_events([pending] if pending else [], startup=False)
        return bool(pending), offered
    before, after = run_ledger_both(backend, scope_a, monkeypatch, REOPEN_ROWS, legacy,
                                    lambda c: c.reopen_admitted_completion(delegation_id=delegation_id))
    assert_same(before, after, project)


# ── reads ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("delegation_id", ["done", "running", "missing"])
def test_get_delegation(backend, scope_a, monkeypatch, delegation_id):
    rows = [row("done", origin_session_id="o-1", result_json=json.dumps({"summary": "sk-ant-api03-" + "f" * 40})),
            row("running", state="running", result_json=None)]

    def project(record):
        if record is None:
            return None
        return {"delegation_id": record.delegation_id, "origin_session": record.origin_session, "state": record.state,
                "dispatched_at": record.dispatched_at, "completed_at": record.completed_at,
                "result": redact_child_result(record.result) if record.result is not None else None,
                "delivery_state": record.delivery_state.value, "delivery_attempts": record.delivery_attempts,
                "origin_session_id": record.origin_session_id}
    before, after = run_ledger_both(backend, scope_a, monkeypatch, rows, lambda: ad.get_durable_delegation(delegation_id),
                                    lambda c: c.get_delegation(delegation_id=delegation_id))
    assert_same(before, after, project)


# ── transaction shape ─────────────────────────────────────────────────────────

class _Forbidden:
    """Raises on use: a liveness probe or a new connection inside a verb."""

    def __init__(self, what: str) -> None:
        self.what = what

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{self.what} called inside a coordination verb")


def _forbid_probes_and_connections(monkeypatch) -> None:
    from gateway import status
    for name in ("_pid_exists", "get_process_start_time", "start_time_fingerprints_match"):
        monkeypatch.setattr(status, name, _Forbidden(f"gateway.status.{name}"))
    monkeypatch.setattr(ad, "_owner_liveness", _Forbidden("the owner liveness probe"))
    monkeypatch.setattr(sqlite3, "connect", _Forbidden("sqlite3.connect"))


def _ready(backend, scope, rows):
    seed_rows(store_for(scope), rows)
    coord = backend.open(scope).coordination
    coord.get_delegation(delegation_id="warm")  # open the writer and a pooled reader before the forbidden window
    return coord


def test_orphan_sweep_takes_owners_as_data_and_skips_rows_changed_since_the_read(backend, scope_a, monkeypatch, clock):
    coord = _ready(backend, scope_a, [row("a"), row("b"), row("c"), row("live", owner_pid=LIVE_PID)])
    dead = frozenset(o for o in coord.list_orphan_completion_owners(now=NOW) if o.pid == DEAD_PID)
    assert dead == {RecordedOwner(pid=DEAD_PID, started_at=11)}
    assert coord.claim_delivery(delegation_id="b", claim_id="consumer") is True  # claimed after the owner read
    coord.record_delegation_dispatch(delegation_id="c", origin_session="k", origin_ui_session_id="", parent_session_id=None,
                                     dispatched_at=NOW, owner=OwnerStamp(pid=LIVE_PID, started_at=1), task={})
    _forbid_probes_and_connections(monkeypatch)
    offered = coord.sweep_orphaned_completions(now=NOW, dead_owners=dead)
    monkeypatch.undo()
    assert [item.delegation_id for item in offered] == ["a"]
    assert coord.get_delegation(delegation_id="b").delivery_attempts == 1


def test_recovery_writes_only_rows_unchanged_since_the_inflight_read(backend, scope_a, monkeypatch, clock):
    coord = _ready(backend, scope_a, [row("x", state="running", event_json=None),
                                      row("y", state="running", event_json=None)])
    inflight = coord.list_inflight_delegations()
    clock.now += 1
    coord.record_unit_child(delegation_id="y", entry={"task_index": 0, "status": "completed"})
    _forbid_probes_and_connections(monkeypatch)
    written = coord.recover_abandoned_delegations(recoveries=[
        DelegationRecovery(delegation_id=item.delegation_id, expected_state=item.state, expected_updated_at=item.updated_at,
                           completed_at=NOW, event={"status": "unknown"}, result={"status": "unknown"})
        for item in inflight])
    monkeypatch.undo()
    assert written == 1
    assert coord.get_delegation(delegation_id="x").state == "unknown"
    assert coord.get_delegation(delegation_id="y").state == "running"


LEDGER_WRITES: dict[str, Callable[[Any], Any]] = {
    "record_delegation_dispatch": lambda c: c.record_delegation_dispatch(
        delegation_id="n", origin_session="k", origin_ui_session_id="", parent_session_id=None, dispatched_at=NOW,
        owner=OwnerStamp(pid=1, started_at=1), task={}),
    "prune_delegation_ledger": lambda c: c.prune_delegation_ledger(),
    "record_delegation_completion": lambda c: c.record_delegation_completion(
        delegation_id="a", state="completed", completed_at=NOW, event={}, result={}),
    "record_unit_child": lambda c: c.record_unit_child(delegation_id="a", entry={"task_index": 0}),
    "recover_abandoned_delegations": lambda c: c.recover_abandoned_delegations(recoveries=[]),
    "take_replayable_completions": lambda c: c.take_replayable_completions(),
    "sweep_orphaned_completions": lambda c: c.sweep_orphaned_completions(now=NOW, dead_owners=frozenset()),
    "claim_delivery": lambda c: c.claim_delivery(delegation_id="a", claim_id="x"),
    "release_delivery": lambda c: c.release_delivery(delegation_id="a", claim_id="x"),
    "defer_delivery": lambda c: c.defer_delivery(delegation_id="a", claim_id="x"),
    "drop_delivery": lambda c: c.drop_delivery(delegation_id="a", claim_id="x"),
    "complete_delivery": lambda c: c.complete_delivery(delegation_id="a", claim_id="x"),
    "reopen_admitted_completion": lambda c: c.reopen_admitted_completion(delegation_id="a"),
    "delete_unscheduled_delegation": lambda c: c.delete_unscheduled_delegation(delegation_id="a"),
}
LEDGER_READS: dict[str, Callable[[Any], Any]] = {
    "get_delegation": lambda c: c.get_delegation(delegation_id="a"),
    "list_finished_delegations": lambda c: c.list_finished_delegations(origin_ui_session_id="ui-1"),
    "list_inflight_delegations": lambda c: c.list_inflight_delegations(),
    "list_orphan_completion_owners": lambda c: c.list_orphan_completion_owners(now=NOW),
    "list_orphan_completions": lambda c: c.list_orphan_completions(now=NOW),
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
    coord = _ready(backend, scope_a, [row("a")])
    writes, _ = _record_patience(monkeypatch)
    LEDGER_WRITES[verb](coord)
    assert writes == [10.0]


@pytest.mark.parametrize("verb", sorted(LEDGER_READS))
def test_ledger_reads_wait_ten_seconds_for_the_lock(backend, scope_a, monkeypatch, verb):
    coord = _ready(backend, scope_a, [row("a")])
    writes, reads = _record_patience(monkeypatch)
    LEDGER_READS[verb](coord)
    assert (writes, reads) == ([], [10.0])
