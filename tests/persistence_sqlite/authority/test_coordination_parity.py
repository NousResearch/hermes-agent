"""The SQLite ``CoordinationRepository`` against the legacy store: compression lock, turn lease, hygiene streak and
handoffs, verb by verb.

Each case seeds the profile's ``state.db`` with the legacy ``SessionDB`` under a fixed clock, keeps a copy of the
seed, runs the legacy method, restores the seed and runs the backend verb, then compares the outcomes (records
projected to the legacy values) and the rows both runs left. The helpers here serve the ledger parity files too.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pytest

from persistence import PersistenceError
from persistence.authority.coordination import CoordinationRepository, HandoffStatus
from tests.contracts.authority.test_coordination import dead_holder
from tests.persistence_sqlite.authority.conftest import NOW
from tests.persistence_sqlite.conftest import store_for

pytestmark = pytest.mark.usefixtures("clock", "legacy_store")

DAY = 86400.0
LEASE_TABLES = ("compression_locks", "session_turn_leases", "gateway_hygiene_state")


def copy_db(src: Path, dst: Path) -> None:
    with sqlite3.connect(src) as source, sqlite3.connect(dst) as target:
        source.backup(target)


def dump(path: Path, tables: Iterable[str]) -> dict[str, Any]:
    """Every row and the column list of each named table that exists."""
    with sqlite3.connect(path) as conn:
        present = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        out: dict[str, Any] = {}
        for table in tables:
            if table in present:
                out[table] = conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                out[table + ".columns"] = [row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")]
    return out


def outcome(call: Callable[[], Any]) -> tuple[str, Any]:
    try:
        return "ok", call()
    except Exception as exc:  # compared by class and message
        return "error", exc


def assert_same(before: tuple[str, Any], after: tuple[str, Any], project: Callable[[Any], Any] = lambda v: v) -> None:
    """Equal values, or the same refusal: a legacy SQLite error arrives as a contract error chaining it."""
    assert after[0] == before[0], (before, after)
    if before[0] == "ok":
        assert project(after[1]) == before[1]
        return
    legacy_exc, exc = before[1], after[1]
    if isinstance(legacy_exc, sqlite3.Error):
        assert isinstance(exc, PersistenceError) and type(exc.__cause__) is type(legacy_exc)
    else:
        assert isinstance(exc, type(legacy_exc))
    assert str(exc) == str(legacy_exc)


def run_store_both(backend, scope, seed: Callable[[Any], None], legacy: Callable[[Any], Any],
                   verb: Callable[[CoordinationRepository], Any], tables: Iterable[str]):
    """Seed, run the legacy method, restore the seed, run the backend verb; assert equal rows; return both outcomes."""
    from hermes_state import SessionDB

    tables = tuple(tables)
    path = store_for(scope)
    db = SessionDB(db_path=path)
    try:
        seed(db)
    finally:
        db.close()
    snapshot = path.with_name("seed.db")
    copy_db(path, snapshot)
    db = SessionDB(db_path=path)
    try:
        before = outcome(lambda: legacy(db))
    finally:
        db.close()
    legacy_rows = dump(path, tables)
    copy_db(snapshot, path)
    after = outcome(lambda: verb(backend.open(scope).coordination))
    assert dump(path, tables) == legacy_rows
    return before, after


def _sql(db: Any, sql: str, params: Any = ()) -> None:
    db._write_sql(sql, params)


def _lock(table: str, key_col: str, key: str, holder: str, expires_at: float) -> Callable[[Any], None]:
    return lambda db: _sql(db, f"INSERT INTO {table} ({key_col}, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                           (key, holder, NOW - 10, expires_at))


def _nothing(db: Any) -> None:
    return None


# ── compression lock ──────────────────────────────────────────────────────────

LOCK_SEEDS = {
    "free": lambda: _nothing,
    "live-other": lambda: _lock("compression_locks", "session_id", "s", "other", NOW + 60),
    "other-at-expiry": lambda: _lock("compression_locks", "session_id", "s", "other", NOW),
    "expired-other": lambda: _lock("compression_locks", "session_id", "s", "other", NOW - 0.001),
    "dead-holder": lambda: _lock("compression_locks", "session_id", "s", dead_holder("owner"), NOW + 60),
    "mine-expired": lambda: _lock("compression_locks", "session_id", "s", "me", NOW - 5),
    "mine-live": lambda: _lock("compression_locks", "session_id", "s", "me", NOW + 5),
}


@pytest.mark.parametrize("seed", sorted(LOCK_SEEDS))
@pytest.mark.parametrize("holder", ["me", ""])
def test_try_acquire_compression_lock(backend, scope_a, seed, holder):
    before, after = run_store_both(
        backend, scope_a, LOCK_SEEDS[seed](),
        lambda db: db.try_acquire_compression_lock("s", holder, ttl_seconds=60),
        lambda c: c.try_acquire_compression_lock(session_id="s", holder=holder, ttl_seconds=60), LEASE_TABLES)
    assert_same(before, after)


@pytest.mark.parametrize("seed", sorted(LOCK_SEEDS))
def test_refresh_and_release_compression_lock(backend, scope_a, seed):
    def legacy(db):
        return db.refresh_compression_lock("s", "me", ttl_seconds=30), db.release_compression_lock("s", "other")

    def verb(c):
        return (c.refresh_compression_lock(session_id="s", holder="me", ttl_seconds=30),
                c.release_compression_lock(session_id="s", holder="other"))
    before, after = run_store_both(backend, scope_a, LOCK_SEEDS[seed](), legacy, verb, LEASE_TABLES)
    assert_same(before, after)


@pytest.mark.parametrize("seed", sorted(LOCK_SEEDS))
def test_get_compression_lock_holder(backend, scope_a, seed):
    before, after = run_store_both(
        backend, scope_a, LOCK_SEEDS[seed](), lambda db: db.get_compression_lock_holder("s"),
        lambda c: c.get_compression_lock_holder(session_id="s"), LEASE_TABLES)
    assert_same(before, after)


# ── turn lease ────────────────────────────────────────────────────────────────

def _lineage(db: Any) -> None:
    """``root`` ended by compression with the continuation ``tip``, and an explicit branch child of ``tip``."""
    db.create_session("root", "cli")
    db.create_session("tip", "cli", parent_session_id="root")
    db.end_session("root", "compression")
    db.create_session("branch", "cli", parent_session_id="tip", model_config={"_branched_from": "tip"})


def _with(*steps: Callable[[Any], None]) -> Callable[[Any], None]:
    def seed(db: Any) -> None:
        for step in steps:
            step(db)
    return seed


LEASE_SEEDS = {
    "free": lambda: _lineage,
    "held-by-other": lambda: _with(_lineage, _lock("session_turn_leases", "conversation_id", "root", "other", NOW + 60)),
    "other-at-expiry": lambda: _with(_lineage, _lock("session_turn_leases", "conversation_id", "root", "other", NOW)),
    "dead-holder": lambda: _with(_lineage, _lock("session_turn_leases", "conversation_id", "root", dead_holder(), NOW + 60)),
    "mine-expired": lambda: _with(_lineage, _lock("session_turn_leases", "conversation_id", "root", "me", NOW - 30)),
    "swept-a-day-ago": lambda: _with(
        _lineage, _lock("session_turn_leases", "conversation_id", "elsewhere", "gone", NOW - DAY - 1),
        _lock("session_turn_leases", "conversation_id", "recent", "kept", NOW - DAY + 1)),
}


@pytest.mark.parametrize("seed", sorted(LEASE_SEEDS))
@pytest.mark.parametrize("session_id", ["tip", "root", "branch", "unknown"])
def test_try_acquire_turn_lease(backend, scope_a, seed, session_id):
    before, after = run_store_both(
        backend, scope_a, LEASE_SEEDS[seed](),
        lambda db: db.try_acquire_session_turn_lease(session_id, "me", ttl_seconds=0.01),
        lambda c: c.try_acquire_turn_lease(session_id=session_id, holder="me", ttl_seconds=0.01), LEASE_TABLES)
    assert_same(before, after)


@pytest.mark.parametrize("seed", sorted(LEASE_SEEDS))
@pytest.mark.parametrize("session_id", ["tip", "branch"])
def test_refresh_read_and_release_turn_lease(backend, scope_a, seed, session_id):
    def legacy(db):
        return (db.refresh_session_turn_lease(session_id, "me", ttl_seconds=45),
                db.session_turn_lease_expires_at(session_id, "me"),
                db.release_session_turn_lease(session_id, "other"),
                db.session_turn_lease_expires_at(session_id, "other"))

    def verb(c):
        return (c.refresh_turn_lease(session_id=session_id, holder="me", ttl_seconds=45),
                c.turn_lease_expires_at(session_id=session_id, holder="me"),
                c.release_turn_lease(session_id=session_id, holder="other"),
                c.turn_lease_expires_at(session_id=session_id, holder="other"))
    before, after = run_store_both(backend, scope_a, LEASE_SEEDS[seed](), legacy, verb, LEASE_TABLES)
    assert_same(before, after)


def test_turn_lease_empty_arguments(backend, scope_a):
    def legacy(db):
        return (db.try_acquire_session_turn_lease("", "me"), db.try_acquire_session_turn_lease("tip", ""),
                db.refresh_session_turn_lease("tip", ""), db.session_turn_lease_expires_at("", "me"))

    def verb(c):
        return (c.try_acquire_turn_lease(session_id="", holder="me"), c.try_acquire_turn_lease(session_id="tip", holder=""),
                c.refresh_turn_lease(session_id="tip", holder=""), c.turn_lease_expires_at(session_id="", holder="me"))
    before, after = run_store_both(backend, scope_a, LEASE_SEEDS["held-by-other"](), legacy, verb, LEASE_TABLES)
    assert_same(before, after)


# ── hygiene streak ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("existing", [None, 0, 4])
@pytest.mark.parametrize("key", ["agent:main:telegram:dm:1", ""])
def test_hygiene_streak(backend, scope_a, existing, key):
    def seed(db):
        if existing is not None:
            _sql(db, "INSERT INTO gateway_hygiene_state (session_key, failure_streak) VALUES (?, ?)",
                 ("agent:main:telegram:dm:1", existing))

    def legacy(db):
        first = db.increment_hygiene_failure_streak(key)
        second = db.increment_hygiene_failure_streak(key)
        db.reset_hygiene_failure_streak("absent")
        return first, second

    def verb(c):
        first = c.increment_hygiene_failure_streak(session_key=key)
        second = c.increment_hygiene_failure_streak(session_key=key)
        c.reset_hygiene_failure_streak(session_key="absent")
        return first, second
    before, after = run_store_both(backend, scope_a, seed, legacy, verb, LEASE_TABLES)
    assert_same(before, after)


def test_hygiene_streak_reset(backend, scope_a):
    def seed(db):
        _sql(db, "INSERT INTO gateway_hygiene_state (session_key, failure_streak) VALUES ('k', 3), ('j', 1)")
    before, after = run_store_both(
        backend, scope_a, seed, lambda db: db.reset_hygiene_failure_streak("k"),
        lambda c: c.reset_hygiene_failure_streak(session_key="k"), LEASE_TABLES)
    assert_same(before, after)


# ── handoffs ──────────────────────────────────────────────────────────────────

HANDOFF_TABLES = ("sessions",)


def _handoffs(db: Any) -> None:
    """One session per handoff state, started in a known order."""
    for offset, (sid, state) in enumerate([("none", None), ("pending-a", "pending"), ("running", "running"),
                                            ("completed", "completed"), ("failed", "failed"), ("pending-b", "pending")]):
        db.create_session(sid, "cli")
        _sql(db, "UPDATE sessions SET started_at = ?, title = ?, handoff_state = ?, handoff_platform = ?, "
                 "handoff_error = ? WHERE id = ?",
             (NOW - 100 + (5 - offset), f"title {sid}", state, "telegram" if state else None,
              "old error" if state == "failed" else None, sid))


SESSIONS = ["none", "pending-a", "running", "completed", "failed", "missing"]


@pytest.mark.parametrize("session_id", SESSIONS)
def test_request_and_claim_handoff(backend, scope_a, session_id):
    def legacy(db):
        return db.request_handoff(session_id, "discord"), db.claim_handoff(session_id), db.claim_handoff(session_id)

    def verb(c):
        return (c.request_handoff(session_id=session_id, platform="discord"), c.claim_handoff(session_id=session_id),
                c.claim_handoff(session_id=session_id))
    before, after = run_store_both(backend, scope_a, _handoffs, legacy, verb, HANDOFF_TABLES)
    assert_same(before, after)


@pytest.mark.parametrize("session_id", SESSIONS)
@pytest.mark.parametrize("only_states", [None, ("pending",), ("pending", "running")])
def test_fail_and_complete_handoff(backend, scope_a, session_id, only_states):
    error = "x" * 600

    def legacy(db):
        failed = db.fail_handoff(session_id, error, only_states=only_states)
        db.complete_handoff("completed")
        return failed

    def verb(c):
        states = tuple(HandoffStatus(s) for s in only_states) if only_states else None
        failed = c.fail_handoff(session_id=session_id, error=error, only_states=states)
        c.complete_handoff(session_id="completed")
        return failed
    before, after = run_store_both(backend, scope_a, _handoffs, legacy, verb, HANDOFF_TABLES)
    assert_same(before, after)


def test_reclaim_stale_running_handoffs(backend, scope_a):
    before, after = run_store_both(
        backend, scope_a, _handoffs, lambda db: db.reclaim_stale_running_handoffs("y" * 700),
        lambda c: c.reclaim_stale_running_handoffs(error="y" * 700), HANDOFF_TABLES)
    assert_same(before, after, list)


def _legacy_handoff(state):
    return None if state is None or state["state"] is None else state


def test_handoff_reads(backend, scope_a):
    def legacy(db):
        return ([_legacy_handoff(db.get_handoff_state(sid)) for sid in SESSIONS],
                [(row["id"], row["handoff_platform"], row["title"], row["started_at"]) for row in db.list_pending_handoffs()],
                db.has_pending_handoffs())

    def verb(c):
        return ([c.get_handoff(session_id=sid) for sid in SESSIONS], c.list_pending_handoffs(), c.has_pending_handoffs())

    def project(value):
        states, pending, has = value
        return ([None if s is None else {"state": s.state.value, "platform": s.platform, "error": s.error} for s in states],
                [(p.session_id, p.platform, p.title, p.started_at) for p in pending], has)
    before, after = run_store_both(backend, scope_a, _handoffs, legacy, verb, HANDOFF_TABLES)
    assert_same(before, after, project)
