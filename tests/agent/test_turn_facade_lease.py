"""Unit tests for agent.turn_facade_lease (admission + lease bracket)."""
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    DurableTurnLease,
    admit_durable_turn_lease,
)
from tests.fixtures.storage_holder import raw_holder


class _Db:
    def __init__(self, exists=True, acquired=True):
        self.exists = exists
        self.acquired = acquired
        self.events = []

    def get_session(self, session_id):
        return {"id": session_id} if self.exists else None

    def acquire_session_turn_lease(self, session_id, holder, **kwargs):
        self.events.append(("acquire", session_id, holder))
        return self.acquired

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        return True

    def release_session_turn_lease(self, session_id, holder):
        self.events.append(("release", session_id, holder))


class _Unused:
    """An owner's legacy store: the owner path must not call it."""

    def __getattr__(self, name):
        raise AssertionError(f"legacy store called on the owner path: {name}")


def _handle(db):
    """A writer's interface handle whose verbs land on ``db``'s fakes."""
    return SimpleNamespace(
        read_only=False,
        coordination=SimpleNamespace(
            try_acquire_turn_lease=lambda *, session_id, holder, **kw: db.acquire_session_turn_lease(
                session_id, holder),
            refresh_turn_lease=lambda *, session_id, holder, **kw: db.refresh_session_turn_lease(
                session_id, holder, **kw),
            release_turn_lease=lambda *, session_id, holder: db.release_session_turn_lease(session_id, holder),
            turn_lease_expires_at=lambda *, session_id, holder: None,
        ),
        sessions=SimpleNamespace(get=db.get_session),
    )


def _wired(db, kind, **overrides):
    """``client``: ``db`` is the agent's client store. ``owner``: ``db`` backs the agent's handle."""
    if kind == "client":
        return _agent(db, **overrides)
    return _agent(_Unused(), persistence=_handle(db), **overrides)


def _agent(db, **overrides):
    agent = SimpleNamespace(
        _session_db=db,
        session_id="s1",
        _persist_disabled=False,
        _interrupt_requested=False,
        _interrupt_message=None,
        _execution_thread_id=None,
        _session_turn_lease_refresh_interval=60.0,
        statuses=[],
    )
    agent._emit_status = agent.statuses.append
    agent._emit_warning = agent.statuses.append
    agent._touch_activity = lambda *a, **k: None
    agent._liveness_activity_lock = lambda: threading.Lock()
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _admit(agent, history=None):
    return admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "cli"},
        conversation_history=history,
    )


def test_fresh_row_leases_with_seed_kept_and_persist_disabled_skips(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    seed = [{"role": "user", "content": "hi"}]
    agent = _agent(_Db(exists=False))
    admission = _admit(agent, seed)
    assert admission.lease is not None and admission.early_result is None
    assert admission.conversation_history is seed
    assert getattr(agent, "_session_db_created", False) is False
    admission.lease.release()

    db = _Db()
    admission = _admit(_agent(db, _persist_disabled=True), seed)
    assert admission.lease is None and db.events == []


@pytest.mark.parametrize("kind", ["client", "owner"])
def test_admission_sets_holder_attrs_and_release_clears_them(monkeypatch, kind):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    db = _Db()
    agent = _wired(db, kind)
    admission = _admit(agent)
    lease = admission.lease
    assert isinstance(lease, DurableTurnLease)
    assert agent._session_db_created is True
    assert agent._active_session_turn_lease_holder == lease.holder
    assert agent._active_session_turn_lease_ttl_seconds == LEASE_TTL_SECONDS
    assert lease.holder.startswith("pid=") and ":platform=cli" in lease.holder
    assert lease.watchdog is None and lease.timer_handles == []
    assert lease.is_turn_active() is False

    lease.stop_refresher()
    lease.join_threads()
    lease.clear_interrupt()
    lease.release()
    assert db.events == [("acquire", "s1", lease.holder), ("release", "s1", lease.holder)]
    assert agent._active_session_turn_lease_holder is None
    assert agent._active_session_turn_lease_ttl_seconds is None


@pytest.mark.parametrize("kind", ["client", "owner"])
def test_timeout_and_interrupt_early_results(monkeypatch, kind):
    # The owner's wait loop runs here; a zero wait makes the first miss its timeout.
    monkeypatch.setattr("agent.turn_facade_lease.LEASE_WAIT_SECONDS", 0.0)
    agent = _wired(_Db(acquired=False), kind)
    admission = _admit(agent, [{"role": "user", "content": "x"}])
    assert admission.lease is None
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    # Stamped so UI descriptors show "session busy" instead of code="unknown".
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _wired(_Db(acquired=False), kind, _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(agent)
    assert admission.early_result["interrupted"] is True
    assert admission.early_result["interrupt_message"] == "stop"


def test_interrupt_before_any_wait_does_not_blame_another_process():
    """Ctrl+C right after submit lands before the first lease attempt (tracker #134889 N4): nothing
    else held the session, so the reply must not say it stopped waiting for another process."""
    agent = _agent(_Db(acquired=False), _interrupt_requested=True)
    agent.clear_interrupt = lambda: None
    reply = _admit(agent).early_result["final_response"]
    assert "another Hermes process" not in reply and "not processed" in reply

    class _HeldDb(_Db):
        def acquire_session_turn_lease(self, session_id, holder, **kwargs):
            kwargs["on_wait"](0.5)  # another holder seen, then the user interrupted the wait
            return False

    agent = _agent(_HeldDb(), _interrupt_requested=True)
    agent.clear_interrupt = lambda: None
    assert "another Hermes process" in _admit(agent).early_result["final_response"]


def test_owner_interrupt_before_any_wait_does_not_blame_another_process():
    agent = _wired(_Db(acquired=False), "owner", _interrupt_requested=True)
    agent.clear_interrupt = lambda: None
    reply = _admit(agent).early_result["final_response"]
    assert "another Hermes process" not in reply and "not processed" in reply

    class _HeldDb(_Db):
        def acquire_session_turn_lease(self, session_id, holder, **kwargs):
            agent._interrupt_requested = True  # another holder seen, then the user interrupted the wait
            return False

    agent = _wired(_HeldDb(), "owner")
    agent.clear_interrupt = lambda: None
    assert "another Hermes process" in _admit(agent).early_result["final_response"]


def _active_lease(db, kind="client"):
    agent = _wired(db, kind)
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h", persistence=getattr(agent, "persistence", None))
    lease.turn_active = True
    return lease, calls


@pytest.mark.parametrize("kind", ["client", "owner"])
def test_refresh_tick_sqlite_lock_keeps_the_turn(kind):
    from persistence.core.errors import StoreContention

    db = _Db()

    def locked(session_id, holder, **kwargs):
        # The client store raises sqlite3's lock error; the owner's handle raises the contract's.
        if kind == "client":
            raise sqlite3.OperationalError("database is locked")
        raise StoreContention("database is locked")

    db.refresh_session_turn_lease = locked
    lease, calls = _active_lease(db, kind)

    # False would cancel the refresher; a lock is a missed tick, not a lost lease.
    assert lease.refresh_tick() is None
    assert calls == []
    assert lease.interrupt_message is None
    assert lease.stop.is_set() is False


class _MovingClock:
    """Wall clock that runs at real speed from a chosen instant, so the store's expiry stamps and
    the lease's deadline arithmetic see the same time while real lock waits elapse."""

    def __init__(self, at):
        self._at, self._anchor = at, time.monotonic()

    def time(self):
        return self._at + (time.monotonic() - self._anchor)

    def set(self, at):
        self._at, self._anchor = at, time.monotonic()


def test_lock_tolerance_never_outlives_the_committed_row_expiry(tmp_path, monkeypatch):
    """Real store handle + a real second-connection write lock. The local deadline never runs past the
    row's committed expiry (a renewal that waited for the lock), and a renewal blocked near expiry
    gives up and stops the turn BEFORE the row becomes reclaimable, never after a successor could
    take it. Control: the lock clearing lets the delayed renewal succeed with no interrupt."""
    from agent import turn_facade_lease
    from persistence.sqlite.authority import coordination as sqlite_coordination
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_seed import seed_session

    clock = _MovingClock(1000.0)
    fake_time = SimpleNamespace(time=clock.time, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(turn_facade_lease, "time", fake_time)
    monkeypatch.setattr(sqlite_coordination, "time", fake_time)

    path = tmp_path / "state.db"
    handle = open_persistence(profile_scope(home=tmp_path))
    seed_session(handle, "s1", "test")
    coordination = handle.coordination
    assert coordination.try_acquire_turn_lease(session_id="s1", holder="h")
    events = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: events.append((clock.time(), msg)))
    lease = DurableTurnLease(agent, _Unused(), "s1", "h",
                             expires_at=coordination.turn_lease_expires_at(session_id="s1", holder="h"),
                             persistence=handle)
    lease.turn_active = True

    def row_expiry() -> float:
        committed = coordination.turn_lease_expires_at(session_id="s1", holder="h")
        assert committed is not None, "the turn lost its lease row"
        return committed

    def write_lock():
        conn = sqlite3.connect(path, timeout=0)
        conn.execute("BEGIN IMMEDIATE")
        return conn

    # A renewal that waits on the lock still succeeds, and the deadline it adopts is not later than
    # the expiry the store committed (stamped before the wait). The lock is released only once the
    # renewal has provably been refused by it (its first retry sleep), never on a timer.
    clock.set(1060.0)
    admitted_expiry = row_expiry()
    blocker = write_lock()
    refused = threading.Event()
    holder = raw_holder(handle)
    real_retry_sleep = holder._sleep_before_write_retry

    def retry_sleep(deadline, patience_s):
        refused.set()
        return real_retry_sleep(deadline, patience_s)

    monkeypatch.setattr(holder, "_sleep_before_write_retry", retry_sleep)
    worker = threading.Thread(target=lease.refresh_tick)
    worker.start()
    assert refused.wait(10), "renewal never reached the locked write"
    blocker.rollback()
    blocker.close()
    worker.join(10)
    assert not worker.is_alive()
    assert events == []
    assert row_expiry() > admitted_expiry, "the delayed renewal must have renewed the row"
    assert lease._authority_deadline <= row_expiry()

    # Near expiry with the lock held throughout: the renewal must give up and interrupt while the
    # row is still unexpired, so no successor can reclaim it under a running turn.
    clock.set(lease._authority_deadline - 3.0)
    blocker = write_lock()
    try:
        assert lease.refresh_tick() is False
    finally:
        blocker.rollback()
        blocker.close()
    assert [msg for _, msg in events] == [
        "Session turn lease could not be refreshed; stopping to protect the transcript."
    ]
    assert events[0][0] < row_expiry()
    handle.close()


def test_interrupt_turn_only_while_active():
    agent = _agent(_Db())
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, agent._session_db, "s1", "h")
    lease._interrupt_turn("lost")  # inactive: ignored
    assert calls == [] and lease.interrupt_message is None
    lease.turn_active = True
    lease._interrupt_turn("lost")
    assert calls == ["lost"] and lease.interrupt_message == "lost"
    lease.deactivate_after_liveness_abort()
    assert lease.stop.is_set() and lease.is_turn_active() is False
