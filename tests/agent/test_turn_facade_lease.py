"""Unit tests for agent.turn_facade_lease (admission + lease bracket)."""
import sqlite3
import threading
import time
from types import SimpleNamespace

from agent.turn_facade_lease import (
    LEASE_TTL_SECONDS,
    DurableTurnLease,
    admit_durable_turn_lease,
)


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


def test_no_lease_without_durable_row_or_when_persist_disabled():
    seed = [{"role": "user", "content": "hi"}]
    admission = _admit(_agent(_Db(exists=False)), seed)
    assert admission.lease is None and admission.early_result is None
    assert admission.conversation_history is seed

    db = _Db()
    admission = _admit(_agent(db, _persist_disabled=True), seed)
    assert admission.lease is None and db.events == []


def test_admission_sets_holder_attrs_and_release_clears_them(monkeypatch):
    monkeypatch.setattr(
        "agent.turn_liveness.resolve_turn_liveness_settings", lambda cfg: (None, 1.0)
    )
    db = _Db()
    agent = _agent(db)
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


def test_timeout_and_interrupt_early_results():
    agent = _agent(_Db(acquired=False))
    admission = _admit(agent, [{"role": "user", "content": "x"}])
    assert admission.lease is None
    assert admission.early_result["failed"] is True
    assert admission.early_result["error"] == "session_turn_lease_timeout:s1"
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
    agent.clear_interrupt = lambda: None
    admission = _admit(agent)
    assert admission.early_result["interrupted"] is True
    assert admission.early_result["interrupt_message"] == "stop"


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


# ── refresh_tick resilience (#turn-lease-fix) ─────────────────────────────
# A lease-refresh that fails because another process holds the state.db write
# lock must NOT kill a still-owned turn — the write-side fence (turn lease lost
# on the next append) is the correctness boundary. Interruption is reserved for
# verifiable lease loss (holder changed / row gone) or an unreachable store.

class _ContendedDb(_Db):
    """Fake store whose refresh raises a writable-lock error until armed otherwise."""

    def __init__(self, owner_holder=None, refresh_raises=True, owner_probe_raises=False):
        super().__init__()
        self.owner_holder = owner_holder  # None -> probe reports the lease as absent
        self.refresh_raises = refresh_raises
        self.owner_probe_raises = owner_probe_raises

    def refresh_session_turn_lease(self, session_id, holder, **kwargs):
        if self.refresh_raises:
            raise sqlite3.OperationalError(
                "database is locked (another Hermes process held the state.db write lock)"
            )
        return super().refresh_session_turn_lease(session_id, holder, **kwargs)

    def get_session_turn_lease_owner(self, session_id):
        if self.owner_probe_raises:
            raise sqlite3.OperationalError("disk I/O error")
        if self.owner_holder is None:
            return None
        return (self.owner_holder, time.time() + 300.0)


def _active_lease(agent, db):
    agent.interrupt = lambda msg, **kw: agent.statuses.append(("interrupt", msg))
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    return lease


def test_refresh_tick_contention_does_not_interrupt_owned_turn():
    """A 'database is locked' refresh failure with ownership intact continues the turn."""
    db = _ContendedDb(owner_holder="h")
    agent = _agent(db)
    lease = _active_lease(agent, db)
    assert lease.refresh_tick() is None  # timer stays alive
    assert lease.interrupt_message is None
    assert not lease.stop.is_set()
    assert agent.statuses == []
    lease.stop_refresher()


def test_refresh_tick_interrupts_when_holder_changed():
    """A refresh failure PLUS read-verify showing a different holder interrupts."""
    db = _ContendedDb(owner_holder="someone-else")
    agent = _agent(db)
    lease = _active_lease(agent, db)
    assert lease.refresh_tick() is False  # timer stops
    assert lease.interrupt_message is not None
    assert ("interrupt", lease.interrupt_message) in agent.statuses
    lease.stop_refresher()


def test_refresh_tick_interrupts_when_lease_row_gone():
    """A refresh failure PLUS absent lease row interrupts (lease no longer exists)."""
    db = _ContendedDb(owner_holder=None)
    agent = _agent(db)
    lease = _active_lease(agent, db)
    assert lease.refresh_tick() is False
    assert lease.interrupt_message is not None
    lease.stop_refresher()


def test_refresh_tick_interrupts_when_ownership_unverifiable():
    """Fail closed: an unreachable store (owner probe raises) must not run unsynchronized."""
    db = _ContendedDb(owner_holder="h", owner_probe_raises=True)
    agent = _agent(db)
    lease = _active_lease(agent, db)
    assert lease.refresh_tick() is False
    assert lease.interrupt_message is not None
    lease.stop_refresher()


def test_refresh_tick_refresh_false_still_interrupts():
    """A cleanly-returned False (holder-qualified UPDATE matched 0 rows) still interrupts."""
    db = _ContendedDb(owner_holder="h", refresh_raises=False)
    db.refresh_result = False

    def refresh_false(*a, **k):
        return db.refresh_result

    db.refresh_session_turn_lease = refresh_false
    agent = _agent(db)
    lease = _active_lease(agent, db)
    assert lease.refresh_tick() is False
    assert lease.interrupt_message is not None
    lease.stop_refresher()
