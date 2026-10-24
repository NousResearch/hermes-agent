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
    # Stamped so UI descriptors show "session busy" instead of code="unknown".
    assert admission.early_result["failure_reason"] == "session_busy"
    assert admission.early_result["failure_retryable"] is True
    assert admission.early_result["messages"] == [{"role": "user", "content": "x"}]

    agent = _agent(_Db(acquired=False), _interrupt_requested=True, _interrupt_message="stop")
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


def _active_lease(db):
    agent = _agent(db)
    calls = []
    agent.interrupt = lambda msg, **kw: calls.append(msg)
    lease = DurableTurnLease(agent, db, "s1", "h")
    lease.turn_active = True
    return lease, calls


def test_refresh_tick_sqlite_lock_keeps_the_turn():
    db = _Db()

    def locked(session_id, holder, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    db.refresh_session_turn_lease = locked
    lease, calls = _active_lease(db)

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
    """A real write lock cannot stop a turn while the row still belongs to it, even after expiry.

    Once the lock clears, the next scheduled renewal can recover the same holder's expired row;
    holder-fenced writes still prevent revival after a real takeover.
    """
    import hermes_state
    import hermes_state_compression
    from agent import turn_facade_lease

    clock = _MovingClock(1000.0)
    fake_time = SimpleNamespace(time=clock.time, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(turn_facade_lease, "time", fake_time)
    monkeypatch.setattr(hermes_state_compression, "time", fake_time)

    path = tmp_path / "state.db"
    db = hermes_state.SessionDB(path)
    db.create_session("s1", source="test")
    assert db.try_acquire_session_turn_lease("s1", "h")
    events = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: events.append((clock.time(), msg)))
    lease = DurableTurnLease(agent, db, "s1", "h", expires_at=db.session_turn_lease_expires_at("s1", "h"))
    lease.turn_active = True

    def row_expiry() -> float:
        committed = db.session_turn_lease_expires_at("s1", "h")
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
    real_retry_sleep = db._sleep_before_write_retry

    def retry_sleep(deadline, patience_s):
        refused.set()
        return real_retry_sleep(deadline, patience_s)

    monkeypatch.setattr(db, "_sleep_before_write_retry", retry_sleep)
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

    # Near expiry with the lock held: the bounded renewal gives up, then the ownership read sees
    # that the lease row still belongs to this turn. That is a missed write, not lease loss.
    current_expiry = row_expiry()
    clock.set(lease._authority_deadline - 3.0)
    blocker = write_lock()
    try:
        assert lease.refresh_tick() is None
        assert row_expiry() == current_expiry
        assert events == []
    finally:
        blocker.rollback()
        blocker.close()

    # The scheduled retry is allowed to renew after the cached deadline when no successor has
    # reclaimed the row. This is a real SQLite UPDATE, not a synthetic success result.
    clock.set(current_expiry + 5.0)
    assert lease.refresh_tick() is None
    assert events == []
    assert row_expiry() > current_expiry
    assert lease._authority_deadline <= row_expiry()
    db.close()


def test_unreadable_exhausted_lease_recheck_is_unknown_not_loss(tmp_path, monkeypatch):
    """A locked read-only lease recheck is unknown, not evidence of a takeover."""
    import hermes_state
    import hermes_state_compression
    from agent import turn_facade_lease

    clock = _MovingClock(1000.0)
    fake_time = SimpleNamespace(time=clock.time, monotonic=time.monotonic, sleep=time.sleep)
    monkeypatch.setattr(turn_facade_lease, "time", fake_time)
    monkeypatch.setattr(hermes_state_compression, "time", fake_time)
    monkeypatch.setattr(turn_facade_lease, "_REFRESH_WRITE_PATIENCE_S", 0.01)

    path = tmp_path / "state.db"
    writer = hermes_state.SessionDB(path)
    writer.create_session("s1", source="test")
    assert writer.try_acquire_session_turn_lease("s1", "h")
    writer.close()
    # Switch the disposable database after the writer has closed, then attach only the established
    # read-only SessionDB interface. The rollback-journal exclusive lock makes its bounded SELECT
    # raise SQLITE_BUSY without adding any production store API or schema.
    with sqlite3.connect(path, timeout=0) as setup:
        journal_mode = setup.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
    assert journal_mode.lower() == "delete"
    reader = hermes_state.SessionDB(path, read_only=True)

    class _LockedWriteDb:
        def session_turn_lease_expires_at(self, session_id, holder):
            return reader.session_turn_lease_expires_at(session_id, holder)

        def session_turn_lease_expires_at_bounded(self, session_id, holder):
            return reader.session_turn_lease_expires_at_bounded(session_id, holder)

        def refresh_session_turn_lease(self, session_id, holder, **kwargs):
            conn = sqlite3.connect(path, timeout=0, isolation_level=None)
            try:
                conn.execute("BEGIN IMMEDIATE")
                updated = conn.execute(
                    "UPDATE session_turn_leases SET expires_at = ? "
                    "WHERE conversation_id = ? AND holder = ?",
                    (clock.time() + LEASE_TTL_SECONDS, session_id, holder),
                ).rowcount
                conn.commit()
                return updated > 0
            finally:
                conn.close()

    events = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: events.append(msg))
    lease = DurableTurnLease(agent, _LockedWriteDb(), "s1", "h", expires_at=1000.0)
    lease.turn_active = True
    clock.set(1001.0)
    blocker = sqlite3.connect(path, timeout=0, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")
    try:
        read_started = time.monotonic()
        try:
            reader.session_turn_lease_expires_at_bounded("s1", "h")
        except (sqlite3.OperationalError, TimeoutError) as exc:
            assert any(token in str(exc).lower() for token in ("locked", "deadline"))
        else:
            raise AssertionError("exclusive lock did not make the read-only lease check unreadable")
        assert time.monotonic() - read_started < (
            hermes_state_compression._TURN_LEASE_RECHECK_TIMEOUT_S + 2.0
        ), "SQLite busy timeout exceeded the bounded ownership-read deadline"

        started = time.monotonic()
        # The bounded ownership SELECT encounters the real lock. It is unknown, not proof that the
        # row is absent or belongs to another process, so this tick returns for a later retry.
        assert lease.refresh_tick() is None
        assert events == []
        assert lease.stop.is_set() is False
    finally:
        blocker.rollback()
        blocker.close()
        reader.close()
    assert time.monotonic() - started < 2.0, "read-only ownership probe exceeded its bound"


def test_exhausted_recheck_bounds_sessiondb_lock_and_later_renews(tmp_path):
    """A fallback mutex timeout returns the tick; a later tick renews the same expired holder."""
    import hermes_state
    import hermes_state_compression

    path = tmp_path / "state.db"
    db = hermes_state.SessionDB(path)
    db.create_session("s1", source="test")
    assert db.try_acquire_session_turn_lease("s1", "h")
    db._wal_active = False
    interrupts = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: interrupts.append(msg))
    lease = DurableTurnLease(agent, db, "s1", "h", expires_at=0.0)
    lease.turn_active = True
    assert db._lock.acquire(blocking=False)
    finished = threading.Event()
    tick_results = []

    def run_tick():
        tick_results.append(lease.refresh_tick())
        finished.set()

    worker = threading.Thread(target=run_tick)
    started = time.monotonic()
    worker.start()
    try:
        # The 0.5s production bound plus scheduling margin is well below the legacy unbounded wait.
        assert finished.wait(2.5), "refresh callback blocked behind SessionDB._lock"
        assert tick_results == [None]
        assert interrupts == [] and lease.stop.is_set() is False
        recheck_timeout = getattr(hermes_state_compression, "_TURN_LEASE_RECHECK_TIMEOUT_S", 0.5)
        assert time.monotonic() - started < recheck_timeout + 2.0
    finally:
        db._lock.release()
        worker.join(10)
    assert not worker.is_alive()

    previous_expiry = db.session_turn_lease_expires_at("s1", "h")
    assert previous_expiry is not None
    assert lease.refresh_tick() is None
    renewed_expiry = db.session_turn_lease_expires_at("s1", "h")
    assert renewed_expiry is not None and renewed_expiry > previous_expiry
    assert lease._authority_deadline <= renewed_expiry
    assert interrupts == [] and lease.stop.is_set() is False
    db.close()


def test_missing_bounded_recheck_reader_still_retries_holder_fenced_renewal():
    """A store adapter without the optional reader remains on its renewal fallback path."""

    class _AdapterDb:
        def __init__(self):
            self.attempts = 0

        def refresh_session_turn_lease(self, session_id, holder, **kwargs):
            self.attempts += 1
            return True

    db = _AdapterDb()
    interrupts = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: interrupts.append(msg))
    lease = DurableTurnLease(agent, db, "s1", "h", expires_at=0.0)
    lease.turn_active = True

    assert lease.refresh_tick() is None
    assert db.attempts == 1
    assert lease._authority_deadline > 0.0
    assert interrupts == [] and lease.stop.is_set() is False


def test_expired_owned_row_retries_after_non_lock_renewal_error():
    """A failed retry does not stop an expired row that still names this holder."""

    class _RetryDb:
        def __init__(self):
            self.attempts = 0

        def session_turn_lease_expires_at(self, session_id, holder):
            return 0.0  # expired, but still held by this turn

        def session_turn_lease_expires_at_bounded(self, session_id, holder):
            return self.session_turn_lease_expires_at(session_id, holder)

        def refresh_session_turn_lease(self, session_id, holder, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise OSError("temporary store error")
            return True

    db = _RetryDb()
    interrupts = []
    agent = SimpleNamespace(session_id="s1", interrupt=lambda msg, **kw: interrupts.append(msg))
    lease = DurableTurnLease(agent, db, "s1", "h", expires_at=0.0)
    lease.turn_active = True

    assert lease.refresh_tick() is None
    assert db.attempts == 1
    assert lease._authority_deadline == 0.0, "a failed update must not extend cached authority"
    assert interrupts == []
    assert lease.stop.is_set() is False

    assert lease.refresh_tick() is None
    assert db.attempts == 2
    assert lease._authority_deadline > 0.0
    assert interrupts == []


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
