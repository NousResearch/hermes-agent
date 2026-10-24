"""Tests for the durable turn-lease orphan fix (kanban t_1915c9a3).

The defect: a turn that dies WITHOUT running its ``finally`` leaks its durable lease
forever. The refresher timer (``DurableTurnLease.refresh_tick``) is only cancelled from
that ``finally``, so it keeps renewing the row every 60s. The row never expires, the
conversation cannot be leased again, and every inbound on that session waits out the
full budget and produces nothing.

Observed 2026-10-23 on the homelab gateway: three telegram conversation leases held by
the live gateway pid, still being refreshed 91 / 45 / 32 minutes after their turns died.

The fix has two halves, both exercised here:
  1. ``agent.turn_lease_liveness`` — a process-local registry of live holders. The
     reclaim predicate (``hermes_state._compression_lock_holder_process_is_dead``) asks it
     "is the TURN behind this own-pid holder still running?" and frees the row when the
     answer is a provable no.
  2. ``DurableTurnLease.refresh_tick`` — refuses to renew a lease whose turn is gone, and
     releases the row instead, so the next inbound is not blocked.
"""
from __future__ import annotations

import os
import re
import threading
from types import SimpleNamespace

import pytest

from agent import turn_lease_liveness
from agent.turn_facade_lease import DurableTurnLease, admit_durable_turn_lease


@pytest.fixture(autouse=True)
def _clean_registry():
    turn_lease_liveness._reset_for_tests()
    yield
    turn_lease_liveness._reset_for_tests()


def _own_pid_holder(token: str = "turn-1") -> str:
    """A holder shaped exactly like the real one: our own pid, a namespace token, a turn id."""
    return f"pid={os.getpid()}:pidns=4026531836:turn={token}:platform=telegram"


def _dead_thread_ident() -> int:
    """An ident of a thread that no longer exists.

    Started and joined: the ident is real (so it cannot collide with a live thread's)
    but the thread is gone, which is exactly the state a turn that died mid-flight
    leaves behind.
    """
    ident_box: list = []
    started = threading.Event()

    def _run():
        ident_box.append(threading.current_thread().ident)
        started.set()

    thread = threading.Thread(target=_run, name="turn-lease-orphan-probe", daemon=True)
    thread.start()
    started.wait(5.0)
    thread.join(5.0)
    assert ident_box, "probe thread did not start"
    return ident_box[0]


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
        self.events.append(("refresh", session_id, holder))
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


# ── registry semantics ─────────────────────────────────────────────────────────────


def test_own_pid_holder_never_registered_is_unknown_not_dead():
    """Absence from the registry is NOT proof of death.

    A lease written by another build, or one whose registration raced the probe, must
    keep the pre-existing "own pid is not reclaimable" guarantee. Only an EXPLICIT
    release proves the turn is gone.
    """
    holder = _own_pid_holder()
    assert turn_lease_liveness.is_holder_live(holder) is None


def test_released_own_pid_holder_is_provably_dead():
    """A holder whose turn explicitly exited is an orphan."""
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    turn_lease_liveness.unregister_holder(holder)
    assert turn_lease_liveness.is_holder_live(holder) is False


def test_registered_own_pid_holder_is_live():
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    assert turn_lease_liveness.is_holder_live(holder) is True


def test_re_registering_a_released_holder_makes_it_live_again():
    """A holder string can be reused across turns (same relay_turn_id); re-registering
    must clear the released mark, or the new turn's lease would be stolen."""
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    turn_lease_liveness.unregister_holder(holder)
    turn_lease_liveness.register_holder(holder, "s1")
    assert turn_lease_liveness.is_holder_live(holder) is True


def test_unregistered_foreign_pid_is_unknown_not_dead():
    """A foreign pid is the kernel's business; the registry must not speak for it."""
    holder = "pid=999999:pidns=4026531836:turn=x:platform=telegram"
    assert turn_lease_liveness.is_holder_live(holder) is None


def test_unstructured_holder_is_unknown():
    """A holder with no pid (e.g. a test fixture) cannot be judged."""
    assert turn_lease_liveness.is_holder_live("fixture-client-turn") is None
    assert turn_lease_liveness.is_holder_live("") is None


# ── the reclaim predicate ──────────────────────────────────────────────────────────


def test_reclaim_predicate_keeps_an_unregistered_own_pid_lease():
    """The pre-existing guarantee: an own-pid lease is not reclaimable on a guess.

    ``tests/hermes_state/test_session_turn_lease.py`` pins this. The orphan fix must not
    weaken it: only an EXPLICIT release proves the turn is gone.
    """
    from hermes_state import _compression_lock_holder_process_is_dead

    holder = _own_pid_holder()
    assert _compression_lock_holder_process_is_dead(holder) is False


def test_reclaim_predicate_frees_a_released_own_pid_lease():
    """A holder whose turn explicitly exited is an orphan — the row may be reclaimed."""
    from hermes_state import _compression_lock_holder_process_is_dead

    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    turn_lease_liveness.unregister_holder(holder)
    assert _compression_lock_holder_process_is_dead(holder) is True


def test_reclaim_predicate_keeps_a_live_own_pid_lease():
    """A turn that IS running must never have its lease stolen."""
    from hermes_state import _compression_lock_holder_process_is_dead

    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    assert _compression_lock_holder_process_is_dead(holder) is False


def test_reclaim_predicate_still_defers_to_the_kernel_for_foreign_pids():
    """A foreign pid that is genuinely gone is still reclaimed by the kernel probe."""
    from hermes_state import _compression_lock_holder_process_is_dead

    holder = "pid=999999:pidns=4026531836:turn=x:platform=telegram"
    assert _compression_lock_holder_process_is_dead(holder) is True


# ── the refresher ──────────────────────────────────────────────────────────────────


def test_refresher_refuses_to_renew_when_the_turn_thread_is_gone():
    """The heartbeat must not keep a dead turn's lease alive — and must release the row.

    This is the incident's exact shape: the turn's thread died without running the
    facade's finally, so the timer is the only surviving artifact of that turn.
    """
    db = _Db()
    agent = _agent(db)
    holder = _own_pid_holder()
    lease = DurableTurnLease(agent, db, "s1", holder)
    # The turn ran on a thread that no longer exists.
    lease._turn_thread_ident = _dead_thread_ident()

    assert lease.refresh_tick() is False, "an orphaned refresher must stop itself"
    assert ("refresh", "s1", holder) not in db.events, "an orphaned lease must not be renewed"
    assert ("release", "s1", holder) in db.events, "the orphaned row must be released"


def test_refresher_renews_while_the_turn_thread_is_alive():
    """A live turn keeps its lease alive exactly as before."""
    db = _Db()
    agent = _agent(db)
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    lease = DurableTurnLease(agent, db, "s1", holder)
    lease._turn_thread_ident = threading.current_thread().ident

    assert lease.refresh_tick() is None, "a live refresher must keep rescheduling"
    assert ("refresh", "s1", holder) in db.events
    assert ("release", "s1", holder) not in db.events


def test_refresher_refuses_to_renew_a_released_holder():
    """A holder whose turn exited is an orphan even if its thread is somehow still around."""
    db = _Db()
    agent = _agent(db)
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    turn_lease_liveness.unregister_holder(holder)
    lease = DurableTurnLease(agent, db, "s1", holder)
    lease._turn_thread_ident = threading.current_thread().ident

    assert lease.refresh_tick() is False
    assert ("refresh", "s1", holder) not in db.events
    assert ("release", "s1", holder) in db.events


def test_refresher_leaves_unstructured_holders_alone():
    """A holder with no pid (no liveness signal) keeps the pre-fix behaviour."""
    db = _Db()
    agent = _agent(db)
    lease = DurableTurnLease(agent, db, "s1", "fixture-client-turn")
    lease._turn_thread_ident = threading.current_thread().ident

    assert lease.refresh_tick() is None
    assert ("refresh", "s1", "fixture-client-turn") in db.events


# ── the admission bracket ──────────────────────────────────────────────────────────


def test_admission_registers_the_holder_before_writing_the_row():
    """The holder must be registered BEFORE the row exists.

    Otherwise a concurrent acquirer can observe "own pid, not registered" and free a
    lease that a live turn is about to take.
    """
    db = _Db()
    agent = _agent(db)
    seen: list = []

    original_acquire = db.acquire_session_turn_lease

    def _acquire(session_id, holder, **kwargs):
        seen.append(turn_lease_liveness.is_holder_live(holder))
        return original_acquire(session_id, holder, **kwargs)

    db.acquire_session_turn_lease = _acquire

    admission = admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "telegram"},
        conversation_history=[],
    )

    assert admission.lease is not None
    assert seen == [True], "the holder must already be live when the row is written"


def test_failed_admission_unregisters_the_holder():
    """An admission that never got the lease must not leave a live-looking holder."""
    db = _Db(acquired=False)
    agent = _agent(db)

    admission = admit_durable_turn_lease(
        agent,
        session_id="s1",
        relay_turn_id="s1:t:abcd",
        task_context={"session_id": "s1", "task_id": "t", "platform": "telegram"},
        conversation_history=[],
    )

    assert admission.lease is None
    assert admission.early_result is not None
    assert turn_lease_liveness.live_holder_count() == 0


def test_release_unregisters_the_holder():
    """The normal exit path clears the registration so the row is reclaimable."""
    db = _Db()
    agent = _agent(db)
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    lease = DurableTurnLease(agent, db, "s1", holder)

    lease.release()

    assert turn_lease_liveness.is_holder_live(holder) is False


def test_release_unregisters_even_when_the_db_write_fails():
    """A failed release must not leave the holder registered — that would keep a dead
    turn's lease alive forever, which is the exact failure this fix exists to stop."""

    class _FailingDb(_Db):
        def release_session_turn_lease(self, session_id, holder):
            raise RuntimeError("db down")

    db = _FailingDb()
    agent = _agent(db)
    holder = _own_pid_holder()
    turn_lease_liveness.register_holder(holder, "s1")
    lease = DurableTurnLease(agent, db, "s1", holder)

    lease.release()

    assert turn_lease_liveness.is_holder_live(holder) is False


# ── end-to-end: the orphaned lease no longer blocks the next inbound ───────────────


def test_a_released_lease_does_not_block_the_next_acquisition(tmp_path, monkeypatch):
    """The user-visible property from the incident.

    A lease row left behind by a dead turn must not stop the next inbound from being
    serviced on that session. Before the fix the row was renewed forever and every
    inbound waited out the budget.
    """
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB()
    holder = _own_pid_holder("dead-turn")

    # A dead turn's row: our pid, and the turn explicitly exited (registered, released).
    turn_lease_liveness.register_holder(holder, "s1")
    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=300.0) is True
    turn_lease_liveness.unregister_holder(holder)

    # The next inbound must be able to take the lease immediately.
    new_holder = _own_pid_holder("live-turn")
    turn_lease_liveness.register_holder(new_holder, "s1")
    assert db.try_acquire_session_turn_lease("s1", new_holder, ttl_seconds=300.0) is True

    row = db._read_one(
        "SELECT holder FROM session_turn_leases WHERE conversation_id = ?", ("s1",)
    )
    assert row is not None and row[0] == new_holder, "the orphaned row must have been replaced"


def test_an_unregistered_own_pid_lease_still_blocks(tmp_path, monkeypatch):
    """The conservative case: a holder we cannot vouch for keeps its row.

    "Unknown" is not "dead" — the pre-existing mutual-exclusion guarantee must survive.
    """
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB()
    holder = _own_pid_holder("unregistered")

    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=300.0) is True
    other = _own_pid_holder("other")
    turn_lease_liveness.register_holder(other, "s2")
    assert db.try_acquire_session_turn_lease("s1", other, ttl_seconds=300.0) is False


def test_a_live_lease_still_blocks_a_second_acquisition(tmp_path, monkeypatch):
    """The fix must not weaken mutual exclusion between two live turns."""
    from hermes_state import SessionDB

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = SessionDB()
    holder = _own_pid_holder("live-turn")
    turn_lease_liveness.register_holder(holder, "s1")

    assert db.try_acquire_session_turn_lease("s1", holder, ttl_seconds=300.0) is True
    other = _own_pid_holder("other-live-turn")
    turn_lease_liveness.register_holder(other, "s2")
    assert db.try_acquire_session_turn_lease("s1", other, ttl_seconds=300.0) is False
