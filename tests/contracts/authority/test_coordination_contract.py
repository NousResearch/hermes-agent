"""Executable contract suite for the compression lock and the handoff verbs of ``CoordinationRepository``.

Backend cases run a real assembled backend from the shared fixture. Contested cases require ``transactional_claims``, race contenders on threads (``run_concurrently``), collect every outcome and assert the exact partition (one winner, typed losers). The cross-process cases at the end re-prove the partitions under real process contention on backends declaring ``cross_process``. The delegation ledger is pinned in ``test_delegation_ledger_contract.py``, the obligation ledger in ``test_delivery_ledger_contract.py``.
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pytest

from persistence import DelegationId, PersistenceError, SessionId, StoreReplaced
from persistence.authority.coordination import HandoffState, HandoffStatus, OwnerStamp, PendingHandoff
from persistence.conversation.sessions import NewSession
from tests.contracts._concurrency import run_concurrently
from tests.contracts._scopes import profile_scope
from tests.contracts._waiting import wait_past
from tests.contracts.conftest import require_capability

N_RACERS = 6
S1, S2, SX, SY = (SessionId(f"compress-{name}") for name in ("s1", "s2", "sx", "sy"))


# ── compression lock ──────────────────────────────────────────────────────────


def test_compression_lock_lifecycle_single_winner_and_owner_release(backend, scope_a):
    coord = backend.open(scope_a).coordination
    assert coord.get_compression_lock_holder(session_id=S1) is None
    assert coord.try_acquire_compression_lock(session_id=S1, holder="holder-a") is True
    # CONTRACT: a second holder is refused while the lock is live; a live holder's re-acquire is True.
    assert coord.try_acquire_compression_lock(session_id=S1, holder="holder-b") is False
    assert coord.try_acquire_compression_lock(session_id=S1, holder="holder-a") is True
    assert coord.get_compression_lock_holder(session_id=S1) == "holder-a"
    assert coord.try_acquire_compression_lock(session_id=S2, holder="holder-b") is True
    # CONTRACT: release is holder-qualified and idempotent.
    coord.release_compression_lock(session_id=S1, holder="holder-b")
    assert coord.get_compression_lock_holder(session_id=S1) == "holder-a"
    coord.release_compression_lock(session_id=S1, holder="holder-a")
    assert coord.get_compression_lock_holder(session_id=S1) is None
    assert coord.try_acquire_compression_lock(session_id=S1, holder="holder-b") is True
    coord.release_compression_lock(session_id=S1, holder="holder-b")
    coord.release_compression_lock(session_id=S1, holder="holder-b")
    assert coord.get_compression_lock_holder(session_id=S1) is None


def test_compression_lock_expiry_reclaim_and_owner_revive_via_refresh(backend, scope_a):
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id=SX, holder="holder-a", ttl_seconds=0.01) is True
    wait_past(time.time() + 0.01)
    # CONTRACT: an expired lock reads as no holder; its holder revives it by refresh (holder alone decides); another holder's refresh is refused.
    assert coord.get_compression_lock_holder(session_id=SX) is None
    assert coord.refresh_compression_lock(session_id=SX, holder="holder-a", ttl_seconds=300.0) is True
    assert coord.get_compression_lock_holder(session_id=SX) == "holder-a"
    assert coord.refresh_compression_lock(session_id=SX, holder="holder-b", ttl_seconds=300.0) is False
    assert coord.try_acquire_compression_lock(session_id=SX, holder="holder-c") is False
    # CONTRACT: another claimant reclaims an expired lock in its own acquisition.
    assert coord.try_acquire_compression_lock(session_id=SY, holder="holder-a", ttl_seconds=0.01) is True
    wait_past(time.time() + 0.01)
    assert coord.try_acquire_compression_lock(session_id=SY, holder="holder-b") is True
    assert coord.get_compression_lock_holder(session_id=SY) == "holder-b"


def test_compression_lock_empty_arguments(backend, scope_a):
    # CONTRACT: an empty session id is False / None / a no-op without a write; an empty holder is admitted (only the id is checked) but cannot refresh.
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_compression_lock(session_id=SessionId(""), holder="holder-x") is False
    assert coord.refresh_compression_lock(session_id=SessionId(""), holder="holder-x") is False
    coord.release_compression_lock(session_id=SessionId(""), holder="holder-x")
    assert coord.get_compression_lock_holder(session_id=SessionId("")) is None
    assert coord.try_acquire_compression_lock(session_id=S1, holder="") is True
    assert coord.refresh_compression_lock(session_id=S1, holder="") is False


def test_compression_lock_verbs_never_raise_on_a_torn_down_store(backend, scope_a):
    # CONTRACT: a store failure is logged and answers the empty value: False for acquire and refresh, nothing for release.
    coord = backend.open(scope_a).coordination
    backend.close()
    assert coord.try_acquire_compression_lock(session_id=S1, holder="h") is False
    assert coord.refresh_compression_lock(session_id=S1, holder="h") is False
    coord.release_compression_lock(session_id=S1, holder="h")


def test_compression_lock_writes_raise_on_a_replaced_store(backend, scope_a):
    # CONTRACT: a store replaced under the handle raises from each lock write; the holder read still answers from the store the handle opened.
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=S1, source="cli"))
    assert store.coordination.try_acquire_compression_lock(session_id=S1, holder="h") is True
    backend.replace_store(scope_a)
    coord = store.coordination
    with pytest.raises(StoreReplaced):
        coord.try_acquire_compression_lock(session_id=S1, holder="h")
    with pytest.raises(StoreReplaced):
        coord.refresh_compression_lock(session_id=S1, holder="h")
    with pytest.raises(StoreReplaced):
        coord.release_compression_lock(session_id=S1, holder="h")
    assert coord.get_compression_lock_holder(session_id=S1) == "h"


def test_compression_lock_holder_read_raises_on_a_torn_down_store(backend, scope_a):
    # CONTRACT: the holder read catches nothing, so a store failure raises instead of reading as no holder.
    coord = backend.open(scope_a).coordination
    backend.close()
    with pytest.raises((PersistenceError, RuntimeError)):
        coord.get_compression_lock_holder(session_id=S1)


def test_compression_lock_contested_acquire_partitions_one_winner(backend, scope_a):
    coord = backend.open(scope_a).coordination
    require_capability(backend, "transactional_claims")
    storm = SessionId("compress-storm")
    outcomes = run_concurrently(*[lambda i=i: coord.try_acquire_compression_lock(session_id=storm, holder=f"racer-{i}") for i in range(N_RACERS)])
    assert outcomes.count(True) == 1 and outcomes.count(False) == N_RACERS - 1
    winner = f"racer-{outcomes.index(True)}"
    assert coord.get_compression_lock_holder(session_id=storm) == winner
    coord.release_compression_lock(session_id=storm, holder=winner)
    assert coord.get_compression_lock_holder(session_id=storm) is None


def test_compression_lock_two_profile_isolation(backend, scope_a, scope_b):
    coord_a, coord_b = backend.open(scope_a).coordination, backend.open(scope_b).coordination
    shared = SessionId("compress-shared")
    assert coord_a.try_acquire_compression_lock(session_id=shared, holder="holder-a") is True
    assert coord_b.try_acquire_compression_lock(session_id=shared, holder="holder-b") is True
    assert coord_b.refresh_compression_lock(session_id=shared, holder="holder-a") is False
    coord_b.release_compression_lock(session_id=shared, holder="holder-a")
    assert coord_a.get_compression_lock_holder(session_id=shared) == "holder-a"
    assert coord_b.get_compression_lock_holder(session_id=shared) == "holder-b"


# ── handoffs ──────────────────────────────────────────────────────────────────


def test_handoff_transitions_and_cas_scoped_fail(backend, scope_a):
    store_a = backend.open(scope_a)
    coord = store_a.coordination
    sid = SessionId("sess-ho1")
    store_a.sessions.create(NewSession(sid, "cli"))
    store_a.sessions.set_title(sid, "Handoff chat")
    assert coord.get_handoff(session_id=sid) is None
    assert coord.has_pending_handoffs() is False
    assert coord.request_handoff(session_id=sid, platform="telegram") is True
    assert coord.get_handoff(session_id=sid) == HandoffState(state=HandoffStatus.PENDING, platform="telegram", error=None)
    started = store_a.sessions.get(sid).started_at
    # CONTRACT: the dispatcher's read carries id, platform, title and started_at.
    assert coord.list_pending_handoffs() == (PendingHandoff(session_id=sid, platform="telegram", title="Handoff chat", started_at=started),)
    assert coord.has_pending_handoffs() is True
    assert coord.request_handoff(session_id=sid, platform="discord") is False
    assert coord.claim_handoff(session_id=sid) is True
    assert coord.get_handoff(session_id=sid).state is HandoffStatus.RUNNING
    assert coord.claim_handoff(session_id=sid) is False
    assert coord.list_pending_handoffs() == () and coord.has_pending_handoffs() is False
    # CONTRACT: a fail scoped to pending cannot steal a running row.
    assert coord.fail_handoff(session_id=sid, error="gave up", only_states=(HandoffStatus.PENDING,)) is False
    assert coord.get_handoff(session_id=sid).state is HandoffStatus.RUNNING
    # CONTRACT: an empty ``only_states`` is no condition: the fail lands.
    assert coord.fail_handoff(session_id=sid, error="dispatch died", only_states=()) is True
    assert coord.get_handoff(session_id=sid) == HandoffState(state=HandoffStatus.FAILED, platform="telegram", error="dispatch died")
    assert coord.request_handoff(session_id=sid, platform="telegram") is True
    assert coord.claim_handoff(session_id=sid) is True
    coord.complete_handoff(session_id=sid)
    assert coord.get_handoff(session_id=sid) == HandoffState(state=HandoffStatus.COMPLETED, platform="telegram", error=None)
    assert coord.request_handoff(session_id=sid, platform="telegram") is True
    never = SessionId("sess-never")
    assert coord.request_handoff(session_id=never, platform="telegram") is False
    assert coord.claim_handoff(session_id=never) is False
    assert coord.fail_handoff(session_id=never, error="nope") is False
    assert coord.get_handoff(session_id=never) is None


def test_pending_handoffs_list_oldest_first(backend, scope_a):
    store_a = backend.open(scope_a)
    for sid in ("ho-a", "ho-b"):
        store_a.sessions.create(NewSession(SessionId(sid), "cli"))
        wait_past(time.time())
    for sid in ("ho-b", "ho-a"):
        assert store_a.coordination.request_handoff(session_id=SessionId(sid), platform="slack") is True
    assert [item.session_id for item in store_a.coordination.list_pending_handoffs()] == ["ho-a", "ho-b"]


def test_handoff_contested_claim_partitions_one_winner_and_rearms(backend, scope_a):
    store_a = backend.open(scope_a)
    coord = store_a.coordination
    require_capability(backend, "transactional_claims")
    sid = SessionId("sess-ho-storm")
    store_a.sessions.create(NewSession(sid, "cli"))
    for _round in range(3):
        assert coord.request_handoff(session_id=sid, platform="telegram") is True
        outcomes = run_concurrently(*[lambda: coord.claim_handoff(session_id=sid) for _ in range(N_RACERS)])
        assert sorted(outcomes) == [False] * (N_RACERS - 1) + [True]
        coord.complete_handoff(session_id=sid)
        assert coord.get_handoff(session_id=sid).state is HandoffStatus.COMPLETED


def test_handoff_two_profile_isolation(backend, scope_a, scope_b):
    store_a, store_b = backend.open(scope_a), backend.open(scope_b)
    sid = SessionId("sess-shared")
    store_a.sessions.create(NewSession(sid, "cli"))
    store_b.sessions.create(NewSession(sid, "cli"))
    assert store_a.coordination.request_handoff(session_id=sid, platform="telegram") is True
    assert store_b.coordination.request_handoff(session_id=sid, platform="discord") is True
    assert store_a.coordination.get_handoff(session_id=sid).platform == "telegram"
    assert store_b.coordination.get_handoff(session_id=sid).platform == "discord"
    assert store_b.coordination.claim_handoff(session_id=sid) is True
    assert store_a.coordination.get_handoff(session_id=sid).state is HandoffStatus.PENDING
    assert [item.session_id for item in store_a.coordination.list_pending_handoffs()] == [sid]
    assert store_b.coordination.list_pending_handoffs() == ()
    assert backend.open(profile_scope("profile-never-requested")).coordination.get_handoff(session_id=sid) is None


def test_handoff_reads_never_raise_on_a_torn_down_store(backend, scope_a):
    # CONTRACT: the handoff reads swallow every error.
    coord = backend.open(scope_a).coordination
    backend.close()
    assert coord.get_handoff(session_id=SessionId("x")) is None
    assert coord.list_pending_handoffs() == ()


# ── cross-process contested claims (re-proof of the single-winner partitions) ─

_XP_CTX = multiprocessing.get_context("spawn")
_XP_RACERS = 4
_XP_BARRIER_TIMEOUT_S = 120.0
_XP_JOIN_TIMEOUT_S = 240.0


def _xp_backend(root, profile, lock):
    """Child-side assembly over the parent's root; construction serialized."""
    from tests.contracts._sqlite_seams import ContractSQLiteBackend

    scope = profile_scope(profile)
    with lock:
        backend = ContractSQLiteBackend({}, root=Path(root))
        backend.session_db(scope)
    return backend, scope


def _xp_child(kind, root, profile, key, idx, lock, start, queue):
    try:
        backend, scope = _xp_backend(root, profile, lock)
        start.wait(timeout=_XP_BARRIER_TIMEOUT_S)
        coord = backend.open(scope).coordination
        if kind == "lock":
            won = coord.try_acquire_compression_lock(session_id=SessionId(key), holder=f"xp-holder-{idx}")
        elif kind == "handoff":
            won = coord.claim_handoff(session_id=SessionId(key))
        else:
            won = coord.claim_delivery(delegation_id=DelegationId(key), claim_id=f"xp-claim-{idx}")
        queue.put(("result", idx, bool(won)))
        backend.close()
    except BaseException as exc:
        queue.put(("error", type(exc).__name__, repr(exc)))
        raise


def _xp_race(static_args):
    """Spawn one child per racer, release them together, drain results, join bounded."""
    lock = _XP_CTX.Lock()
    start = _XP_CTX.Barrier(_XP_RACERS)
    queue = _XP_CTX.Queue()
    procs = [_XP_CTX.Process(target=_xp_child, args=(*static_args, idx, lock, start, queue)) for idx in range(_XP_RACERS)]
    try:
        for proc in procs:
            proc.start()
        results = []
        deadline = time.monotonic() + _XP_JOIN_TIMEOUT_S
        while len(results) < _XP_RACERS and time.monotonic() < deadline:
            try:
                results.append(queue.get(timeout=1.0))
            except Exception:
                if not any(p.is_alive() for p in procs):
                    break
        for proc in procs:
            proc.join(timeout=max(1.0, deadline - time.monotonic()))
        exitcodes = [proc.exitcode for proc in procs]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10.0)
    assert [r for r in results if r and r[0] == "error"] == []
    assert len(results) == _XP_RACERS
    assert exitcodes == [0] * _XP_RACERS
    return results


def _xp_root(backend) -> str:
    require_capability(backend, "cross_process")
    require_capability(backend, "transactional_claims")
    if not isinstance(backend, pytest.importorskip("persistence.sqlite").SQLiteBackend):
        pytest.skip("the multiprocess re-proof binds to the SQLite assembly (children re-assemble it by import)")
    return str(backend.root)


def test_compression_lock_contested_acquire_across_processes(backend, scope_a):
    coord = backend.open(scope_a).coordination
    root = _xp_root(backend)
    results = _xp_race(("lock", root, scope_a.name, "xp-lock"))
    winners = [r[1] for r in results if r[2]]
    assert len(winners) == 1
    assert coord.get_compression_lock_holder(session_id=SessionId("xp-lock")) == f"xp-holder-{winners[0]}"


def test_handoff_contested_claim_across_processes(backend, scope_a):
    store_a = backend.open(scope_a)
    root = _xp_root(backend)
    sid = SessionId("sess-xp-handoff")
    store_a.sessions.create(NewSession(sid, "cli"))
    assert store_a.coordination.request_handoff(session_id=sid, platform="telegram") is True
    results = _xp_race(("handoff", root, scope_a.name, str(sid)))
    assert sorted(r[2] for r in results) == [False] * (_XP_RACERS - 1) + [True]
    assert store_a.coordination.get_handoff(session_id=sid).state is HandoffStatus.RUNNING


def test_delivery_contested_claim_across_processes(backend, scope_a):
    coord = backend.open(scope_a).coordination
    root = _xp_root(backend)
    did = DelegationId("deleg-xp")
    coord.record_delegation_dispatch(delegation_id=did, origin_session="k", origin_ui_session_id="", parent_session_id=None,
                                     dispatched_at=time.time(), owner=OwnerStamp(pid=1, started_at=1), task={})
    coord.record_delegation_completion(delegation_id=did, state="completed", completed_at=time.time(), event={}, result={})
    results = _xp_race(("delivery", root, scope_a.name, str(did)))
    winners = [r[1] for r in results if r[2]]
    assert len(winners) == 1
    assert coord.get_delegation(delegation_id=did).delivery_attempts == 1
    assert coord.complete_delivery(delegation_id=did, claim_id=f"xp-claim-{winners[0]}") is True
