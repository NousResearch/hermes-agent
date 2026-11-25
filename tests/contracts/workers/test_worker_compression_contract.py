"""Contract suite for the worker compression state (``persistence.workers.compression.WorkerCompressionRepository``).

The worker writes through its receipts and each check reads the state through the non-worker verb that owns it (coordination lock, messages cooldown and watermark, lineage counters and reopen), so an operation that wrote somewhere private, or whose replay re-executed, fails here.
"""

from __future__ import annotations

import time

import pytest

from persistence import AuthorityEpoch, SessionId
from persistence.conversation.messages import ConversationPeer
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from persistence.core.errors import CooldownRestoreRefused
from persistence.workers.compression import WorkerCooldown, WorkerCooldownSnapshot
from persistence.workers.executions import WorkerExecutionId, WorkerTurnLease
from tests.contracts._reference._authority import InjectedFailure, authority_state
from tests.contracts._reference.workers.receipt import WRITE_FAILPOINT
from tests.contracts._waiting import wait_past
from tests.contracts.conftest import require_capability, require_failpoints
from tests.contracts.workers._support import SESSION, begin_epoch, ident, last_sequence, refused, register, worker

HOLDER = "compression-holder-a"
T1 = 2_000_000_000.0
T0 = 1_900_000_000.0


def _setup(backend, scope, **fields):
    require_capability(backend, "transactional_claims")
    return worker(backend, scope, **fields)


def _call(store, epoch, sequence, verb, **kwargs):
    return getattr(store.worker_executions, verb)(**ident(epoch, sequence=sequence), **kwargs)


def _holder(store, session_id=SESSION):
    return store.coordination.get_compression_lock_holder(session_id=SessionId(str(session_id)))


# ── compression lock ──────────────────────────────────────────────────────────


def test_compression_lock_operations_share_the_coordination_lock(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    coordination = store.coordination
    assert _call(store, epoch, 1, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER, 300.0)) is True
    assert _holder(store) == HOLDER
    assert coordination.try_acquire_compression_lock(session_id=SESSION, holder="other", ttl_seconds=300.0) is False
    assert _call(store, epoch, 2, "compression_lock_holder") == HOLDER
    assert _call(store, epoch, 3, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER, 300.0)) is True
    assert _call(store, epoch, 4, "renew_compression_lock", lease=WorkerTurnLease(HOLDER, 60)) is True
    assert _call(store, epoch, 5, "renew_compression_lock", lease=WorkerTurnLease("other", 60)) is False
    _call(store, epoch, 6, "cleanup_compression_lock", target_session_id=SESSION, holder="other")
    assert _holder(store) == HOLDER
    _call(store, epoch, 7, "cleanup_compression_lock", target_session_id=SESSION, holder=HOLDER)
    assert _holder(store) is None
    assert _call(store, epoch, 8, "compression_lock_holder") is None
    # A live lock held by someone else: the worker's acquire loses and leaves it alone.
    assert coordination.try_acquire_compression_lock(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    assert _call(store, epoch, 9, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER, 300.0)) is False
    assert (_call(store, epoch, 10, "compression_lock_holder"), _holder(store)) == ("other", "other")
    assert last_sequence(backend, scope_a, epoch) == 10


def test_compression_lock_replay_returns_the_recorded_outcome(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    lease = WorkerTurnLease(HOLDER, 300.0)
    assert _call(store, epoch, 1, "acquire_compression_lock", lease=lease) is True
    assert _call(store, epoch, 2, "compression_lock_holder") == HOLDER
    store.coordination.release_compression_lock(session_id=SESSION, holder=HOLDER)
    assert store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="other", ttl_seconds=300.0) is True
    # Exact duplicates replay the recorded outcomes without re-deciding against the new holder.
    assert _call(store, epoch, 1, "acquire_compression_lock", lease=lease) is True
    assert _call(store, epoch, 2, "compression_lock_holder") == HOLDER
    assert _holder(store) == "other"
    with refused("admission_conflict"):
        _call(store, epoch, 1, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER, 60.0))
    with refused("admission_conflict"):
        _call(store, epoch, 2, "cleanup_compression_lock", target_session_id=SESSION, holder=HOLDER)
    assert last_sequence(backend, scope_a, epoch) == 2


def test_compression_lock_renew_extends_the_expiry_even_after_it_lapsed(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    assert _call(store, epoch, 1, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER, 0.3)) is True
    first_expiry = time.time() + 0.3
    assert _call(store, epoch, 2, "renew_compression_lock", lease=WorkerTurnLease(HOLDER, 300.0)) is True
    wait_past(first_expiry)
    assert _holder(store) == HOLDER
    assert store.coordination.refresh_compression_lock(session_id=SESSION, holder=HOLDER, ttl_seconds=0.2) is True
    wait_past(time.time() + 0.2)
    assert _holder(store) is None
    # Holder-matched renewal of a lapsed row revives it; another holder cannot.
    assert _call(store, epoch, 3, "renew_compression_lock", lease=WorkerTurnLease("other", 300.0)) is False
    assert _call(store, epoch, 4, "renew_compression_lock", lease=WorkerTurnLease(HOLDER, 300.0)) is True
    assert _holder(store) == HOLDER


def test_compression_lock_acquire_reclaims_an_expired_holder(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    assert store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="short", ttl_seconds=0.2) is True
    wait_past(time.time() + 0.2)
    assert _call(store, epoch, 1, "compression_lock_holder") is None
    assert _call(store, epoch, 2, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER)) is True
    assert _holder(store) == HOLDER


def test_compression_lock_payload_checks(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    for lease in (WorkerTurnLease(""), WorkerTurnLease(HOLDER, 0.05), WorkerTurnLease(HOLDER, 3601), WorkerTurnLease(HOLDER, float("inf")), WorkerTurnLease(HOLDER, True)):
        with refused("invalid_params"):
            _call(store, epoch, 1, "acquire_compression_lock", lease=lease)
    assert (_holder(store), last_sequence(backend, scope_a, epoch)) == (None, 0)


def test_compression_cleanup_releases_only_along_the_compression_lineage(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    child, foreign = SessionId("sess-worker-cont"), SessionId("sess-foreign")
    store.sessions.end(SESSION, END_REASON_COMPRESSION)
    store.sessions.create(NewSession(session_id=child, source="contract", parent_session_id=SESSION))
    store.sessions.create(NewSession(session_id=foreign, source="contract"))
    assert store.coordination.try_acquire_compression_lock(session_id=child, holder=HOLDER, ttl_seconds=300.0) is True
    assert store.coordination.try_acquire_compression_lock(session_id=foreign, holder=HOLDER, ttl_seconds=300.0) is True
    for target in (foreign, SessionId("sess-missing")):
        with refused("permission_denied"):
            _call(store, epoch, 1, "cleanup_compression_lock", target_session_id=target, holder=HOLDER)
    assert _holder(store, foreign) == HOLDER
    # Holder-matched: another holder's cleanup releases nothing.
    _call(store, epoch, 1, "cleanup_compression_lock", target_session_id=child, holder="other")
    assert _holder(store, child) == HOLDER
    _call(store, epoch, 2, "cleanup_compression_lock", target_session_id=child, holder=HOLDER)
    assert _holder(store, child) is None
    # The replay returns without re-authorizing or releasing again.
    assert store.coordination.try_acquire_compression_lock(session_id=child, holder=HOLDER, ttl_seconds=300.0) is True
    _call(store, epoch, 2, "cleanup_compression_lock", target_session_id=child, holder=HOLDER)
    assert _holder(store, child) == HOLDER
    assert last_sequence(backend, scope_a, epoch) == 2


# ── cooldown ──────────────────────────────────────────────────────────────────


def test_compression_cooldown_record_restore_clear_land_on_the_messages_cooldown(backend, scope_a):
    store, epoch = _setup(backend, scope_a)

    def snapshot():
        snap = store.messages.read_compression_cooldown_snapshot(SESSION)
        return snap.present, snap.until, snap.error

    _call(store, epoch, 1, "record_compression_cooldown", cooldown=WorkerCooldown(T1, "e1"))
    assert snapshot() == (True, T1, "e1")
    # The deadline merge-maxes, the diagnostic takes the latest report.
    _call(store, epoch, 2, "record_compression_cooldown", cooldown=WorkerCooldown(T0, "e2"))
    assert snapshot() == (True, T1, "e2")
    _call(store, epoch, 3, "restore_compression_cooldown", snapshot=WorkerCooldownSnapshot(True, T0, "restored"))
    assert snapshot() == (True, T0, "restored")
    # A snapshot taken without a session row cannot land on the worker's session: refused on the session, nothing written, sequence free.
    with pytest.raises(CooldownRestoreRefused) as caught:
        _call(store, epoch, 4, "restore_compression_cooldown", snapshot=WorkerCooldownSnapshot(False, None, None))
    assert (caught.value.entity, caught.value.entity_id) == ("session", str(SESSION))
    assert snapshot() == (True, T0, "restored")
    _call(store, epoch, 4, "clear_compression_cooldown")
    assert snapshot() == (True, None, None)
    # Replaying the first record does not re-apply it.
    _call(store, epoch, 1, "record_compression_cooldown", cooldown=WorkerCooldown(T1, "e1"))
    assert snapshot() == (True, None, None)
    assert last_sequence(backend, scope_a, epoch) == 4


def test_compression_cooldown_payload_checks_refuse_non_finite_and_malformed_values(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    for until in (True, float("nan"), float("inf"), "1"):
        with refused("invalid_params"):
            _call(store, epoch, 1, "record_compression_cooldown", cooldown=WorkerCooldown(until))
    with refused("invalid_params"):
        _call(store, epoch, 1, "record_compression_cooldown", cooldown=WorkerCooldown(T1, 5))
    for snap in (WorkerCooldownSnapshot("yes"), WorkerCooldownSnapshot(True, float("nan")), WorkerCooldownSnapshot(True, None, 5)):
        with refused("invalid_params"):
            _call(store, epoch, 1, "restore_compression_cooldown", snapshot=snap)
    # Any finite number is a stored deadline, negative and far-future ones included.
    _call(store, epoch, 1, "record_compression_cooldown", cooldown=WorkerCooldown(-1.0))
    _call(store, epoch, 2, "restore_compression_cooldown", snapshot=WorkerCooldownSnapshot(True, 2e11, None))
    assert store.messages.read_compression_cooldown_snapshot(SESSION).until == 2e11


# ── counters, watermark, reopen, boundary ─────────────────────────────────────


def test_compression_counters_land_on_the_lineage_counters(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    lineage = store.lineage
    _call(store, epoch, 1, "set_compression_fallback_streak", streak=3)
    _call(store, epoch, 2, "set_compression_ineffective_count", count=2)
    _call(store, epoch, 3, "set_compression_recovery_deadline", deadline=T1)
    assert (lineage.compression_fallback_streak(SESSION), lineage.compression_ineffective_count(SESSION), lineage.compression_recovery_deadline(SESSION)) == (3, 2, T1)
    _call(store, epoch, 4, "set_compression_recovery_deadline", deadline=None)
    _call(store, epoch, 5, "set_compression_fallback_streak", streak=0)
    assert (lineage.compression_recovery_deadline(SESSION), lineage.compression_fallback_streak(SESSION)) == (0.0, 0)
    for verb, kwargs in (("set_compression_fallback_streak", {"streak": -1}), ("set_compression_fallback_streak", {"streak": True}),
                         ("set_compression_fallback_streak", {"streak": None}), ("set_compression_ineffective_count", {"count": 1.0}),
                         ("set_compression_recovery_deadline", {"deadline": "soon"}), ("set_compression_recovery_deadline", {"deadline": float("nan")})):
        with refused("invalid_params"):
            _call(store, epoch, 6, verb, **kwargs)
    assert (lineage.compression_fallback_streak(SESSION), last_sequence(backend, scope_a, epoch)) == (0, 5)
    # The recovery deadline is written as sent, any finite number; the owner's read clamps it at 0.
    _call(store, epoch, 6, "set_compression_recovery_deadline", deadline=-5.0)
    assert lineage.compression_recovery_deadline(SESSION) == 0.0
    assert _call(store, epoch, 7, "compression_session_context", target_session_id=SESSION)["compression_recovery_deadline"] == -5.0


def test_compression_watermark_reads_the_active_watermark_and_replays_it(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    assert _call(store, epoch, 1, "compression_watermark") == 0
    first = _call(store, epoch, 2, "append_transcript", messages=[{"role": "user", "content": "hello"}]).annotations[0].row_id
    assert _call(store, epoch, 3, "compression_watermark") == first == store.messages.active_watermark(SESSION).last_active
    _call(store, epoch, 4, "append_transcript", messages=[{"role": "assistant", "content": "hi"}])
    # The replay returns the watermark captured at the original call, not the current one.
    assert (_call(store, epoch, 3, "compression_watermark"), _call(store, epoch, 1, "compression_watermark")) == (first, 0)
    assert store.messages.active_watermark(SESSION).last_active != first


def test_compression_reopen_is_the_lineage_reopen_inside_the_receipt(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    assert _call(store, epoch, 1, "reopen_orphaned_compression") is False  # never ended
    store.sessions.end(SESSION, END_REASON_COMPRESSION)
    assert store.coordination.try_acquire_compression_lock(session_id=SESSION, holder="compressor", ttl_seconds=300.0) is True
    assert _call(store, epoch, 2, "reopen_orphaned_compression") is False  # a live lock means a compression in flight
    store.coordination.release_compression_lock(session_id=SESSION, holder="compressor")
    assert _call(store, epoch, 3, "reopen_orphaned_compression") is True
    assert store.sessions.get(SESSION).ended_at is None
    assert (_call(store, epoch, 2, "reopen_orphaned_compression"), _call(store, epoch, 3, "reopen_orphaned_compression")) == (False, True)
    assert _call(store, epoch, 4, "reopen_orphaned_compression") is False


def test_compression_boundary_reads_only_the_worker_peer(backend, scope_a):
    require_capability(backend, "transactional_claims")
    store = backend.open(scope_a)
    store.sessions.create(NewSession(session_id=SESSION, source="telegram", session_key="agent:main:telegram:dm:1"))
    epoch = begin_epoch(backend, scope_a)
    register(backend, scope_a, epoch)
    mine = ConversationPeer(session_key="agent:main:telegram:dm:1", source="telegram")
    for peer in (ConversationPeer(session_key="agent:main:telegram:dm:1", source="api"), ConversationPeer(session_key="agent:main:telegram:dm:2", source="telegram")):
        with refused("permission_denied"):
            _call(store, epoch, 1, "compression_boundary", peer=peer)
    assert _call(store, epoch, 1, "compression_boundary", peer=mine) is None
    # A reset of another session routed to the same peer advances the generation the worker reads.
    for n, earlier in ((1, "sess-earlier-1"), (2, "sess-earlier-2")):
        store.sessions.create(NewSession(session_id=SessionId(earlier), source="telegram", session_key="agent:main:telegram:dm:1"))
        store.sessions.end(SessionId(earlier), "session_reset")
        assert store.messages.latest_conversation_boundary(mine) == n
    assert _call(store, epoch, 2, "compression_boundary", peer=mine) == 2
    assert _call(store, epoch, 1, "compression_boundary", peer=mine) is None
    assert last_sequence(backend, scope_a, epoch) == 2


# ── shared precedence ─────────────────────────────────────────────────────────


def test_compression_operations_fence_before_any_domain_work(backend, scope_a):
    store, epoch = _setup(backend, scope_a)
    with refused("stale_epoch"):
        store.worker_executions.acquire_compression_lock(**ident(AuthorityEpoch(epoch + 1)), lease=WorkerTurnLease(HOLDER))
    with refused("not_found"):
        store.worker_executions.record_compression_cooldown(**ident(epoch, execution_id=WorkerExecutionId("ghost")), cooldown=WorkerCooldown(T1))
    with refused("invalid_params"):  # not the next sequence
        _call(store, epoch, 2, "set_compression_fallback_streak", streak=1)
    with refused("invalid_params"):
        _call(store, epoch, 1, "cleanup_compression_lock", target_session_id=SESSION, holder="")
    assert (_holder(store), store.lineage.compression_fallback_streak(SESSION)) == (None, 0)
    _call(store, epoch, 1, "finish")
    with refused("stale_generation"):
        _call(store, epoch, 2, "compression_lock_holder")


def test_compression_crash_window_publishes_nothing_then_retries_exactly_once(backend, scope_a):
    require_failpoints(backend)
    store, epoch = _setup(backend, scope_a)
    state = authority_state(backend, scope_a)
    state.failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _call(store, epoch, 1, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER))
    assert _holder(store) is None
    state.failpoints.arm(WRITE_FAILPOINT)
    with pytest.raises(InjectedFailure):
        _call(store, epoch, 1, "record_compression_cooldown", cooldown=WorkerCooldown(T1, "e"))
    assert store.messages.read_compression_cooldown_snapshot(SESSION).until is None
    with refused("invalid_params"):  # the sequence did not advance
        _call(store, epoch, 2, "compression_lock_holder")
    assert _call(store, epoch, 1, "acquire_compression_lock", lease=WorkerTurnLease(HOLDER)) is True
    assert _holder(store) == HOLDER
