"""Authority vocabulary and shared reference-state tests.

Checks the vocabulary and shared infrastructure the authority repositories use: the fence types and their error discrimination, the three authority capability flags on every assembled backend, the retired-outcome type, and the shared reference authority state with copy-on-write publication over the whole scope namespace, failure injection, replay, sequencing, and retirement helpers.

The runtime, admissions, worker-execution, and binding protocols exist in their capability modules. The crash-window proofs here exercise the shared reference transaction boundary; each capability suite tests its own composite verbs end to end. Nothing here counts as that coverage.
"""

from __future__ import annotations

import pytest

from persistence import (
    AuthorityEpoch,
    Conflict,
    ExecutionGeneration,
    FenceStale,
    PersistenceError,
    ProofRejected,
    RetiredSessionError,
    SessionId,
    SessionRevision,
    StaleFenceKind,
    UnknownExecution,
)
from tests.contracts._reference._authority import (
    RESULT_SCRUBBED,
    InjectedFailure,
    authority_state,
    record_retirement,
    replay,
    require_next_sequence,
    require_not_retired,
    session_runtime,
)
from tests.contracts._scopes import profile_scope
from tests.contracts.conftest import require_failpoints

# ── Fence vocabulary ────────────────────────────────────────


def test_fence_stale_discriminates_the_three_staleness_kinds():
    # One catchable class, three discriminated kinds carrying the guarded
    # entity; the stale_epoch/stale_generation/
    # revision_conflict reasons map onto exactly these.
    for kind in StaleFenceKind:
        error = FenceStale(kind, "session", "sess-1")
        assert error.kind is kind
        assert (error.entity, error.entity_id) == ("session", "sess-1")
        assert kind.value in str(error)
    assert {k.value for k in StaleFenceKind} == {
        "stale_epoch",
        "stale_generation",
        "revision_conflict",
    }


def test_authority_errors_are_persistence_errors_but_never_each_other():
    # FenceStale never masquerades as a lease loss or StoreReplaced and the barrier,
    # proof, and retirement refusals stay their own catchable conditions; an
    # application boundary still contains the whole family via PersistenceError.
    from persistence import CompressionBusy, StoreReplaced, TurnLeaseLost

    fence = FenceStale(StaleFenceKind.STALE_EPOCH, "runtime_epoch", "singleton")
    barrier = UnknownExecution("admission", "adm-1")
    proof = ProofRejected("worker_execution", "exec-1")
    retired = RetiredSessionError(SessionId("sess-gone"))
    errors = [fence, barrier, proof, retired]
    for error in errors:
        assert isinstance(error, PersistenceError)
        assert not isinstance(error, (TurnLeaseLost, CompressionBusy, StoreReplaced))
    distinct = {type(error) for error in errors}
    assert len(distinct) == 4
    assert retired.session_id == SessionId("sess-gone")


def test_fence_counters_are_distinct_spine_types():
    # The three fence values flow as plain ints under distinct NewType names.
    epoch = AuthorityEpoch(3)
    revision = SessionRevision(7)
    generation = ExecutionGeneration(2)
    assert (epoch, revision, generation) == (3, 7, 2)


# ── Capability flags on every assembled backend ──────────────────

# Flag -> the assembled attributes whose registered repositories earn it (the
# tests.contracts.conftest._AUTHORITY_SUITE_FLAGS map's assembly-side mirror). receipted_writes needs at
# least one of its two owners implemented and registered.
_FLAG_OWNERS = {
    "authority_fences": ("runtime",),
    "durable_admissions": ("admissions",),
    "receipted_writes": ("worker_executions", "bindings"),
}


def _folded(backend, attribute: str) -> bool:
    if attribute in getattr(backend, "unfolded_capabilities", frozenset()):
        return False
    return getattr(backend.open(profile_scope("authority-spine")), attribute, None) is not None


def test_authority_flags_declared_and_honest(backend):
    # The three flags must be present and boolean. A true flag without an available owning repository fails; a false flag with an available repository is allowed.
    for flag, owners in _FLAG_OWNERS.items():
        value = getattr(backend.capabilities, flag)
        assert isinstance(value, bool)
        if not any(_folded(backend, attribute) for attribute in owners):
            assert value is False, f"{flag} promised with no folded owner among {owners}"


# ── Shared reference authority state ───────────────────────


def test_authority_state_is_scope_isolated(backend, scope_a, scope_b):
    require_failpoints(backend)
    state_a = authority_state(backend, scope_a)
    state_b = authority_state(backend, scope_b)
    assert state_a is authority_state(backend, scope_a)
    assert state_a is not state_b
    epoch_a = state_a.begin_epoch("instance-a")
    assert epoch_a == 1
    # Scope B's epoch singleton is untouched by scope A's acquisition.
    assert state_b.snapshot["epoch"] == 0


def test_begin_epoch_is_monotonic_and_fences_stale_holders(backend, scope_a):
    require_failpoints(backend)
    state = authority_state(backend, scope_a)
    # Fail-closed on a fresh scope: no epoch was ever issued, so nothing
    # validates, zero included.
    with pytest.raises(FenceStale):
        state.require_epoch(state.snapshot, 0)
    first = state.begin_epoch("instance-a")
    second = state.begin_epoch("instance-b")
    assert second == first + 1
    assert state.snapshot["instance_id"] == "instance-b"
    state.require_epoch(state.snapshot, int(second))
    # The predicate is type(epoch) is not int or a mismatch: the
    # previous holder, a future value, and non-int equivalents (bool, float)
    # are all the epoch-kind stale fence even when they compare equal.
    for stale in (int(first), int(second) + 1, 0, True, float(second), None, str(second)):
        with pytest.raises(FenceStale) as refused:
            state.require_epoch(state.snapshot, stale)
        assert refused.value.kind is StaleFenceKind.STALE_EPOCH


def test_transaction_publishes_atomically_and_failpoint_leaves_state_unchanged(backend, scope_a):
    # Inject a failure between staged mutations, prove the
    # committed state shows no partial transition, then retry exactly once and
    # prove the whole transition applied. The staged view spans the whole scope
    # namespace: a sibling domain-table write and the
    # authority receipt publish together or not at all.
    require_failpoints(backend)
    state = authority_state(backend, scope_a)
    state.begin_epoch("instance-a")
    sid = SessionId("sess-crash")
    scope_store = backend.scope_store(scope_a)
    scope_store.setdefault("messages", {})

    def composite() -> None:
        # Three writes that must publish together: a domain-table row, the
        # session counter bump, and the receipt, with the crash window between.
        with state.transaction() as staged:
            staged.scope["messages"][sid] = ["m1"]
            entry = session_runtime(staged.authority, sid)
            entry["revision"] += 1
            state.failpoints.hit("between-mutation-and-receipt")
            staged.authority["receipts"][("crash", 1)] = {"digest": "d1", "result": {"ok": True}}

    state.failpoints.arm("between-mutation-and-receipt")
    with pytest.raises(InjectedFailure):
        composite()
    # Neither-or-both, across both stores: no domain row, no revision bump,
    # no receipt.
    assert sid not in backend.scope_store(scope_a)["messages"]
    assert sid not in state.snapshot["sessions"]
    assert ("crash", 1) not in state.snapshot["receipts"]
    # Exactly-once retry: the failpoint disarmed itself, the retry commits all.
    composite()
    assert backend.scope_store(scope_a)["messages"][sid] == ["m1"]
    assert state.snapshot["sessions"][sid]["revision"] == 1
    assert state.snapshot["receipts"][("crash", 1)]["result"] == {"ok": True}


def test_publication_failpoint_discards_a_fully_staged_transition(backend, scope_a):
    # The crash window at the publication boundary itself: everything staged,
    # nothing published, in either store.
    require_failpoints(backend)
    state = authority_state(backend, scope_a)
    scope_store = backend.scope_store(scope_a)
    state.failpoints.arm(state.PUBLISH)
    with pytest.raises(InjectedFailure):
        with state.transaction() as staged:
            staged.scope["sessions_table"] = {"sess-pub": {"title": "t"}}
            session_runtime(staged.authority, SessionId("sess-pub"))["generation"] += 1
    assert "sessions_table" not in scope_store
    assert SessionId("sess-pub") not in state.snapshot["sessions"]
    with state.transaction() as staged:
        staged.scope["sessions_table"] = {"sess-pub": {"title": "t"}}
        session_runtime(staged.authority, SessionId("sess-pub"))["generation"] += 1
    assert backend.scope_store(scope_a)["sessions_table"] == {"sess-pub": {"title": "t"}}
    assert state.snapshot["sessions"][SessionId("sess-pub")]["generation"] == 1


def test_replay_returns_detached_result_and_conflicts_on_digest_mismatch(backend, scope_a):
    # The shared reference replay helper returns None for an absent key so its caller can execute; a match replays a detached copy. Mutating a replayed result must not alter committed evidence. A mismatch conflicts, and a retirement-scrubbed result refuses with the generation-kind fence after the digest check so a mismatched late duplicate still surfaces as a conflict. Binding reservations use their own replay checks, not this helper.
    require_failpoints(backend)
    state = authority_state(backend, scope_a)
    with state.transaction() as staged:
        staged.authority["receipts"]["req-1"] = {"digest": "d1", "result": {"value": 1}}
        staged.authority["receipts"]["req-none"] = {"digest": "d1", "result": None}
        staged.authority["receipts"]["req-scrubbed"] = {
            "digest": "d1",
            "result": None,
            RESULT_SCRUBBED: True,
        }
    receipts = state.snapshot["receipts"]
    assert replay(receipts, "req-absent", "d1", entity="receipt") is None
    hit = replay(receipts, "req-1", "d1", entity="receipt")
    assert hit is not None and hit.result == {"value": 1}
    hit.result["value"] = 9
    again = replay(receipts, "req-1", "d1", entity="receipt")
    assert again is not None and again.result == {"value": 1}
    # An ordinary stored None result replays; only the scrubbed marker refuses.
    none_hit = replay(receipts, "req-none", "d1", entity="receipt")
    assert none_hit is not None and none_hit.result is None
    with pytest.raises(Conflict):
        replay(receipts, "req-1", "d2", entity="receipt")
    with pytest.raises(Conflict):
        replay(receipts, "req-scrubbed", "d2", entity="receipt")
    with pytest.raises(FenceStale) as refused:
        replay(receipts, "req-scrubbed", "d1", entity="receipt")
    assert refused.value.kind is StaleFenceKind.STALE_GENERATION


def test_sequence_gate_accepts_only_the_strict_successor(backend, scope_a):
    require_failpoints(backend)
    worker = {"last_sequence": 2}
    require_next_sequence(worker, 3)
    for wrong in (2, 4, 0):
        with pytest.raises(ValueError):
            require_next_sequence(worker, wrong)


def test_retirement_marker_is_permanent_and_fences_creation(backend, scope_a, scope_b):
    # The marker refuses re-creation with the typed retired outcome,
    # re-recording is a fact not a conflict, and the fence is scope-local.
    require_failpoints(backend)
    state = authority_state(backend, scope_a)
    sid = SessionId("sess-retired")
    with state.transaction() as staged:
        record_retirement(staged.authority, sid)
        record_retirement(staged.authority, sid)
    with pytest.raises(RetiredSessionError) as refused:
        require_not_retired(state.snapshot, sid)
    assert refused.value.session_id == sid
    require_not_retired(state.snapshot, SessionId("sess-live"))
    # The same id in the other scope is untouched: retirement is per-scope state.
    require_not_retired(authority_state(backend, scope_b).snapshot, sid)
