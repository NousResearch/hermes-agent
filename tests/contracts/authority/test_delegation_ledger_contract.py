"""Contract cases for the delegation-ledger verbs of ``CoordinationRepository``.

The clock is pinned through ``time.time`` so ages and windows are exact; the owner probe arrives as data (``dead_owners``), so no case needs a dead process. Record validation is pinned without a backend.
"""

from __future__ import annotations

import pytest

from persistence import DelegationId, SessionId
from persistence.authority.coordination import (
    DELEGATION_MAX_DELIVERY_ATTEMPTS,
    DelegationRecord,
    DelegationRecovery,
    DeliveryState,
    OwnerStamp,
    PendingCompletion,
    RecordedOwner,
)

NOW = 1_800_000_000.0
OWNER = OwnerStamp(pid=4242, started_at=777)
DEAD = frozenset({RecordedOwner(4242, 777)})


@pytest.fixture
def clock(monkeypatch):
    now = [NOW]
    monkeypatch.setattr("time.time", lambda: now[0])
    return now


def dispatch(coord, did: str, *, at: float = NOW, ui: str = "ui-1", parent: str | None = "parent-1", task=None) -> DelegationId:
    delegation_id = DelegationId(did)
    coord.record_delegation_dispatch(delegation_id=delegation_id, origin_session="route-1", origin_ui_session_id=ui,
                                     parent_session_id=None if parent is None else SessionId(parent), dispatched_at=at, owner=OWNER,
                                     task={"goal": "g"} if task is None else task, origin_session_id="origin-1")
    return delegation_id


def complete(coord, did: DelegationId, *, at: float = NOW, state: str = "completed") -> None:
    coord.record_delegation_completion(delegation_id=did, state=state, completed_at=at, event={"delegation_id": did, "status": state}, result={"status": state})


def test_delegation_record_validates_identity_and_counters():
    with pytest.raises(ValueError):
        DelegationRecord(delegation_id=DelegationId(""), origin_session="k", state="running", dispatched_at=NOW, completed_at=None, result=None,
                         delivery_state=DeliveryState.PENDING, delivery_attempts=0, origin_session_id="")
    with pytest.raises(ValueError):
        DelegationRecord(delegation_id=DelegationId("d"), origin_session="k", state="running", dispatched_at=NOW, completed_at=None, result=None,
                         delivery_state=DeliveryState.PENDING, delivery_attempts=-1, origin_session_id="")


def test_dispatch_creates_a_running_row_and_replaces_a_reused_id(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-1")
    assert coord.get_delegation(delegation_id=did) == DelegationRecord(
        delegation_id=did, origin_session="route-1", state="running", dispatched_at=NOW, completed_at=None, result=None,
        delivery_state=DeliveryState.PENDING, delivery_attempts=0, origin_session_id="origin-1")
    (inflight,) = coord.list_inflight_delegations()
    assert (inflight.origin_ui_session_id, inflight.parent_session_id, inflight.owner_pid, inflight.owner_started_at, inflight.task, inflight.updated_at) == (
        "ui-1", "parent-1", 4242, 777, {"goal": "g"}, NOW)
    # CONTRACT: a reused id replaces the row and starts over (no Conflict).
    complete(coord, did)
    dispatch(coord, "d-1", task={"goal": "again"})
    assert coord.get_delegation(delegation_id=did).state == "running"
    assert coord.get_delegation(delegation_id=DelegationId("missing")) is None


def test_completion_attaches_the_payloads_and_a_missing_row_writes_nothing(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-2")
    clock[0] += 5
    complete(coord, did, at=NOW + 4, state="error")
    record = coord.get_delegation(delegation_id=did)
    assert (record.state, record.completed_at, record.result, record.delivery_state) == ("error", NOW + 4, {"status": "error"}, DeliveryState.PENDING)
    assert coord.list_inflight_delegations() == ()
    complete(coord, DelegationId("ghost"))
    assert coord.get_delegation(delegation_id=DelegationId("ghost")) is None


def test_unit_child_merges_by_task_index_only_while_running(backend, scope_a, clock):
    # CONTRACT: the stored partial gains the entry, replacing one with the same task_index; a missing or non-running row writes nothing.
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-unit")
    coord.record_unit_child(delegation_id=did, entry={"task_index": 0, "status": "error"})
    coord.record_unit_child(delegation_id=did, entry={"task_index": 1, "status": "completed"})
    coord.record_unit_child(delegation_id=did, entry={"task_index": 0, "status": "completed"})
    (inflight,) = coord.list_inflight_delegations()
    assert inflight.result == {"results": [{"task_index": 1, "status": "completed"}, {"task_index": 0, "status": "completed"}], "partial": True}
    complete(coord, did)
    coord.record_unit_child(delegation_id=did, entry={"task_index": 2})
    assert coord.get_delegation(delegation_id=did).result == {"status": "completed"}
    coord.record_unit_child(delegation_id=DelegationId("ghost"), entry={"task_index": 0})
    assert coord.get_delegation(delegation_id=DelegationId("ghost")) is None


def test_recover_abandoned_writes_only_unchanged_rows(backend, scope_a, clock):
    # CONTRACT: each recovery lands only while state and updated_at still match the read; the row becomes unknown with a pending delivery.
    coord = backend.open(scope_a).coordination
    kept, moved = dispatch(coord, "d-kept"), dispatch(coord, "d-moved")
    rows = {row.delegation_id: row for row in coord.list_inflight_delegations()}
    clock[0] += 10
    coord.record_unit_child(delegation_id=moved, entry={"task_index": 0})
    recoveries = [DelegationRecovery(delegation_id=did, expected_state=rows[did].state, expected_updated_at=rows[did].updated_at, completed_at=NOW + 10,
                                     event={"status": "unknown", "delegation_id": did}, result={"status": "unknown"}) for did in (kept, moved)]
    assert coord.recover_abandoned_delegations(recoveries=recoveries) == 1
    record = coord.get_delegation(delegation_id=kept)
    assert (record.state, record.completed_at, record.result, record.delivery_state) == ("unknown", NOW + 10, {"status": "unknown"}, DeliveryState.PENDING)
    assert coord.get_delegation(delegation_id=moved).state == "running"


def test_replayable_completions_order_and_age_drop(backend, scope_a, clock):
    # CONTRACT: the startup read takes rows (state not running, pending, an event) by completion time then id; rows past 48 h are dropped, the rest returned and left pending.
    coord = backend.open(scope_a).coordination
    old, b, a = dispatch(coord, "d-old"), dispatch(coord, "d-b"), dispatch(coord, "d-a")
    complete(coord, old, at=NOW - 49 * 3600)
    complete(coord, b, at=NOW - 10)
    complete(coord, a, at=NOW - 10)
    dispatch(coord, "d-running")
    offered = coord.take_replayable_completions()
    assert [item.delegation_id for item in offered] == ["d-a", "d-b"]
    assert offered[0] == PendingCompletion(delegation_id=a, event={"delegation_id": "d-a", "status": "completed"}, completed_at=NOW - 10, dispatched_at=NOW)
    assert coord.get_delegation(delegation_id=old).delivery_state is DeliveryState.DROPPED
    assert coord.get_delegation(delegation_id=a).delivery_state is DeliveryState.PENDING


def test_has_replayable_completions_answers_the_startup_select(backend, scope_a, clock):
    # CONTRACT: true exactly when take_replayable_completions has a row to return or drop (state not running, pending, an event).
    coord = backend.open(scope_a).coordination
    assert coord.has_replayable_completions() is False
    did = dispatch(coord, "d-1")
    assert coord.has_replayable_completions() is False
    complete(coord, did, at=NOW - 49 * 3600)
    assert coord.has_replayable_completions() is True
    coord.take_replayable_completions()
    assert coord.has_replayable_completions() is False


def test_orphan_sweep_takes_dead_owners_idle_rows(backend, scope_a, clock):
    # CONTRACT: candidates are terminal, pending, with an event, idle past 60 s and without a claim younger than 300 s; a row in exclude_ids or with a live owner (not in dead_owners) is skipped.
    coord = backend.open(scope_a).coordination
    ready, fresh, excluded, claimed = (dispatch(coord, f"d-{name}") for name in ("ready", "fresh", "excluded", "claimed"))
    for did in (ready, excluded, claimed):
        complete(coord, did)
    clock[0] += 61
    assert coord.claim_delivery(delegation_id=claimed, claim_id="live-claim") is True
    complete(coord, fresh)
    clock[0] += 1
    assert coord.list_orphan_completion_owners(now=clock[0]) == DEAD
    assert coord.sweep_orphaned_completions(now=clock[0], dead_owners=frozenset()) == ()
    swept = coord.sweep_orphaned_completions(now=clock[0], exclude_ids=frozenset({excluded}), dead_owners=DEAD)
    assert [item.delegation_id for item in swept] == [ready]
    assert coord.get_delegation(delegation_id=ready).delivery_state is DeliveryState.PENDING
    assert coord.list_orphan_completion_owners(now=NOW) == frozenset()


def test_orphan_candidates_name_their_rows_and_a_lost_copy_is_swept_whatever_its_owner(backend, scope_a, clock):
    # CONTRACT: list_orphan_completions reads each candidate's id, owner stamp and updated_at, by completion time then id; a candidate in lost_ids is taken even when exclude_ids holds it or its owner is not in dead_owners; a row in lost_ids that is no candidate is not.
    coord = backend.open(scope_a).coordination
    kept, lost, fresh = (dispatch(coord, f"d-{name}") for name in ("kept", "lost", "fresh"))
    for did in (kept, lost):
        complete(coord, did)
    clock[0] += 61
    complete(coord, fresh)
    clock[0] += 1
    candidates = coord.list_orphan_completions(now=clock[0])
    assert [(item.delegation_id, item.owner, item.updated_at) for item in candidates] == [(kept, RecordedOwner(4242, 777), NOW), (lost, RecordedOwner(4242, 777), NOW)]
    swept = coord.sweep_orphaned_completions(now=clock[0], exclude_ids=frozenset({lost}), dead_owners=frozenset(), lost_ids=frozenset({lost, fresh}))
    assert [item.delegation_id for item in swept] == [lost]
    assert coord.get_delegation(delegation_id=kept).delivery_state is DeliveryState.PENDING


def test_orphan_sweep_drops_a_row_whose_owner_died_holding_the_last_attempt(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-last")
    complete(coord, did)
    for attempt in range(DELEGATION_MAX_DELIVERY_ATTEMPTS):
        assert coord.claim_delivery(delegation_id=did, claim_id=f"c{attempt}") is True
        clock[0] += 301
    assert coord.get_delegation(delegation_id=did).delivery_attempts == DELEGATION_MAX_DELIVERY_ATTEMPTS
    assert coord.sweep_orphaned_completions(now=clock[0], dead_owners=DEAD) == ()
    assert coord.get_delegation(delegation_id=did).delivery_state is DeliveryState.DROPPED


def test_orphan_offers_read_what_the_sweep_returns_without_its_drops(backend, scope_a, clock):
    # CONTRACT: list_orphan_completion_offers returns what sweep_orphaned_completions returns for the same arguments, in its order, and writes nothing: a spent row and a row past the replay age stay pending until the sweep drops them.
    coord = backend.open(scope_a).coordination
    ready, excluded, spent, old = (dispatch(coord, f"d-{name}") for name in ("ready", "excluded", "spent", "old"))
    for did in (ready, excluded, spent):
        complete(coord, did)
    complete(coord, old, at=NOW - 49 * 3600)
    for attempt in range(DELEGATION_MAX_DELIVERY_ATTEMPTS):
        assert coord.claim_delivery(delegation_id=spent, claim_id=f"c{attempt}") is True
        clock[0] += 301
    sweep = dict(now=clock[0], exclude_ids=frozenset({excluded}), dead_owners=DEAD)
    offers = coord.list_orphan_completion_offers(**sweep)
    assert [item.delegation_id for item in offers] == [ready]
    assert {coord.get_delegation(delegation_id=did).delivery_state for did in (spent, old)} == {DeliveryState.PENDING}
    assert [item.delegation_id for item in coord.list_orphan_completion_offers(
        now=clock[0], exclude_ids=frozenset({excluded}), dead_owners=frozenset(), lost_ids=frozenset({excluded}))] == [excluded]
    assert coord.sweep_orphaned_completions(**sweep) == offers
    assert {coord.get_delegation(delegation_id=did).delivery_state for did in (spent, old)} == {DeliveryState.DROPPED}


def test_delivery_claim_settle_lifecycle(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-claim")
    complete(coord, did)
    # CONTRACT: one claimant at a time, one attempt spent per claim; a missing row is admitted without a write.
    assert coord.claim_delivery(delegation_id=did, claim_id="c1") is True
    assert coord.claim_delivery(delegation_id=did, claim_id="c2") is False
    assert coord.claim_delivery(delegation_id=DelegationId("legacy"), claim_id="c") is True
    assert coord.get_delegation(delegation_id=DelegationId("legacy")) is None
    # CONTRACT: every settle needs the claim; defer refunds the attempt.
    for settle in (coord.release_delivery, coord.defer_delivery, coord.drop_delivery, coord.complete_delivery):
        assert settle(delegation_id=did, claim_id="wrong") is False
    assert coord.defer_delivery(delegation_id=did, claim_id="c1") is True
    assert coord.get_delegation(delegation_id=did).delivery_attempts == 0
    assert coord.claim_delivery(delegation_id=did, claim_id="c2") is True
    assert coord.release_delivery(delegation_id=did, claim_id="c2") is True
    assert coord.claim_delivery(delegation_id=did, claim_id="c3") is True
    assert coord.complete_delivery(delegation_id=did, claim_id="c3") is True
    record = coord.get_delegation(delegation_id=did)
    assert (record.delivery_state, record.delivery_attempts) == (DeliveryState.DELIVERED, 2)
    assert coord.claim_delivery(delegation_id=did, claim_id="c4") is False
    dropped = dispatch(coord, "d-drop")
    complete(coord, dropped)
    assert coord.claim_delivery(delegation_id=dropped, claim_id="c") is True
    assert coord.drop_delivery(delegation_id=dropped, claim_id="c") is True
    assert coord.get_delegation(delegation_id=dropped).delivery_state is DeliveryState.DROPPED


def _delivered_after(coord, did: DelegationId, claims: int) -> None:
    """Spend ``claims`` attempts (all but the last released), then complete under the last claim."""
    for attempt in range(claims - 1):
        assert coord.claim_delivery(delegation_id=did, claim_id=f"r{attempt}") is True
        assert coord.release_delivery(delegation_id=did, claim_id=f"r{attempt}") is True
    assert coord.claim_delivery(delegation_id=did, claim_id="last") is True
    assert coord.complete_delivery(delegation_id=did, claim_id="last") is True


def test_admitted_completion_reads_what_the_reopen_returns_without_its_write(backend, scope_a, clock):
    # CONTRACT: get_admitted_completion returns what reopen_admitted_completion returns and writes nothing: None for a row not delivered, a spent budget, a row past the replay age or a missing row.
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-admitted")
    complete(coord, did)
    assert coord.get_admitted_completion(delegation_id=did) is None
    _delivered_after(coord, did, 1)
    admitted = coord.get_admitted_completion(delegation_id=did)
    assert admitted is not None and admitted.delegation_id == did
    assert coord.get_delegation(delegation_id=did).delivery_state is DeliveryState.DELIVERED
    assert coord.reopen_admitted_completion(delegation_id=did) == admitted
    spent = dispatch(coord, "d-admitted-spent")
    complete(coord, spent)
    _delivered_after(coord, spent, DELEGATION_MAX_DELIVERY_ATTEMPTS)
    old = dispatch(coord, "d-admitted-old")
    complete(coord, old, at=NOW - 49 * 3600)
    _delivered_after(coord, old, 1)
    for unoffered in (spent, old):
        assert coord.get_admitted_completion(delegation_id=unoffered) is None
        assert coord.get_delegation(delegation_id=unoffered).delivery_state is DeliveryState.DELIVERED
    assert coord.get_admitted_completion(delegation_id=DelegationId("ghost")) is None


def test_reopen_admitted_completion(backend, scope_a, clock):
    # CONTRACT: a delivered row with an event returns to pending and is returned; a spent budget drops it; a row not delivered, or missing, returns None.
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-reopen")
    complete(coord, did)
    assert coord.reopen_admitted_completion(delegation_id=did) is None
    _delivered_after(coord, did, 1)
    reopened = coord.reopen_admitted_completion(delegation_id=did)
    assert reopened is not None and reopened.delegation_id == did
    assert coord.get_delegation(delegation_id=did).delivery_state is DeliveryState.PENDING
    spent = dispatch(coord, "d-spent")
    complete(coord, spent)
    _delivered_after(coord, spent, DELEGATION_MAX_DELIVERY_ATTEMPTS)
    assert coord.reopen_admitted_completion(delegation_id=spent) is None
    assert coord.get_delegation(delegation_id=spent).delivery_state is DeliveryState.DROPPED
    old = dispatch(coord, "d-old")
    complete(coord, old, at=NOW - 49 * 3600)
    _delivered_after(coord, old, 1)
    assert coord.reopen_admitted_completion(delegation_id=old) is None
    assert coord.get_delegation(delegation_id=old).delivery_state is DeliveryState.DROPPED
    assert coord.reopen_admitted_completion(delegation_id=DelegationId("ghost")) is None


def test_delete_unscheduled_delegation_removes_the_row(backend, scope_a, clock):
    coord = backend.open(scope_a).coordination
    did = dispatch(coord, "d-unscheduled")
    coord.delete_unscheduled_delegation(delegation_id=did)
    assert coord.get_delegation(delegation_id=did) is None
    coord.delete_unscheduled_delegation(delegation_id=did)


def test_prune_keeps_bounded_terminal_history(backend, scope_a, clock):
    # CONTRACT: delivered rows untouched for 7 days go; then terminal rows past 50 go, delivered and oldest first; running rows stay.
    coord = backend.open(scope_a).coordination
    stale = dispatch(coord, "d-stale")
    complete(coord, stale)
    _delivered_after(coord, stale, 1)
    clock[0] += 7 * 86400 + 1
    running = dispatch(coord, "d-live")
    for index in range(52):
        did = dispatch(coord, f"d-t{index:02d}")
        complete(coord, did)
        clock[0] += 1
    _delivered_after(coord, DelegationId("d-t51"), 1)
    coord.prune_delegation_ledger()
    assert coord.get_delegation(delegation_id=stale) is None
    assert coord.get_delegation(delegation_id=running) is not None
    assert coord.get_delegation(delegation_id=DelegationId("d-t51")) is None
    assert coord.get_delegation(delegation_id=DelegationId("d-t00")) is None
    assert coord.get_delegation(delegation_id=DelegationId("d-t01")) is not None


def test_list_finished_delegations_reads_recent_terminal_rows_by_selector(backend, scope_a, clock):
    # CONTRACT: either selector matches; terminal rows completed within 24 h, newest first, at most ``limit``; no selector reads nothing; task and result are objects.
    coord = backend.open(scope_a).coordination
    first = dispatch(coord, "d-f1", ui="ui-x", parent=None)
    second = dispatch(coord, "d-f2", ui="other", parent="parent-x")
    old = dispatch(coord, "d-f3", ui="ui-x")
    dispatch(coord, "d-f4", ui="ui-x")
    complete(coord, first, at=NOW - 10, state="error")
    complete(coord, second, at=NOW - 5)
    complete(coord, old, at=NOW - 25 * 3600)
    rows = coord.list_finished_delegations(origin_ui_session_id="ui-x", parent_session_id="parent-x")
    assert [(row.delegation_id, row.state, row.task, row.result) for row in rows] == [
        (second, "completed", {"goal": "g"}, {"status": "completed"}), (first, "error", {"goal": "g"}, {"status": "error"})]
    assert [row.delegation_id for row in coord.list_finished_delegations(origin_ui_session_id="ui-x", limit=1)] == [first]
    assert coord.list_finished_delegations() == ()
    assert [row.delegation_id for row in coord.list_finished_delegations(origin_ui_session_id="ui-x", now=NOW - 24 * 3600)] == [first, old]



def test_delegation_ledger_two_profile_isolation(backend, scope_a, scope_b, clock):
    # CONTRACT: the ledger is per profile: a colliding delegation id is two rows, and a claim in one profile leaves the other's row claimable.
    coord_a, coord_b = backend.open(scope_a).coordination, backend.open(scope_b).coordination
    did = dispatch(coord_a, "d-shared")
    dispatch(coord_b, "d-shared", task={"goal": "b"})
    complete(coord_a, did)
    complete(coord_b, did, state="error")
    assert coord_b.claim_delivery(delegation_id=did, claim_id="b") is True
    assert coord_a.get_delegation(delegation_id=did).delivery_attempts == 0
    assert coord_a.claim_delivery(delegation_id=did, claim_id="a") is True
    assert (coord_a.get_delegation(delegation_id=did).state, coord_b.get_delegation(delegation_id=did).state) == ("completed", "error")
    assert coord_a.complete_delivery(delegation_id=did, claim_id="b") is False
