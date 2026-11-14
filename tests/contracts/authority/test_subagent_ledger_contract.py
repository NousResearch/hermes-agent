"""Contract cases for the subagent-record verbs of the delegation ledger (``persistence.authority.subagents.SubagentLedger`` on ``CoordinationRepository``).

Rows are seeded through ``put_subagent`` and every write is read back through ``get_subagent``. Times are injected through ``now``; ``updated_at`` anchors are the values a seeding write stamped.
"""

from __future__ import annotations

import pytest

from persistence import DelegationId, SessionId
from persistence.authority.coordination import DelegationRecovery, DeliveryState, OwnerStamp
from persistence.authority.subagents import (
    AUTO_RESUME_CANCELLED,
    AUTO_RESUME_MAX_ATTEMPTS,
    GATEWAY_SHUTDOWN,
    SUBAGENT_EXPIRED_RETENTION_SECONDS,
    AutoRefusal,
    AutoStart,
    GaveUpCompletion,
    GiveUpOutcome,
    SubagentLimits,
    SubagentRecovery,
    SubagentState,
    SubagentWrite,
    parent_superseded_note,
)
from persistence.conversation.sessions import NewSession

S = SubagentState
T0 = 1_700_000_000.0
LIMITS = SubagentLimits(ttl_hours=48.0, max_resumable=32)
PID, BIRTH = 4242, 1_699_000_000


def write(state=S.RESUMABLE, *, pinned=True, stop_cause=None, owner=(PID, BIRTH), expires_at=None, **document):
    return SubagentWrite(state=state, pinned=pinned, stop_cause=stop_cause, owner_pid=owner[0], owner_started_at=owner[1],
                         document={"goal": "g", **document}, expires_at=expires_at)


def seed(coord, sid="sa-1", state=S.RESUMABLE, *, now=T0, **kwargs):
    assert coord.put_subagent(subagent_id=sid, write=write(state, **kwargs), expected=coord.get_subagent(subagent_id=sid), now=now) is True
    return coord.get_subagent(subagent_id=sid)


def doc(coord, sid="sa-1"):
    return dict(coord.get_subagent(subagent_id=sid).document)


@pytest.fixture
def coord(backend, scope_a):
    return backend.open(scope_a).coordination


@pytest.fixture
def store(backend, scope_a):
    return backend.open(scope_a)


# ── put_subagent / get_subagent ───────────────────────────────────────────────


def test_put_inserts_then_updates_under_compare_and_swap(coord):
    assert coord.get_subagent(subagent_id="sa-1") is None
    first = seed(coord, state=S.RUNNING, now=T0, cwd="/a", auto_claim="p", auto_cancel={"state": "stopped"}, stop_cause_extra=1)
    # CONTRACT: columns are typed; the document drops state, stop_cause, auto_claim and auto_cancel; insert stamps both times.
    assert (first.state, first.pinned, first.owner_pid, first.owner_started_at, first.created_at, first.updated_at) == (S.RUNNING, True, PID, BIRTH, T0, T0)
    assert "auto_claim" not in first.document and "auto_cancel" not in first.document and first.document["cwd"] == "/a"
    # CONTRACT: a write on a stale expectation (absent, or an older row) writes nothing.
    assert coord.put_subagent(subagent_id="sa-1", write=write(S.STOPPED), expected=None, now=T0 + 1) is False
    assert coord.put_subagent(subagent_id="sa-1", write=write(S.STOPPED, document_only=1), expected=first, now=T0 + 2) is True
    stale = first
    assert coord.put_subagent(subagent_id="sa-1", write=write(S.COMPLETED), expected=stale, now=T0 + 3) is False
    row = coord.get_subagent(subagent_id="sa-1")
    # CONTRACT: an update keeps created_at and stamps updated_at.
    assert (row.state, row.created_at, row.updated_at) == (S.STOPPED, T0, T0 + 2)


def test_put_with_limits_applies_them_in_the_same_write(coord):
    seed(coord, "old", S.RESUMABLE, now=T0, expires_at=T0 + 10)
    expected = coord.get_subagent(subagent_id="new")
    assert coord.put_subagent(subagent_id="new", write=write(S.RESUMABLE, expires_at=T0 + 1000), expected=expected, limits=LIMITS, now=T0 + 20) is True
    # CONTRACT: the old row's TTL passed by the write's time: expired and unpinned.
    old = coord.get_subagent(subagent_id="old")
    assert (old.state, old.pinned) == (S.EXPIRED, False)


# ── reads ─────────────────────────────────────────────────────────────────────


def test_inflight_and_continuable_reads(coord):
    seed(coord, "run", S.RUNNING, now=T0)
    seed(coord, "clm", S.CLAIMED, now=T0)
    seed(coord, "r2", S.RESUMABLE, now=T0 + 2, unit_id="u-1")
    seed(coord, "r1", S.RESUMABLE, now=T0 + 1, delegation_id="u-1")
    seed(coord, "r3", S.RESUMABLE, now=T0 + 3, unit_id="u-2", delegation_id="u-1")
    seed(coord, "st", S.STOPPED, now=T0, unit_id="u-1")
    # CONTRACT: in flight is running or claimed; continuable is resumable by unit_id, else delegation_id, oldest created first.
    assert {r.subagent_id for r in coord.list_inflight_subagents()} == {"run", "clm"}
    assert coord.list_continuable_subagents(unit_id="u-1") == ("r1", "r2")


def test_pinned_read_filters_orders_and_caps(coord):
    seed(coord, "a", S.RESUMABLE, now=T0 + 1)
    seed(coord, "b", S.RUNNING, now=T0 + 3)
    seed(coord, "c", S.SUPERSEDED, now=T0 + 2, expires_at=T0 + 100)
    seed(coord, "gone", S.RESUMABLE, now=T0 + 4, expires_at=T0 + 50)
    seed(coord, "unpinned", S.RESUMABLE, now=T0 + 5, pinned=False)
    seed(coord, "done", S.COMPLETED, now=T0 + 6)
    # CONTRACT: pinned live states, unexpired, newest first, capped.
    assert [r.subagent_id for r in coord.list_pinned_subagents(max_resumable=10, now=T0 + 60)] == ["b", "c", "a"]
    assert [r.subagent_id for r in coord.list_pinned_subagents(max_resumable=2, now=T0 + 60)] == ["b", "c"]


def test_owed_reads(coord):
    seed(coord, "owed", S.RESUMABLE, now=T0 + 2, auto_promised=True)
    seed(coord, "given-up", S.RESUMABLE, now=T0 + 1, auto_promised=True, auto_gave_up="x")
    seed(coord, "plain", S.RESUMABLE, now=T0 + 3)
    seed(coord, "exp-owed", S.EXPIRED, now=T0, auto_promised=True)
    seed(coord, "exp", S.EXPIRED, now=T0 + 4)
    seed(coord, "late", S.RESUMABLE, now=T0 + 5, auto_promised=True, expires_at=T0 + 10)
    assert coord.get_subagent(subagent_id="owed").owed is True and coord.get_subagent(subagent_id="given-up").owed is False
    # CONTRACT: candidates are resumable rows plus owed expired ones, oldest updated first; owed_only keeps the owed.
    assert [r.subagent_id for r in coord.list_auto_resume_candidates()] == ["exp-owed", "given-up", "owed", "plain", "late"]
    assert [r.subagent_id for r in coord.list_auto_resume_candidates(owed_only=True)] == ["exp-owed", "owed", "late"]
    assert coord.has_owed_subagents() is True
    # CONTRACT: waiting rows are owed resumable rows not expired.
    assert {r.subagent_id for r in coord.list_waiting_subagents(now=T0 + 20)} == {"owed"}


def test_has_owed_is_false_on_an_empty_store(coord):
    assert coord.has_owed_subagents() is False
    assert coord.list_pinned_subagents(max_resumable=5) == ()


def test_parent_supersession_follows_the_compression_tip(store):
    coord = store.coordination
    store.sessions.create(NewSession(session_id=SessionId("p-root"), source="cli"))
    assert coord.subagent_parent_superseded(parent_session_id=SessionId("p-root")) is None
    store.sessions.end(SessionId("p-root"), "compression")
    store.sessions.create(NewSession(session_id=SessionId("p-tip"), source="cli", parent_session_id=SessionId("p-root")))
    # CONTRACT: the live tip decides: a compressed root with a live tip is wanted; an automatic end is wanted; a deliberate end or no row supersedes.
    assert coord.subagent_parent_superseded(parent_session_id=SessionId("p-root")) is None
    store.sessions.end(SessionId("p-tip"), "tui_shutdown")
    assert coord.subagent_parent_superseded(parent_session_id=SessionId("p-root")) is None
    store.sessions.create(NewSession(session_id=SessionId("p-2"), source="cli"))
    store.sessions.end(SessionId("p-2"), "new_session")
    assert coord.subagent_parent_superseded(parent_session_id=SessionId("p-2")) == "its parent session ended (new_session)"
    assert coord.subagent_parent_superseded(parent_session_id=SessionId("nobody")) == "its parent session no longer exists"


def test_parent_superseded_note_wording():
    assert parent_superseded_note(None, tip_exists=False) == "its parent session no longer exists"
    assert parent_superseded_note("idle_timeout", tip_exists=True) is None
    assert parent_superseded_note("deleted", tip_exists=True) == "its parent session ended (deleted)"


def _dispatch(coord, did="u-1", *, parent="p-1", origin_session="sk", origin_ui="ui"):
    coord.record_delegation_dispatch(delegation_id=DelegationId(did), origin_session=origin_session, origin_ui_session_id=origin_ui,
                                     parent_session_id=SessionId(parent) if parent else None, dispatched_at=T0, owner=OwnerStamp(pid=1, started_at=1),
                                     task={"goal": "g", "task_indexes": [0]}, origin_session_id="os")


def test_delegation_route_reads_the_unit_row(coord):
    assert coord.get_delegation_route(delegation_id=DelegationId("u-1")) is None
    _dispatch(coord)
    route = coord.get_delegation_route(delegation_id=DelegationId("u-1"))
    # CONTRACT: the route columns, decoded task and result, dispatch time and updated_at.
    assert (route.origin_session, route.origin_ui_session_id, route.parent_session_id, route.origin_session_id, route.task, route.result, route.dispatched_at) == (
        "sk", "ui", "p-1", "os", {"goal": "g", "task_indexes": [0]}, None, T0)
    assert route.updated_at > 0


# ── run-time writes ───────────────────────────────────────────────────────────


def test_checkpoint_moves_a_running_row_only(coord):
    seed(coord, "run", S.RUNNING, now=T0, cwd="/a")
    seed(coord, "res", S.RESUMABLE, now=T0, cwd="/a")
    # CONTRACT: cwd and baselines land on a running row with updated_at; another state is untouched.
    assert coord.checkpoint_subagent(subagent_id="run", cwd="/b", repo_baselines={"/repo": ["/repo"]}, now=T0 + 5) is True
    assert (doc(coord, "run")["cwd"], doc(coord, "run")["repo_baselines"], coord.get_subagent(subagent_id="run").updated_at) == ("/b", {"/repo": ["/repo"]}, T0 + 5)
    assert coord.checkpoint_subagent(subagent_id="res", cwd="/b", repo_baselines={}, now=T0 + 5) is False
    assert doc(coord, "res")["cwd"] == "/a"


def test_gateway_shutdown_marks_this_owners_running_rows(coord):
    seed(coord, "mine", S.RUNNING, now=T0)
    seed(coord, "other", S.RUNNING, now=T0, owner=(99, 1))
    seed(coord, "res", S.RESUMABLE, now=T0)
    # CONTRACT: only running rows of the owner pid take the cause; updated_at does not move.
    assert coord.mark_subagents_gateway_shutdown(owner_pid=PID) == 1
    mine = coord.get_subagent(subagent_id="mine")
    assert (mine.stop_cause, mine.updated_at) == (GATEWAY_SHUTDOWN, T0)
    assert coord.get_subagent(subagent_id="other").stop_cause is None and coord.get_subagent(subagent_id="res").stop_cause is None


def test_limits_expire_delete_and_cap(coord):
    ttl = LIMITS.ttl_hours * 3600
    now = T0 + SUBAGENT_EXPIRED_RETENTION_SECONDS + ttl + 100
    seed(coord, "expiring", S.STOPPED, now=now - 50, expires_at=now - 1)
    seed(coord, "running", S.RUNNING, now=now - 50, expires_at=now - 1)
    seed(coord, "old-exp", S.EXPIRED, now=T0)
    seed(coord, "old-owed", S.EXPIRED, now=T0, auto_promised=True)
    for i in range(3):
        seed(coord, f"pin-{i}", S.RESUMABLE, now=now - 10 + i)
    row = coord.apply_subagent_limits(limits=SubagentLimits(ttl_hours=LIMITS.ttl_hours, max_resumable=2), subagent_id="pin-0", now=now)
    # CONTRACT: TTL expiry only from the finished states; old expired rows go unless owed; pins beyond the cap drop, newest kept.
    assert (coord.get_subagent(subagent_id="expiring").state, coord.get_subagent(subagent_id="expiring").pinned) == (S.EXPIRED, False)
    assert coord.get_subagent(subagent_id="running").state is S.RUNNING
    assert coord.get_subagent(subagent_id="old-exp") is None and coord.get_subagent(subagent_id="old-owed") is not None
    assert [coord.get_subagent(subagent_id=f"pin-{i}").pinned for i in range(3)] == [False, True, True]
    assert row == coord.get_subagent(subagent_id="pin-0")
    assert coord.apply_subagent_limits(limits=LIMITS, now=now) is None


# ── continue ──────────────────────────────────────────────────────────────────


def test_claim_moves_an_expected_state_once_and_clears_the_debt(coord):
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True)
    assert coord.claim_subagent(subagent_id="sa-1", expected_state=S.STOPPED, owner_pid=7, owner_started_at=8, now=T0 + 1) is False
    assert coord.claim_subagent(subagent_id="sa-1", expected_state=S.RESUMABLE, owner_pid=7, owner_started_at=8, now=T0 + 1) is True
    row = coord.get_subagent(subagent_id="sa-1")
    # CONTRACT: claimed by the new owner, updated_at stamped, auto_promised false.
    assert (row.state, row.owner_pid, row.owner_started_at, row.updated_at, row.document["auto_promised"]) == (S.CLAIMED, 7, 8, T0 + 1, False)
    assert coord.claim_subagent(subagent_id="sa-1", expected_state=S.RESUMABLE, owner_pid=9, owner_started_at=9, now=T0 + 2) is False


def _claimed(coord, sid="sa-1"):
    seed(coord, sid, S.RESUMABLE, now=T0)
    assert coord.claim_subagent(subagent_id=sid, expected_state=S.RESUMABLE, owner_pid=PID, owner_started_at=BIRTH, now=T0 + 1)
    return coord.get_subagent(subagent_id=sid)


def test_hand_back_returns_a_claim_with_its_promise(coord):
    _claimed(coord)
    coord.hand_back_subagent(subagent_id="sa-1", state=S.RESUMABLE, promised=True, owner_pid=1)  # another owner: untouched
    assert coord.get_subagent(subagent_id="sa-1").state is S.CLAIMED
    coord.hand_back_subagent(subagent_id="sa-1", state=S.RESUMABLE, promised=True, owner_pid=PID)
    row = coord.get_subagent(subagent_id="sa-1")
    # CONTRACT: back to the state with the promise; updated_at unchanged.
    assert (row.state, row.document["auto_promised"], row.updated_at) == (S.RESUMABLE, True, T0 + 1)


def test_hand_back_after_a_recorded_cancel_takes_the_cancel(coord):
    _claimed(coord)
    assert coord.cancel_manual_subagent_claim(subagent_id="sa-1", state=S.STOPPED, note="dismissed by its parent", now=T0 + 2) is True
    coord.hand_back_subagent(subagent_id="sa-1", state=S.RESUMABLE, promised=True)
    row = coord.get_subagent(subagent_id="sa-1")
    # CONTRACT: the cancel wins: its state and note, no debt, unpinned for stopped, claim marks gone.
    assert (row.state, row.pinned, row.document["auto_note"], row.document["auto_promised"]) == (S.STOPPED, False, "dismissed by its parent", False)
    assert "auto_cancel" not in row.document and "auto_claim" not in row.document


# ── automatic resume ──────────────────────────────────────────────────────────


def test_note_moves_from_states_and_clears_the_debt_on_cancel(coord):
    seed(coord, "a", S.RESUMABLE, now=T0, auto_promised=True)
    seed(coord, "b", S.STOPPED, now=T0)
    assert coord.note_subagent(subagent_id="a", note="later", now=T0 + 1) is True
    assert (doc(coord, "a")["auto_note"], coord.get_subagent(subagent_id="a").state, doc(coord, "a")["auto_promised"]) == ("later", S.RESUMABLE, True)
    assert coord.note_subagent(subagent_id="b", note="x", now=T0 + 1) is False
    # CONTRACT: superseded/stopped clear the debt; unpin applies with a state.
    assert coord.note_subagent(subagent_id="a", note="gone", state=S.SUPERSEDED, unpin=True, now=T0 + 2) is True
    row = coord.get_subagent(subagent_id="a")
    assert (row.state, row.pinned, row.document["auto_promised"], row.updated_at) == (S.SUPERSEDED, False, False, T0 + 2)


def test_note_reaches_an_unstarted_automatic_claim(store):
    coord = store.coordination
    store.sessions.create(NewSession(session_id=SessionId("p-1"), source="cli"))
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True, auto_attempts=0)
    assert coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=0, parent_session_id=SessionId("p-1"), owner_pid=PID, owner_started_at=BIRTH, now=T0 + 1).record is not None
    assert coord.note_subagent(subagent_id="sa-1", note="n", state=S.SUPERSEDED, from_states=(), now=T0 + 2) is False
    # CONTRACT: unstarted_claim admits a claimed row with auto_claim; the claim mark is removed.
    assert coord.note_subagent(subagent_id="sa-1", note="n", state=S.SUPERSEDED, from_states=(), unstarted_claim=True, now=T0 + 2) is True
    assert coord.get_subagent(subagent_id="sa-1").state is S.SUPERSEDED and "auto_claim" not in doc(coord)


def test_cancel_expired_needs_an_owed_expired_row(coord):
    seed(coord, "owed", S.EXPIRED, now=T0, auto_promised=True)
    seed(coord, "free", S.EXPIRED, now=T0)
    assert coord.cancel_expired_subagent(subagent_id="free", note="n", now=T0 + 1) is False
    assert coord.cancel_expired_subagent(subagent_id="owed", note="n", now=T0 + 1) is True
    # CONTRACT: the debt clears and the note lands.
    assert (doc(coord, "owed")["auto_promised"], doc(coord, "owed")["auto_note"]) == (False, "n")


def test_manual_claim_cancel_skips_an_automatic_claim(store):
    coord = store.coordination
    store.sessions.create(NewSession(session_id=SessionId("p-1"), source="cli"))
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True)
    coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=0, parent_session_id=SessionId("p-1"), owner_pid=PID, owner_started_at=BIRTH, now=T0 + 1)
    # CONTRACT: only a claim without auto_claim records the cancel.
    assert coord.cancel_manual_subagent_claim(subagent_id="sa-1", state=S.SUPERSEDED, note="n") is False


def test_auto_claim_claims_spends_an_attempt_and_refuses_a_moved_row(store):
    coord = store.coordination
    store.sessions.create(NewSession(session_id=SessionId("p-1"), source="cli"))
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True, auto_attempts=1)
    moved = coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=0, parent_session_id=SessionId("p-1"), owner_pid=PID, owner_started_at=BIRTH, now=T0 + 1)
    assert (moved.record, moved.superseded) == (None, None)
    claim = coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=1, parent_session_id=SessionId("p-1"), owner_pid=7, owner_started_at=8, now=T0 + 1)
    # CONTRACT: claimed by the owner, attempt spent before the run, parent stored as auto_claim.
    row = coord.get_subagent(subagent_id="sa-1")
    assert claim.record == row
    assert (row.state, row.owner_pid, row.updated_at, row.document["auto_attempts"], row.document["auto_last_at"], row.document["auto_claim"]) == (
        S.CLAIMED, 7, T0 + 1, 2, T0 + 1, "p-1")


@pytest.mark.parametrize("document", [{"auto_promised": False}, {"auto_promised": True, "auto_off": "x"}, {"auto_promised": True, "auto_attempts": AUTO_RESUME_MAX_ATTEMPTS}])
def test_auto_claim_refuses_ineligible_rows(store, document):
    coord = store.coordination
    store.sessions.create(NewSession(session_id=SessionId("p-1"), source="cli"))
    seed(coord, state=S.RESUMABLE, now=T0, **document)
    attempts = document.get("auto_attempts", 0)
    claim = coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=attempts, parent_session_id=SessionId("p-1"), owner_pid=7, owner_started_at=8, now=T0 + 1)
    assert (claim.record, claim.superseded, coord.get_subagent(subagent_id="sa-1").state) == (None, None, S.RESUMABLE)


def test_auto_claim_supersedes_for_a_parent_that_moved_on(store):
    coord = store.coordination
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True)
    claim = coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=0, parent_session_id=SessionId("gone"), owner_pid=7, owner_started_at=8, now=T0 + 1)
    # CONTRACT: the supersession is noted in the same write and returned.
    assert (claim.record, claim.superseded) == (None, "its parent session no longer exists")
    row = coord.get_subagent(subagent_id="sa-1")
    assert (row.state, row.document["auto_note"], row.document["auto_promised"]) == (S.SUPERSEDED, "its parent session no longer exists", False)


def _auto_claimed(store, parent="p-1", **document):
    coord = store.coordination
    store.sessions.create(NewSession(session_id=SessionId(parent), source="cli"))
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True, auto_attempts=0, **document)
    assert coord.claim_auto_resume(subagent_id="sa-1", expected_attempts=0, parent_session_id=SessionId(parent), owner_pid=PID, owner_started_at=BIRTH, now=T0 + 1).record
    return coord.get_subagent(subagent_id="sa-1")


def test_confirm_auto_start(store):
    coord = store.coordination
    _auto_claimed(store)
    # CONTRACT: None while the claim stands and the parent wants it.
    assert coord.confirm_auto_start(subagent_id="sa-1") is None
    store.sessions.end(SessionId("p-1"), "deleted")
    assert coord.confirm_auto_start(subagent_id="sa-1") == f"{AUTO_RESUME_CANCELLED}: its parent session ended (deleted)"
    assert coord.get_subagent(subagent_id="sa-1").state is S.SUPERSEDED
    assert coord.confirm_auto_start(subagent_id="sa-1") == AUTO_RESUME_CANCELLED
    assert coord.confirm_auto_start(subagent_id="none") == AUTO_RESUME_CANCELLED


def test_auto_start_writes_the_run_record_while_the_claim_stands(store):
    coord = store.coordination
    claimed = _auto_claimed(store)
    run = write(S.RUNNING, expires_at=T0 + 100, auto_attempts=1, cwd="/w")
    outcome = coord.start_auto_resumed_run(subagent_id="sa-1", write=run, expected=claimed, owner_pid=PID, now=T0 + 2)
    # CONTRACT: written as running; the claim marks are dropped by the write.
    assert (outcome.written, outcome.cancelled) == (True, None)
    row = coord.get_subagent(subagent_id="sa-1")
    assert (row.state, row.document["cwd"], "auto_claim" in row.document) == (S.RUNNING, "/w", False)


def test_auto_start_lets_the_parent_cancel_and_refunds_the_attempt(store):
    coord = store.coordination
    claimed = _auto_claimed(store)
    store.sessions.end(SessionId("p-1"), "deleted")
    outcome = coord.start_auto_resumed_run(subagent_id="sa-1", write=write(S.RUNNING), expected=claimed, owner_pid=PID, now=T0 + 2)
    # CONTRACT: superseded in the same write, the attempt refunded, no run record.
    assert (outcome.written, outcome.cancelled) == (False, f"{AUTO_RESUME_CANCELLED}: its parent session ended (deleted)")
    row = coord.get_subagent(subagent_id="sa-1")
    assert (row.state, row.document["auto_attempts"]) == (S.SUPERSEDED, 0)


def test_auto_start_on_a_stopped_or_foreign_row(store):
    coord = store.coordination
    claimed = _auto_claimed(store)
    # CONTRACT: a claim another process holds is a moved record, with nothing written.
    assert coord.start_auto_resumed_run(subagent_id="sa-1", write=write(S.RUNNING), expected=claimed, owner_pid=1, now=T0 + 2) == AutoStart(
        written=False, cancelled=f"{AUTO_RESUME_CANCELLED}: the record moved (claimed)")
    assert coord.get_subagent(subagent_id="sa-1") == claimed
    assert coord.note_subagent(subagent_id="sa-1", note="dismissed by its parent", state=S.STOPPED, from_states=(), unstarted_claim=True, now=T0 + 3)
    outcome = coord.start_auto_resumed_run(subagent_id="sa-1", write=write(S.RUNNING), expected=claimed, owner_pid=PID, now=T0 + 4)
    # CONTRACT: a stopped row answers with its note and has its attempt refunded.
    assert outcome.cancelled == f"{AUTO_RESUME_CANCELLED}: dismissed by its parent"
    assert doc(coord)["auto_attempts"] == 0
    missing = coord.start_auto_resumed_run(subagent_id="none", write=write(S.RUNNING), expected=claimed, owner_pid=PID)
    assert missing.cancelled == f"{AUTO_RESUME_CANCELLED}: the record moved (gone)"


@pytest.mark.parametrize(("kind", "expected"), [
    (AutoRefusal.CANCELLED, {"auto_attempts": 0}),
    (AutoRefusal.TRANSIENT, {"auto_attempts": 0, "auto_note": "automatic continue refused: busy"}),
    (AutoRefusal.PERMANENT, {"auto_attempts": 1, "auto_off": "automatic continue refused: busy", "auto_note": "automatic continue refused: busy"}),
])
def test_settle_refusal(store, kind, expected):
    coord = store.coordination
    _auto_claimed(store)
    coord.settle_auto_refusal(subagent_id="sa-1", kind=kind, attempts=0, note="automatic continue refused: busy", owner_pid=PID)
    row = coord.get_subagent(subagent_id="sa-1")
    # CONTRACT: the claim returns to resumable and the refusal's fields land; updated_at unchanged.
    assert (row.state, row.updated_at, "auto_claim" in row.document) == (S.RESUMABLE, T0 + 1, False)
    assert {k: row.document.get(k) for k in expected} == expected


def test_settle_refusal_leaves_another_owners_state(store):
    coord = store.coordination
    _auto_claimed(store)
    coord.settle_auto_refusal(subagent_id="sa-1", kind=AutoRefusal.CANCELLED, attempts=0, note="n", owner_pid=1)
    assert coord.get_subagent(subagent_id="sa-1").state is S.CLAIMED


def _route_completion(coord, unit="u-1", sid="sa-1"):
    route = coord.get_delegation_route(delegation_id=DelegationId(unit))
    evt = {"type": "async_delegation", "delegation_id": f"{unit}:auto:{sid}", "session_key": "sk", "origin_ui_session_id": "ui",
           "parent_session_id": "p-1", "origin_session_id": "os", "goal": "g", "status": "failed", "dispatched_at": T0, "completed_at": T0 + 9}
    return GaveUpCompletion(unit_id=DelegationId(unit), unit_updated_at=route.updated_at, event=evt, result={"status": "failed"}, owner_pid=5, owner_started_at=6)


def test_give_up_records_one_failed_completion(coord):
    _dispatch(coord)
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True, unit_id="u-1")
    completion = _route_completion(coord)
    assert coord.give_up_auto_resume(subagent_id="sa-1", reason="gave up: x", completion=completion, now=T0 + 10) is GiveUpOutcome.RECORDED
    # CONTRACT: the record is marked and the failed completion is pending on its own row.
    row = coord.get_subagent(subagent_id="sa-1")
    assert (row.document["auto_gave_up"], row.document["auto_off"], row.document["auto_note"], row.owed) == ("gave up: x", "gave up: x", "gave up: x", False)
    failed = coord.get_delegation(delegation_id=DelegationId("u-1:auto:sa-1"))
    assert (failed.state, failed.delivery_state, failed.origin_session, failed.completed_at) == ("failed", DeliveryState.PENDING, "sk", T0 + 9)
    # CONTRACT: no longer owed, a second give-up writes nothing.
    assert coord.give_up_auto_resume(subagent_id="sa-1", reason="again", completion=completion, now=T0 + 11) is GiveUpOutcome.NOT_OWED


def test_give_up_without_a_route_marks_only(coord):
    seed(coord, state=S.EXPIRED, now=T0, auto_promised=True)
    assert coord.give_up_auto_resume(subagent_id="sa-1", reason="r", completion=None, now=T0 + 1) is GiveUpOutcome.NO_ROUTE
    assert doc(coord)["auto_gave_up"] == "r"


def test_give_up_with_a_moved_route_writes_nothing(coord):
    _dispatch(coord)
    seed(coord, state=S.RESUMABLE, now=T0, auto_promised=True, unit_id="u-1")
    completion = _route_completion(coord)
    coord.record_unit_child(delegation_id=DelegationId("u-1"), entry={"task_index": 0, "status": "completed"})
    if coord.get_delegation_route(delegation_id=DelegationId("u-1")).updated_at == completion.unit_updated_at:
        pytest.skip("the unit's updated_at did not advance within the clock's resolution")
    # CONTRACT: a unit row that changed since the read answers route_moved and leaves the record owed.
    assert coord.give_up_auto_resume(subagent_id="sa-1", reason="r", completion=completion, now=T0 + 1) is GiveUpOutcome.ROUTE_MOVED
    assert coord.get_subagent(subagent_id="sa-1").owed is True
    assert coord.get_delegation(delegation_id=DelegationId("u-1:auto:sa-1")) is None


def test_supersede_for_session_cancels_every_waiting_child_of_its_units(coord):
    _dispatch(coord, "u-1", parent="p-1", origin_session="sk-1")
    _dispatch(coord, "u-2", parent="p-2", origin_session="sk-2")
    seed(coord, "res", S.RESUMABLE, now=T0, unit_id="u-1")
    seed(coord, "owed", S.RESUMABLE, now=T0, unit_id="u-1", auto_promised=True)
    seed(coord, "exp", S.EXPIRED, now=T0, unit_id="u-1", auto_promised=True)
    seed(coord, "other", S.RESUMABLE, now=T0, unit_id="u-2")
    seed(coord, "manual", S.RESUMABLE, now=T0, unit_id="u-1")
    assert coord.claim_subagent(subagent_id="manual", expected_state=S.RESUMABLE, owner_pid=1, owner_started_at=1, now=T0)
    assert coord.supersede_subagents_for_session(note="n") == 0
    # CONTRACT: with promised_only only owed resumable rows move; claims and owed expired rows always do.
    assert coord.supersede_subagents_for_session(note="n", session_key="sk-1", promised_only=True) == 3
    assert coord.get_subagent(subagent_id="res").state is S.RESUMABLE
    assert coord.get_subagent(subagent_id="owed").state is S.SUPERSEDED
    assert (coord.get_subagent(subagent_id="exp").state, doc(coord, "exp")["auto_promised"]) == (S.EXPIRED, False)
    assert doc(coord, "manual")["auto_cancel"] == {"state": "superseded", "note": "n"}
    assert coord.supersede_subagents_for_session(note="n", parent_session_id="p-1") == 2  # res, and the manual claim again
    assert coord.get_subagent(subagent_id="other").state is S.RESUMABLE


def test_dismiss_under_compare_and_swap(coord):
    seed(coord, "res", S.RESUMABLE, now=T0, auto_promised=True)
    expected = coord.get_subagent(subagent_id="res")
    assert coord.note_subagent(subagent_id="res", note="moved", now=T0 + 1)
    # CONTRACT: a row that moved since the read is not dismissed.
    assert coord.dismiss_subagent(expected=expected, note="dismissed by its parent", now=T0 + 2) is None
    current = coord.get_subagent(subagent_id="res")
    assert coord.dismiss_subagent(expected=current, note="dismissed by its parent", now=T0 + 2) == "dismissed"
    row = coord.get_subagent(subagent_id="res")
    assert (row.state, row.pinned, row.document["auto_promised"]) == (S.STOPPED, False, False)
    seed(coord, "exp", S.EXPIRED, now=T0, auto_promised=True)
    assert coord.dismiss_subagent(expected=coord.get_subagent(subagent_id="exp"), note="d", now=T0 + 3) == "dismissed"
    assert doc(coord, "exp")["auto_promised"] is False
    seed(coord, "man", S.RESUMABLE, now=T0)
    coord.claim_subagent(subagent_id="man", expected_state=S.RESUMABLE, owner_pid=1, owner_started_at=1, now=T0)
    assert coord.dismiss_subagent(expected=coord.get_subagent(subagent_id="man"), note="d", now=T0 + 3) == "recorded"
    assert doc(coord, "man")["auto_cancel"] == {"state": "stopped", "note": "d"}


# ── boot recovery composite ───────────────────────────────────────────────────


def test_recovery_finishes_dead_owner_records_with_the_ledger_in_one_write(coord):
    _dispatch(coord, "u-1")
    seed(coord, "dead", S.RUNNING, now=T0, unit_id="u-1")
    seed(coord, "moved", S.RUNNING, now=T0, owner=(77, 1))
    unit = next(r for r in coord.list_inflight_delegations() if r.delegation_id == "u-1")
    finished = write(S.RESUMABLE, expires_at=4_000_000_000.0, unit_id="u-1", outcome="unknown", auto_promised=True)
    plan = (SubagentRecovery(subagent_id="dead", owner_pid=PID, owner_started_at=BIRTH, write=finished),
            SubagentRecovery(subagent_id="moved", owner_pid=PID, owner_started_at=BIRTH, write=finished))
    recovery = DelegationRecovery(delegation_id=DelegationId("u-1"), expected_state="running", expected_updated_at=unit.updated_at, completed_at=T0 + 5,
                                  event={"delegation_id": "u-1"}, result={"status": "unknown"}, delivery="delivered")
    assert coord.recover_abandoned_delegations(recoveries=[recovery], subagent_recoveries=plan, limits=LIMITS) == 1
    # CONTRACT: a record still held by its dead owner is finished; one another owner holds is left; the ledger row recovers in the same write.
    dead = coord.get_subagent(subagent_id="dead")
    assert (dead.state, dead.document["outcome"], dead.document["auto_promised"]) == (S.RESUMABLE, "unknown", True)
    assert coord.get_subagent(subagent_id="moved").state is S.RUNNING
    ledger = coord.get_delegation(delegation_id=DelegationId("u-1"))
    assert (ledger.state, ledger.delivery_state) == ("unknown", DeliveryState.DELIVERED)
    assert coord.list_continuable_subagents(unit_id="u-1") == ("dead",)


def test_recovery_without_subagents_keeps_todays_signature(coord):
    _dispatch(coord, "u-1")
    unit = next(r for r in coord.list_inflight_delegations() if r.delegation_id == "u-1")
    recovery = DelegationRecovery(delegation_id=DelegationId("u-1"), expected_state="running", expected_updated_at=unit.updated_at, completed_at=T0 + 5,
                                  event={}, result={})
    # CONTRACT: subagent recoveries and limits are optional; the ledger recovery alone is unchanged.
    assert coord.recover_abandoned_delegations(recoveries=[recovery]) == 1
    assert coord.get_delegation(delegation_id=DelegationId("u-1")).delivery_state is DeliveryState.PENDING
