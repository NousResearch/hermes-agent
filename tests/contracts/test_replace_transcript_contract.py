"""Contract cases for transcript replacement: ``MessageRepository.replace_transcript`` returns the survivor ids and provides the holder-blind ``reject_active_turn_lease`` refusal the tui edit/regenerate path needs (the ``tui_gateway/methods_prompt.py`` truncation write); ``clear_transcript`` empties a transcript with no write guard.

Every case runs both registered backends, seeds through public protocol operations only and observes through the verb plus re-reads; refusals are proven non-mutating against the ALL view, the counters and the session row.
"""

from __future__ import annotations

import time
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pytest

from persistence import CompressionClosed, CompressionInProgress, Conflict, ProfileScope, RowVisibility, SessionId, TurnLeaseLost
from tests.contracts.authority.test_coordination import pidns_token
from persistence.conversation.messages import NewMessage, ReplaceMode, TranscriptRead, TranscriptView
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession
from tests.contracts.conftest import require_capability
from tests.contracts._waiting import wait_past
from tests.contracts.conversation.test_messages import totals


T0 = datetime(2026, 9, 23, 12, 0, 0, tzinfo=UTC)
SID = SessionId("t2-session")
ROOT = SessionId("t2-root")
CHILD = SessionId("t2-child")


def _msg(content: str, i: int, role: str = "user") -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=(T0 + timedelta(seconds=i)).timestamp())


def _seed(backend, scope: ProfileScope, session_id: SessionId = SID) -> None:
    """A compacted session with a live tail: two compaction-archived rows, a summary and a live user/assistant pair."""
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=session_id, source="tui"))
    for i, (content, role) in enumerate([("old-u", "user"), ("old-a", "assistant")]):
        store.messages.append(session_id, _msg(content, i, role))
    store.messages.compact(session_id, [_msg("SUMMARY", 5, "assistant")])
    store.messages.append(session_id, _msg("live-u", 6))
    store.messages.append(session_id, _msg("live-a", 7, "assistant"))


def _snapshot(backend, scope: ProfileScope, session_id: SessionId = SID):
    store = backend.open(scope)
    rows = store.messages.read_transcript(session_id, TranscriptRead(view=TranscriptView.ALL))
    return [(r.id, r.content, r.visibility) for r in rows], totals(backend, scope, session_id)


@pytest.mark.parametrize("mode", list(ReplaceMode))
def test_survivor_ids_are_the_rows_left_active_in_replacement_order(backend, scope_a, mode):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a)
    before = {r.id for r in store_a.messages.read_transcript(SID, TranscriptRead(view=TranscriptView.ALL))}
    replacement = [_msg("SUMMARY", 10, "assistant"), _msg("edited-u", 11), _msg("edited-a", 12, "assistant")]
    result = store_a.messages.replace_transcript(SID, replacement, mode)
    active = store_a.messages.read_transcript(SID)
    assert result.survivor_ids == tuple(r.id for r in active)
    assert [r.content for r in active] == ["SUMMARY", "edited-u", "edited-a"]
    assert len(set(result.survivor_ids)) == len(replacement)
    carried = set(result.survivor_ids) & before
    if mode is ReplaceMode.ARCHIVE_DROPPED:
        # The live prefix the replacement repeats (the SUMMARY row) keeps its stored id; every divergent row is fresh.
        assert carried == {result.survivor_ids[0]}
    else:
        # The other modes insert every row fresh, so holders of old ids must rebind.
        assert not carried


def test_archive_dropped_keeps_the_compaction_archive_and_the_dropped_turns(backend, scope_a):
    """The tui edit mode: the live prefix the replacement repeats stays in place, the divergent live rows turn REWOUND, and the compaction archive is untouched."""
    store_a = backend.open(scope_a)
    _seed(backend, scope_a)
    rows_before, _ = _snapshot(backend, scope_a)
    result = store_a.messages.replace_transcript(
        SID, [_msg("SUMMARY", 10, "assistant"), _msg("edited-u", 11)], ReplaceMode.ARCHIVE_DROPPED,
        reject_active_turn_lease=True)
    rows_after, totals = _snapshot(backend, scope_a)
    by_id = {row_id: (content, vis) for row_id, content, vis in rows_after}
    for row_id, content, vis in rows_before:
        expected = RowVisibility.REWOUND if vis is RowVisibility.ACTIVE and content != "SUMMARY" else vis
        assert by_id[row_id] == (content, expected)
    assert result.survivor_ids[0] in {row_id for row_id, content, vis in rows_before if content == "SUMMARY"}
    assert [by_id[i] for i in result.survivor_ids] == [("SUMMARY", RowVisibility.ACTIVE), ("edited-u", RowVisibility.ACTIVE)]
    assert totals.message_count == 2


def test_empty_replacement_returns_no_survivors(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a)
    result = store_a.messages.replace_transcript(SID, [], ReplaceMode.ARCHIVE_DROPPED)
    assert result.survivor_ids == ()
    assert store_a.messages.read_transcript(SID) == ()


def test_reject_refuses_under_a_held_turn_lease_and_proceeds_once_released(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed(backend, scope_a)
    token = "live-turn" if store_a.coordination.try_acquire_turn_lease(session_id=SID, holder="live-turn", ttl_seconds=300.0) else None
    assert token is not None
    before = _snapshot(backend, scope_a)
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        store_a.messages.replace_transcript(
            SID, [_msg("edited-u", 11)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert isinstance(refused.value, TurnLeaseLost)
    assert isinstance(refused.value, Conflict) and (refused.value.entity, refused.value.entity_id) == ("session", str(SID))
    assert _snapshot(backend, scope_a) == before
    # The refusal left the lease with its holder.
    assert store_a.coordination.refresh_turn_lease(session_id=SID, holder="live-turn", ttl_seconds=300.0) is True
    # Without the flag no lease is consulted.
    unguarded = store_a.messages.replace_transcript(SID, [_msg("forced", 12)], ReplaceMode.ARCHIVE_DROPPED)
    assert len(unguarded.survivor_ids) == 1
    store_a.coordination.release_turn_lease(session_id=SID, holder="live-turn")
    result = store_a.messages.replace_transcript(
        SID, [_msg("edited-u", 13)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert result.survivor_ids == tuple(r.id for r in store_a.messages.read_transcript(SID))


def test_reject_keys_on_the_conversation_not_the_segment(backend, scope_a):
    """A turn lease taken on the live continuation holds the whole compression lineage: the reject resolves the same key."""
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    store_a.sessions.create(NewSession(session_id=ROOT, source="tui"))
    store_a.sessions.end(ROOT, END_REASON_COMPRESSION)
    store_a.sessions.create(NewSession(session_id=CHILD, source="tui", parent_session_id=ROOT))
    store_a.messages.append(CHILD, _msg("child-u", 0))
    assert store_a.coordination.try_acquire_turn_lease(session_id=ROOT, holder="live-turn", ttl_seconds=300.0) is True
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        store_a.messages.replace_transcript(
            CHILD, [_msg("edited", 1)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert isinstance(refused.value, TurnLeaseLost)
    assert [r.content for r in store_a.messages.read_transcript(CHILD)] == ["child-u"]


def test_reject_refuses_under_a_live_compaction_lease(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed(backend, scope_a)
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(SID)), holder="compressor", ttl_seconds=300.0) is True
    # Both leases live: the compaction arm is checked first.
    assert store_a.coordination.try_acquire_turn_lease(session_id=SID, holder="live-turn", ttl_seconds=300.0) is True
    before = _snapshot(backend, scope_a)
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        store_a.messages.replace_transcript(
            SID, [_msg("edited-u", 11)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert isinstance(refused.value, CompressionInProgress)
    assert _snapshot(backend, scope_a) == before
    assert store_a.coordination.get_compression_lock_holder(session_id=SessionId(str(SID))) == "compressor"


def test_reject_reclaims_a_stale_turn_lease_and_proceeds(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed(backend, scope_a)
    assert store_a.coordination.try_acquire_turn_lease(session_id=SID, holder="dead-turn", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    result = store_a.messages.replace_transcript(
        SID, [_msg("edited-u", 11)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert len(result.survivor_ids) == 1
    # The expired row is gone, not merely passed over: its holder can no longer renew it (the reclaim fences the late turn).
    assert store_a.coordination.refresh_turn_lease(session_id=SID, holder="dead-turn", ttl_seconds=300.0) is False


def test_reject_reclaims_a_stale_compaction_lease_and_a_dead_holder(backend, scope_a):
    """Stale = expired OR a ``pid=<n>`` holder naming a provably dead local process, before expiry."""
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed(backend, scope_a)
    finished = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"], capture_output=True, text=True, check=True)
    dead_holder = f"pid={int(finished.stdout)}{pidns_token()}:turn=gone"
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(SID)), holder="old-compressor", ttl_seconds=0.1) is True
    expires = time.time() + 0.1
    assert store_a.coordination.try_acquire_turn_lease(session_id=SID, holder=dead_holder, ttl_seconds=300.0) is True
    wait_past(expires)
    result = store_a.messages.replace_transcript(
        SID, [_msg("edited-u", 11)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert len(result.survivor_ids) == 1
    assert store_a.coordination.refresh_compression_lock(session_id=SessionId(str(SID)), holder="old-compressor", ttl_seconds=300.0) is False
    assert store_a.coordination.refresh_turn_lease(session_id=SID, holder=dead_holder, ttl_seconds=300.0) is False


def test_a_later_refusal_keeps_the_stale_lease_it_passed_over(backend, scope_a):
    """The reclaim belongs to the successful write: a stale turn lease survives a refusal on the compaction arm."""
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _seed(backend, scope_a)
    assert store_a.coordination.try_acquire_turn_lease(session_id=SID, holder="slow-turn", ttl_seconds=0.1) is True
    wait_past(time.time() + 0.1)
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(SID)), holder="compressor", ttl_seconds=300.0) is True
    with pytest.raises((TurnLeaseLost, CompressionInProgress)) as refused:
        store_a.messages.replace_transcript(
            SID, [_msg("edited-u", 11)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert isinstance(refused.value, CompressionInProgress)
    # The expired row still names its holder, so its late refresh revives it (the coordination contract's refresh arm).
    assert store_a.coordination.refresh_turn_lease(session_id=SID, holder="slow-turn", ttl_seconds=300.0) is True


def test_refused_reject_on_an_absent_session_springs_no_row(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    ghost = SessionId("t2-ghost")
    assert store_a.coordination.try_acquire_turn_lease(session_id=ghost, holder="live-turn", ttl_seconds=300.0) is True
    with pytest.raises((TurnLeaseLost, CompressionInProgress)):
        store_a.messages.replace_transcript(
            ghost, [_msg("x", 0)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert store_a.sessions.get(ghost) is None
    assert store_a.messages.read_transcript(ghost, TranscriptRead(view=TranscriptView.ALL)) == ()
    # Nothing retired either: the id is still creatable once the turn ends (create refuses a retired id).
    store_a.coordination.release_turn_lease(session_id=ghost, holder="live-turn")
    assert store_a.sessions.create(NewSession(session_id=ghost, source="cli")) == ghost
    result = store_a.messages.replace_transcript(
        ghost, [_msg("x", 0)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=True)
    assert result.survivor_ids == tuple(r.id for r in store_a.messages.read_transcript(ghost))


@pytest.mark.parametrize("reject", [False, True])
def test_replace_on_a_compression_closed_session_names_the_tip(backend, scope_a, reject):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(session_id=ROOT, source="tui"))
    store_a.messages.append(ROOT, _msg("before", 0))
    store_a.sessions.end(ROOT, END_REASON_COMPRESSION)
    store_a.sessions.create(NewSession(session_id=CHILD, source="tui", parent_session_id=ROOT))
    with pytest.raises(CompressionClosed) as refused:
        store_a.messages.replace_transcript(
            ROOT, [_msg("late", 1)], ReplaceMode.ARCHIVE_DROPPED, reject_active_turn_lease=reject)
    assert refused.value.tip == CHILD
    assert [r.content for r in store_a.messages.read_transcript(ROOT, TranscriptRead(view=TranscriptView.ALL))] == ["before"]


def test_clear_deletes_every_row_and_zeroes_the_counters_of_a_compression_closed_session(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a)
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    store_a.messages.append(SID, NewMessage(role="assistant", content="", tool_calls=(call,),
                                            timestamp=(T0 + timedelta(seconds=8)).timestamp()))
    store_a.sessions.end(SID, END_REASON_COMPRESSION)
    assert totals(backend, scope_a, SID)[:2] != (0, 0)
    assert store_a.messages.clear_transcript(SID) is None
    assert store_a.messages.read_transcript(SID, TranscriptRead(view=TranscriptView.ALL)) == ()
    assert totals(backend, scope_a, SID)[:2] == (0, 0)
    assert store_a.sessions.get(SID).end_reason == END_REASON_COMPRESSION
    absent = SessionId("t2-absent")
    assert store_a.messages.clear_transcript(absent) is None
    assert store_a.sessions.get(absent) is None
