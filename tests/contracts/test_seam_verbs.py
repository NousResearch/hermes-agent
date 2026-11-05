"""Contract cases for agent persistence operations.

Cases using the shared backend fixture run against every registered backend (the reference and the SQLite adapter). The related legacy cases live in ``tests/characterization/test_l3d_seam_verbs.py``, using the public ``SessionDB`` surface. Protocol cases assert observable outcomes: cases checking non-mutation compare re-read state, and race cases assert the full partition of contender results. The quarantine case selects SQLite and replaces its connection; usage-input and artifact-layout cases do not use a backend.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from persistence import CompressionClosed, CompressionInProgress, Conflict, ProfileScope, RetiredSessionError, SessionArtifacts, SessionRowMissing, TurnLeaseLost
from persistence.core.identity import MessageId, SessionId
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.conversation.messages import NewMessage, RowDisposition, RowOutcome, TranscriptRead, TranscriptView
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind
from persistence.conversation.usage import TokenCounts, UsageDelta
from tests.contracts.conftest import require_capability
from tests.contracts._concurrency import run_concurrently
from tests.contracts._waiting import wait_past
from tests.contracts.conversation.test_messages import Totals, totals
from tests.contracts.conversation.test_usage import accounting
from tests.contracts._scopes import MEDIA_ROOT

ROOT = SessionId("seam-root")
CHILD = SessionId("seam-child")
OTHER = SessionId("seam-other")


def ts(i: int) -> float:
    return datetime(2026, 9, 22, 10, 0, i, tzinfo=UTC).timestamp()


def msg(content: str, i: int = 0, role: str = "user", **kwargs: object) -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=ts(i), **kwargs)  # type: ignore[arg-type]


def contents(repo, session_id: SessionId) -> list:
    return [r.content for r in repo.read_transcript(session_id)]


# ── the turn-lease fence inside persistence.conversation.messages.MessageRepository.append_batch ──────────────────────────────


def test_g1_two_contenders_one_winner_loser_batch_lands_nothing(backend, scope_a):
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    outcomes = run_concurrently(lambda: coord.try_acquire_turn_lease(session_id=ROOT, holder="turn-a", ttl_seconds=300.0), lambda: coord.try_acquire_turn_lease(session_id=ROOT, holder="turn-b", ttl_seconds=300.0))
    assert sorted(outcomes) == [False, True]
    winner = "turn-a" if outcomes[0] else "turn-b"
    loser = "turn-b" if winner == "turn-a" else "turn-a"
    # The loser flushes first: refused, nothing lands, counters untouched.
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("lost-0"), msg("lost-1", 1, role="assistant")], turn_lease_holder=loser)
    assert contents(repo, ROOT) == []
    assert totals(backend, scope_a, ROOT) == Totals(0, 0, 0)
    # The winner's batch lands whole.
    assert (repo.append_batch(ROOT, [msg("won-0"), msg("won-1", 1, role="assistant")], turn_lease_holder=winner)).inserted == 2
    assert contents(repo, ROOT) == ["won-0", "won-1"]
    assert totals(backend, scope_a, ROOT) == Totals(2, 0, 0)
    # Order does not matter: the loser is refused after the winner too.
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("late", 2)], turn_lease_holder=loser)
    assert contents(repo, ROOT) == ["won-0", "won-1"]


def test_g1_released_or_reacquired_lease_fails_the_fence(backend, scope_a):
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    token = "turn-a" if coord.try_acquire_turn_lease(session_id=ROOT, holder="turn-a", ttl_seconds=300.0) else None
    assert token is not None
    coord.release_turn_lease(session_id=ROOT, holder="turn-a")
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("after-release")], turn_lease_holder=token)
    assert contents(repo, ROOT) == []
    # Another party acquired: the old holder is foreign now.
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="turn-b", ttl_seconds=300.0) is True
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("after-takeover")], turn_lease_holder=token)
    assert contents(repo, ROOT) == []
    # No lease at all is not the same as unfenced: a holder without a live row loses; None appends.
    assert (repo.append_batch(ROOT, [msg("unfenced")])).inserted == 1
    assert contents(repo, ROOT) == ["unfenced"]


def test_g1_expired_but_still_owned_lease_is_renewed_not_lost(backend, scope_a):
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    token = "slow-turn" if coord.try_acquire_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=0.1) else None
    expires = time.time() + 0.1
    assert token is not None
    wait_past(expires)
    # Expiry alone never fences the owner out: the flush lands and re-arms the lease for turn_lease_ttl_seconds.
    assert (repo.append_batch(ROOT, [msg("recovered")], turn_lease_holder=token, turn_lease_ttl_seconds=300.0)).inserted == 1
    assert contents(repo, ROOT) == ["recovered"]
    assert coord.try_acquire_turn_lease(session_id=ROOT, holder="contender", ttl_seconds=300.0) is False
    assert coord.refresh_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=300.0) is True


def test_g1_expired_lease_reclaimed_by_a_contender_fences_the_late_flush(backend, scope_a):
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    token = "slow-turn" if coord.try_acquire_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=0.1) else None
    expires = time.time() + 0.1
    assert token is not None
    wait_past(expires)
    contender = "contender" if coord.try_acquire_turn_lease(session_id=ROOT, holder="contender", ttl_seconds=300.0) else None
    assert contender is not None
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("late-flush")], turn_lease_holder=token)
    assert contents(repo, ROOT) == []
    assert (repo.append_batch(ROOT, [msg("contender-flush")], turn_lease_holder=contender)).inserted == 1
    assert contents(repo, ROOT) == ["contender-flush"]


def test_g1_fence_keys_on_the_conversation_not_the_segment(backend, scope_a):
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT))
    # The lease is taken on the child (the live segment); the fence resolves the same conversation key.
    token = "turn-a" if coord.try_acquire_turn_lease(session_id=CHILD, holder="turn-a", ttl_seconds=300.0) else None
    assert token is not None
    assert (repo.append_batch(CHILD, [msg("child-row")], turn_lease_holder=token)).inserted == 1
    # The holder of another conversation's live lease is foreign to the child's append: the fence reads only the child's conversation lease.
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    assert coord.try_acquire_turn_lease(session_id=OTHER, holder="turn-other", ttl_seconds=300.0) is True
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(CHILD, [msg("foreign-conversation")], turn_lease_holder="turn-other")
    assert contents(repo, CHILD) == ["child-row"]


def test_g1_empty_fenced_batch_is_a_no_op_even_with_a_lost_lease(backend, scope_a):
    require_capability(backend, "transactional_claims")
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    assert (repo.append_batch(OTHER, [], turn_lease_holder="nobody")).inserted == 0
    assert contents(repo, OTHER) == []


def test_g1_refused_append_commits_no_renewal(backend, scope_a):
    """The lease check, the closed-parent refusal, and the renewal share the legacy write transaction (hermes_state_messages.SessionMessagesMixin._check_transcript_write_guards): an append refused for the closed parent rolls its renewal back, so a contender reclaims the expired lease."""
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    token = "slow-turn" if coord.try_acquire_turn_lease(session_id=ROOT, holder="slow-turn", ttl_seconds=0.1) else None
    expires = time.time() + 0.1
    assert token is not None
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT))
    wait_past(expires)
    with pytest.raises(CompressionClosed):
        repo.append_batch(ROOT, [msg("late")], turn_lease_holder=token, turn_lease_ttl_seconds=300.0)
    assert contents(repo, ROOT) == []
    contender = "contender" if coord.try_acquire_turn_lease(session_id=ROOT, holder="contender", ttl_seconds=300.0) else None
    assert contender is not None


def test_g1_refused_append_on_an_absent_id_leaves_it_creatable(backend, scope_a):
    """A refusal appends nothing and retires nothing: the id an unfenced writer never got to use is still creatable afterwards (the constructor's tombstone check, hermes_state_worker_lifecycle.insert_session_row_in_transaction, only refuses deleted sessions)."""
    require_capability(backend, "transactional_claims")
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("never")], turn_lease_holder="nobody")
    # create refuses a retired id, so a landed create proves the refusal retired nothing.
    assert sessions.create(NewSession(session_id=ROOT, source="cli")) == ROOT
    assert (repo.append_batch(ROOT, [msg("first")])).inserted == 1


# ── persistence.conversation.messages.MessageRepository.append_batch refuses a missing session ─


def test_a1_required_session_absent_refuses_and_springs_nothing(backend, scope_a, scope_b):
    store_b = backend.open(scope_b)
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    store_b.sessions.create(NewSession(session_id=ROOT, source="cli"))  # another scope's row vouches for nothing
    with pytest.raises(SessionRowMissing) as refused:
        repo.append_batch(ROOT, [msg("orphan-0"), msg("orphan-1", 1, role="assistant")])
    assert (refused.value.entity, refused.value.entity_id, refused.value.cause) == ("session", str(ROOT), "session_row_missing")
    assert sessions.get(ROOT) is None
    assert contents(repo, ROOT) == []
    assert totals(backend, scope_a, ROOT) == Totals(0, 0, 0)
    with pytest.raises(SessionRowMissing):
        repo.append_batch(OTHER, [msg("sprung")])
    assert sessions.get(OTHER) is None
    assert contents(repo, OTHER) == []
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    assert (repo.append_batch(ROOT, [msg("kept")])).inserted == 1
    assert contents(repo, ROOT) == ["kept"]


def test_a1_required_session_deleted_refuses_and_stays_retired(backend, scope_a):
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    repo.append_batch(ROOT, [msg("before")])
    assert sessions.delete(ROOT, media_root=MEDIA_ROOT) == (ROOT,)
    after_delete = repo.read_transcript(ROOT, TranscriptRead(view=TranscriptView.ALL))
    with pytest.raises(SessionRowMissing):
        repo.append_batch(ROOT, [msg("stale-writer")])
    assert sessions.get(ROOT) is None
    assert repo.read_transcript(ROOT, TranscriptRead(view=TranscriptView.ALL)) == after_delete
    with pytest.raises(RetiredSessionError):
        sessions.create(NewSession(session_id=ROOT, source="cli"))


def test_a1_fences_refuse_before_the_missing_session_check(backend, scope_a):
    """The legacy transaction runs the lease guard and the closed-parent guard before the insert whose FK finds the row missing, so a lost lease or a closed session wins; neither refusal writes or springs anything."""
    require_capability(backend, "transactional_claims")
    coord, repo, sessions = backend.open(scope_a).coordination, backend.open(scope_a).messages, backend.open(scope_a).sessions
    with pytest.raises(TurnLeaseLost):
        repo.append_batch(ROOT, [msg("unleased")], turn_lease_holder="nobody")
    assert sessions.get(ROOT) is None
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    token = "turn-a" if coord.try_acquire_turn_lease(session_id=ROOT, holder="turn-a", ttl_seconds=300.0) else None
    assert token is not None
    assert (repo.append_batch(ROOT, [msg("fenced")], turn_lease_holder=token)).inserted == 1
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT))
    with pytest.raises(CompressionClosed):
        repo.append_batch(ROOT, [msg("closed")], turn_lease_holder=token)
    assert contents(repo, ROOT) == ["fenced"]


# ── repair / adopt in place with per-row outcomes ─────────────────────────


def test_g2_blank_assistant_row_is_repaired_in_place_counters_untouched(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=ROOT, source="cli"))
    # First flush: the provider had streamed nothing yet, so the assistant row landed blank.
    first = repo.append_batch(ROOT, [msg("ask"), msg("", 1, role="assistant")])
    assert [o.disposition for o in first.outcomes] == [RowDisposition.APPENDED, RowDisposition.APPENDED]
    blank_id = first.outcomes[1].id
    counted = totals(backend, scope_a, ROOT)
    # Second flush carries the same assistant message, now with content and its stored id.
    second = repo.append_batch(ROOT, [msg("answer", 1, role="assistant", repair_target=blank_id), msg("next ask", 2)])
    assert [(o.disposition, o.id, o.canonical_row) for o in second.outcomes[:1]] == [(RowDisposition.REPAIRED, blank_id, None)]
    assert second.outcomes[1].disposition is RowDisposition.APPENDED
    assert second.inserted == 1
    assert contents(repo, ROOT) == ["ask", "answer", "next ask"]
    assert totals(backend, scope_a, ROOT) == Totals(counted.message_count + 1, 0, 0)


def test_g2_non_blank_target_is_adopted_never_overwritten(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=ROOT, source="cli"))
    first = repo.append_batch(ROOT, [msg("ask"), msg("winner text", 1, role="assistant")])
    target = first.outcomes[1].id
    result = repo.append_batch(ROOT, [msg("loser text", 1, role="assistant", repair_target=target)])
    assert result.outcomes == (RowOutcome(
        id=target, disposition=RowDisposition.ADOPTED, timestamp=first.outcomes[1].timestamp,
        db_row_snapshot=first.outcomes[1].db_row_snapshot, canonical_row={"content": "winner text", "_content_only": True},
        message_uid=first.outcomes[1].message_uid),)
    assert result.inserted == 0
    assert contents(repo, ROOT) == ["ask", "winner text"]


def test_g2_unresolvable_or_non_assistant_target_appends_fresh(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=ROOT, source="cli"))
    first = repo.append_batch(ROOT, [msg("ask"), msg("", 1, role="assistant")])
    user_id, blank_id = first.outcomes[0].id, first.outcomes[1].id
    # Legacy repair acts on every role: a user row carrying its own row's id resolves to that row and is kept as stored. A target of another role appends fresh. A missing target id appends fresh.
    result = repo.append_batch(ROOT, [
        msg("user again", 2, repair_target=user_id),
        msg("aimed at a user row", 3, role="assistant", repair_target=user_id),
        msg("aimed at nothing", 4, role="assistant", repair_target=MessageId(999_999)),
    ])
    assert [o.disposition for o in result.outcomes] == [RowDisposition.REPAIRED, RowDisposition.APPENDED, RowDisposition.APPENDED]
    assert result.outcomes[0].id == user_id and result.outcomes[0].canonical_row is None
    assert result.outcomes[0].db_row_snapshot == first.outcomes[0].db_row_snapshot
    assert result.inserted == 2
    assert len({o.id for o in result.outcomes} | {user_id, blank_id}) == 4
    assert contents(repo, ROOT) == ["ask", "", "aimed at a user row", "aimed at nothing"]


def test_g2_target_archived_by_watermark_compaction_resolves_to_its_active_clone(backend, scope_a):
    store_a = backend.open(scope_a)
    repo = backend.open(scope_a).messages
    store_a.sessions.create(NewSession(session_id=ROOT, source="cli"))
    repo.append_batch(ROOT, [msg("old ask"), msg("old answer", 1, role="assistant")])
    watermark = repo.active_watermark(ROOT)
    # The turn's first flush lands a blank assistant row AFTER the compression watermark...
    late = repo.append_batch(ROOT, [msg("ask"), msg("", 3, role="assistant")])
    blank_id = late.outcomes[1].id
    # ...then the slow compaction commits and re-sequences the tail as fresh rows (originals superseded).
    repo.compact(ROOT, [msg("summary", 2)], watermark=watermark)
    active_before = repo.read_transcript(ROOT)
    clone = next(r for r in active_before if r.role == "assistant" and r.content == "")
    assert clone.id != blank_id
    # The second flush still carries the ORIGINAL id: it resolves to the clone and repairs it.
    result = repo.append_batch(ROOT, [msg("answer", 3, role="assistant", repair_target=blank_id)])
    assert [(o.disposition, o.id, o.canonical_row) for o in result.outcomes] == [(RowDisposition.REPAIRED, clone.id, None)]
    assert contents(repo, ROOT) == ["summary", "ask", "answer"]
    assert [r.id for r in repo.read_transcript(ROOT)] == [r.id for r in active_before]
    archived = [r for r in repo.read_transcript(ROOT, TranscriptRead(view=TranscriptView.ALL)) if r.id == blank_id]
    assert archived[0].content == ""  # the superseded original is untouched


# ── NewSession carries system_prompt / display_name / origin into persistence.conversation.sessions.SessionRepository.create ──

PROMPT = "You are Hermes.\n\n  spaced\u00a0nbsp\ttab\n"
ORIGIN = {"platform": "telegram", "chat_id": "c9", "thread_id": None}


def gateway_entry(backend, scope: ProfileScope, session_id: SessionId):
    store = backend.open(scope)
    return next(e.session for e in store.sessions.list_gateway_sessions() if e.session.session_id == session_id)


def test_g3_create_writes_the_constructor_facets_in_the_creation_step(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession(
        session_id=ROOT, source="telegram", session_key=f"telegram:{ROOT}", chat_id="c9",
        system_prompt=PROMPT, display_name="Ada", origin=ORIGIN))
    assert sessions.get(ROOT).system_prompt == PROMPT  # byte-stable
    entry = gateway_entry(backend, scope_a, ROOT)
    assert entry.display_name == "Ada"
    assert dict(entry.origin) == ORIGIN


def test_g3_facets_are_keep_existing_on_upsert_and_fill_nulls(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="telegram", session_key=f"telegram:{ROOT}", chat_id="c9"))
    assert (sessions.get(ROOT)).system_prompt is None
    # The constructor's upsert fills the null facets...
    sessions.create(NewSession(session_id=ROOT, source="telegram", system_prompt=PROMPT, display_name="Ada", origin=ORIGIN))
    assert (sessions.get(ROOT)).system_prompt == PROMPT
    # ...and a later create never overwrites the first writer.
    sessions.create(NewSession(session_id=ROOT, source="telegram", system_prompt="other", display_name="Bob", origin={"platform": "x"}))
    assert (sessions.get(ROOT)).system_prompt == PROMPT
    entry = gateway_entry(backend, scope_a, ROOT)
    assert (entry.display_name, dict(entry.origin)) == ("Ada", ORIGIN)


def test_g3_compression_continuation_inherits_peer_facets_not_the_prompt(backend, scope_a):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession(
        session_id=ROOT, source="telegram", session_key=f"telegram:{ROOT}", chat_id="c9",
        system_prompt=PROMPT, display_name="Ada", origin=ORIGIN))
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="telegram", parent_session_id=ROOT))
    entry = gateway_entry(backend, scope_a, CHILD)
    assert (entry.display_name, dict(entry.origin)) == ("Ada", ORIGIN)
    # The prompt snapshot is per segment: the continuation's own constructor writes it.
    assert (sessions.get(CHILD)).system_prompt is None


def test_g3_late_parent_facets_are_inherited_on_the_next_construction(backend, scope_a):
    """The legacy constructor runs the compression-fork inheritance after the keep-existing upsert whenever a parent is named (hermes_state_worker_lifecycle.insert_session_row_in_transaction): a continuation constructed before its parent carried the peer facets fills them on the next construction, and never overwrites its own."""
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="telegram", session_key=f"telegram:{ROOT}", chat_id="c9"))
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="telegram", parent_session_id=ROOT))
    entry = gateway_entry(backend, scope_a, CHILD)
    assert (entry.display_name, entry.origin) == (None, None)
    sessions.create(NewSession(session_id=ROOT, source="telegram", display_name="Ada", origin=ORIGIN))
    sessions.create(NewSession(session_id=CHILD, source="telegram", parent_session_id=ROOT))
    entry = gateway_entry(backend, scope_a, CHILD)
    assert (entry.display_name, dict(entry.origin)) == ("Ada", ORIGIN)


# ── persistence.conversation.messages.MessageRepository.compact merges model_config_patch in the compaction transition ────────


def test_g4_compact_merges_the_patch_with_the_compaction(backend, scope_a):
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli", model_config={"keep": 1, "rearm": "armed"}))
    repo.append_batch(ROOT, [msg("a"), msg("b", 1, role="assistant")])
    # The compressor disarms its re-arm key (None deletes) and records a token count in the same commit.
    assert (repo.compact(ROOT, [msg("summary", 2)], model_config_patch={"rearm": None, "compacted_tokens": 42})).active_count == 1
    assert contents(repo, ROOT) == ["summary"]
    assert sessions.get_model_config_value(ROOT, "rearm", "gone") == "gone"
    assert sessions.get_model_config_value(ROOT, "compacted_tokens") == 42
    assert sessions.get_model_config_value(ROOT, "keep") == 1
    # No patch: config untouched by a later compaction.
    repo.compact(ROOT, [msg("summary-2", 3)])
    assert sessions.get_model_config_value(ROOT, "compacted_tokens") == 42


def test_g4_fenced_compaction_that_loses_the_lease_patches_nothing(backend, scope_a):
    repo, sessions, coord = backend.open(scope_a).messages, backend.open(scope_a).sessions, backend.open(scope_a).coordination
    sessions.create(NewSession(session_id=ROOT, source="cli", model_config={"rearm": "armed"}))
    repo.append_batch(ROOT, [msg("a"), msg("b", 1, role="assistant")])
    token = "compressor-a" if coord.try_acquire_compression_lock(session_id=SessionId(str(ROOT)), holder="compressor-a", ttl_seconds=300.0) else None
    assert token is not None
    coord.release_compression_lock(session_id=SessionId(str(ROOT)), holder="compressor-a")
    assert coord.try_acquire_compression_lock(session_id=SessionId(str(ROOT)), holder="compressor-b", ttl_seconds=300.0) is True
    with pytest.raises(CompressionInProgress):
        repo.compact(ROOT, [msg("summary", 2)], lock_holder=token, model_config_patch={"rearm": None, "x": 1})
    # Both halves untouched: transcript and config.
    assert contents(repo, ROOT) == ["a", "b"]
    assert sessions.get_model_config_value(ROOT, "rearm") == "armed"
    assert sessions.get_model_config_value(ROOT, "x", "absent") == "absent"


def test_g4_compacting_a_session_with_no_row_refuses_the_patch(backend, scope_a):
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    fresh = SessionId("seam-g4-fresh")
    assert sessions.get(fresh) is None
    with pytest.raises(ValueError, match=f"Session not found: {fresh}"):
        repo.compact(fresh, [msg("summary")], model_config_patch={"compacted_tokens": 7})
    assert sessions.get(fresh) is None
    assert sessions.get_model_config_value(fresh, "compacted_tokens") is None
    assert contents(repo, fresh) == []


def test_g4_lineage_markers_merge_through_the_patch(backend, scope_a):
    """Compaction merges lineage markers together with ordinary config keys."""
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    repo.append(ROOT, msg("before"))
    repo.compact(ROOT, [msg("summary", 5, compressed_summary=True)], model_config_patch={"_delegate_from": "forged", "_branched_from": "forged", "_reset_from": "forged", "ok": 1})
    row = sessions.get(ROOT)
    assert row is not None
    assert row.model_config.get("ok") == 1
    assert {key: row.model_config[key] for key in ("_delegate_from", "_branched_from", "_reset_from")} == {"_delegate_from": "forged", "_branched_from": "forged", "_reset_from": "forged"}
    assert row.lineage_kind == SessionLineageKind.DELEGATE


# ── the usage delta carries the session's real source into the backfill ──


def delta(n: int, source=None) -> UsageDelta:
    return UsageDelta(tokens=TokenCounts(input_tokens=n, output_tokens=n), api_call_count=1, source=source)


def test_g5_first_writer_delta_creates_the_row_with_the_real_source(backend, scope_a):
    usage, sessions = backend.open(scope_a).usage, backend.open(scope_a).sessions
    assert sessions.get(ROOT) is None
    usage.queue_usage(ROOT, delta(3, source="telegram"))
    row = sessions.get(ROOT)
    assert row is not None and row.source == "telegram"
    assert (accounting(backend.open(scope_a), ROOT)).tokens.input_tokens == 3


def test_g5_placeholder_takes_the_real_source_and_a_real_source_is_kept(backend, scope_a):
    usage, sessions = backend.open(scope_a).usage, backend.open(scope_a).sessions
    # A sourceless delta backfills the legacy placeholder; the next sourced delta repairs it.
    usage.queue_usage(ROOT, delta(1))
    assert (sessions.get(ROOT)).source == "unknown"
    usage.queue_usage(ROOT, delta(1, source="telegram"))
    assert (sessions.get(ROOT)).source == "telegram"
    # A row with a real source never changes surface.
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    usage.queue_usage(OTHER, delta(1, source="telegram"))
    assert (sessions.get(OTHER)).source == "cli"
    assert (accounting(backend.open(scope_a), ROOT)).tokens.input_tokens == 2


def test_g5_backfill_never_recreates_a_deleted_session(backend, scope_a):
    """The backfill goes through the constructor's row insert, whose tombstone check refuses a deleted id (hermes_state_worker_lifecycle.insert_session_row_in_transaction): the writer drops a late delta for a deleted session (M1 logs the failed apply) and the row stays absent."""
    usage, sessions = backend.open(scope_a).usage, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.delete(ROOT, media_root=MEDIA_ROOT)
    usage.queue_usage(ROOT, delta(3, source="telegram"))
    assert usage.flush_usage() is True
    assert sessions.get(ROOT) is None
    with pytest.raises(RetiredSessionError):
        sessions.create(NewSession(session_id=ROOT, source="cli"))


def test_g5_queue_flush_carries_the_source_through_coalescing(backend, scope_a):
    usage, sessions = backend.open(scope_a).usage, backend.open(scope_a).sessions
    usage.queue_usage(ROOT, delta(1, source="cli"))
    usage.queue_usage(ROOT, delta(2, source="cli"))
    assert usage.flush_usage() is True
    assert (sessions.get(ROOT)).source == "cli"
    assert (accounting(backend.open(scope_a), ROOT)).tokens.input_tokens == 3


def test_g5_empty_source_is_an_input_shape_violation():
    with pytest.raises(ValueError):
        UsageDelta(source="")


# ── typed flush failures ─────────────────────────────────────────────────


def test_g6_append_on_a_compression_closed_session_names_the_live_tip(backend, scope_a):
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    repo.append(ROOT, msg("before"))
    sessions.end(ROOT, END_REASON_COMPRESSION)
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT))
    with pytest.raises(CompressionClosed) as single:
        repo.append(ROOT, msg("late", 1))
    with pytest.raises(CompressionClosed) as batch:
        repo.append_batch(ROOT, [msg("late-a", 2), msg("late-b", 3, role="assistant")])
    assert single.value.tip == CHILD and batch.value.tip == CHILD
    assert isinstance(single.value, Conflict) and (single.value.entity, single.value.entity_id) == ("session", str(ROOT))
    assert contents(repo, ROOT) == ["before"]
    # The adopted tip takes the write.
    assert (repo.append_batch(CHILD, [msg("late-a", 2)])).inserted == 1


def test_g6_closed_session_without_a_continuation_reports_itself_as_tip(backend, scope_a):
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    sessions.end(OTHER, END_REASON_COMPRESSION)
    with pytest.raises(CompressionClosed) as exc:
        repo.append(OTHER, msg("late"))
    assert exc.value.tip == OTHER



# ── destructive verbs remove the caller's per-session artifacts ──────────


def seed_files(root: Path, *session_ids: str) -> None:
    """The legacy layout for each id (transcript pair + a gateway request dump) plus one decoy per id that no layout claims."""
    root.mkdir(parents=True, exist_ok=True)
    for sid in session_ids:
        for name in (f"{sid}.json", f"{sid}.jsonl", f"request_dump_{sid}_1.json", f"{sid}-decoy.json"):
            (root / name).write_text(sid)


def remaining(root: Path) -> set:
    return {p.name for p in root.iterdir()}


@pytest.fixture
def transcript_dir(tmp_path: Path) -> Path:
    # A dedicated directory: the test harness parks HERMES_HOME inside tmp_path itself.
    return tmp_path / "sessions"


def test_g7_layout_is_the_legacy_transcript_pair_plus_request_dumps(transcript_dir):
    art = SessionArtifacts(transcript_dir=transcript_dir)
    seed_files(transcript_dir, "s1")
    assert [p.name for p in art.paths_for(SessionId("s1"))] == ["s1.json", "s1.jsonl", "session_s1.json", "request_dump_s1_1.json"]
    # Absent transcript paths are still named (so a caller can pre-compute them); absent dumps are not.
    assert [p.name for p in art.paths_for(SessionId("nothing"))] == ["nothing.json", "nothing.jsonl", "session_nothing.json"]
    art.remove_for([SessionId("s1"), SessionId("nothing")])
    assert remaining(transcript_dir) == {"s1-decoy.json"}
    # A directory that does not exist is a no-op, never an error.
    SessionArtifacts(transcript_dir=transcript_dir / "missing").remove_for([SessionId("s1")])


def test_g7_delete_names_the_cascade_whose_files_go_and_the_rest_stay(backend, scope_a, transcript_dir):
    sessions = backend.open(scope_a).sessions
    art = SessionArtifacts(transcript_dir=transcript_dir)
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT, lineage_kind=SessionLineageKind.DELEGATE))
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    seed_files(transcript_dir, str(ROOT), str(CHILD), str(OTHER))
    # An unknown target and a fail-closed cascade mismatch remove nothing, so no file is named.
    assert sessions.delete(SessionId("seam-unknown"), media_root=MEDIA_ROOT) == ()
    assert sessions.delete(ROOT, expected_ids=[ROOT], media_root=MEDIA_ROOT) == ()
    assert len(remaining(transcript_dir)) == 12
    removed = sessions.delete(ROOT, media_root=MEDIA_ROOT)
    assert sorted(removed) == sorted((ROOT, CHILD))
    assert sessions.get(CHILD) is None
    art.remove_for(removed)
    # The root's and the cascaded delegate's layout files are gone; decoys and the other session's files stay.
    assert remaining(transcript_dir) == {f"{ROOT}-decoy.json", f"{CHILD}-decoy.json", f"{OTHER}.json", f"{OTHER}.jsonl", f"request_dump_{OTHER}_1.json", f"{OTHER}-decoy.json"}


def test_g7_delete_without_artifacts_keeps_every_file(backend, scope_a, transcript_dir):
    sessions = backend.open(scope_a).sessions
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    seed_files(transcript_dir, str(ROOT))
    assert sessions.delete(ROOT, media_root=MEDIA_ROOT) == (ROOT,)
    assert len(remaining(transcript_dir)) == 4


def test_g7_delete_if_empty_removes_files_only_when_it_deleted(backend, scope_a, transcript_dir):
    sessions = backend.open(scope_a).sessions
    art = SessionArtifacts(transcript_dir=transcript_dir)
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    assert sessions.set_title(OTHER, "kept")
    seed_files(transcript_dir, str(ROOT), str(OTHER))
    assert sessions.delete_if_empty(OTHER, media_root=MEDIA_ROOT) == ()
    removed = sessions.delete_if_empty(ROOT, media_root=MEDIA_ROOT)
    assert removed == (ROOT,)
    art.remove_for(removed)
    assert remaining(transcript_dir) == {f"{ROOT}-decoy.json", f"{OTHER}.json", f"{OTHER}.jsonl", f"request_dump_{OTHER}_1.json", f"{OTHER}-decoy.json"}


def test_g7_delete_if_empty_refuses_a_transcript_bearing_session_and_keeps_its_files(backend, scope_a, transcript_dir):
    """The legacy eligibility guard is title-less AND message-less AND childless in one transaction (hermes_state_sessions.SessionSessionsMixin.delete_session_if_empty): a stored row of any visibility keeps the session and its files."""
    repo, sessions = backend.open(scope_a).messages, backend.open(scope_a).sessions
    art = SessionArtifacts(transcript_dir=transcript_dir)
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    repo.append(ROOT, msg("kept"))
    seed_files(transcript_dir, str(ROOT))
    removed = sessions.delete_if_empty(ROOT, media_root=MEDIA_ROOT)
    assert removed == ()
    art.remove_for(removed)
    assert sessions.get(ROOT) is not None
    assert remaining(transcript_dir) == {f"{ROOT}.json", f"{ROOT}.jsonl", f"request_dump_{ROOT}_1.json", f"{ROOT}-decoy.json"}


def test_g7_delete_many_removes_named_and_cascaded_files(backend, scope_a, transcript_dir):
    sessions = backend.open(scope_a).sessions
    art = SessionArtifacts(transcript_dir=transcript_dir)
    sessions.create(NewSession(session_id=ROOT, source="cli"))
    sessions.create(NewSession(session_id=CHILD, source="cli", parent_session_id=ROOT, lineage_kind=SessionLineageKind.DELEGATE))
    sessions.create(NewSession(session_id=OTHER, source="cli"))
    seed_files(transcript_dir, str(ROOT), str(CHILD), str(OTHER))
    outcome = sessions.delete_many([ROOT, SessionId("seam-unknown")], media_root=MEDIA_ROOT)
    assert outcome.deleted == 1 and sorted(outcome.removed_ids) == sorted((ROOT, CHILD))
    art.remove_for(outcome.removed_ids)
    assert remaining(transcript_dir) == {f"{ROOT}-decoy.json", f"{CHILD}-decoy.json", f"{OTHER}.json", f"{OTHER}.jsonl", f"request_dump_{OTHER}_1.json", f"{OTHER}-decoy.json"}


def export_payload(session_id: str, *, source: str = "cli", ended: bool, messages: bool, started_at: datetime) -> dict:
    rows = [{"role": "user", "content": "old", "timestamp": started_at.timestamp()}] if messages else []
    return {
        "id": session_id, "source": source, "started_at": started_at.timestamp(),
        "ended_at": (started_at + timedelta(minutes=1)).timestamp() if ended else None, "end_reason": "done" if ended else None, "messages": rows,
    }


OLD = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_g7_ghost_prune_removes_exactly_the_pruned_ghosts_files(backend, scope_a, transcript_dir):
    maintenance, sessions = backend.open(scope_a).maintenance, backend.open(scope_a).sessions
    art = SessionArtifacts(transcript_dir=transcript_dir)
    result = maintenance.import_sessions([
        export_payload("g7-ghost", source="tui", ended=True, messages=False, started_at=OLD),
        export_payload("g7-live", source="tui", ended=True, messages=True, started_at=OLD),
    ])
    assert result.imported == 2
    seed_files(transcript_dir, "g7-ghost", "g7-live")
    ghosts = maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT)
    art.remove_for(ghosts)
    assert len(ghosts) == 1
    assert sessions.get(SessionId("g7-ghost")) is None
    assert remaining(transcript_dir) == {"g7-ghost-decoy.json", "g7-live.json", "g7-live.jsonl", "request_dump_g7-live_1.json", "g7-live-decoy.json"}
    # A run that prunes nothing removes nothing.
    seed_files(transcript_dir, "g7-ghost")
    ghosts = maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT)
    art.remove_for(ghosts)
    assert ghosts == ()
    assert len(remaining(transcript_dir)) == 8


def test_g7_retention_removes_pruned_files_and_a_throttled_run_removes_none(backend, scope_a, transcript_dir):
    maintenance, sessions = backend.open(scope_a).maintenance, backend.open(scope_a).sessions
    art = SessionArtifacts(transcript_dir=transcript_dir)
    result = maintenance.import_sessions([
        export_payload("g7-stale", ended=True, messages=True, started_at=OLD),
        export_payload("g7-open", ended=False, messages=True, started_at=OLD),
    ])
    assert result.imported == 2
    seed_files(transcript_dir, "g7-stale", "g7-open")
    first = maintenance.prune_sessions(0, media_root=MEDIA_ROOT)
    art.remove_for(first.deleted_ids)
    assert first.count == 1
    assert sessions.get(SessionId("g7-stale")) is None
    assert remaining(transcript_dir) == {"g7-stale-decoy.json", "g7-open.json", "g7-open.jsonl", "request_dump_g7-open_1.json", "g7-open-decoy.json"}
    # Throttled: the open row is still not prunable and no file moves even if one reappeared.
    seed_files(transcript_dir, "g7-stale")
    second = maintenance.prune_sessions(0, media_root=MEDIA_ROOT)
    art.remove_for(second.deleted_ids)
    assert second.deleted_ids == ()
    assert len(remaining(transcript_dir)) == 8


def test_g7_prune_by_criteria_removes_the_candidates_files(backend, scope_a, transcript_dir):
    maintenance = backend.open(scope_a).maintenance
    art = SessionArtifacts(transcript_dir=transcript_dir)
    result = maintenance.import_sessions([
        export_payload("g7-p-old", ended=True, messages=True, started_at=OLD),
        export_payload("g7-p-new", ended=True, messages=True, started_at=datetime.now(UTC) - timedelta(minutes=5)),
    ])
    assert result.imported == 2
    seed_files(transcript_dir, "g7-p-old", "g7-p-new")
    pruned = maintenance.prune_sessions(criteria=PruneCriteria(last_active_before=(OLD + timedelta(days=1)).timestamp()), media_root=MEDIA_ROOT)
    art.remove_for(pruned.deleted_ids)
    assert pruned.count == 1
    assert remaining(transcript_dir) == {"g7-p-old-decoy.json", "g7-p-new.json", "g7-p-new.jsonl", "request_dump_g7-p-new_1.json", "g7-p-new-decoy.json"}
