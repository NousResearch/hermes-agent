"""Contract cases for the guarded delete and prune verbs.

Seeding uses public protocol operations only, and assertions check return values, refusals, and repository re-reads for ``persistence.conversation.sessions.SessionRepository.delete(refuse_guarded=, expected_display=)``, ``persistence.conversation.sessions.SessionRepository.delete_many(skip_guarded=)``, ``persistence.housekeeping.maintenance.SessionMaintenance.list_prune_candidates(whole_lineages=)``, ``persistence.housekeeping.maintenance.SessionMaintenance.prune_sessions(exclude_active_write_guards=)`` and ``PruneCriteria.lineage_tips_only``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from persistence import Conflict, SessionId, SessionWriteGuarded, ProfileScope
from persistence.housekeeping.maintenance import PruneCriteria
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind
from tests.contracts.conftest import require_capability
from tests.contracts.conversation._session_probes import is_retired
from tests.contracts._waiting import wait_past
from tests.contracts._scopes import MEDIA_ROOT


T0 = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
ROOT = SessionId("gd-root")
DELEGATE = SessionId("gd-delegate")
TIP = SessionId("gd-tip")
OTHER = SessionId("gd-other")


def _msg(content: str, i: int, role: str = "user") -> NewMessage:
    return NewMessage(role=role, content=content, timestamp=(T0 + timedelta(seconds=i)).timestamp())


def _session(backend, scope: ProfileScope, sid: SessionId, *, parent: SessionId | None = None,
                   delegate: bool = False, ended: str | None = None, content: str = "hi") -> None:
    store = backend.open(scope)
    store.sessions.create(NewSession(
        session_id=sid, source="cli", parent_session_id=parent,
        lineage_kind=SessionLineageKind.DELEGATE if delegate else SessionLineageKind.NONE))
    store.messages.append(sid, _msg(content, 0))
    if ended:
        store.sessions.end(sid, ended)


def _alive(backend, scope: ProfileScope, *sids: SessionId) -> list[bool]:
    store = backend.open(scope)
    return [store.sessions.get(sid) is not None for sid in sids]


def _display(backend, scope: ProfileScope, sid: SessionId) -> list[dict]:
    store = backend.open(scope)
    rows = store.messages.read_transcript(sid, TranscriptRead(view=TranscriptView.DISPLAY))
    return [row.as_wire_row() for row in rows]


# ── delete(refuse_guarded=) ──────────────────────────────────────────────────


@pytest.mark.parametrize("held_on", ["root", "delegate"])
@pytest.mark.parametrize("lease", ["turn", "compaction"])
def test_a_live_lease_anywhere_in_the_cascade_refuses_the_guarded_delete(backend, scope_a, held_on, lease):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _session(backend, scope_a, ROOT)
    _session(backend, scope_a, DELEGATE, parent=ROOT, delegate=True)
    target = ROOT if held_on == "root" else DELEGATE
    if lease == "turn":
        assert store_a.coordination.try_acquire_turn_lease(session_id=target, holder="live", ttl_seconds=300.0)
    else:
        assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(target)), holder="live", ttl_seconds=300.0)
    with pytest.raises(SessionWriteGuarded) as refused:
        store_a.sessions.delete(ROOT, refuse_guarded=True, media_root=MEDIA_ROOT)
    # The refusal names the requested root and carries the text the CLI prints.
    assert isinstance(refused.value, Conflict) and refused.value.entity_id == str(ROOT)
    assert str(refused.value) == f"session '{ROOT}' (or a delegate child) has an active turn lease or compression lock"
    assert _alive(backend, scope_a, ROOT, DELEGATE) == [True, True]
    assert not is_retired(backend, scope_a, ROOT)
    # Without the flag no lease is consulted.
    assert store_a.sessions.delete(ROOT, media_root=MEDIA_ROOT) == (ROOT, DELEGATE)
    assert _alive(backend, scope_a, ROOT, DELEGATE) == [False, False]


def test_a_live_turn_on_the_tip_guards_its_compression_ancestor(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _session(backend, scope_a, ROOT, ended=END_REASON_COMPRESSION)
    _session(backend, scope_a, TIP, parent=ROOT)
    assert store_a.coordination.try_acquire_turn_lease(session_id=TIP, holder="live", ttl_seconds=300.0)
    with pytest.raises(SessionWriteGuarded):
        store_a.sessions.delete(ROOT, refuse_guarded=True, media_root=MEDIA_ROOT)
    store_a.coordination.release_turn_lease(session_id=TIP, holder="live")
    # Closed by compression and unleased: deletable, the continuation orphaned.
    assert store_a.sessions.delete(ROOT, refuse_guarded=True, media_root=MEDIA_ROOT) == (ROOT,)
    tip = store_a.sessions.get(TIP)
    assert tip is not None and tip.parent_session_id is None


def test_an_expired_lease_does_not_guard(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    _session(backend, scope_a, ROOT)
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(ROOT)), holder="gone", ttl_seconds=0.001)
    wait_past(time.time() + 0.001)
    assert store_a.sessions.delete(ROOT, refuse_guarded=True, media_root=MEDIA_ROOT) == (ROOT,)
    assert _alive(backend, scope_a, ROOT) == [False]


def test_the_guarded_delete_of_an_unknown_id_removes_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    assert store_a.sessions.delete(SessionId("gd-missing"), refuse_guarded=True, expected_display={}, media_root=MEDIA_ROOT) == ()


# ── delete(expected_display=) ───────────────────────────────────────────────


def test_an_unchanged_display_snapshot_permits_the_delete(backend, scope_a):
    store_a = backend.open(scope_a)
    _session(backend, scope_a, ROOT, content="one")
    store_a.messages.append(ROOT, _msg("two", 1, "assistant"))
    store_a.messages.compact(ROOT, [_msg("SUMMARY", 2, "assistant")])
    _session(backend, scope_a, OTHER, content="other")
    snapshot = {ROOT: _display(backend, scope_a, ROOT), OTHER: _display(backend, scope_a, OTHER)}
    # Coverage may name a session outside the cascade; it is checked, never deleted.
    assert store_a.sessions.delete(ROOT, expected_display=snapshot, media_root=MEDIA_ROOT) == (ROOT,)
    assert _alive(backend, scope_a, ROOT, OTHER) == [False, True]


@pytest.mark.parametrize("drift", ["append_to_target", "append_to_covered", "covered_missing"])
def test_a_display_drift_after_capture_fails_closed(backend, scope_a, drift):
    store_a = backend.open(scope_a)
    _session(backend, scope_a, ROOT, content="one")
    _session(backend, scope_a, OTHER, content="other")
    snapshot = {ROOT: _display(backend, scope_a, ROOT), OTHER: _display(backend, scope_a, OTHER)}
    if drift == "append_to_target":
        store_a.messages.append(ROOT, _msg("late", 5))
    elif drift == "append_to_covered":
        store_a.messages.append(OTHER, _msg("late", 5))
    else:
        assert store_a.sessions.delete(OTHER, media_root=MEDIA_ROOT) == (OTHER,)
    assert store_a.sessions.delete(ROOT, expected_display=snapshot, media_root=MEDIA_ROOT) == ()
    assert _alive(backend, scope_a, ROOT) == [True]
    assert not is_retired(backend, scope_a, ROOT)


def test_an_empty_display_expectation_checks_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    _session(backend, scope_a, ROOT)
    store_a.messages.append(ROOT, _msg("late", 5))
    assert store_a.sessions.delete(ROOT, expected_display={}, media_root=MEDIA_ROOT) == (ROOT,)


# ── delete_many(skip_guarded=) ─────────────────────────────────────────────────────────


def test_delete_many_skip_guarded_keeps_leased_selections_and_deletes_the_rest(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    idle, leased, guarded_parent = SessionId("gd-idle"), SessionId("gd-leased"), SessionId("gd-parent")
    for sid in (idle, leased, guarded_parent):
        _session(backend, scope_a, sid)
    _session(backend, scope_a, DELEGATE, parent=guarded_parent, delegate=True)
    assert store_a.coordination.try_acquire_turn_lease(session_id=leased, holder="live", ttl_seconds=300.0)
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(DELEGATE)), holder="live", ttl_seconds=300.0)
    outcome = store_a.sessions.delete_many(
        [leased, idle, SessionId("gd-missing"), idle, guarded_parent], skip_guarded=True, media_root=MEDIA_ROOT)
    # Missing and duplicate ids are ignored; a guarded delegate protects its selected parent; skipped ids are sorted.
    assert outcome.deleted == 1 and outcome.removed_ids == (idle,)
    assert outcome.skipped_guarded == tuple(sorted((leased, guarded_parent)))
    assert _alive(backend, scope_a, idle, leased, guarded_parent, DELEGATE) == [False, True, True, True]
    assert is_retired(backend, scope_a, idle)


def test_delete_many_skip_guarded_with_nothing_leased_deletes_like_delete(backend, scope_a):
    store_a = backend.open(scope_a)
    _session(backend, scope_a, ROOT)
    _session(backend, scope_a, DELEGATE, parent=ROOT, delegate=True)
    outcome = store_a.sessions.delete_many([ROOT], skip_guarded=True, media_root=MEDIA_ROOT)
    assert (outcome.deleted, outcome.skipped_guarded, outcome.removed_ids) == (1, (), (ROOT, DELEGATE))
    assert _alive(backend, scope_a, ROOT, DELEGATE) == [False, False]


# ── prune(refuse_leased=), whole lineages, lineage_tips_only ────────────────


def _everything() -> PruneCriteria:
    return PruneCriteria(last_active_before=(datetime.now(UTC) + timedelta(minutes=5)).timestamp())


def _lineage(backend, scope: ProfileScope, prefix: str, *, tip_ended: bool) -> tuple[SessionId, SessionId]:
    root, tip = SessionId(f"{prefix}-root"), SessionId(f"{prefix}-tip")
    _session(backend, scope, root, ended=END_REASON_COMPRESSION)
    _session(backend, scope, tip, parent=root, ended="done" if tip_ended else None)
    return root, tip


def test_whole_lineage_preview_keeps_the_ancestors_of_a_continuation_still_in_use(backend, scope_a):
    store_a = backend.open(scope_a)
    live_root, live_tip = _lineage(backend, scope_a, "gd-live", tip_ended=False)
    done_root, done_tip = _lineage(backend, scope_a, "gd-done", tip_ended=True)
    plain = {c.session_id for c in store_a.maintenance.list_prune_candidates(criteria=_everything())}
    whole = {c.session_id for c in store_a.maintenance.list_prune_candidates(criteria=_everything(), whole_lineages=True)}
    assert plain == {live_root, done_root, done_tip}
    assert whole == {done_root, done_tip}
    # prune always deletes whole lineages.
    assert store_a.maintenance.prune_sessions(criteria=_everything(), media_root=MEDIA_ROOT).count == 2
    assert _alive(backend, scope_a, live_root, live_tip, done_root, done_tip) == [True, True, False, False]


def test_a_matching_suffix_stays_eligible_while_its_ancestor_is_kept(backend, scope_a):
    """Selection is not all-or-nothing: only ancestors of an unmatched continuation are kept, so the matched suffix below an unmatched (here filtered-out) root stays eligible."""
    store_a = backend.open(scope_a)
    root, mid, tip = SessionId("gd-s-root"), SessionId("gd-s-mid"), SessionId("gd-s-tip")
    _session(backend, scope_a, root, ended=END_REASON_COMPRESSION)
    _session(backend, scope_a, mid, parent=root, ended=END_REASON_COMPRESSION)
    _session(backend, scope_a, tip, parent=mid, ended="done")
    assert store_a.sessions.set_title(mid, "x-mid") and store_a.sessions.set_title(tip, "x-tip")
    titled = PruneCriteria(last_active_before=_everything().last_active_before, title_like="x-")
    whole = {c.session_id for c in store_a.maintenance.list_prune_candidates(criteria=titled, whole_lineages=True)}
    assert whole == {mid, tip}


def test_prune_refuse_leased_skips_leased_candidates(backend, scope_a):
    store_a = backend.open(scope_a)
    require_capability(backend, "transactional_claims")
    leased, idle = SessionId("gd-p-leased"), SessionId("gd-p-idle")
    _session(backend, scope_a, leased, ended="done")
    _session(backend, scope_a, idle, ended="done")
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId(str(leased)), holder="live", ttl_seconds=300.0)
    assert store_a.maintenance.prune_sessions(criteria=_everything(), exclude_active_write_guards=True, media_root=MEDIA_ROOT).count == 1
    assert _alive(backend, scope_a, leased, idle) == [True, False]
    # Without the flag the lease is not consulted.
    assert store_a.maintenance.prune_sessions(criteria=_everything(), media_root=MEDIA_ROOT).count == 1
    assert _alive(backend, scope_a, leased) == [False]


def test_lineage_tips_only_drops_compression_ended_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _root, tip = _lineage(backend, scope_a, "gd-t", tip_ended=True)
    criteria = PruneCriteria(last_active_before=(datetime.now(UTC) + timedelta(minutes=5)).timestamp(), lineage_tips_only=True)
    assert [c.session_id for c in store_a.maintenance.list_prune_candidates(criteria=criteria)] == [tip]
    assert store_a.maintenance.count_prune_matches(criteria=criteria) == 1
