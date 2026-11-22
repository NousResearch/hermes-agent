"""Contract cases for the maintenance capability's operational verbs: heartbeats, the orphan sweep, the ghost prune, the prune counts and the foreign import.

Runs against every registered backend through public protocol operations only. Old rows enter through ``import_sessions`` (the one public door that lands a past ``started_at``), liveness rows through ``register_backend_heartbeat``, leases through the coordination repository; every observation goes through a different public operation than the one that seeded it. Race cases run two contenders on their own threads against one handle.
"""

from __future__ import annotations

import time
from datetime import datetime, UTC
from functools import partial
from typing import Any, Callable

import pytest

from persistence import RetiredSessionError, SessionId
from persistence.conversation.sessions import NewSession
from persistence.housekeeping.maintenance import (
    GHOST_SESSION_MIN_AGE,
    IMPORT_MAX_MESSAGES_PER_SESSION,
    IMPORT_MAX_SESSION_BYTES,
    ORPHAN_REAP_END_REASON,
    ForeignImportResult,
    ForeignOrigin,
    ForeignTurn,
    PruneCriteria,
)
from tests.contracts._concurrency import run_concurrently
from tests.contracts._scopes import MEDIA_ROOT


_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
_DAY = 86400.0
_HOUR = 3600.0


def _msg(role: str, content: str, offset: float) -> dict[str, Any]:
    return {"role": role, "content": content, "timestamp": _BASE + offset}


def _payload(session_id: str, *, source: str = "tui", messages: list[dict[str, Any]] | None = None, ended_at: float | None = None, end_reason: str | None = None, started_at: float = _BASE, **fields: Any) -> dict[str, Any]:
    msgs = [_msg("user", "old alpha", 0), _msg("assistant", "old bravo", 1)] if messages is None else messages
    return {"id": session_id, "source": source, "started_at": started_at, "ended_at": ended_at, "end_reason": end_reason, "model": "model-a", "messages": msgs, **fields}


def _seed(backend, scope, *payloads: dict[str, Any]) -> None:
    result = backend.open(scope).maintenance.import_sessions(list(payloads))
    assert result.imported == len(payloads), result


def _race(*calls: Callable[[], Any]) -> list:
    """Run each call on its own thread, released together by a barrier; results in call order."""
    return run_concurrently(*calls)


def _hb(store, backend_id: str, *, started_at: float, last_heartbeat: float | None = None, pid: int = 4242, profile: str = "p") -> None:
    store.maintenance.register_backend_heartbeat(backend_id=backend_id, pid=pid, started_at=started_at, last_heartbeat=last_heartbeat, profile=profile, host="h")


def _sweep(store, **kw: Any) -> tuple[SessionId, ...]:
    return store.maintenance.sweep_orphaned_sessions(max_idle_seconds=_HOUR, **kw)


# ── backend liveness ────────────────────────────────────────────────────────


def test_heartbeat_register_upserts_by_backend_id(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _hb(store_a, "b1", started_at=now - _DAY, profile="first")
    _hb(store_a, "b1", started_at=now, profile="second")
    # CONTRACT: one row per backend_id; a re-register overwrites the whole row (last write wins), so the first owner tag is gone.
    assert store_a.profile_identity.purge_profile("first").gateway_heartbeats == 0
    assert store_a.profile_identity.purge_profile("second").gateway_heartbeats == 1
    assert store_a.maintenance.clear_backend_heartbeat("b1") is False


def test_heartbeat_last_heartbeat_none_stamps_now(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a, _payload("hb-now"))
    # CONTRACT: last_heartbeat None stamps the current time, so the row is a live backend that started before the session and protects it.
    _hb(store_a, "owner", started_at=_BASE - _DAY)
    assert _sweep(store_a) == ()


def test_heartbeat_two_writers_one_id_leave_one_row(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _race(lambda: _hb(store_a, "b-race", started_at=now, pid=11), lambda: _hb(store_a, "b-race", started_at=now, pid=12))
    # RACE OUTCOME: two writers for one id leave exactly one row.
    assert store_a.profile_identity.purge_profile("p").gateway_heartbeats == 1


def test_heartbeat_clear_true_exactly_once(backend, scope_a):
    store_a = backend.open(scope_a)
    _hb(store_a, "b2", started_at=time.time())
    # CONTRACT: clear removes the row and reports True once; a repeat, an unknown id and an empty id report False and write nothing.
    assert store_a.maintenance.clear_backend_heartbeat("b2") is True
    assert store_a.maintenance.clear_backend_heartbeat("b2") is False
    assert store_a.maintenance.clear_backend_heartbeat("never") is False
    assert store_a.maintenance.clear_backend_heartbeat("") is False


def test_heartbeat_clear_race_exactly_one_winner(backend, scope_a):
    store_a = backend.open(scope_a)
    _hb(store_a, "b3", started_at=time.time())
    outcomes = _race(lambda: store_a.maintenance.clear_backend_heartbeat("b3"), lambda: store_a.maintenance.clear_backend_heartbeat("b3"))
    # RACE OUTCOME: exactly one of two concurrent clears removes the row.
    assert sorted(outcomes) == [False, True]


def test_heartbeat_empty_backend_id_registers_nothing(backend, scope_a):
    store_a = backend.open(scope_a)
    _hb(store_a, "", started_at=_BASE)
    # CONTRACT: an empty backend_id is a no-op; no anonymous row appears.
    assert store_a.profile_identity.purge_profile("p").gateway_heartbeats == 0


# ── orphan sweep ──────────────────────────────────────────────────────────────


def _open_ids(backend, scope, ids: tuple[str, ...]) -> set[str]:
    store = backend.open(scope)
    return {sid for sid in ids if (row := store.sessions.get(SessionId(sid))) is not None and row.ended_at is None}


def test_sweep_reaps_only_stale_owned_open_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _seed(
        backend, scope_a,
        _payload("sw-victim"),
        _payload("sw-cli", source="cli"),
        _payload("sw-ended", ended_at=_BASE + _HOUR, end_reason="done"),
        _payload("sw-fresh-msg", messages=[{"role": "user", "content": "recent", "timestamp": now}]),
        _payload("sw-young", started_at=now, messages=[]),
        _payload("sw-excluded"),
    )
    victims = _sweep(store_a, exclude_ids=("sw-excluded",))
    # CONTRACT: only open rows of an owned source, stale on BOTH clocks (started_at and recency), not excluded, are reaped.
    assert victims == (SessionId("sw-victim"),)
    reaped = store_a.sessions.get(SessionId("sw-victim"))
    # CONTRACT: the reaped row is ended with the orphan-reap reason, transcript intact.
    assert reaped is not None and reaped.ended_at is not None and reaped.end_reason == ORPHAN_REAP_END_REASON
    exported = store_a.maintenance.export_session(SessionId("sw-victim"))
    assert exported is not None and len(exported.messages) == 2
    # CONTRACT: the spared rows are untouched, and the previously ended row keeps its own reason (first reason wins).
    seeded = ("sw-victim", "sw-cli", "sw-ended", "sw-fresh-msg", "sw-young", "sw-excluded")
    assert _open_ids(backend, scope_a, seeded) == {"sw-cli", "sw-fresh-msg", "sw-young", "sw-excluded"}
    assert store_a.sessions.get(SessionId("sw-ended")).end_reason == "done"
    # CONTRACT: the exclusion belongs to the caller, not the row: a sweep without it reaps the row; then nothing is left.
    assert _sweep(store_a) == (SessionId("sw-excluded"),)
    assert _sweep(store_a) == ()


def test_sweep_respects_live_backend_heartbeats(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _seed(backend, scope_a, _payload("sw-owned"))
    # A live backend that started before the row (within grace) could own it: spared.
    _hb(store_a, "owner", started_at=_BASE - _DAY, last_heartbeat=now)
    assert _sweep(store_a) == ()
    # A stale heartbeat protects nothing, nor does a live backend that started long after the row plus grace.
    _hb(store_a, "owner", started_at=_BASE - _DAY, last_heartbeat=now - _DAY)
    _hb(store_a, "respawn", started_at=now, last_heartbeat=now)
    # CONTRACT: the row is reaped once no live heartbeat satisfies BOTH the staleness and the ownership-grace predicate.
    assert _sweep(store_a) == (SessionId("sw-owned"),)


def test_sweep_heartbeat_gate_can_be_disabled(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a, _payload("sw-gate"))
    _hb(store_a, "owner", started_at=_BASE - _DAY, last_heartbeat=time.time())
    # CONTRACT: respect_gateway_heartbeats=False reaps as if no heartbeat existed.
    assert _sweep(store_a, respect_gateway_heartbeats=False) == (SessionId("sw-gate"),)


def test_sweep_spares_leased_and_pinned_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a, _payload("sw-leased"), _payload("sw-pinned"), _payload("sw-plain"))
    assert store_a.coordination.try_acquire_compression_lock(session_id=SessionId("sw-leased"), holder="h1", ttl_seconds=60) is True
    assert store_a.sessions.set_pinned(SessionId("sw-pinned"), True)
    # CONTRACT: a row under a live compression lock is spared; exclude_pinned spares pinned rows.
    assert _sweep(store_a, exclude_pinned=True) == (SessionId("sw-plain"),)
    store_a.coordination.release_compression_lock(session_id=SessionId("sw-leased"), holder="h1")
    # CONTRACT: pinned rows are reaped when the sweep does not exclude them; the released row is reaped too.
    assert set(_sweep(store_a)) == {SessionId("sw-leased"), SessionId("sw-pinned")}


def test_sweep_two_contenders_disjoint_union_is_victim_set(backend, scope_a):
    store_a = backend.open(scope_a)
    ids = [f"sw-race-{i}" for i in range(6)]
    _seed(backend, scope_a, *(_payload(sid) for sid in ids))
    a, b = _race(lambda: _sweep(store_a), lambda: _sweep(store_a))
    # RACE OUTCOME: disjoint id sets whose union is the victim set; every victim ended exactly once with the reap reason.
    assert set(a).isdisjoint(set(b))
    assert set(a) | set(b) == {SessionId(sid) for sid in ids}
    assert all(store_a.sessions.get(SessionId(sid)).end_reason == ORPHAN_REAP_END_REASON for sid in ids)


def test_sweep_degenerate_arguments_are_a_noop(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a, _payload("sw-noop"))
    # CONTRACT: a non-positive idle window or no non-empty source sweeps nothing and touches nothing.
    assert store_a.maintenance.sweep_orphaned_sessions(max_idle_seconds=0) == ()
    assert store_a.maintenance.sweep_orphaned_sessions(max_idle_seconds=_HOUR, sources=()) == ()
    assert store_a.maintenance.sweep_orphaned_sessions(max_idle_seconds=_HOUR, sources=("",)) == ()
    assert store_a.sessions.get(SessionId("sw-noop")).ended_at is None


def test_sweep_degenerate_heartbeat_windows_fall_back_to_the_defaults(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _seed(backend, scope_a, _payload("sw-owned-win"))
    _hb(store_a, "owner", started_at=_BASE - _DAY, last_heartbeat=now)
    # CONTRACT: a non-positive staleness or a negative grace is not honoured literally; the defaults (staleness 2 * max_idle, grace = staleness) apply, so the live owner still protects the row.
    assert _sweep(store_a, heartbeat_staleness_seconds=0) == ()
    assert _sweep(store_a, heartbeat_ownership_grace_seconds=-1) == ()
    # CONTRACT: the default staleness is 2 * max_idle: a heartbeat 90 minutes old still protects at max_idle one hour...
    _hb(store_a, "owner", started_at=_BASE - _DAY, last_heartbeat=now - 1.5 * _HOUR)
    assert _sweep(store_a) == ()
    # ...and an explicit positive staleness is honoured: a heartbeat older than it protects nothing.
    assert _sweep(store_a, heartbeat_staleness_seconds=300) == (SessionId("sw-owned-win"),)


# ── ghost prune and prune counts ─────────────────────────────────────────────


def _ghost(session_id: str, **kw: Any) -> dict[str, Any]:
    return _payload(session_id, messages=[], ended_at=_BASE + 60, end_reason="done", **kw)


def test_ghost_prune_deletes_only_true_ghosts(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _seed(
        backend, scope_a,
        _ghost("gh-ghost"),
        _ghost("gh-titled"),
        _ghost("gh-cli", source="cli"),
        _payload("gh-with-msgs", ended_at=_BASE + 60, end_reason="done"),
        _payload("gh-open", messages=[]),
        _ghost("gh-young", started_at=now - GHOST_SESSION_MIN_AGE.total_seconds() + _HOUR),
    )
    assert store_a.sessions.set_title(SessionId("gh-titled"), "kept")
    # CONTRACT: only ended, untitled, message-less tui rows past the age floor are deleted, and their ids are returned.
    assert store_a.maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT) == (SessionId("gh-ghost"),)
    assert store_a.sessions.get(SessionId("gh-ghost")) is None
    # CONTRACT: a pruned id is retired like a deleted one.
    with pytest.raises(RetiredSessionError):
        store_a.sessions.create(NewSession(SessionId("gh-ghost"), "tui"))
    for sid in ("gh-titled", "gh-cli", "gh-with-msgs", "gh-open", "gh-young"):
        assert store_a.sessions.get(SessionId(sid)) is not None, sid
    # CONTRACT: a second run finds nothing.
    assert store_a.maintenance.prune_empty_ghost_sessions(media_root=MEDIA_ROOT) == ()


def test_ghost_prune_two_contenders_disjoint(backend, scope_a):
    store_a = backend.open(scope_a)
    ids = [f"gh-race-{i}" for i in range(5)]
    _seed(backend, scope_a, *(_ghost(sid) for sid in ids))
    prune = partial(store_a.maintenance.prune_empty_ghost_sessions, media_root=MEDIA_ROOT)
    a, b = _race(prune, prune)
    # RACE OUTCOME: each ghost is deleted exactly once; the two id sets are disjoint and cover the ghost set.
    assert set(a).isdisjoint(b) and set(a) | set(b) == {SessionId(sid) for sid in ids}
    assert all(store_a.sessions.get(SessionId(sid)) is None for sid in ids)


def test_prune_counts_mirror_candidates_and_open_skips(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(
        backend, scope_a,
        _payload("pc-ended-a", ended_at=_BASE + _HOUR, end_reason="done"),
        _payload("pc-ended-b", ended_at=_BASE + _HOUR, end_reason="done"),
        _payload("pc-open"),
        _payload("pc-cli", source="cli", ended_at=_BASE + _HOUR, end_reason="done"),
    )
    for sid, title in (("pc-ended-a", "Alpha work"), ("pc-ended-b", "beta work"), ("pc-open", "alpha open")):
        assert store_a.sessions.set_title(SessionId(sid), title)
    criteria = PruneCriteria(last_active_before=time.time())
    candidates = store_a.maintenance.list_prune_candidates(None, "tui", criteria=criteria)
    # CONTRACT: count_prune_matches equals the candidate count for the same arguments.
    assert store_a.maintenance.count_prune_matches(None, "tui", criteria=criteria) == len(candidates) == 2
    # CONTRACT: count_open_prune_matches reports the OPEN rows the same selection would otherwise match, nothing else.
    assert store_a.maintenance.count_open_prune_matches(None, "tui", criteria=criteria) == 1
    # CONTRACT: text filters are case-insensitive substrings and compose with the rest.
    alpha = PruneCriteria(last_active_before=time.time(), title_like="ALPHA")
    assert store_a.maintenance.count_prune_matches(criteria=alpha) == 1
    assert store_a.maintenance.count_open_prune_matches(criteria=alpha) == 1
    # CONTRACT: include_open lifts the ended guard, so the open count is 0 without a read.
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(include_open=True)) == 4
    assert store_a.maintenance.count_open_prune_matches(criteria=PruneCriteria(include_open=True)) == 0
    # CONTRACT: the counts never write.
    assert store_a.maintenance.count_prune_matches() == 3
    assert store_a.sessions.get(SessionId("pc-ended-a")) is not None


def test_prune_age_window_and_its_override(backend, scope_a):
    store_a = backend.open(scope_a)
    now = time.time()
    _seed(backend, scope_a, _payload("pw-old", ended_at=_BASE + 60, end_reason="done"), _payload("pw-new", started_at=now - _HOUR, messages=[], ended_at=now - _HOUR, end_reason="done"))
    # CONTRACT: older_than_days is the inactivity window: one day admits the old row only, None admits both.
    assert [str(c.session_id) for c in store_a.maintenance.list_prune_candidates(1)] == ["pw-old"]
    assert store_a.maintenance.count_prune_matches(None) == 2
    # CONTRACT: an explicit started_before overrides the window.
    assert store_a.maintenance.count_prune_matches(1, criteria=PruneCriteria(started_before=now)) == 2
    # CONTRACT: a negative window is a ValueError on every prune verb.
    for verb in (store_a.maintenance.list_prune_candidates, store_a.maintenance.count_prune_matches, store_a.maintenance.count_open_prune_matches,
                 partial(store_a.maintenance.prune_sessions, media_root=MEDIA_ROOT)):
        with pytest.raises(ValueError):
            verb(-1)
    # CONTRACT: prune_sessions defaults to 90 days and refuses include_open before touching anything.
    with pytest.raises(ValueError):
        store_a.maintenance.prune_sessions(criteria=PruneCriteria(include_open=True), media_root=MEDIA_ROOT)
    result = store_a.maintenance.prune_sessions(media_root=MEDIA_ROOT)
    assert result.count == 1 and result.deleted_ids == (SessionId("pw-old"),)
    assert store_a.sessions.get(SessionId("pw-new")) is not None


def test_prune_criteria_pinned_archived_and_message_bounds(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(
        backend, scope_a,
        _payload("pf-pinned", ended_at=_BASE + _HOUR, end_reason="done"),
        _payload("pf-archived", ended_at=_BASE + _HOUR, end_reason="done"),
        _payload("pf-empty", messages=[], ended_at=_BASE + _HOUR, end_reason="done"),
    )
    assert store_a.sessions.set_pinned(SessionId("pf-pinned"), True)
    assert store_a.sessions.set_archived(SessionId("pf-archived"), True)
    # CONTRACT: pinned rows are excluded unless include_pinned; pinned_only counts the pinned rows themselves.
    assert {str(c.session_id) for c in store_a.maintenance.list_prune_candidates()} == {"pf-archived", "pf-empty"}
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(include_pinned=True)) == 3
    assert store_a.maintenance.count_prune_matches(pinned_only=True) == 1
    # CONTRACT: archived is tri-state and the candidate carries the stored 0/1.
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(archived=True)) == 1
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(archived=False)) == 1
    assert {c.archived for c in store_a.maintenance.list_prune_candidates()} == {0, 1}
    # CONTRACT: message bounds are inclusive on the stored count.
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(max_messages=0)) == 1
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(min_messages=2, include_pinned=True)) == 2
    # CONTRACT: prune honours the same criteria (only the empty, unpinned, unarchived row goes).
    assert store_a.maintenance.prune_sessions(None, criteria=PruneCriteria(archived=False, max_messages=0), media_root=MEDIA_ROOT).count == 1
    assert store_a.sessions.get(SessionId("pf-empty")) is None
    assert store_a.sessions.get(SessionId("pf-pinned")) is not None


def test_prune_last_active_before_ages_swept_rows_from_their_sweep(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed(backend, scope_a, _payload("pa-swept"))
    assert _sweep(store_a) == (SessionId("pa-swept"),)
    stale_cutoff = PruneCriteria(last_active_before=_BASE + _DAY)
    # CONTRACT: the swept row's old activity is before the cutoff, but it ended (was swept) after it: not a candidate yet.
    assert store_a.maintenance.count_prune_matches(criteria=stale_cutoff) == 0
    assert store_a.maintenance.list_prune_candidates(criteria=stale_cutoff) == ()
    # CONTRACT: a cutoff after the sweep admits it.
    assert store_a.maintenance.count_prune_matches(criteria=PruneCriteria(last_active_before=time.time() + 5)) == 1


# ── foreign import ───────────────────────────────────────────────────────────


_ORIGIN = ForeignOrigin(tool="claude-code", path="/tmp/foreign/a.jsonl", foreign_session_id="cc-1")
_TURNS = (ForeignTurn("user", "hello from claude"), ForeignTurn("assistant", "hello back"))


def _foreign(store, origin: ForeignOrigin = _ORIGIN, turns: tuple[ForeignTurn, ...] = _TURNS, *, title: str = "Imported chat", cwd: str | None = None, profile: str = "work") -> ForeignImportResult:
    return store.maintenance.import_foreign_history(origin, turns, title=title, cwd=cwd, profile=profile)


def test_foreign_import_mints_session_with_provenance(backend, scope_a):
    store_a = backend.open(scope_a)
    # CONTRACT: an unknown origin resolves to None before import.
    assert store_a.maintenance.find_foreign_import(_ORIGIN) is None
    result = _foreign(store_a, cwd="/repo")
    assert isinstance(result, ForeignImportResult) and result.already_imported is False
    row = store_a.sessions.get(result.session_id)
    # CONTRACT: the session carries the tool as source, the title, the cwd, the given profile as owner, the origin as provenance, and the turns as its transcript with counters rebuilt.
    assert row is not None and (row.source, row.title, row.cwd, row.profile_name) == ("claude-code", "Imported chat", "/repo", "work")
    assert row.origin == {"imported_from": {"tool": "claude-code", "path": "/tmp/foreign/a.jsonl", "foreign_session_id": "cc-1"}}
    assert row.message_count == 2
    exported = store_a.maintenance.export_session(result.session_id)
    assert [(m.role, m.content) for m in exported.messages] == [("user", "hello from claude"), ("assistant", "hello back")]
    # CONTRACT: the id is 28 characters and the origin now resolves to it.
    assert len(str(result.session_id)) == 28
    assert store_a.maintenance.find_foreign_import(_ORIGIN) == result.session_id


def test_foreign_import_is_idempotent_per_origin_and_identity_rules(backend, scope_a):
    store_a = backend.open(scope_a)
    first = _foreign(store_a)
    second = _foreign(store_a)
    # CONTRACT: a second import of the same origin returns the existing id and writes nothing.
    assert second == ForeignImportResult(session_id=first.session_id, already_imported=True)
    # CONTRACT: identity is foreign_session_id when present: a different path with the same foreign id is the same import; the same path with another foreign id is not.
    assert store_a.maintenance.find_foreign_import(ForeignOrigin(tool="claude-code", path="/elsewhere/b.jsonl", foreign_session_id="cc-1")) == first.session_id
    assert store_a.maintenance.find_foreign_import(ForeignOrigin(tool="claude-code", path=_ORIGIN.path, foreign_session_id="cc-2")) is None
    # CONTRACT: another tool never matches, even on the same path and id.
    assert store_a.maintenance.find_foreign_import(ForeignOrigin(tool="codex", path=_ORIGIN.path, foreign_session_id="cc-1")) is None
    # CONTRACT: without a foreign id the path is the identity.
    by_path = ForeignOrigin(tool="codex", path="/tmp/foreign/c.jsonl")
    minted = _foreign(store_a, by_path, title="Codex chat")
    assert store_a.maintenance.find_foreign_import(by_path) == minted.session_id
    assert store_a.maintenance.find_foreign_import(ForeignOrigin(tool="codex", path="/tmp/foreign/other.jsonl")) is None


def test_foreign_import_suffixes_taken_title(backend, scope_a):
    store_a = backend.open(scope_a)
    first = _foreign(store_a, title="Shared title")
    second = _foreign(store_a, ForeignOrigin(tool="claude-code", path="/tmp/foreign/z.jsonl", foreign_session_id="cc-9"), title="Shared title")
    assert second.already_imported is False and second.session_id != first.session_id
    # CONTRACT: a taken title is suffixed with the new id's last 12 characters so titles stay unique.
    assert store_a.sessions.get(second.session_id).title == f"Shared title ({str(second.session_id)[-12:]})"
    assert store_a.sessions.get(first.session_id).title == "Shared title"


def test_foreign_import_two_contenders_one_winner_same_id(backend, scope_a):
    store_a = backend.open(scope_a)
    a, b = _race(lambda: _foreign(store_a, title="Race"), lambda: _foreign(store_a, title="Race"))
    # RACE OUTCOME: exactly one already_imported=False; both return the same id; one row exists for the origin.
    assert sorted([a.already_imported, b.already_imported]) == [False, True]
    assert a.session_id == b.session_id
    assert store_a.maintenance.find_foreign_import(_ORIGIN) == a.session_id
    assert store_a.sessions.count_by_source().get("claude-code") == 1


def test_foreign_import_refuses_oversize_snapshot_before_writing(backend, scope_a):
    store_a = backend.open(scope_a)
    too_many = tuple(ForeignTurn("user", "x") for _ in range(IMPORT_MAX_MESSAGES_PER_SESSION + 1))
    # CONTRACT: more turns than the per-session import limit is a ValueError with the validator's text, and nothing is written.
    with pytest.raises(ValueError, match="messages exceeds the per-session import limit"):
        _foreign(store_a, turns=too_many, title="Huge")
    assert store_a.maintenance.find_foreign_import(_ORIGIN) is None
    assert store_a.sessions.get_by_title("Huge") is None
    # CONTRACT: the byte limit measures the real UTF-8 bytes, so a multilingual snapshot under it (4 MB of two-byte letters) imports...
    assert _foreign(store_a, turns=(ForeignTurn("user", "\u00e9" * 2_000_000),), title="Wide").already_imported is False
    # ...and a snapshot over it is refused before any write, even when its turns alone would fit (the title counts).
    other = ForeignOrigin(tool="claude-code", path="/tmp/foreign/b.jsonl", foreign_session_id="cc-2")
    with pytest.raises(ValueError, match="session exceeds the import size limit"):
        _foreign(store_a, other, (ForeignTurn("user", "x"),), title="t" * IMPORT_MAX_SESSION_BYTES)
    assert store_a.maintenance.find_foreign_import(other) is None


def test_foreign_records_refuse_empty_identity():
    # CONTRACT: origin and turn records refuse empty identity fields before any store access.
    with pytest.raises(ValueError):
        ForeignOrigin(tool="", path="/x")
    with pytest.raises(ValueError):
        ForeignOrigin(tool="codex", path="")
    with pytest.raises(ValueError):
        ForeignTurn(role="", content="x")
