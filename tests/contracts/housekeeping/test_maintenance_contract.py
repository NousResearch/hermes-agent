"""Contract suite for ``persistence.housekeeping.maintenance.SessionMaintenance``: logical export and import, prune, idle archive and the read-only probes.

Runs the real fixture backend end to end. Sessions and transcripts enter through ``import_sessions`` (export dicts) or the sessions and messages repositories; each observation uses a different public operation than the one that seeded the state.
"""

from __future__ import annotations

import time
from datetime import datetime, UTC
from typing import Any

import pytest

from persistence import ExportTooLarge, ProfileScope, SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import CANONICAL_BOT_CHAT_TITLE, END_REASON_COMPRESSION, NewSession
from persistence.housekeeping.maintenance import IMPORT_MAX_SESSIONS, SESSION_PROJECTION_FIELDS, ImportResult, PruneCriteria, TreeServeFacts

from tests.contracts._scopes import MEDIA_ROOT


_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
_HOUR = 3600.0
_FAR_FUTURE = datetime(2100, 1, 1, tzinfo=UTC).timestamp()


def _msg(role: str, content: str, offset: float, tool_name: str | None = None, **fields: Any) -> dict[str, Any]:
    return {"role": role, "content": content, "tool_name": tool_name, "timestamp": _BASE + offset, **fields}


def _transcript() -> list[dict[str, Any]]:
    """The five-message shape: user/assistant/tool/user/assistant."""
    return [
        _msg("user", "port alpha", 0),
        _msg("assistant", "port bravo", 1),
        _msg("tool", "port tool result", 2, tool_name="web_search"),
        _msg("user", "port charlie", 3),
        _msg("assistant", "port delta", 4),
    ]


def _payload(session_id: str, messages: list[dict[str, Any]] | None = None, parent: str | None = None, source: str = "cli", ended_at: float | None = None, end_reason: str | None = None, **fields: Any) -> dict[str, Any]:
    return {
        "id": session_id, "source": source, "started_at": _BASE, "ended_at": ended_at, "end_reason": end_reason, "model": "model-a",
        "parent_session_id": parent, "messages": _transcript() if messages is None else messages, **fields,
    }


def _messages(backend, scope: ProfileScope, session_id: str, **flags: bool) -> tuple:
    exported = backend.open(scope).maintenance.export_session(SessionId(session_id), **flags)
    return () if exported is None else exported.messages


def _view(messages) -> list:
    """Order-preserving (role, content, tool_name) projection over exported rows or payload dicts."""
    return [(m["role"], m["content"], m["tool_name"]) if isinstance(m, dict) else (m.role, m.content, m.tool_name) for m in messages]


# ── import ───────────────────────────────────────────────────────────────────


def test_import_export_round_trip_field_level_equality(backend, scope_a):
    store_a = backend.open(scope_a)
    payload = _payload("s-port-1", ended_at=_BASE + _HOUR, end_reason="done")
    result = store_a.maintenance.import_sessions([payload])
    # CONTRACT: a clean import reports success with the imported ids and no errors.
    assert result == ImportResult(ok=True, imported=1, skipped=0, detached=0, imported_ids=(SessionId("s-port-1"),), skipped_ids=(), errors=())
    exported = store_a.maintenance.export_session(SessionId("s-port-1"))
    # CONTRACT: message content, order, roles, tool names and timestamps survive the round trip exactly.
    assert _view(exported.messages) == _view(payload["messages"])
    assert [m.timestamp for m in exported.messages] == [m["timestamp"] for m in payload["messages"]]
    # CONTRACT: identity and descriptive fields survive field-by-field; the counters count the live rows (no tool_calls were imported, so 0 tool calls).
    row = exported.session
    assert (row.id, row.source, row.model, row.end_reason, row.started_at, row.ended_at, row.archived) == ("s-port-1", "cli", "model-a", "done", _BASE, _BASE + _HOUR, 0)
    assert (row.message_count, row.tool_call_count) == (5, 0)
    assert store_a.sessions.get(SessionId("s-port-1")).end_reason == "done"


def test_import_counters_rebuilt_not_trusted(backend, scope_a):
    store_a = backend.open(scope_a)
    tool_turn = _msg("assistant", "calling", 5, tool_calls=[{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}])
    store_a.maintenance.import_sessions([_payload("s-port-lie", messages=[*_transcript(), tool_turn], message_count=999, tool_call_count=999)])
    # CONTRACT: counters are rebuilt from the imported live rows, never trusted from the payload; tool calls count the parsed tool_calls.
    record = store_a.sessions.get(SessionId("s-port-lie"))
    assert (record.message_count, record.tool_call_count) == (6, 1)


def test_import_resets_live_activity_and_runtime_fields(backend, scope_a):
    store_a = backend.open(scope_a)
    payload = _payload(
        "s-port-act", last_activity_at=_BASE + 300, last_activity_description="working on export", last_activity_provenance="tool",
        session_key="agent:main:x", chat_id="c1", title_source="user", handoff_state="pending",
    )
    store_a.maintenance.import_sessions([payload])
    record = store_a.sessions.get(SessionId("s-port-act"))
    # CONTRACT: import resets the live activity, routing and handoff columns to None even when the payload carries values: it restores history, not ownership.
    assert (record.last_activity_at, record.last_activity_description, record.last_activity_provenance) == (None, None, None)
    assert (record.session_key, record.chat_id, record.handoff_state) == (None, None, None)


def test_import_keeps_flags_and_restores_archived_messages(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = [_msg("user", "live", 0), _msg("assistant", "compacted", 1, active=0, compacted=1), _msg("user", "rewound", 2, active=0, compacted=0)]
    store_a.maintenance.import_sessions([
        _payload("s-port-flags", messages=rows, archived=1, auto_archived=1, pinned=1, hidden=1),
        _payload("s-port-auto", messages=[], auto_archived=1),
    ])
    flagged = store_a.sessions.get(SessionId("s-port-flags"))
    # CONTRACT: archived, pinned and hidden survive; auto_archived survives only on an archived row.
    assert (flagged.archived, flagged.auto_archived, flagged.pinned, flagged.hidden) == (True, True, True, True)
    assert store_a.sessions.get(SessionId("s-port-auto")).auto_archived is False
    # CONTRACT: a message with a false ``active`` is restored archived (compacted per its flag); the counters count the live rows.
    assert flagged.message_count == 1
    assert _view(_messages(backend, scope_a, "s-port-flags")) == [("user", "live", None)]
    every = _messages(backend, scope_a, "s-port-flags", include_inactive=True)
    assert [(m.content, m.as_wire_row()["active"], m.as_wire_row()["compacted"]) for m in every] == [("live", 1, 0), ("compacted", 0, 1), ("rewound", 0, 0)]


def test_import_skips_existing_ids_and_rejects_invalid_payloads_wholesale(backend, scope_a):
    store_a = backend.open(scope_a)
    payload = _payload("s-port-dup")
    assert store_a.maintenance.import_sessions([payload]).imported == 1
    result = store_a.maintenance.import_sessions([payload])
    # CONTRACT: re-importing an existing id skips it without error and without duplication.
    assert (result.ok, result.imported, result.skipped, result.skipped_ids) == (True, 0, 1, (SessionId("s-port-dup"),))
    assert store_a.sessions.get(SessionId("s-port-dup")).message_count == 5
    # CONTRACT: any invalid entry rejects the whole payload with one error per rejected entry (index, text, id when known); even a valid record lands nothing.
    rejected = store_a.maintenance.import_sessions([_payload("s-port-dup2"), _payload("s-port-innocent"), _payload("s-port-dup2"), {"id": ""}, {"id": "s-bad", "messages": [{"role": ""}]}])
    assert (rejected.ok, rejected.imported) == (False, 0)
    assert rejected.errors == (
        {"index": 2, "error": "duplicate session id", "session_id": "s-port-dup2"},
        {"index": 3, "error": "session id is required"},
        {"index": 4, "error": "messages[0].role must be a non-empty string", "session_id": "s-bad"},
    )
    assert store_a.sessions.get(SessionId("s-port-dup2")) is None and store_a.sessions.get(SessionId("s-port-innocent")) is None
    # CONTRACT: a payload that is not a list, or holds too many sessions, is a ValueError.
    with pytest.raises(ValueError):
        store_a.maintenance.import_sessions({"id": "x"})
    with pytest.raises(ValueError):
        store_a.maintenance.import_sessions([_payload(f"s-many-{i}") for i in range(IMPORT_MAX_SESSIONS + 1)])


def test_import_refuses_a_deleted_id_by_name(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([_payload("s-gone", ended_at=_BASE + 60, end_reason="done")])
    assert store_a.maintenance.prune_sessions(None, media_root=MEDIA_ROOT).deleted_ids == (SessionId("s-gone"),)
    result = store_a.maintenance.import_sessions([_payload("s-gone"), _payload("s-fresh")])
    # CONTRACT: an id deleted here refuses the whole payload by name and writes nothing.
    assert result == ImportResult(ok=False, imported=0, skipped=0, detached=0, errors=("session s-gone was deleted here; a deleted session is not restored",))
    assert store_a.sessions.get(SessionId("s-fresh")) is None


def test_import_detaches_missing_parent_keeps_payload_parent(backend, backend_factory, scope_a):
    store_a = backend.open(scope_a)
    child = _payload("s-port-child", parent="s-port-parent")
    parent = _payload("s-port-parent")
    # CONTRACT: a child whose parent is absent from both the store and the payload imports detached and is counted.
    solo = store_a.maintenance.import_sessions([child])
    assert (solo.imported, solo.detached) == (1, 1)
    assert store_a.sessions.get(SessionId("s-port-child")).parent_session_id is None
    # CONTRACT: when the parent arrives in the same payload, the lineage edge is kept in either payload order.
    for batch in ([child, parent], [parent, child]):
        with backend_factory() as fresh:
            both = fresh.open(scope_a).maintenance.import_sessions(batch)
            assert (both.imported, both.detached) == (2, 0)
            assert fresh.open(scope_a).sessions.get(SessionId("s-port-child")).parent_session_id == "s-port-parent"


def test_import_keeps_store_parent_and_drops_only_the_closing_edge_of_a_cycle(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([_payload("s-port-parent2")])
    # CONTRACT: a parent already present in the store keeps the lineage edge; nothing is detached.
    assert store_a.maintenance.import_sessions([_payload("s-port-child2", parent="s-port-parent2")]).detached == 0
    assert store_a.sessions.get(SessionId("s-port-child2")).parent_session_id == "s-port-parent2"
    # CONTRACT: of two entries naming each other, the first edge would close the cycle and is dropped; the later entry then attaches to the now-root session.
    cycle = store_a.maintenance.import_sessions([_payload("s-cyc-a", parent="s-cyc-b"), _payload("s-cyc-b", parent="s-cyc-a")])
    assert cycle.detached == 1
    assert store_a.sessions.get(SessionId("s-cyc-a")).parent_session_id is None
    assert store_a.sessions.get(SessionId("s-cyc-b")).parent_session_id == "s-cyc-a"


# ── export ───────────────────────────────────────────────────────────────────


def test_export_guard_counts_every_stored_row(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([_payload("s-port-cap", messages=[*_transcript(), _msg("user", "archived", 9, active=0, compacted=1)])])
    sid = SessionId("s-port-cap")
    # CONTRACT: the guard counts every stored row, archived included, up to limit + 1, and raises the structured error over the limit.
    with pytest.raises(ExportTooLarge) as excinfo:
        store_a.maintenance.assert_export_safe(sid, 3)
    assert (excinfo.value.session_id, excinfo.value.message_count, excinfo.value.limit) == (sid, 4, 3)
    assert isinstance(excinfo.value, ValueError)
    assert store_a.maintenance.assert_export_safe(sid, 6) == 6
    assert store_a.maintenance.assert_export_safe(SessionId("s-nope"), 6) == 0
    # CONTRACT: 0 disables the guard; a negative limit is an input-shape ValueError, not an oversized export.
    assert store_a.maintenance.assert_export_safe(sid, 0) == 0
    with pytest.raises(ValueError) as negative:
        store_a.maintenance.assert_export_safe(sid, -1)
    assert not isinstance(negative.value, ExportTooLarge)


def test_export_missing_session_is_none(backend, scope_a):
    # CONTRACT: exporting an unknown session id is a typed absence, not an error.
    store_a = backend.open(scope_a)
    assert store_a.maintenance.export_session(SessionId("s-nope")) is None
    assert store_a.maintenance.export_session_lineage(SessionId("s-nope")) is None


def test_export_flags_select_the_rows(backend, scope_a):
    store_a = backend.open(scope_a)
    rows = [_msg("user", "live", 0), _msg("assistant", "compacted", 1, active=0, compacted=1), _msg("user", "rewound", 2, active=0, compacted=0)]
    store_a.maintenance.import_sessions([_payload("s-flags", messages=rows)])
    # CONTRACT: the default export is the live rows; include_compacted adds compaction-archived rows, never rewound ones; include_inactive is every row in storage order.
    assert [m.content for m in _messages(backend, scope_a, "s-flags")] == ["live"]
    assert [m.content for m in _messages(backend, scope_a, "s-flags", include_compacted=True)] == ["live", "compacted"]
    assert [m.content for m in _messages(backend, scope_a, "s-flags", include_inactive=True)] == ["live", "compacted", "rewound"]


def test_export_row_is_the_stored_row(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession(SessionId("s-row"), "cli", model_config={"_reset_from": "s-prev", "k": 1}, system_prompt="be brief"))
    store_a.sessions.touch_activity(SessionId("s-row"), _BASE, description="busy", provenance="agent")
    row = store_a.maintenance.export_session(SessionId("s-row")).session.as_dict()
    # CONTRACT: the row dict holds the 65 sessions columns with the resolved system prompt, the stored activity and the lifted reset marker, and no last_active.
    assert len([key for key in row if not key.startswith("_")]) == 65 and "last_active" not in row
    assert (row["id"], row["system_prompt"], row["last_activity_description"], row["_reset_from"]) == ("s-row", "be brief", "busy", "s-prev")
    assert "_branched_from" not in row
    # CONTRACT: export_all rows carry last_active: the freshest of the activity stamp and the messages, the start time only without either.
    (only,) = store_a.maintenance.export_all()
    assert only.session.last_active == pytest.approx(_BASE)


def test_export_lineage_merges_segments_root_to_tip(backend, scope_a):
    store_a = backend.open(scope_a)
    root, tip = SessionId("s-lin-root"), SessionId("s-lin-tip")
    store_a.sessions.create(NewSession(root, "cli"))
    store_a.messages.append(root, NewMessage(role="user", content="first", timestamp=_BASE))
    store_a.sessions.end(root, END_REASON_COMPRESSION)
    store_a.sessions.create(NewSession(tip, "cli", parent_session_id=root))
    store_a.messages.append(tip, NewMessage(role="user", content="second", timestamp=_BASE + 1))
    for asked in (root, tip):
        lineage = store_a.maintenance.export_session_lineage(asked)
        # CONTRACT: the lineage is exported root to tip whichever segment is asked, with the merged messages and their count.
        assert lineage.lineage_session_ids == (root, tip)
        assert [segment.session.id for segment in lineage.segments] == [root, tip]
        assert ([m.content for m in lineage.messages], lineage.message_count) == (["first", "second"], 2)
    # CONTRACT: a session off any chain is its own one-segment lineage.
    store_a.sessions.create(NewSession(SessionId("s-lin-solo"), "cli"))
    assert store_a.maintenance.export_session_lineage(SessionId("s-lin-solo")).lineage_session_ids == (SessionId("s-lin-solo"),)


def test_export_all_orders_by_recency_and_filters_by_source(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([
        _payload("s-all-old", messages=[_msg("user", "old", 0)]),
        _payload("s-all-new", messages=[_msg("user", "new", 100)]),
        _payload("s-all-tg", source="telegram", messages=[_msg("user", "tg", 50)]),
    ])
    # CONTRACT: newest recency first, each with its messages; ``source`` narrows the set.
    assert [e.session.id for e in store_a.maintenance.export_all()] == ["s-all-new", "s-all-tg", "s-all-old"]
    assert [e.session.last_active for e in store_a.maintenance.export_all()] == [_BASE + 100, _BASE + 50, _BASE]
    (tg,) = store_a.maintenance.export_all("telegram")
    assert (tg.session.id, [m.content for m in tg.messages]) == ("s-all-tg", ["tg"])


# ── prune ────────────────────────────────────────────────────────────────────


def test_list_prune_candidates_and_prune(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([_payload("s-prune-old", ended_at=_BASE + _HOUR, end_reason="done"), _payload("s-prune-open")])
    criteria = PruneCriteria(last_active_before=time.time())
    candidates = store_a.maintenance.list_prune_candidates(criteria=criteria)
    # CONTRACT: only ended sessions are prune candidates; the dry run reports identity and bookkeeping fields per candidate.
    assert [c.session_id for c in candidates] == ["s-prune-old"]
    assert (candidates[0].source, candidates[0].message_count, candidates[0].ended_at, candidates[0].last_active) == ("cli", 5, _BASE + _HOUR, _BASE + 4)
    removed = store_a.maintenance.prune_sessions(criteria=criteria, media_root=MEDIA_ROOT)
    # CONTRACT: prune deletes exactly the candidate set, returns its ids, and leaves everything else intact.
    assert (removed.count, removed.deleted_ids) == (1, (SessionId("s-prune-old"),))
    assert store_a.sessions.get(SessionId("s-prune-old")) is None
    assert store_a.maintenance.assert_export_safe(SessionId("s-prune-old"), 10) == 0
    assert len(_messages(backend, scope_a, "s-prune-open")) == 5


def test_prune_orphans_children_instead_of_cascading(backend, scope_a):
    store_a = backend.open(scope_a)
    old = _payload("s-prune-parent", messages=[_msg("user", "old parent", 0)], ended_at=_BASE + _HOUR, end_reason="done")
    live_child = _payload("s-prune-child", messages=[_msg("user", "live child", 0)], parent="s-prune-parent")
    store_a.maintenance.import_sessions([old, live_child])
    assert store_a.sessions.get(SessionId("s-prune-child")).parent_session_id == "s-prune-parent"
    removed = store_a.maintenance.prune_sessions(None, "cli", media_root=MEDIA_ROOT)
    # CONTRACT: pruning the parent orphans the open child (parent cleared) instead of deleting it.
    assert removed.count == 1
    assert store_a.sessions.get(SessionId("s-prune-parent")) is None
    assert store_a.sessions.get(SessionId("s-prune-child")).parent_session_id is None


def test_prune_source_filter_composes(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.maintenance.import_sessions([
        _payload("s-prune-cli", ended_at=_BASE + _HOUR, end_reason="done"),
        _payload("s-prune-tg", source="telegram", ended_at=_BASE + _HOUR, end_reason="done"),
    ])
    # CONTRACT: the source argument narrows the candidate set; other-source rows survive.
    assert store_a.maintenance.prune_sessions(None, "telegram", media_root=MEDIA_ROOT).deleted_ids == (SessionId("s-prune-tg"),)
    assert store_a.sessions.get(SessionId("s-prune-cli")) is not None


def test_prune_spares_the_unpinned_continuations_of_a_pinned_segment(backend, scope_a):
    store_a = backend.open(scope_a)
    root, tip = SessionId("s-pin-root"), SessionId("s-pin-tip")
    store_a.maintenance.import_sessions([
        _payload(root, ended_at=_BASE + 10, end_reason=END_REASON_COMPRESSION, pinned=1),
        _payload(tip, parent=root, ended_at=_BASE + 20, end_reason="done"),
    ])
    assert store_a.sessions.get(tip).pinned is False
    everything = PruneCriteria(last_active_before=time.time() + 60)
    # CONTRACT: a pin covers the continuations published after the pinned segment.
    assert store_a.maintenance.count_prune_matches(criteria=everything) == 0
    assert store_a.maintenance.count_prune_matches(criteria=everything, pinned_only=True) == 1
    assert {c.session_id for c in store_a.maintenance.list_prune_candidates(criteria=PruneCriteria(last_active_before=time.time() + 60, include_pinned=True))} == {root, tip}


def test_two_scopes_import_export_prune_stay_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    store_b = backend.open(scope_b)
    shared = SessionId("s-shared")
    sentinel_a, sentinel_b = [_msg("user", "sentinel-a", 0)], [_msg("user", "sentinel-b", 0)]
    store_a.maintenance.import_sessions([_payload("s-shared", messages=sentinel_a, ended_at=_BASE + _HOUR, end_reason="done")])
    store_b.maintenance.import_sessions([_payload("s-shared", messages=sentinel_b, ended_at=_BASE + _HOUR, end_reason="done"), _payload("s-only-b", ended_at=_BASE + _HOUR, end_reason="done")])
    store_a.maintenance.import_sessions([_payload("s-only-a")])
    # Positive reads per scope: each sees its own sentinel transcript; a session imported into one scope is absent from the other.
    assert _view(_messages(backend, scope_a, "s-shared")) == _view(sentinel_a)
    assert _view(_messages(backend, scope_b, "s-shared")) == _view(sentinel_b)
    assert store_b.maintenance.export_session(SessionId("s-only-a")) is None
    assert store_a.maintenance.export_session(SessionId("s-only-b")) is None
    # CONTRACT: B's prune really fires (deletes B's own ended rows, its colliding id included) yet cannot touch A's row.
    assert store_b.maintenance.prune_sessions(None, media_root=MEDIA_ROOT).count == 2
    assert store_b.maintenance.export_session(shared) is None
    assert _view(_messages(backend, scope_a, "s-shared")) == _view(sentinel_a)


# ── idle archive ─────────────────────────────────────────────────────────────


def _live(backend, scope, sid: str, **kw) -> SessionId:
    session_id = SessionId(sid)
    backend.open(scope).sessions.create(NewSession(session_id=session_id, source="cli", **kw))
    return session_id


def _archived(backend, scope, sid: str) -> bool:
    return backend.open(scope).sessions.get(SessionId(sid)).archived


def _run_archive(store, idle_days: float, **kw: Any) -> int:
    """The idle sweep as its helper runs it: the candidate read, then one archive per match."""
    matches = store.maintenance.list_idle_archive_candidates(idle_days, **kw)
    for sid in matches:
        assert store.maintenance.auto_archive_lineage(sid) is True
    return len(matches)


def test_idle_archive_archives_stale_tips_and_keeps_fresh_ones(backend, scope_a):
    store_a = backend.open(scope_a)
    stale = _live(backend, scope_a, "ai-stale")
    fresh = _live(backend, scope_a, "ai-fresh")
    # A row whose newest transcript row is in the future is not idle whatever its start time.
    store_a.messages.append(fresh, NewMessage(role="user", content="recent", timestamp=_FAR_FUTURE))
    # CONTRACT: the stale tip is the one candidate; archiving it hides it and deletes nothing.
    assert store_a.maintenance.list_idle_archive_candidates(0) == (stale,)
    assert _run_archive(store_a, 0) == 1
    assert _archived(backend, scope_a, "ai-stale") is True and _archived(backend, scope_a, "ai-fresh") is False
    assert store_a.sessions.get(stale).auto_archived is True
    # CONTRACT: an archived row is no candidate again; a negative idle window matches nothing.
    assert store_a.maintenance.list_idle_archive_candidates(0) == ()
    assert store_a.maintenance.list_idle_archive_candidates(-1) == ()
    # CONTRACT: archiving an unknown id reports False.
    assert store_a.maintenance.auto_archive_lineage(SessionId("ai-none")) is False


def test_idle_archive_recency_uses_started_at_only_as_a_fallback(backend, scope_a):
    store_a = backend.open(scope_a)
    replayed = _live(backend, scope_a, "ai-replayed")
    store_a.messages.append(replayed, NewMessage(role="user", content="old", timestamp=datetime(2000, 1, 1, tzinfo=UTC).timestamp()))
    _live(backend, scope_a, "ai-bare")
    # CONTRACT: a just-created row whose only transcript is ancient is idle since that transcript (started_at is not a floor); a bare just-created row falls back to started_at and is not idle.
    assert store_a.maintenance.list_idle_archive_candidates(1) == (replayed,)


def test_idle_archive_pinned_and_bot_chat_exemptions(backend, scope_a):
    store_a = backend.open(scope_a)
    pinned = _live(backend, scope_a, "ai-pinned")
    store_a.sessions.set_pinned(pinned, True)
    plain = _live(backend, scope_a, "ai-plain")
    store_a.maintenance.import_sessions([_payload("ai-bot", messages=[], title=CANONICAL_BOT_CHAT_TITLE, hidden=1)])
    # CONTRACT: pinned rows are exempt by default and included on request; the hidden canonical Bot Chat is always exempt.
    assert store_a.maintenance.list_idle_archive_candidates(0) == (plain,)
    assert set(store_a.maintenance.list_idle_archive_candidates(0, exclude_pinned=False)) == {pinned, plain}


def test_idle_archive_lineage_tip_rule_and_provenance(backend, scope_a):
    store_a = backend.open(scope_a)
    root = _live(backend, scope_a, "ai-root")
    store_a.sessions.end(root, END_REASON_COMPRESSION)
    tip = _live(backend, scope_a, "ai-tip", parent_session_id=root)
    store_a.messages.append(tip, NewMessage(role="user", content="recent", timestamp=_FAR_FUTURE))
    # CONTRACT: the old root is never matched on its own (only lineage tips are candidates).
    assert store_a.maintenance.list_idle_archive_candidates(0) == ()
    # A second lineage whose tip IS stale, with a deliberately archived root.
    root2, tip2 = SessionId("ai-root2"), SessionId("ai-tip2")
    store_a.maintenance.import_sessions([_payload(root2, messages=[], ended_at=_BASE + 1, end_reason=END_REASON_COMPRESSION, archived=1), _payload(tip2, parent=root2)])
    assert store_a.maintenance.list_idle_archive_candidates(0) == (tip2,)
    # CONTRACT: one match archives the whole compression lineage; auto_archived is stamped only on the rows this call archived.
    assert store_a.maintenance.auto_archive_lineage(tip2) is True
    assert (_archived(backend, scope_a, "ai-tip2"), _archived(backend, scope_a, "ai-root2")) == (True, True)
    assert (store_a.sessions.get(tip2).auto_archived, store_a.sessions.get(root2).auto_archived) == (True, False)
    assert _archived(backend, scope_a, "ai-tip") is False


def test_idle_archive_is_scope_isolated(backend, scope_a, scope_b):
    store_a = backend.open(scope_a)
    _live(backend, scope_a, "ai-a")
    _live(backend, scope_b, "ai-b")
    assert _run_archive(store_a, 0) == 1
    # CONTRACT: the other scope's rows are untouched.
    assert _archived(backend, scope_b, "ai-b") is False
    assert _run_archive(backend.open(scope_b), 0) == 1


# ── read-only probes ─────────────────────────────────────────────────────────


def test_session_projection_digest_tracks_the_listed_columns_only(backend, scope_a):
    store_a = backend.open(scope_a)
    sid = _live(backend, scope_a, "pd-one")
    fields, first = store_a.maintenance.session_projection_digest()
    # CONTRACT: the digest names the projected columns the store has and is 16 bytes.
    assert fields == SESSION_PROJECTION_FIELDS and len(first) == 16
    # CONTRACT: an activity heartbeat leaves the digest as it was; a projected column change moves it.
    store_a.sessions.touch_activity(sid, time.time(), description="busy", provenance="agent")
    assert store_a.maintenance.session_projection_digest()[1] == first
    store_a.sessions.set_title(sid, "renamed")
    assert store_a.maintenance.session_projection_digest()[1] != first


def test_retained_tree_is_live_reads_sessions_inside_the_tree(backend, scope_a, tmp_path):
    store_a = backend.open(scope_a)
    tree = tmp_path / "wt"
    store_a.maintenance.import_sessions([
        _payload("rt-inside", cwd=str(tree / "pkg")),
        _payload("rt-sibling", cwd=str(tmp_path / "wt-other")),
        _payload("rt-empty", messages=[], cwd=str(tree)),
    ])
    # CONTRACT: a session with history whose cwd is the tree or below it keeps the tree; a sibling path with the same prefix does not.
    assert store_a.maintenance.retained_tree_is_live(str(tree)) is True
    assert store_a.maintenance.retained_tree_is_live(str(tmp_path / "elsewhere")) is False
    # CONTRACT: pending_only asks only for pending admissions: none at first, then the empty row's queued admission keeps the tree.
    assert store_a.maintenance.retained_tree_is_live(str(tree), pending_only=True) is False
    epoch = store_a.runtime.begin_epoch(instance_id="tree-test")
    store_a.admissions.admit(epoch=epoch, principal_id="human", session_id="rt-empty", request_id="req", payload={"input": "work"})
    assert store_a.maintenance.retained_tree_is_live(str(tree), pending_only=True) is True


def test_tree_served_sessions_reads_open_rows_and_active_schedules(backend, scope_a, tmp_path):
    store_a = backend.open(scope_a)
    tree = str(tmp_path / "wt")
    served = _live(backend, scope_a, "ts-open", cwd=tree, chat_id="local-1")
    # CONTRACT: an open session inside the tree serves it.
    assert store_a.maintenance.tree_served_sessions(tree) == TreeServeFacts(served=True)
    store_a.sessions.end(served, "tui_shutdown")
    # CONTRACT: once ended, it serves the tree only through an active loop, heartbeat or goal row; otherwise the chat ids come back for the cron check.
    assert store_a.maintenance.tree_served_sessions(tree) == TreeServeFacts(served=False, chat_ids=frozenset({"local-1"}))
    store_a.state_meta.set("goal:ts-open", '{"status": "done"}')
    assert store_a.maintenance.tree_served_sessions(tree).served is False
    store_a.state_meta.set("loop:ts-open", '{"interval": 5}')
    assert store_a.maintenance.tree_served_sessions(tree) == TreeServeFacts(served=True)
