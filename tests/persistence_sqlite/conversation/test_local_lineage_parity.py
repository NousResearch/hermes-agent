"""The local lineage ids of the SQLite listings equal the legacy ``SessionDB`` rows read from the same store file.

``list_sessions``, ``search_by_id`` and ``summary`` are compared with ``list_sessions_rich``, ``search_sessions_by_id`` and ``get_session_rich_row``, whose listing names every segment of a local conversation on the row that represents it (``hermes_state_local.annotate_local_lineages``). The store holds local conversations built through the binding verbs (resets, compression steps, both, neither), a branch and a delegate of a lineage head, a deleted conversation, a pinned head and an archived head, and receipts stored raw: one naming a segment row that does not exist, one with a non-string owner, one with a non-string segment and one with a single segment.
"""

from __future__ import annotations

import json

from persistence import SessionId
from persistence.authority.bindings import local_session_id
from persistence.conversation.sessions import (
    END_REASON_COMPRESSION,
    END_REASON_SESSION_RESET,
    NewSession,
    SessionLineageKind,
    SessionOrder,
    SessionQuery,
    SessionWindow,
)
from tests.contracts._scopes import MEDIA_ROOT
from tests.contracts.authority.test_bindings_contract import make_binding
from tests.persistence_sqlite._legacy_store import legacy_store
from tests.persistence_sqlite.conftest import store_for

__all__ = ["legacy_store"]

PRINCIPAL = "user-1"
POLICY_PREFIX = "gateway.local_policy.v1:"


def _local(store, epoch, request_id):
    sid = local_session_id("default", PRINCIPAL, request_id)
    binding = make_binding(sid, route="route-" + request_id, request_id=request_id)
    store.bindings.reserve(epoch=epoch, binding=binding, owner=None)
    return sid, binding


def _reset(store, epoch, parent, child, binding):
    store.bindings.reset_target(epoch=epoch, parent_session_id=SessionId(parent), entry={**dict(binding.entry), "session_id": child})


def _step(store, owner, parent, child, binding):
    """A compression step past the receipt's tip: the child carries the route, owner and principal, the parent ends by compression."""
    store.sessions.create(NewSession(session_id=SessionId(child), source="cli", parent_session_id=SessionId(parent),
                                     session_key=binding.route, chat_id=owner, user_id=PRINCIPAL))
    store.sessions.end(SessionId(parent), END_REASON_COMPRESSION)


def _compress(store, epoch, owner, parent, child, binding):
    _step(store, owner, parent, child, binding)
    store.bindings.adopt_untracked_compression(epoch=epoch, session_id=SessionId(owner), expected=store.bindings.receipt(SessionId(owner)))


def _raw(store, key_id, owner, lineage, *ended):
    """A receipt stored as written, for rows ``ended`` by reset that the store does not otherwise link."""
    for sid in ended:
        store.sessions.create(NewSession(session_id=SessionId(sid), source="cli"))
        store.sessions.end(SessionId(sid), END_REASON_SESSION_RESET)
    store.state_meta.set(POLICY_PREFIX + key_id, json.dumps({"session_id": owner, "lineage": lineage}))


def _seed(store):
    s = store.sessions
    epoch = store.runtime.begin_epoch(instance_id="lineage-parity")
    owners = {}
    owners["solo"], _ = _local(store, epoch, "req-solo")
    owners["r"], rb = _local(store, epoch, "req-r")
    _reset(store, epoch, owners["r"], "r-1", rb)
    owners["q"], qb = _local(store, epoch, "req-q")
    _reset(store, epoch, owners["q"], "q-1", qb)
    _reset(store, epoch, "q-1", "q-2", qb)
    owners["c"], cb = _local(store, epoch, "req-c")
    _compress(store, epoch, owners["c"], owners["c"], "c-1", cb)
    owners["m"], mb = _local(store, epoch, "req-m")
    _compress(store, epoch, owners["m"], owners["m"], "m-1", mb)
    _reset(store, epoch, "m-1", "m-2", mb)
    _compress(store, epoch, owners["m"], "m-2", "m-3", mb)
    _step(store, owners["m"], "m-3", "m-4", mb)
    owners["x"], xb = _local(store, epoch, "req-x")
    _reset(store, epoch, owners["x"], "x-1", xb)
    assert set(s.delete(SessionId("x-1"), media_root=MEDIA_ROOT)) == {"x-1", owners["x"]}
    s.create(NewSession(session_id=SessionId("r-branch"), source="cli", parent_session_id=SessionId("r-1"), lineage_kind=SessionLineageKind.BRANCH))
    s.create(NewSession(session_id=SessionId("r-delegate"), source="cli", parent_session_id=SessionId("r-1"), lineage_kind=SessionLineageKind.DELEGATE))
    s.set_pinned(SessionId("r-1"), True)
    s.set_pinned(SessionId(owners["r"]), True)
    s.set_archived(SessionId("q-2"), True)
    _raw(store, "g-0", "g-0", ["g-0", "g-ghost"], "g-0")
    _raw(store, "h-0", "h-0", ["h-0", "h-ghost", "h-2"], "h-0")
    s.create(NewSession(session_id=SessionId("h-2"), source="cli"))
    _raw(store, "n-0", 7, ["n-0", "n-1"], "n-0")
    s.create(NewSession(session_id=SessionId("n-1"), source="cli"))
    _raw(store, "e-0", "e-0", ["e-0", 5], "e-0")
    _raw(store, "s-0", "s-0", ["s-0"], "s-0")
    return owners


def _stamps(rows):
    return [(row["id"], row.get("_lineage_root_id"), row.get("_lineage_ids")) for row in rows]


_LISTINGS = (
    (SessionQuery(), {}),
    (SessionQuery(project_compression_tips=False), {"project_compression_tips": False}),
    (SessionQuery(order=SessionOrder.LAST_ACTIVE), {"order_by_last_active": True}),
    (SessionQuery(include_children=True), {"include_children": True}),
    (SessionQuery(include_subagents=True), {"include_subagents": True}),
    (SessionQuery(include_archived=True), {"include_archived": True}),
    (SessionQuery(archived_only=True), {"archived_only": True}),
)


def test_listings_stamp_the_legacy_local_lineages(backend, scope_a, legacy_store):
    from hermes_state import SessionDB

    store = backend.open(scope_a)
    owners = _seed(store)
    legacy = SessionDB(db_path=store_for(scope_a))
    try:
        plain = legacy.list_sessions_rich(limit=50)
        assert {row["id"]: row.get("_lineage_ids") for row in plain if row.get("_lineage_ids")} == {
            "r-1": [owners["r"], "r-1"],
            "c-1": [owners["c"], "c-1"],
            "m-4": [owners["m"], "m-1", "m-2", "m-3", "m-4"],
        }
        for query, kwargs in _LISTINGS:
            for compact in (False, True):
                served = store.sessions.list_sessions(query, SessionWindow(limit=50), compact_rows=compact)
                expected = legacy.list_sessions_rich(limit=50, compact_rows=compact, **kwargs)
                assert _stamps(e.as_list_row() for e in served) == _stamps(expected), (query, compact)
                assert [list(e.lineage_ids) for e in served] == [row.get("_lineage_ids") or [] for row in expected], (query, compact)
        pinned = store.sessions.list_sessions(SessionQuery(include_pinned=True), SessionWindow(limit=1))
        assert _stamps(e.as_list_row() for e in pinned) == _stamps(legacy.list_sessions_rich(limit=1, include_pinned=True))
    finally:
        legacy.close()


def test_id_search_and_summary_stamp_as_the_legacy_reads(backend, scope_a, legacy_store):
    from hermes_state import SessionDB

    store = backend.open(scope_a)
    owners = _seed(store)
    legacy = SessionDB(db_path=store_for(scope_a))
    try:
        for needle in ("r-1", owners["r"], "m-", "q-2", "h-", "n-"):
            served = store.sessions.search_by_id(needle)
            assert _stamps(e.as_list_row() for e in served) == _stamps(legacy.search_sessions_by_id(needle)), needle
        for sid in ("r-1", owners["c"], "m-2", "q-2"):
            served = store.sessions.summary(SessionId(sid)).as_list_row()
            expected = legacy.get_session_rich_row(sid)
            assert (served.get("_lineage_root_id"), served.get("_lineage_ids")) == (expected.get("_lineage_root_id"), expected.get("_lineage_ids")), sid
    finally:
        legacy.close()
