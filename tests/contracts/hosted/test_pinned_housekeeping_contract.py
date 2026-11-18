"""Pinned (hosted) scopes under housekeeping and delegation: a retention or prune sweep run through one pin's handle
reaches only that pin's rows, and a delegated child session never crosses from one pin to another. Every case runs on
every registered backend: on a backend declaring ``scope_pinning`` against real pinned handles, on any other as the
refusal of the pin at ``open``, never a skip.

The two pins hold rows under the same session ids, so a sweep or a lineage read that chose its store by anything but
the pin would delete or reveal the other pin's rows and fail these cases.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from persistence import DelegationId, SessionId
from persistence.authority.coordination import OwnerStamp
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession, SessionLineageKind, SessionQuery, SessionWindow
from tests.contracts._scopes import MEDIA_ROOT
from tests.contracts.hosted.test_scope_pinning_contract import _open_pinned_or_refused, _pinned

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC).timestamp()
_HOUR = 3600.0


def _payload(session_id: str, text: str, *, ended: bool) -> dict[str, Any]:
    """An old imported session: ended long ago when ``ended``, else left open (an orphan the sweep would close)."""
    return {"id": session_id, "source": "tui", "started_at": _BASE, "model": "model-a",
            "ended_at": _BASE + 60 if ended else None, "end_reason": "done" if ended else None,
            "messages": [{"role": "user", "content": text, "timestamp": _BASE},
                         {"role": "assistant", "content": f"{text} reply", "timestamp": _BASE + 1}]}


def _seed_old(store: Any, prefix: str) -> None:
    """Ended and open old sessions, two under ids every pin shares."""
    result = store.maintenance.import_sessions([
        _payload("shared-ended", f"{prefix} ended", ended=True), _payload("shared-open", f"{prefix} open", ended=False),
        _payload(f"{prefix}-ended", f"{prefix} own", ended=True)])
    assert result.imported == 3, result


def _rows(store: Any, *ids: str) -> dict[str, Any]:
    """Each id's row facts and transcript as the store answers them (``None`` for a missing row)."""
    out: dict[str, Any] = {}
    for sid in ids:
        row = store.sessions.get(SessionId(sid))
        out[sid] = None if row is None else (row.ended_at, row.end_reason, row.message_count,
                                             [str(r.content) for r in store.messages.read_transcript(SessionId(sid))])
    return out


_SEEDED = ("shared-ended", "shared-open")


def test_a_pinned_sweep_never_deletes_another_pins_rows(backend_name: str, backend, scope_a) -> None:
    """The prune and the orphan sweep, each run through tenant A's pinned
    handle, change A's rows only: tenant B's rows under the same ids, and the local profile the pins were built in,
    read exactly as before."""
    local = backend.open(scope_a)
    _seed_old(local, "local")
    handles = _open_pinned_or_refused(backend, _pinned(scope_a, "tenant-a", "agent-1"), _pinned(scope_a, "tenant-b", "agent-1"))
    if handles is None:
        return
    store_a, store_b = handles
    _seed_old(store_a, "a")
    _seed_old(store_b, "b")
    b_before = _rows(store_b, *_SEEDED, "b-ended")
    local_before = _rows(local, *_SEEDED, "local-ended")

    pruned = store_a.maintenance.prune_sessions(None, media_root=MEDIA_ROOT)
    assert set(map(str, pruned.deleted_ids)) == {"shared-ended", "a-ended"}
    swept = store_a.maintenance.sweep_orphaned_sessions(max_idle_seconds=_HOUR, sources=("tui",), respect_gateway_heartbeats=False)

    # A's sweep did its work on A's own rows ...
    assert set(map(str, swept)) == {"shared-open"}
    assert _rows(store_a, "shared-ended", "a-ended") == {"shared-ended": None, "a-ended": None}
    assert store_a.sessions.get(SessionId("shared-open")).ended_at is not None
    # ... and nothing of anyone else's.
    assert _rows(store_b, *_SEEDED, "b-ended") == b_before
    assert _rows(local, *_SEEDED, "local-ended") == local_before


def _listed_with_children(store: Any) -> set[str]:
    query = SessionQuery(include_children=True, include_subagents=True)
    return {str(entry.session.session_id) for entry in store.sessions.list_sessions(query, SessionWindow(limit=100))}


def test_a_delegated_child_never_crosses_pins(backend_name: str, backend, scope_a) -> None:
    """A delegate child created under tenant A's parent lives in A only: B (holding a parent under the same id) never
    reads, lists or reaches it through its parent, its lineage or the delegation ledger; and a child B creates naming
    that parent id is B's own, never one of A's parent's children."""
    handles = _open_pinned_or_refused(backend, _pinned(scope_a, "tenant-a", "agent-1"), _pinned(scope_a, "tenant-b", "agent-1"))
    if handles is None:
        return
    store_a, store_b = handles
    parent = SessionId("parent")
    for store, tag in ((store_a, "a"), (store_b, "b")):
        store.sessions.create(NewSession(session_id=parent, source="cli"))
        store.messages.append(parent, NewMessage(role="user", content=f"{tag} parent text"))
    child_a = SessionId("child-of-a")
    store_a.sessions.create(NewSession(session_id=child_a, source="subagent", parent_session_id=parent,
                                       lineage_kind=SessionLineageKind.DELEGATE))
    store_a.messages.append(child_a, NewMessage(role="assistant", content="a delegated work"))
    store_a.coordination.record_delegation_dispatch(
        delegation_id=DelegationId("d-a"), origin_session="route-a", origin_ui_session_id="ui-a", parent_session_id=parent,
        dispatched_at=_BASE, owner=OwnerStamp(pid=4242, started_at=777), task={"goal": "a"})

    # B learns nothing of A's child.
    assert store_b.sessions.get(child_a) is None
    assert store_b.messages.read_transcript(child_a) == ()
    assert child_a not in {str(edge.session_id) for edge in store_b.sessions.descendants(parent)}
    assert store_b.lineage.lineage(child_a) == store_b.lineage.lineage(SessionId("never-created"))
    assert store_b.coordination.get_delegation(delegation_id=DelegationId("d-a")) is None
    assert _listed_with_children(store_b) == {"parent"}

    # A child B creates under the same parent id stays B's.
    child_b = SessionId("child-of-b")
    store_b.sessions.create(NewSession(session_id=child_b, source="subagent", parent_session_id=parent,
                                       lineage_kind=SessionLineageKind.DELEGATE))
    store_b.messages.append(child_b, NewMessage(role="assistant", content="b delegated work"))
    assert store_a.sessions.get(child_b) is None
    assert _listed_with_children(store_a) == {"parent", "child-of-a"}
    assert _listed_with_children(store_b) == {"parent", "child-of-b"}
    assert store_a.sessions.get(child_a).parent_session_id == parent
    assert [str(r.content) for r in store_a.messages.read_transcript(child_a)] == ["a delegated work"]
    assert [str(r.content) for r in store_a.messages.read_transcript(parent)] == ["a parent text"]
    assert [str(r.content) for r in store_b.messages.read_transcript(parent)] == ["b parent text"]
