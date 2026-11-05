"""Contract cases for the reset lineage intent: ``SessionLineageKind.RESET`` on create and ``SessionRecord.lineage_origin_id`` (the ``persistence/conversation/sessions.py`` docstrings).

Backend cases run both registered backends, seed through public protocol operations only and observe through ``get``, ``list_sessions``, ``delete`` and lineage projections. The parent-requirement case validates ``NewSession`` without a backend.
"""

from __future__ import annotations

import pytest

from persistence import SessionId
from persistence.conversation.sessions import END_REASON_COMPRESSION, END_REASON_SESSION_RESET, NewSession, SessionLineageKind, SessionQuery, SessionWindow

from tests.contracts._scopes import MEDIA_ROOT



def _seed_reset_pair(backend, scope, parent: str, child: str) -> None:
    store = backend.open(scope)
    store.sessions.create(NewSession(session_id=SessionId(parent), source="telegram", session_key="k"))
    store.sessions.end(SessionId(parent), END_REASON_SESSION_RESET)
    store.sessions.create(NewSession(
        session_id=SessionId(child), source="telegram", session_key="k",
        parent_session_id=SessionId(parent), lineage_kind=SessionLineageKind.RESET))


@pytest.mark.parametrize("kind", [SessionLineageKind.RESET, SessionLineageKind.BRANCH, SessionLineageKind.DELEGATE])
def test_an_intent_names_its_parent_as_the_origin(backend, scope_a, kind):
    store_a = backend.open(scope_a)
    parent, child = SessionId(f"origin-parent-{kind.value}"), SessionId(f"origin-child-{kind.value}")
    store_a.sessions.create(NewSession(session_id=parent, source="cli"))
    store_a.sessions.create(NewSession(
        session_id=child, source="cli", parent_session_id=parent, lineage_kind=kind, model_config={"temperature": 0.1}))
    record = store_a.sessions.get(child)
    assert (record.lineage_kind, record.parent_session_id, record.lineage_origin_id) == (kind, parent, parent)
    marker = {SessionLineageKind.RESET: "_reset_from", SessionLineageKind.BRANCH: "_branched_from", SessionLineageKind.DELEGATE: "_delegate_from"}[kind]
    assert dict(record.model_config) == {"temperature": 0.1, marker: parent}
    root = store_a.sessions.get(parent)
    assert (root.lineage_kind, root.lineage_origin_id) == (SessionLineageKind.NONE, None)


def test_reset_under_a_compressed_parent_is_not_a_continuation(backend, scope_a):
    store_a = backend.open(scope_a)
    parent, child = SessionId("compressed-parent"), SessionId("reset-child")
    store_a.sessions.create(NewSession(session_id=parent, source="cli"))
    store_a.sessions.end(parent, END_REASON_COMPRESSION)
    store_a.sessions.create(NewSession(
        session_id=child, source="cli", parent_session_id=parent, lineage_kind=SessionLineageKind.RESET))
    assert store_a.lineage.lineage(child) == (child,)
    assert store_a.lineage.lineage(parent) == (parent,)
    assert store_a.lineage.tip(parent) == parent
    rows = store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))
    listed = {row.session.session_id: row for row in rows}
    assert {parent, child} <= listed.keys()
    assert not listed[parent].lineage_ids


def test_reset_requires_a_parent():
    with pytest.raises(ValueError):
        NewSession(session_id=SessionId("orphan-reset"), source="cli", lineage_kind=SessionLineageKind.RESET)


def test_a_reset_child_is_listed_as_its_own_conversation(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_reset_pair(backend, scope_a, "rl-parent", "rl-child")
    rows = store_a.sessions.list_sessions(SessionQuery(), SessionWindow(limit=50))
    listed = {str(r.session.session_id): r for r in rows}
    assert {"rl-parent", "rl-child"} <= set(listed)
    assert not listed["rl-child"].lineage_ids and listed["rl-child"].session.lineage_kind is SessionLineageKind.RESET


def test_deleting_the_parent_orphans_a_reset_child(backend, scope_a):
    store_a = backend.open(scope_a)
    _seed_reset_pair(backend, scope_a, "rd-parent", "rd-child")
    assert store_a.sessions.delete(SessionId("rd-parent"), media_root=MEDIA_ROOT) == (SessionId("rd-parent"),)
    child = store_a.sessions.get(SessionId("rd-child"))
    assert child is not None and child.parent_session_id is None and child.lineage_kind is SessionLineageKind.RESET
