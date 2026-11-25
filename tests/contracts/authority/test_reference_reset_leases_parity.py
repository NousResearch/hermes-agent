from datetime import datetime, UTC

import pytest

from persistence import CompressionInProgress, SessionId, TurnLeaseLost
from persistence.conversation.messages import NewMessage, ReplaceMode
from persistence.conversation.sessions import END_REASON_COMPRESSION, NewSession, SessionLineageKind


@pytest.mark.parametrize("kind", [SessionLineageKind.RESET, SessionLineageKind.BRANCH, SessionLineageKind.DELEGATE])
@pytest.mark.parametrize("raw_marker", [False, True])
def test_explicit_child_has_its_own_turn_lease(backend, scope_a, kind, raw_marker):
    store_a = backend.open(scope_a)
    parent, child = SessionId("parent"), SessionId("child")
    marker = {SessionLineageKind.RESET: "_reset_from", SessionLineageKind.BRANCH: "_branched_from", SessionLineageKind.DELEGATE: "_delegate_from"}[kind]
    store_a.sessions.create(NewSession(parent, source="cli"))
    store_a.sessions.create(NewSession(child, source="cli", parent_session_id=parent, lineage_kind=kind, model_config={marker: str(parent)} if raw_marker else None))
    store_a.sessions.end(parent, END_REASON_COMPRESSION)
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=parent, holder="parent-owner", ttl_seconds=60) is True
    assert coord.try_acquire_turn_lease(session_id=child, holder="child-owner", ttl_seconds=60) is True
    replacement = [NewMessage(role="user", content="replacement", timestamp=datetime(2026, 1, 1, tzinfo=UTC))]
    with pytest.raises((TurnLeaseLost, CompressionInProgress)):
        store_a.messages.replace_transcript(child, replacement, ReplaceMode.DESTRUCTIVE, reject_active_turn_lease=True)
    coord.release_turn_lease(session_id=child, holder="child-owner")
    store_a.messages.replace_transcript(child, replacement, ReplaceMode.DESTRUCTIVE, reject_active_turn_lease=True)
    assert [row.content for row in store_a.messages.read_transcript(child)] == ["replacement"]
    assert coord.refresh_turn_lease(session_id=parent, holder="parent-owner", ttl_seconds=60)


@pytest.mark.parametrize("inherited_reset", [False, True])
def test_continuation_keeps_parent_turn_lease(backend, scope_a, inherited_reset):
    store_a = backend.open(scope_a)
    parent, child = SessionId("parent"), SessionId("continuation")
    store_a.sessions.create(NewSession(parent, source="cli"))
    store_a.sessions.create(NewSession(child, source="cli", parent_session_id=parent, model_config={"_reset_from": "older-origin"} if inherited_reset else None))
    store_a.sessions.end(parent, END_REASON_COMPRESSION)
    coord = backend.open(scope_a).coordination
    assert coord.try_acquire_turn_lease(session_id=parent, holder="parent-owner", ttl_seconds=60) is True
    assert coord.try_acquire_turn_lease(session_id=child, holder="child-owner", ttl_seconds=60) is False
    coord.release_turn_lease(session_id=child, holder="child-owner")
    with pytest.raises((TurnLeaseLost, CompressionInProgress)):
        store_a.messages.replace_transcript(child, [], ReplaceMode.DESTRUCTIVE, reject_active_turn_lease=True)
    assert coord.refresh_turn_lease(session_id=child, holder="parent-owner", ttl_seconds=60)
    assert store_a.messages.read_transcript(child) == ()


@pytest.mark.parametrize("origin,independent", [("parent", True), ("older-origin", False)])
@pytest.mark.parametrize("raw_marker", [False, True])
def test_reset_origin_must_name_the_parent(origin, independent, raw_marker):
    from tests.contracts._reference.authority.coordination import _is_explicit_fork_child

    row = {"parent_session_id": "parent", "lineage_kind": SessionLineageKind.NONE if raw_marker else SessionLineageKind.RESET, "lineage_origin_id": origin, "model_config": {"_reset_from": origin} if raw_marker else {}}
    assert _is_explicit_fork_child(row) is independent
