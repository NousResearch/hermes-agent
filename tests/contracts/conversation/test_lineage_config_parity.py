"""Stored lineage markers survive typed reads and compaction merges."""

import pytest

from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession, SessionLineageKind



@pytest.mark.parametrize("kind, marker", [(SessionLineageKind.DELEGATE, "_delegate_from"), (SessionLineageKind.BRANCH, "_branched_from"), (SessionLineageKind.RESET, "_reset_from")])
def test_created_lineage_marker_is_returned_and_can_be_inherited(backend, scope_a, kind, marker):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("parent", "cli"))
    assert store_a.sessions.create(NewSession("child", "cli", parent_session_id="parent", lineage_kind=kind, model_config={"temperature": 0.1})) == "child"
    created = store_a.sessions.get("child")
    assert created.model_config == {"temperature": 0.1, marker: "parent"}
    assert (created.lineage_kind, created.lineage_origin_id) == (kind, "parent")
    store_a.sessions.end("child", "compression")
    store_a.sessions.create(NewSession("continuation", "cli", parent_session_id="child", model_config=created.model_config))
    inherited = store_a.sessions.get("continuation")
    assert inherited.model_config == created.model_config
    assert (inherited.lineage_kind, inherited.lineage_origin_id) == (kind, "parent")
    assert store_a.lineage.is_fork_child("continuation") is False
    assert store_a.lineage.tip("child") == "child"


def test_compaction_merges_and_removes_lineage_markers_with_config(backend, scope_a):
    store_a = backend.open(scope_a)
    store_a.sessions.create(NewSession("session", "cli", model_config={"keep": 1}))
    message = NewMessage("user", "summary", timestamp=1790553600.0)
    store_a.messages.compact("session", [message], model_config_patch={"_delegate_from": "origin", "_branched_from": "branch", "keep": None})
    record = store_a.sessions.get("session")
    assert record.model_config == {"_delegate_from": "origin", "_branched_from": "branch"}
    assert (record.lineage_kind, record.lineage_origin_id) == (SessionLineageKind.DELEGATE, "origin")
    store_a.messages.compact("session", [message], model_config_patch={"_delegate_from": None, "ok": 2})
    record = store_a.sessions.get("session")
    assert record.model_config == {"_branched_from": "branch", "ok": 2}
    assert (record.lineage_kind, record.lineage_origin_id) == (SessionLineageKind.BRANCH, "branch")
    assert len(store_a.messages.read_transcript("session", TranscriptRead(view=TranscriptView.ALL))) == 2
    assert store_a.sessions.get("session").message_count == 1
