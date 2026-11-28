"""Reference lineage writes: the state they leave and the order of their refusals."""

from unittest.mock import patch

import pytest

from persistence import CompressionBusy
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage, TranscriptRead, TranscriptView
from persistence.conversation.sessions import NewSession, SessionLineageKind
from tests.contracts._reference import open_reference_backend
from tests.contracts._scopes import profile_scope


def test_reset_child_does_not_block_orphaned_compression_reopen():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("parent", "cli"))
        backend.open(scope).sessions.end("parent", "compression")
        backend.open(scope).sessions.create(NewSession("reset", "cli", parent_session_id="parent", lineage_kind=SessionLineageKind.RESET))
        assert backend.open(scope).lineage.reopen_orphaned_compression("parent") is True
        assert backend.open(scope).lineage.reopen_orphaned_compression("parent") is False


def test_failed_publication_keeps_child_and_transcript_absent():
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("parent", "cli"))
        handoff = [NewMessage("user", "summary", timestamp=1.0)]
        with patch("tests.contracts._reference.conversation.transcript_store._row_snapshot", side_effect=RuntimeError("snapshot failed")):
            with pytest.raises(RuntimeError, match="snapshot failed"):
                backend.open(scope).lineage.publish_compression_child("parent", CompressionChild("child", "cli"), handoff, require_compression_lease=False)
        assert backend.open(scope).sessions.get("child") is None
        assert backend.open(scope).messages.read_transcript("child", TranscriptRead(view=TranscriptView.ALL)) == ()
        assert (backend.open(scope).sessions.get("parent")).ended_at is None


@pytest.mark.parametrize("empty", [False, True])
def test_publication_checks_lease_before_missing_parent_and_handoff(empty):
    with open_reference_backend() as backend:
        scope = profile_scope("default")
        handoff = [] if empty else [NewMessage("user", "summary", timestamp=1.0)]
        with pytest.raises(CompressionBusy):
            backend.open(scope).lineage.publish_compression_child("missing", CompressionChild("child", "cli"), handoff, compression_lock_holder="foreign")
        assert backend.open(scope).sessions.get("child") is None


def test_publication_inherits_peer_display_and_origin():
    from persistence.conversation.sessions import SessionQuery, SessionWindow

    with open_reference_backend() as backend:
        scope = profile_scope("default")
        backend.open(scope).sessions.create(NewSession("parent", "cli", display_name="Peer", origin={"platform": "telegram"}))
        backend.open(scope).lineage.publish_compression_child("parent", CompressionChild("child", "cli"), [NewMessage("user", "summary", timestamp=1.0)], require_compression_lease=False)
        entries = backend.open(scope).sessions.list_sessions(SessionQuery(include_children=True, project_compression_tips=False), SessionWindow(limit=10))
        child = next(entry.session for entry in entries if entry.session.session_id == "child")
        assert child.display_name == "Peer"
        assert child.origin == {"platform": "telegram"}
