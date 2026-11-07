"""Imported transcripts share the peer conversation generation counter."""

from datetime import UTC, datetime


from persistence.conversation.messages import ConversationPeer
from persistence.conversation.sessions import NewSession
from tests.contracts.conversation.test_messages import Totals, totals


def test_imported_sessions_advance_and_preserve_peer_generations(backend, scope_a):
    store_a = backend.open(scope_a)
    peer = ConversationPeer(source="telegram", session_key="peer")
    for generation, session_id in enumerate(("first", "second"), 1):
        imported = store_a.maintenance.import_sessions([{"id": session_id, "source": "telegram", "started_at": datetime(2026, 3, 1, tzinfo=UTC).timestamp(), "messages": []}])
        assert imported.ok and imported.imported == 1
        store_a.sessions.create(NewSession(session_id, "telegram", session_key="peer"))
        store_a.sessions.end(session_id, "session_reset")
        assert (store_a.sessions.get(session_id)).end_reason == "session_reset"
        assert store_a.messages.latest_conversation_boundary(peer) == generation
        assert store_a.messages.read_transcript(session_id) == ()
        assert totals(backend, scope_a, session_id) == Totals(0, 0, 0)
