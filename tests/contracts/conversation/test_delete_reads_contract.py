"""Contract cases for what a session delete leaves readable.

Seeding uses public protocol operations only (``sessions.create``, ``messages.append``, ``usage.queue_usage``), and assertions read back through ``messages.read_transcript`` and ``usage.usage_totals`` after ``persistence.conversation.sessions.SessionRepository.delete``.
"""

from __future__ import annotations

from persistence import SessionId
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from tests.contracts._scopes import MEDIA_ROOT


def test_a_deleted_sessions_transcript_and_usage_are_gone(backend, scope_a):
    store_a = backend.open(scope_a)
    gone, kept = SessionId("sess-del-read"), SessionId("sess-del-keep")
    for sid in (gone, kept):
        store_a.sessions.create(NewSession(session_id=sid, source="cli"))
        store_a.messages.append(sid, NewMessage(role="user", content=f"text of {sid}"))
        store_a.usage.queue_usage(sid, UsageDelta(tokens=TokenCounts(input_tokens=100), api_call_count=1,
                                                  route=BillingRoute(model="m1", billing_provider="p")))
    assert store_a.usage.flush_usage()
    before = store_a.usage.usage_totals().tokens
    store_a.sessions.delete(gone, media_root=MEDIA_ROOT)
    # CONTRACT: a delete removes the session's messages with its row, as main deletes its messages; nothing of the deleted conversation stays readable.
    assert store_a.messages.read_transcript(gone) == ()
    # CONTRACT: usage totals read only stored sessions, so the deleted session's tokens leave them; the kept session's stay.
    assert store_a.usage.usage_totals().tokens == before - 100
    assert [row.content for row in store_a.messages.read_transcript(kept)] == ["text of sess-del-keep"]
