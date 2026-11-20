"""A provider delegation marker freezes the exact observed wire prefix."""

import pytest

from persistence import SessionId
from persistence.conversation.live_voice import VoiceFragment
from persistence.conversation.sessions import NewSession
from persistence.core.errors import Conflict, NotFound
from persistence.sqlite.conversation._store import live_voice


def test_replayed_marker_keeps_the_first_prefix_after_reopen(backend, scope_a, scope_b):
    store = backend.open(scope_a)
    chat = store.sessions.create(NewSession(SessionId('voice'), 'desktop'))
    call = store.messages.start_voice_capture(chat, 'call')
    # End timestamp deliberately exceeds the marker offset, as seen on the live wire.
    store.messages.ingest_voice_fragment(call, VoiceFragment('prefix', 'user', 'Denver.', 5400, 5600))
    assert store.messages.record_voice_marker(call, 'delegation', 'provider-marker', 5400)
    store.messages.ingest_voice_fragment(call, VoiceFragment('late', 'user', ' More.', 5600, 5800))
    backend.reopen()
    store = backend.open(scope_a)
    assert not store.messages.record_voice_marker(call, 'delegation', 'provider-marker', 5400)
    with pytest.raises(Conflict):
        store.messages.record_voice_marker(call, 'delegation', 'provider-marker', 5800)

    with pytest.raises(NotFound):
        backend.open(scope_b).messages.record_voice_marker(call, 'delegation', 'provider-marker', 5400)

    holder = store.holder()

    def resolve(conn, ids, prompt):
        return live_voice.resolve_claim(
            holder, conn, session_id=chat, call_id=call, event_ids=ids, admission_id='request-1',
            expected_prompt=prompt, delegation_id='delegation',
        )

    with pytest.raises(Conflict):
        holder._execute_write(lambda conn: resolve(conn, ('prefix', 'late'), 'Denver. More.'))
    result = holder._execute_write(lambda conn: resolve(conn, ('prefix',), 'Denver.'))
    assert holder._execute_write(lambda conn: resolve(conn, ('prefix',), 'Denver.')) == result
    assert [r.content for r in store.messages.read_transcript(chat)] == ['Denver.']
    assert [f.event_id for f in store.messages.pending_voice_fragments(call)] == ['late']
