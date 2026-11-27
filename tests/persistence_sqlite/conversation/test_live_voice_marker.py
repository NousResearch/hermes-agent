"""A provider delegation marker freezes the exact observed wire prefix."""

import pytest

from persistence import SessionId
from persistence.conversation.live_voice import VoiceFragment
from persistence.conversation.sessions import NewSession
from persistence.core.errors import Conflict, NotFound


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
    marker, prefix, late = holder._execute_write(lambda conn: (
        conn.execute('SELECT fragment_seq FROM live_voice_markers WHERE call_id=?', (call,)).fetchone()[0],
        conn.execute('SELECT seq FROM live_voice_fragments WHERE call_id=? AND event_id=?', (call, 'prefix')).fetchone()[0],
        conn.execute('SELECT seq FROM live_voice_fragments WHERE call_id=? AND event_id=?', (call, 'late')).fetchone()[0],
    ))
    assert marker == prefix < late
    assert [f.event_id for f in store.messages.pending_voice_fragments(call)] == ['prefix', 'late']
