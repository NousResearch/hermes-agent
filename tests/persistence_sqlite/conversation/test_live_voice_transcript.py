"""GPT-Live pending text survives restart and projects once into ordinary history."""

import pytest

from persistence import SessionId
from persistence.conversation.live_voice import VoiceFragment, VoiceTurn
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.errors import Conflict, NotFound, SessionRowMissing, TurnLeaseLost


def test_fragment_replay_recovery_and_publication(backend, scope_a, scope_b):
    from persistence.conversation.live_voice import VoiceFragment, VoiceTurn

    store = backend.open(scope_a)
    session = store.sessions.create(NewSession('synthetic', 'desktop'))
    other = backend.open(scope_b)
    other.sessions.create(NewSession('synthetic', 'desktop'))
    call = store.messages.start_voice_capture(session, 'call-1')
    user = VoiceFragment('e1', 'user', 'Do not email Alice. 🗣', 0, 1000)
    reply = VoiceFragment('e2', 'assistant', 'Understood.', 1100, 1600)
    assert store.messages.ingest_voice_fragment(call, user)
    assert store.messages.ingest_voice_fragment(call, reply)

    backend.reopen()
    store = backend.open(scope_a)
    assert store.messages.pending_voice_fragments(call) == (user, reply)
    assert not store.messages.ingest_voice_fragment(call, user)
    with pytest.raises(Conflict):
        store.messages.ingest_voice_fragment(call, VoiceFragment('e1', 'user', 'Email Alice.', 0, 1000))
    with pytest.raises(NotFound):
        other.messages.pending_voice_fragments(call)

    turns = (VoiceTurn(('e1',)), VoiceTurn(('e2',), partial=True))
    assert store.messages.publish_voice_turns(call, turns) == 2
    backend.reopen()
    store = backend.open(scope_a)
    assert store.messages.publish_voice_turns(call, turns) == 0
    assert not store.messages.ingest_voice_fragment(call, reply)
    assert store.messages.pending_voice_fragments(call) == ()
    history = store.messages.read_conversation(session)
    assert [(row.role, row.content) for row in history] == [('user', user.text), ('assistant', reply.text)]
    assert [row.display_metadata['voice_live']['partial'] for row in history] == [False, True]


def test_rebinding_or_deletion_cannot_send_voice_to_a_different_session(backend, scope_a, tmp_path):
    from persistence.conversation.live_voice import VoiceFragment

    store = backend.open(scope_a)
    session = store.sessions.create(NewSession('original', 'desktop'))
    different = store.sessions.create(NewSession('different', 'desktop'))
    call = store.messages.start_voice_capture(session, 'call-1')
    with pytest.raises(Conflict):
        store.messages.start_voice_capture(different, call)
    assert store.messages.ingest_voice_fragment(call, VoiceFragment('e1', 'user', 'Pending', None, None))
    store.sessions.delete(session, media_root=str(tmp_path))
    with pytest.raises(NotFound):
        store.messages.ingest_voice_fragment(call, VoiceFragment('e2', 'user', 'Never re-home', None, None))
    with pytest.raises(SessionRowMissing):
        store.messages.start_voice_capture(session, call)
    assert not store.messages.read_conversation(different)


def test_rewind_does_not_resurrect_pending_speech(backend, scope_a):
    store = backend.open(scope_a)
    session = store.sessions.create(NewSession(SessionId('rewind-chat'), 'desktop'))
    store.messages.append(session, NewMessage('user', 'Undo this conversation.'))
    call = store.messages.start_voice_capture(session, 'open-before-rewind')
    assert store.messages.ingest_voice_fragment(call, VoiceFragment('pending', 'user', 'Unpublished words', 0, 200))
    store.messages.rewind_user_turn(session, 0)
    backend.reopen()
    store = backend.open(scope_a)
    assert not store.messages.pending_voice_fragments(call)
    with pytest.raises(Conflict):
        store.messages.ingest_voice_fragment(call, VoiceFragment('late', 'user', 'Never restore me', 300, 500))
    assert not store.messages.read_conversation(session)
    new_call = store.messages.start_voice_capture(session, 'normal-after-rewind')
    assert store.messages.ingest_voice_fragment(new_call, VoiceFragment('fresh', 'user', 'New call', 0, 400))


def test_refuses_capture_for_a_finalized_chat(backend, scope_a):
    store = backend.open(scope_a)
    session = store.sessions.create(NewSession(SessionId('closed-chat'), 'desktop'))
    store.sessions.end(session, reason='shutdown')
    assert store.sessions.get(session).ended_at is not None
    with pytest.raises(Conflict):
        store.messages.start_voice_capture(session, 'refused-call')
    with pytest.raises(NotFound):
        store.messages.pending_voice_fragments('refused-call')


def test_busy_projection_leaves_a_recoverable_tail(backend, scope_a):
    store = backend.open(scope_a)
    session = store.sessions.create(NewSession(SessionId('voice-busy'), 'desktop'))
    call = store.messages.start_voice_capture(session, 'call-busy')
    delta = VoiceFragment('e1', 'user', 'Still pending', 0, 400)
    assert store.messages.ingest_voice_fragment(call, delta)
    assert store.coordination.try_acquire_turn_lease(session_id=session, holder='other-turn')
    store.messages.finish_voice_capture(call, incomplete=True)

    with pytest.raises(TurnLeaseLost):
        store.messages.publish_voice_turns(call, (VoiceTurn(('e1',)),))

    assert store.messages.pending_voice_calls() == ((call, True),)
    assert store.messages.pending_voice_fragments(call) == (delta,)
    assert not store.messages.read_conversation(session)
    store.coordination.release_turn_lease(session_id=session, holder='other-turn')
    assert store.messages.publish_voice_turns(call, (VoiceTurn(('e1',)),)) == 1
    assert store.messages.read_conversation(session)[0].display_metadata['voice_live']['partial']
    assert not store.messages.pending_voice_calls()
