"""GPT-Live exact coverage through the real SQLite FIFO claim transaction."""

import os
import sqlite3

import pytest

from persistence import SessionId
from persistence.authority.admissions import GateRefusal, VOICE_CLAIM_V1
from persistence.conversation.live_voice import VoiceFragment, VoiceTurn
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.errors import AuthorityRefusal, PermanentStoreError
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch


@pytest.fixture
def unmarked_voice_fifo(store):
    epoch = begin_runtime_epoch(store.holder(), instance_id='voice-owner')
    chat = store.sessions.create(NewSession(SessionId('voice'), 'desktop'))
    call = store.messages.start_voice_capture(chat, 'call')
    for fragment in (VoiceFragment('a', 'user', ' Same\t', 0, 100),
                     VoiceFragment('b', 'user', 'words. ', 100, 200)):
        store.messages.ingest_voice_fragment(call, fragment)
    return store, epoch, chat, call


@pytest.fixture
def voice_fifo(unmarked_voice_fifo):
    store, _, _, call = unmarked_voice_fifo
    store.messages.record_voice_marker(call, 'delegate', 'marker', 150)
    return unmarked_voice_fifo


def _payload(call, *, marker='delegate', ids=('a', 'b')):
    return {'text': 'Same words.', VOICE_CLAIM_V1: {
        'call_id': call, 'delegation_id': marker, 'event_ids': list(ids),
    }}


def _admit(store, epoch, chat, request, payload, **kwargs):
    return store.admissions.admit(epoch=epoch, principal_id='owner', session_id=chat,
                                  request_id=request, payload=payload, **kwargs)


@pytest.mark.parametrize('projected', [False, True], ids=['pending', 'projected'])
def test_marker_rejects_stale_user_even_after_assistant_reply(unmarked_voice_fifo, projected):
    store, epoch, chat, call = unmarked_voice_fifo
    if projected:
        store.messages.publish_voice_turns(call, (VoiceTurn(('a', 'b')),))
    store.messages.ingest_voice_fragment(call, VoiceFragment('reply', 'assistant', 'Okay.', 200, 400))
    store.messages.ingest_voice_fragment(call, VoiceFragment('new', 'user', 'New request.', 400, 600))
    store.messages.record_voice_marker(call, 'delegate', 'marker', 600)
    pending = store.messages.pending_voice_fragments(call)
    history = store.messages.read_transcript(chat)

    stale, _ = _admit(store, epoch, chat, 'stale', _payload(call))
    assert store.admissions.claim(epoch=epoch, session_id=chat, head=stale.admission_id) is False
    assert store.admissions.get(admission_id=stale.admission_id).outcome == 'rejected'
    assert store.messages.pending_voice_fragments(call) == pending
    assert store.messages.read_transcript(chat) == history

    latest_payload = _payload(call, ids=('new',)) | {'text': 'New request.'}
    latest, _ = _admit(store, epoch, chat, 'latest', latest_payload)
    claimed = store.admissions.claim(epoch=epoch, session_id=chat, head=latest.admission_id)
    rows = store.messages.read_transcript(chat)
    assert [(r.role, r.content) for r in rows] == [
        ('user', ' Same\twords. '), ('assistant', 'Okay.'), ('user', 'New request.'),
    ]
    assert claimed.voice_input.row_id == rows[-1].id
    assert not store.messages.pending_voice_fragments(call)
    replay, created = _admit(store, epoch, chat, 'latest', latest_payload)
    assert not created and replay.admission_id == latest.admission_id
    assert store.messages.read_transcript(chat) == rows


@pytest.mark.parametrize('reply_before', [False, True], ids=['following-reply', 'preceding-reply'])
def test_marker_refuses_overlap_with_any_adjacent_reply_delta(unmarked_voice_fifo, reply_before):
    store, epoch, chat, call = unmarked_voice_fifo
    if reply_before:
        fragments = (
            VoiceFragment('r1', 'assistant', 'Of', 300, 510),
            VoiceFragment('r2', 'assistant', ' course.', 220, 350),
            VoiceFragment('new', 'user', 'New request.', 500, 600),
        )
        payload = _payload(call, ids=('new',)) | {'text': 'New request.'}
    else:
        fragments = (
            VoiceFragment('r1', 'assistant', 'Of', 210, 350),
            VoiceFragment('r2', 'assistant', ' course.', 180, 250),
        )
        payload = _payload(call)
    for fragment in fragments:
        store.messages.ingest_voice_fragment(call, fragment)
    store.messages.record_voice_marker(call, 'delegate', 'marker', 600)
    pending = store.messages.pending_voice_fragments(call)

    admission, _ = _admit(store, epoch, chat, 'overlap', payload)
    assert store.admissions.claim(epoch=epoch, session_id=chat, head=admission.admission_id) is False
    assert store.admissions.get(admission_id=admission.admission_id).outcome == 'rejected'
    assert store.messages.pending_voice_fragments(call) == pending
    assert not store.messages.read_transcript(chat)


@pytest.mark.parametrize('marker_first', [True, False], ids=['marker-before-claim', 'marker-after-claim'])
def test_sideband_marker_committed_after_submit_is_checked_at_claim(unmarked_voice_fifo, marker_first):
    store, epoch, chat, call = unmarked_voice_fifo
    payload = _payload(call)
    admitted, _ = _admit(store, epoch, chat, 'speech', payload)
    if marker_first:
        store.messages.record_voice_marker(call, 'delegate', 'marker', 150)
    claimed = store.admissions.claim(epoch=epoch, session_id=chat, head=admitted.admission_id)
    if marker_first:
        assert claimed.voice_input.text == ' Same\twords. '
    else:
        assert claimed is False
        store.messages.record_voice_marker(call, 'delegate', 'marker', 150)
        replay, created = _admit(store, epoch, chat, 'speech', payload)
        assert not created and replay.outcome == 'rejected'
        assert store.admissions.claim(epoch=epoch, session_id=chat) is None
        assert not store.messages.read_transcript(chat)
        assert [fragment.event_id for fragment in store.messages.pending_voice_fragments(call)] == ['a', 'b']


@pytest.mark.parametrize('published', [False, True], ids=['pending', 'published'])
def test_fifo_claim_owns_exact_coverage_and_exposes_canonical_input(voice_fifo, published):
    store, epoch, chat, call = voice_fifo
    turn = VoiceTurn(('a', 'b'))
    if published:
        assert store.messages.publish_voice_turns(call, (turn,)) == 1
    before = store.messages.read_transcript(chat)
    late = VoiceFragment('late', 'user', ' Later.', 200, 300)
    store.messages.ingest_voice_fragment(call, late)
    payload = _payload(call)
    admitted, created = _admit(store, epoch, chat, 'request', payload)
    assert created
    follower, _ = _admit(store, epoch, chat, 'next', {'text': 'typed'})
    assert store.admissions.claim(epoch=epoch, session_id=chat, head=follower.admission_id) is False
    assert store.messages.read_transcript(chat) == before

    claimed = store.admissions.claim(epoch=epoch, session_id=chat, head=admitted.admission_id)
    assert claimed.admission_id == admitted.admission_id and claimed.status == 'started'
    rows = store.messages.read_transcript(chat)
    assert [(r.role, r.content) for r in rows] == [('user', ' Same\twords. ')]
    if published:
        assert rows == before
    voice = claimed.voice_input
    assert voice is not None
    assert (voice.session_id, voice.row_id, voice.message_uid) == (chat, rows[0].id, rows[0].message_uid)
    assert voice.text == rows[0].content
    assert claimed.transcript_mark >= voice.row_id  # settlement must not close over the saved speech
    assert store.messages.pending_voice_fragments(call) == (late,)
    assert store.messages.publish_voice_turns(call, (turn,)) == 0
    replay, created = _admit(store, epoch, chat, 'request', payload)
    assert not created and replay.admission_id == claimed.admission_id and replay.payload == payload
    assert store.admissions.claim(epoch=epoch, session_id=chat) is None
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        _admit(store, epoch, chat, 'request', _payload(call, ids=('a', 'b', 'late')))


def _malformed_ids(call):
    payload = _payload(call)
    payload[VOICE_CLAIM_V1]['event_ids'] = 'a'
    return payload


# Each unsafe proof: (payload for the call, whether it is admitted into another chat).
_UNSAFE_PROOFS = {
    'unknown-marker': (lambda call: _payload(call, marker='not-observed'), False),
    'not-in-marker': (lambda call: _payload(call, ids=('a', 'b', 'late')) | {'text': 'Same words. More.'}, False),
    'malformed': (_malformed_ids, False),
    'wrong-chat': (_payload, True),
    'wrong-text': (lambda call: _payload(call) | {'text': 'Different words.'}, False),
}


@pytest.mark.parametrize('failure', list(_UNSAFE_PROOFS))
def test_unsafe_proof_rejects_only_that_head_without_consuming_speech(voice_fifo, failure):
    store, epoch, chat, call = voice_fifo
    late = VoiceFragment('late', 'user', ' More.', 200, 300)
    store.messages.ingest_voice_fragment(call, late)
    build, other_chat = _UNSAFE_PROOFS[failure]
    payload = build(call)
    if other_chat:
        chat = store.sessions.create(NewSession(SessionId('other'), 'desktop'))
    before = store.messages.pending_voice_fragments(call)
    admitted, _ = _admit(store, epoch, chat, 'unsafe', payload)
    follower, _ = _admit(store, epoch, chat, 'next', {'text': 'typed'})

    assert store.admissions.claim(epoch=epoch, session_id=chat, head=admitted.admission_id) is False
    terminal = store.admissions.get(admission_id=admitted.admission_id)
    assert (terminal.status, terminal.outcome, terminal.generation) == ('terminal', 'rejected', None)
    verdict = store.admissions.settled_result(admission_id=admitted.admission_id)
    assert verdict.record == terminal and verdict.result['reason'] == 'admission_conflict'
    assert store.messages.pending_voice_fragments(call) == before
    assert not store.messages.read_transcript(chat)
    replay, created = _admit(store, epoch, chat, 'unsafe', payload)
    assert not created and replay == terminal
    claimed = store.admissions.claim(epoch=epoch, session_id=chat, head=follower.admission_id)
    assert claimed.admission_id == follower.admission_id and claimed.generation == 1
    assert claimed.voice_input is None


def test_epoch_gate_and_turn_lease_leave_voice_head_queued(voice_fifo):
    store, epoch, chat, call = voice_fifo
    admitted, _ = _admit(store, epoch, chat, 'speech', _payload(call))
    before = store.messages.pending_voice_fragments(call)
    with pytest.raises(AuthorityRefusal, match='stale_epoch'):
        store.admissions.claim(epoch=epoch - 1, session_id=chat)
    with pytest.raises(AuthorityRefusal, match='runtime_draining'):
        store.admissions.claim(epoch=epoch, session_id=chat, gate=GateRefusal('runtime_draining'))
    holder = f'pid={os.getpid()}:turn=other'
    assert store.coordination.try_acquire_turn_lease(session_id=chat, holder=holder)
    assert store.admissions.claim(epoch=epoch, session_id=chat) is None
    assert store.admissions.get(admission_id=admitted.admission_id) == admitted
    assert store.messages.pending_voice_fragments(call) == before
    assert not store.messages.read_transcript(chat)
    store.coordination.release_turn_lease(session_id=chat, holder=holder)
    assert store.admissions.claim(epoch=epoch, session_id=chat).generation == 1


@pytest.mark.parametrize('published', [False, True], ids=['pending', 'published'])
def test_voice_reservation_rolls_back_when_start_write_aborts(voice_fifo, published):
    store, epoch, chat, call = voice_fifo
    if published:
        store.messages.publish_voice_turns(call, (VoiceTurn(('a', 'b')),))
    admitted, _ = _admit(store, epoch, chat, 'speech', _payload(call))
    transcript = store.messages.read_transcript(chat)
    pending = store.messages.pending_voice_fragments(call)
    with sqlite3.connect(store.store_path) as conn:
        conn.execute("""CREATE TRIGGER abort_voice_start BEFORE UPDATE ON session_admissions
            WHEN NEW.status='started' BEGIN SELECT RAISE(ABORT, 'voice start interrupted'); END""")
    with pytest.raises(PermanentStoreError, match='voice start interrupted'):
        store.admissions.claim(epoch=epoch, session_id=chat)
    assert store.admissions.get(admission_id=admitted.admission_id) == admitted
    assert store.messages.read_transcript(chat) == transcript
    assert store.messages.pending_voice_fragments(call) == pending
    with sqlite3.connect(store.store_path) as conn:
        assert conn.execute('SELECT admission_id FROM live_voice_projections').fetchall() == (
            [(None,)] if published else [])
        conn.execute('DROP TRIGGER abort_voice_start')
    assert store.admissions.claim(epoch=epoch, session_id=chat).generation == 1


@pytest.mark.parametrize('mutation', [None, 'append', 'rewind'])
def test_returned_voice_claim_reuses_only_the_same_live_tail(voice_fifo, backend, scope_a, mutation):
    store, epoch, chat, call = voice_fifo
    admitted, _ = _admit(store, epoch, chat, 'speech', _payload(call))
    claimed = store.admissions.claim(epoch=epoch, session_id=chat)
    returned = store.admissions.return_claim(epoch=epoch, admission_id=claimed.admission_id,
        generation=claimed.generation, queued_owner_epoch=claimed.queued_owner_epoch)
    assert returned == admitted
    if mutation == 'append':
        store.messages.append(chat, NewMessage(role='user', content=claimed.voice_input.text))
    elif mutation == 'rewind':
        store.messages.rewind_user_turn(chat, 0)
    before = store.messages.read_transcript(chat)
    backend.reopen()
    store = backend.open(scope_a)
    reclaimed = store.admissions.claim(epoch=epoch, session_id=chat)
    if mutation is None:
        assert reclaimed.voice_input == claimed.voice_input
        store.admissions.settle(epoch=epoch, admission_id=reclaimed.admission_id,
                                generation=reclaimed.generation, outcome='completed')
        duplicate, _ = _admit(store, epoch, chat, 'duplicate', _payload(call))
        assert store.admissions.claim(epoch=epoch, session_id=chat) is False
        assert store.admissions.get(admission_id=duplicate.admission_id).outcome == 'rejected'
    else:
        assert reclaimed is False
        assert store.admissions.get(admission_id=admitted.admission_id).outcome == 'rejected'
    assert store.messages.read_transcript(chat) == before


# A real provider call recorded on macOS: roles, provider timings, arrival order
# and marker frontier as recorded; provider IDs replaced. Real deltas leave
# 200-400 ms between words, and the assistant's "Hello!" arrived before the
# user's trailing "delegating", which therefore starts the delegated run.
_RECORDED_CALL = (
    ('user', 'My', 3000, 3200), ('user', ' test', 3200, 3400), ('user', ' phrase', 3600, 3800),
    ('user', ' is', 3800, 4000), ('user', ' copper', 4200, 4400), ('user', ' lighthouse', 4800, 5000),
    ('assistant', 'Got', 5200, 5400), ('assistant', ' it,', 5400, 5600), ('assistant', " I'll", 5800, 6000),
    ('assistant', ' keep', 6000, 6200), ('assistant', ' that in', 6200, 6400), ('assistant', ' mind.', 6400, 6600),
    ('user', 'Say', 9400, 9600), ('user', ' hello', 9800, 10000), ('user', ' without', 10200, 10400),
    ('assistant', 'Hello!', 10800, 11000),
    ('user', 'delegating', 11000, 11200), ('user', ' Ask', 20800, 21000), ('user', ' Hermes', 21200, 21400),
    ('user', ' to', 21400, 21600), ('user', ' reply', 21800, 22000), ('user', ' with', 22000, 22200),
    ('user', ' exactly', 22600, 22800), ('user', ' pilot', 23200, 23400), ('user', ' check', 23600, 23800),
)
_RECORDED_DELEGATED = 'delegating Ask Hermes to reply with exactly pilot check'


@pytest.mark.parametrize('reply_started', [False, True], ids=['before-reply', 'after-reply'])
def test_real_word_gaps_in_one_user_run_still_claim_once(store, reply_started):
    """Live regression: the owner refused every delegation whose words were not gapless."""
    epoch = begin_runtime_epoch(store.holder(), instance_id='voice-owner')
    chat = store.sessions.create(NewSession(SessionId('recorded-call'), 'desktop'))
    call = store.messages.start_voice_capture(chat, 'call')
    run, ids = [], []
    for seq, (role, text, start, end) in enumerate(_RECORDED_CALL, 1):
        event = f'ev-{seq}'
        if run and run[0][0] != role:
            # The recorder publishes each completed speaker run when the next one starts.
            store.messages.publish_voice_turns(call, (VoiceTurn(tuple(e for _, e in run)),))
            run = []
        store.messages.ingest_voice_fragment(call, VoiceFragment(event, role, text, start, end))
        run.append((role, event))
        ids.append(event)
    claimed_ids = ids[16:]
    store.messages.record_voice_marker(call, 'delegate', 'marker', 24400)
    if reply_started:
        store.messages.ingest_voice_fragment(call, VoiceFragment('reply', 'assistant', 'Sure.', 24600, 24800))
        store.messages.publish_voice_turns(call, (VoiceTurn(tuple(claimed_ids)),))
    payload = {'text': _RECORDED_DELEGATED, VOICE_CLAIM_V1: {
        'call_id': call, 'delegation_id': 'delegate', 'event_ids': claimed_ids}}

    admitted, _ = _admit(store, epoch, chat, 'voice-live:recorded', payload)
    claimed = store.admissions.claim(epoch=epoch, session_id=chat, head=admitted.admission_id)

    assert claimed, store.admissions.get(admission_id=admitted.admission_id)
    rows = store.messages.read_transcript(chat)
    assert [(r.role, r.content) for r in rows] == [
        ('user', 'My test phrase is copper lighthouse'), ('assistant', "Got it, I'll keep that in mind."),
        ('user', 'Say hello without'), ('assistant', 'Hello!'), ('user', _RECORDED_DELEGATED),
    ]
    assert claimed.voice_input.row_id == rows[-1].id and claimed.voice_input.text == _RECORDED_DELEGATED
    # Each saved run names its first provider event: Desktop replaces exactly that
    # run's live words with it (timing alone repeats across calls).
    assert [r.display_metadata['voice_live']['first_event_id'] for r in rows] == ['ev-1', 'ev-7', 'ev-13', 'ev-16', 'ev-17']
    # One opaque key per call: Desktop never joins rows across calls (their clocks restart).
    keys = {r.display_metadata['voice_live']['call_key'] for r in rows}
    assert len(keys) == 1 and call not in keys.pop()
    replay, created = _admit(store, epoch, chat, 'voice-live:recorded', payload)
    assert not created and replay.admission_id == admitted.admission_id
    assert store.messages.read_transcript(chat) == rows
