"""Exact GPT-Live coverage resolves to one canonical SQLite row."""

from typing import Literal

import pytest

from persistence import SessionId
from persistence.conversation.live_voice import VoiceFragment, VoiceTurn
from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence.core.errors import Conflict
from persistence.sqlite.conversation._store import live_voice


@pytest.fixture
def capture(backend, scope_a):
    store = backend.open(scope_a)
    session = store.sessions.create(NewSession(SessionId('voice-chat'), 'desktop'))
    call = store.messages.start_voice_capture(session, 'call')
    return store, session, call


def _claim(store, session, call, *, ids=('a', 'b'), admission='submit-1', prompt='Same words.', after=None):
    holder = store.holder()

    def write(conn):
        row = live_voice.resolve_claim(
            holder, conn, session_id=session, call_id=call, event_ids=ids,
            admission_id=admission, expected_prompt=prompt,
        )
        if after is not None:
            after()
        return row

    return holder._execute_write(write)


def _words(store, call, *, ids=('a', 'b'), start=0, role: Literal['user', 'assistant'] = 'user'):
    for event, text, offset in zip(ids, (' Same\t', 'words. '), (0, 100)):
        store.messages.ingest_voice_fragment(call, VoiceFragment(event, role, text, start + offset, start + offset + 100))


@pytest.mark.parametrize('published', [False, True], ids=['pending', 'projected'])
def test_exact_claim_replays_after_reopen_without_eating_late_or_identical_speech(capture, backend, scope_a, published):
    store, session, call = capture
    _words(store, call)
    turn = VoiceTurn(('a', 'b'))
    if published:
        assert store.messages.publish_voice_turns(call, (turn,)) == 1
    before = store.messages.read_transcript(session)
    result = _claim(store, session, call)
    rows = store.messages.read_transcript(session)
    assert [(r.role, r.content) for r in rows] == [('user', ' Same\twords. ')]
    assert (result.row_id, result.message_uid) == (rows[0].id, rows[0].message_uid)
    if published:
        assert rows == before
    assert not store.messages.pending_voice_fragments(call)
    assert store.messages.publish_voice_turns(call, (turn,)) == 0

    late = VoiceFragment('late', 'user', ' Plus one.', 200, 300)
    store.messages.ingest_voice_fragment(call, late)
    backend.reopen()
    store = backend.open(scope_a)
    assert _claim(store, session, call) == result
    with pytest.raises(Conflict):
        _claim(store, session, call, ids=('a', 'b', 'late'))
    with pytest.raises(Conflict):
        _claim(store, session, call, admission='other')
    assert store.messages.pending_voice_fragments(call) == (late,)
    store.messages.ingest_voice_fragment(call, VoiceFragment('reply', 'assistant', 'Okay.', 350, 400))
    store.messages.publish_voice_turns(call, (VoiceTurn(('late',)), VoiceTurn(('reply',))))
    _words(store, call, ids=('repeat-a', 'repeat-b'), start=500)
    assert _claim(store, session, call, ids=('repeat-a', 'repeat-b'), admission='repeat') != result
    assert [r.content for r in store.messages.read_transcript(session)] == [
        ' Same\twords. ', ' Plus one.', 'Okay.', ' Same\twords. ',
    ]
    assert _claim(store, session, call) == result


def test_pause_inside_one_user_run_is_not_a_boundary(capture):
    """A pause is never a split: dropping words before it could drop a negation."""
    store, session, call = capture
    store.messages.ingest_voice_fragment(call, VoiceFragment('a', 'user', "Don't ", 0, 100))
    store.messages.ingest_voice_fragment(call, VoiceFragment('b', 'user', 'delete it.', 4000, 4100))
    with pytest.raises(Conflict):
        _claim(store, session, call, ids=('b',), admission='suffix', prompt='delete it.')
    assert not store.messages.read_transcript(session)
    result = _claim(store, session, call, prompt="Don't delete it.")
    rows = store.messages.read_transcript(session)
    assert [(r.role, r.content) for r in rows] == [('user', "Don't delete it.")]
    assert result.row_id == rows[0].id


def test_claim_uses_wire_order_not_timestamps(capture):
    store, session, call = capture
    store.messages.ingest_voice_fragment(call, VoiceFragment('a', 'user', '\ufeffFirst ', 5300, 5600))
    store.messages.ingest_voice_fragment(call, VoiceFragment('b', 'user', 'second. ', 5200, 5450))
    _claim(store, session, call, prompt='First second.')
    assert store.messages.read_transcript(session)[0].content == '\ufeffFirst second. '


@pytest.mark.parametrize('already_claimed', [False, True])
def test_projection_cannot_be_rebound_to_edited_content(capture, already_claimed):
    store, session, call = capture
    _words(store, call)
    if already_claimed:
        _claim(store, session, call)
    else:
        store.messages.publish_voice_turns(call, (VoiceTurn(('a', 'b')),))
    row = store.messages.read_transcript(session)[0]
    assert store.messages.set_user_message_content(session, row.id, 'Edited.')
    with pytest.raises(Conflict):
        _claim(store, session, call, prompt='Edited.')
    assert [r.content for r in store.messages.read_conversation(session)] == ['Edited.']


@pytest.mark.parametrize('case', [
    'subset', 'reordered', 'duplicate', 'assistant', 'wrong-prompt', 'wrong-session',
    'untimed', 'overlap', 'noncontiguous', 'mixed', 'split-projection', 'partial', 'projected-prefix',
    'not-tail', 'equal-text-not-tail', 'rewound', 'deleted-row', 'no-receipt', 'missing-id',
])
def test_inexact_claim_refuses_without_changing_coverage_or_history(capture, case):
    store, session, call = capture
    if case == 'untimed':
        for event, text in [('a', 'Same '), ('b', 'words.')]:
            store.messages.ingest_voice_fragment(call, VoiceFragment(event, 'user', text, None, None))
    elif case in ('noncontiguous', 'overlap'):
        store.messages.ingest_voice_fragment(call, VoiceFragment('a', 'user', 'Same ', 0, 100))
        start = 80 if case == 'overlap' else 100
        store.messages.ingest_voice_fragment(call, VoiceFragment('x', 'assistant', 'Really?', start, 200))
        second_start = 150 if case == 'overlap' else 200
        store.messages.ingest_voice_fragment(call, VoiceFragment('b', 'user', 'words.', second_start, 300))
    else:
        _words(store, call, role='assistant' if case == 'assistant' else 'user')
    if case == 'mixed':
        store.messages.publish_voice_turns(call, (VoiceTurn(('a',)),))
    if case == 'split-projection':
        store.messages.publish_voice_turns(call, (VoiceTurn(('a',)), VoiceTurn(('b',))))
    if case == 'projected-prefix':
        store.messages.publish_voice_turns(call, (VoiceTurn(('a',)),))
    if case in ('partial', 'not-tail', 'equal-text-not-tail', 'rewound', 'deleted-row', 'no-receipt'):
        store.messages.publish_voice_turns(call, (VoiceTurn(('a', 'b'), partial=case == 'partial'),))
    if case == 'not-tail':
        store.messages.append(session, NewMessage('assistant', 'Already answered.'))
    if case == 'equal-text-not-tail':
        store.messages.append(session, NewMessage('user', ' Same\twords. '))
    if case == 'rewound':
        store.messages.rewind_user_turn(session, 0)
    if case == 'deleted-row':
        store.messages.clear_transcript(session)
        store.messages.append(session, NewMessage('user', ' Same\twords. '))
    if case == 'no-receipt':
        # Earlier publishers left fragment coverage without a canonical row address.
        store.holder()._execute_write(lambda conn: conn.execute('DELETE FROM live_voice_projections WHERE call_id=?', (call,)))
    pending = store.messages.pending_voice_fragments(call)
    history = store.messages.read_conversation(session)
    target = store.sessions.create(NewSession(SessionId('other-chat'), 'desktop')) if case == 'wrong-session' else session
    ids = {'subset': ('a',), 'reordered': ('b', 'a'), 'duplicate': ('a', 'a'),
           'projected-prefix': ('a',), 'missing-id': ('a', 'missing')}.get(case, ('a', 'b'))
    prompt = {'wrong-prompt': 'Different words.', 'subset': 'Same', 'projected-prefix': 'Same'}.get(case, 'Same words.')
    with pytest.raises(Conflict):
        _claim(store, target, call, ids=ids, prompt=prompt)
    assert store.messages.pending_voice_fragments(call) == pending
    assert store.messages.read_conversation(session) == history


@pytest.mark.parametrize('published', [False, True], ids=['pending', 'projected'])
def test_claim_rolls_back_with_outer_transaction(capture, published):
    store, session, call = capture
    _words(store, call)
    if published:
        store.messages.publish_voice_turns(call, (VoiceTurn(('a', 'b')),))
    pending = store.messages.pending_voice_fragments(call)
    history = store.messages.read_conversation(session)

    def refuse():
        raise RuntimeError('caller refused after resolving')

    with pytest.raises(RuntimeError, match='caller refused'):
        _claim(store, session, call, after=refuse)
    assert store.messages.pending_voice_fragments(call) == pending
    assert store.messages.read_conversation(session) == history
    _claim(store, session, call, admission='retry-after-rollback')
    assert len(store.messages.read_conversation(session)) == 1
