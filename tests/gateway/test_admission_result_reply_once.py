"""LC-9: a settled turn result stores the reply once, and every reader still gets it exactly."""
import json

import pytest

from gateway.session_api_turn import _settled_api_result
from gateway.session_bot import _admission_outcome
from gateway.session_results import _redacted, admission_result, finish_result, retain_result
from hermes_state_runtime import _json, admit_session_input, claim_session_input, mutate_runtime_session
from hermes_state_terminal import RESULT_PREFIX, compact_result, encode_result, load_result

REPLY = 'x' * 2000


def _turn(owner, sid, n, history, reply=REPLY):
    text = f'question {n}'
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id=sid, request_id=f'r{n}',
                        payload={'text': text})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=sid)
    history += [{'role': 'user', 'content': text},
                {'role': 'assistant', 'content': 'checking', 'tool_calls': [{'id': f'c{n}'}]},
                {'role': 'tool', 'tool_call_id': f'c{n}', 'content': 'tool output'},
                {'role': 'assistant', 'content': reply, 'reasoning': 'why'}]
    result = {'result': {'final_response': reply, 'messages': list(history), 'completed': True,
                         'current_turn_user_idx': len(history) - 4}, 'usage': {'input_tokens': 3}}
    # What the pre-LC-9 format stored for this turn: what every reader must still get back.
    expected = json.loads(_json(_redacted(compact_result(result, user_message=text))))
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    return row, expected


def _stored_bytes(owner, admission_id):
    with owner.db._read_ctx() as conn:
        return conn.execute('SELECT length(value) FROM state_meta WHERE key=?',
                            (RESULT_PREFIX + admission_id,)).fetchone()[0]


def test_per_turn_result_bytes_are_about_one_reply_not_two(owner):
    owner.db.create_session('live', source='api_server')
    history, sizes = [], []
    for n in range(1, 11):
        row, expected = _turn(owner, 'live', n, history)
        assert admission_result(owner.db, row['admission_id']) == expected
        sizes.append(_stored_bytes(owner, row['admission_id']))
    # Flat per turn (never cumulative), and the reply is stored once: ~1x its size, not ~2x.
    assert max(sizes) - min(sizes) <= 8, sizes  # turn 10's ids are one character longer
    assert sizes[0] < len(REPLY) * 1.5, f'result row {sizes[0]} bytes for a {len(REPLY)}-char reply'


def test_every_reader_gets_the_exact_reply(owner):
    owner.db.create_session('live', source='api_server')
    row, expected = _turn(owner, 'live', 1, [])
    admission_id = row['admission_id']
    assert expected['result']['messages'][-1] == {'role': 'assistant', 'content': REPLY, 'reasoning': 'why'}
    assert admission_result(owner.db, admission_id) == expected
    # Delivery after settlement (gateway finish), Bot receipt, Responses/chat replay.
    assert finish_result(owner.db, epoch=owner.epoch, row=row, response='other', outcome='completed')[1] == REPLY
    assert _admission_outcome(owner, admission_id) == ('settled', REPLY)
    assert _settled_api_result(owner, admission_id) == (expected['result'], expected['usage'])
    # An exact retry replays the same admission and the same result.
    replay = admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='live',
                                 request_id='r1', payload={'text': 'question 1'})
    assert replay['admission_id'] == admission_id and replay['status'] == 'terminal'
    assert admission_result(owner.db, admission_id) == expected


def test_a_reply_that_differs_from_the_last_message_is_kept_verbatim(owner):
    """Media tags / sanitizing can make final_response differ from the message: nothing is dropped."""
    owner.db.create_session('live', source='api_server')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='live', request_id='r',
                        payload={'text': 'q'})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id='live')
    result = {'result': {'final_response': 'answer\nMEDIA:/tmp/a.png', 'current_turn_user_idx': 0,
                         'messages': [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'answer'}]},
              'usage': {}}
    expected = json.loads(_json(_redacted(compact_result(result, user_message='q'))))
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    assert admission_result(owner.db, row['admission_id']) == expected


def test_an_old_shape_result_committed_before_the_upgrade_replays_exactly(owner):
    owner.db.create_session('live', source='api_server')
    row, expected = _turn(owner, 'live', 1, [])
    admission_id = row['admission_id']
    # The row as the previous release wrote it: the reply both in final_response and the message.
    old = json.dumps(expected)
    owner.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                                                      (old, RESULT_PREFIX + admission_id)))
    replay = admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id='live',
                                 request_id='r1', payload={'text': 'question 1'})
    assert replay['admission_id'] == admission_id and replay['status'] == 'terminal'
    assert replay['payload'] == {'text': 'question 1'}
    assert admission_result(owner.db, admission_id) == expected
    assert _admission_outcome(owner, admission_id) == ('settled', REPLY)
    assert _settled_api_result(owner, admission_id) == (expected['result'], expected['usage'])
    assert finish_result(owner.db, epoch=owner.epoch, row=row, response='other', outcome='completed')[1] == REPLY
    with owner.db._read_ctx() as conn:  # reading never rewrites the stored row
        assert conn.execute('SELECT value FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone()[0] == old


def test_a_deleted_session_keeps_the_exact_retry_reply(owner):
    """Retirement drops the transcript copies; the reply itself must survive the new format."""
    owner.db.create_session('gone', source='api_server')
    row, expected = _turn(owner, 'gone', 1, [])
    snap = owner.db.get_session('gone')
    mutate_runtime_session(owner.db, epoch=owner.epoch, principal_id='human', session_id='gone', request_id='del',
                           operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                           expected_generation=snap['runtime_generation'])
    retired = admission_result(owner.db, row['admission_id'])
    assert retired['result']['final_response'] == REPLY and retired['result']['messages'] == []
    assert retired['usage'] == expected['usage']
    # Retirement decodes before compacting: its row is what it kept from a pre-LC-9 row, byte for byte.
    from hermes_state_mutation_retirement import _TRANSCRIPT_FIELDS
    kept = {**expected, 'result': {**{k: v for k, v in expected['result'].items() if k not in _TRANSCRIPT_FIELDS},
                                   'messages': []}}
    with owner.db._read_ctx() as conn:
        raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (RESULT_PREFIX + row['admission_id'],)).fetchone()[0]
    assert raw == _json(kept)


def test_every_raw_result_row_read_goes_through_the_decoder(owner, monkeypatch):
    """Delivery after settlement (finish_result) and retirement read stored rows too."""
    import gateway.session_results
    import hermes_state_terminal
    seen = []

    def spy(text):
        seen.append(text)
        return load_result(text)
    monkeypatch.setattr(hermes_state_terminal, 'load_result', spy)
    monkeypatch.setattr(gateway.session_results, 'load_result', spy)
    owner.db.create_session('gone', source='api_server')
    row, _expected = _turn(owner, 'gone', 1, [])
    assert finish_result(owner.db, epoch=owner.epoch, row=row, response='other', outcome='completed')[1] == REPLY
    assert len(seen) == 1, 'finish_result read the stored row without load_result'
    snap = owner.db.get_session('gone')
    mutate_runtime_session(owner.db, epoch=owner.epoch, principal_id='human', session_id='gone', request_id='del',
                           operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                           expected_generation=snap['runtime_generation'])
    assert len(seen) == 2, 'retirement read the stored row without load_result'


def _stored_round_trip(owner, result, request_id='r'):
    owner.db.create_session(request_id, source='api_server')
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='api', session_id=request_id,
                        request_id=request_id, payload={'text': 'q'})
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=request_id)
    expected = json.loads(_json(_redacted(compact_result(result, user_message='q'))))
    retain_result(owner.db, epoch=owner.epoch, row=row, result=result)
    return row['admission_id'], expected


def _result(*messages, **top):
    return {'result': {'final_response': REPLY, 'current_turn_user_idx': 0,
                       'messages': [{'role': 'user', 'content': 'q'}, *messages]}, 'usage': {}, **top}


_LEGACY = {'result': {'final_response': REPLY, '_messages_are_turn_suffix': True, 'messages': [
    {'role': 'tool', 'tool_call_id': 'c1', 'content': 'tool evidence', '_content_is_final_response': True},
    {'role': 'assistant', 'content': None, '_content_is_final_response': True},
    {'role': 'assistant', 'content': REPLY}]}, 'usage': {}}


@pytest.mark.parametrize('result, copies', [
    pytest.param(_result({'role': 'assistant', 'content': 'checking', 'tool_calls': [{'id': 'c'}]},
                         {'role': 'tool', 'tool_call_id': 'c', 'content': 'out'},
                         {'role': 'assistant', 'content': REPLY, 'reasoning': 'why'}), 1, id='reply-stored-once'),
    pytest.param(_result({'role': 'assistant', 'content': 'distinct', '_content_is_final_response': True}), 1,
                 id='distinct-reply-beside-an-old-marker-field'),
    pytest.param(_result({'role': 'assistant', 'content': REPLY, '_content_is_final_response': True}), 1,
                 id='old-marker-field-on-the-reply'),
    *(pytest.param(_result({'role': 'assistant', 'content': None}, _stored=theirs), 1, id=f'own-_stored-{n}')
      for n, theirs in enumerate(['not ours', {'v': 1, 'reply_ref': 0}, {'v': 1, 'raw': 'x'}])),
    pytest.param(_LEGACY, None, id='legacy-row-with-any-message-fields'),
])
def test_any_stored_row_reads_back_as_the_pre_change_shape(owner, result, copies):
    """Every row, new or written by an earlier release, decodes to the identical dict, key order
    included, before and after its session is deleted. Message fields are not format metadata,
    and a producer-built ``_stored`` is never taken for the envelope."""
    admission_id, expected = _stored_round_trip(owner, result)
    if copies is None:  # the row as an earlier release wrote it
        expected = result
        owner.db._execute_write(lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?',
                                                          (json.dumps(result), RESULT_PREFIX + admission_id)))
    with owner.db._read_ctx() as conn:
        raw = conn.execute('SELECT value FROM state_meta WHERE key=?', (RESULT_PREFIX + admission_id,)).fetchone()[0]
    assert copies is None or raw.count(REPLY) == copies
    assert json.dumps(admission_result(owner.db, admission_id)) == json.dumps(expected)
    assert json.dumps(load_result(json.dumps(encode_result(expected)))) == json.dumps(expected)
    # Retirement rewrites the row (keys sorted) without its transcript copies; nothing else changes.
    snap = owner.db.get_session('r')
    mutate_runtime_session(owner.db, epoch=owner.epoch, principal_id='human', session_id='r', request_id='del',
                           operation='delete', payload={}, expected_revision=snap['runtime_revision'],
                           expected_generation=snap['runtime_generation'])
    from hermes_state_mutation_retirement import _TRANSCRIPT_FIELDS
    kept = json.loads(_json({**expected, 'result': {
        **{k: v for k, v in expected['result'].items() if k not in _TRANSCRIPT_FIELDS}, 'messages': []}}))
    assert json.dumps(admission_result(owner.db, admission_id)) == json.dumps(kept)
