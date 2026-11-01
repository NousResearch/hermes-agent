"""The session authority's message.complete carries the turn's committed row receipt and precedes
the idle session.info.

The TUI gateway's own turns publish ``persisted_turn`` (the stored row ids of the turn); the
Desktop binds the streamed reply to its stored row with it, so a transcript read that races the
completion never paints the reply twice. Turns the session authority runs through TurnRunner must
publish the same receipt.
"""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_contract import Principal, SessionRef, Submission
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch
from run_agent import AIAgent

ACTOR = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
REF = SessionRef('owned', 's')


def _flushing_agent(db, session_id):
    """Agent shell owning the real per-turn flush that stamps ``_row_id`` on live dicts."""
    agent = SimpleNamespace(
        _session_db=db, _session_db_created=True, _persist_disabled=False, session_id=session_id,
        _session_persist_lock=None, _flushed_db_message_ids=set(), _flushed_db_message_session_id=None,
        _last_flushed_db_idx=0, _persist_user_message_idx=None, _persist_user_message_override=None,
        _persist_user_message_timestamp=None, _pending_cli_user_message=None,
        context_compressor=SimpleNamespace(compression_count=0))
    agent._ensure_db_session = lambda: None
    agent._flush_messages_to_session_db = AIAgent._flush_messages_to_session_db.__get__(agent, AIAgent)
    agent._flush_messages_to_session_db_unlocked = AIAgent._flush_messages_to_session_db_unlocked.__get__(agent, AIAgent)
    return agent


def _tool_turn(db, session_id):
    """One earlier turn already stored, then a tool turn the agent runs and flushes itself."""
    db.append_message(session_id, 'user', 'earlier')
    db.append_message(session_id, 'assistant', 'before')
    history = [{'role': 'user', 'content': 'earlier'}, {'role': 'assistant', 'content': 'before'}]
    agent = _flushing_agent(db, session_id)
    call = {'id': 'c1', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{}'}}
    messages = list(history) + [
        {'role': 'user', 'content': 'b'},
        {'role': 'assistant', 'content': 'checking', 'tool_calls': [call]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'ok'},
        {'role': 'assistant', 'content': 'finished while away'}]
    agent._persist_user_message_idx = len(history)
    agent._flush_messages_to_session_db(messages, history)
    rows = db.get_messages_as_conversation(session_id, include_row_ids=True)
    return agent, history, messages, [row['_row_id'] for row in rows[-4:]]


def test_turn_runner_result_carries_the_committed_turn_receipt(tmp_path):
    from gateway.run_turn_runner import _persisted_turn
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    agent, history, messages, row_ids = _tool_turn(db, 's')
    result = {'final_response': 'finished while away', 'messages': messages}

    assert _persisted_turn(agent, history, result, 0) == {
        'row_ids': row_ids, 'complete': True, 'user_row_id': row_ids[0], 'final_assistant_row_id': row_ids[-1],
        'user_row_ids': [row_ids[0]]}
    # A compaction during the turn can drop streamed rows: the receipt no longer vouches for all of it.
    assert _persisted_turn(agent, history, result, 1)['complete'] is False


def test_steered_turn_receipt_names_every_user_row_in_order(tmp_path):
    """A steer/redirect adds a second user row, so the turn is never `complete`; the receipt still
    names each committed user row so the client can bind every optimistic bubble it painted."""
    from gateway.run_turn_runner import _persisted_turn
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    agent = _flushing_agent(db, 's')
    messages = [{'role': 'user', 'content': 'slow one'}, {'role': 'assistant', 'content': 'partial'},
                {'role': 'user', 'content': 'change course'}, {'role': 'assistant', 'content': 'steered reply'}]
    agent._persist_user_message_idx = 0
    agent._flush_messages_to_session_db(messages, [])
    row_ids = [row['_row_id'] for row in db.get_messages_as_conversation('s', include_row_ids=True)]
    receipt = _persisted_turn(agent, [], {'final_response': 'steered reply', 'messages': messages}, 0)

    assert receipt['complete'] is False
    assert receipt['user_row_ids'] == [row_ids[0], row_ids[2]]
    # An uncommitted steer row would shift every later pairing: publish no list at all.
    messages.insert(3, {'role': 'user', 'content': 'not flushed'})
    assert 'user_row_ids' not in _persisted_turn(agent, [], {'final_response': 'steered reply', 'messages': messages}, 0)


@pytest.mark.asyncio
async def test_authority_completion_publishes_the_turn_receipt(tmp_path, monkeypatch):
    from gateway import session_finite
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    epoch = begin_runtime_epoch(db, instance_id='current')
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='current', db=db, epoch=epoch)
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'route')
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    receipt = {'row_ids': [3, 4, 5, 6], 'complete': True, 'user_row_id': 3, 'final_assistant_row_id': 6}

    async def execute(authority, ref, row):
        authority.pending_results[row['admission_id']] = {'result': {
            'final_response': 'finished while away', 'messages': [], 'persisted_turn': receipt}, 'usage': {}}
        return 'finished while away'
    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)

    with db:
        await authority.submit(ACTOR, Submission(request_id='r', ref=REF, payload={'text': 'b'}, intent='queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 5)

    [complete] = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
    # The receipt names the client submission whose turn it is: the viewer that sent it binds its
    # optimistic prompt (`user-<submission_id>`) to the stored row, as main's submit ack does.
    assert complete['persisted_turn'] == {**receipt, 'submission_id': 'r'}
    # The idle snapshot follows the completion: a viewer never reads running=false for a turn whose
    # terminal frame has not been published yet.
    order = [(f['params']['type'], f['params']['payload'].get('running')) for f in frames
             if f['params']['type'] in ('message.complete', 'session.info')]
    assert order[-2:] == [('message.complete', None), ('session.info', False)]
