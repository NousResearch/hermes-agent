"""A resume that asks ``omit_messages`` (Desktop, which pages the transcript over REST) gets no transcript rows,
so opening a long compressed chat no longer reads its whole compression lineage; a resume without it still
gets the full display history, and both report the same todo plan.

Real store, authority, local session, compaction verb and ``AuthorityConnection.resume``; the runner is a
stand-in with no cached agent, and the resume ``info`` block is stubbed out.
"""
import json
from types import SimpleNamespace

import pytest

from agent.context_compressor import SUMMARY_PREFIX
from persistence.conversation.lineage import CompressionChild
from persistence.conversation.messages import NewMessage
from tests.fixtures.storage_read import session_row, transcript_rows
from tests.fixtures.storage_seed import seed_messages

TODOS = {'todos': [{'id': '1', 'content': 'ship it', 'status': 'in_progress'}]}


async def _local_session(tmp_path, monkeypatch):
    from gateway import run, session_local
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    monkeypatch.setattr(session_local, 'local_session_info', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store.persistence, adapters={}, _draining=False,
                             _cached_agent_for=lambda *a, **k: None, _evict_cached_agent=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'sock')
    ref = session_local.create_local_session(authority, actor, dict(
        request_id='resume', source='gui', cwd=str(tmp_path), model='frozen', toolsets=[]))
    return authority, actor, ref, authority.persistence.bindings.receipt(ref.session_id).entry['session_id']


async def _resume(authority, actor, ref, **params):
    from gateway.session_controls import AuthorityConnection
    conn = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    conn.actor = actor
    authority.events[actor.transport_id] = object()
    return await conn.resume(ref, {'session_id': ref.session_id, **params})


def _seed_compacted(db, physical):
    for n in range(3):
        seed_messages(db, physical, {'role': 'user', 'content': f'OLD_Q{n}'},
                      {'role': 'assistant', 'content': f'OLD_A{n}'})
    seed_messages(db, physical, {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': 'todo-1', 'type': 'function', 'function': {'name': 'todo', 'arguments': '{}'}}]})
    seed_messages(db, physical, {'role': 'tool', 'content': json.dumps(TODOS), 'tool_name': 'todo',
                                 'tool_call_id': 'todo-1'})
    seed_messages(db, physical, {'role': 'user', 'content': 'TAIL_Q'}, {'role': 'assistant', 'content': 'TAIL_A'})
    db.messages.compact(physical, [NewMessage.from_live(m) for m in [
        {'role': 'user', 'content': SUMMARY_PREFIX + '\nsummary'}, {'role': 'user', 'content': 'TAIL_Q'},
        {'role': 'assistant', 'content': 'TAIL_A'}]], carried_tail_count=2)


@pytest.mark.asyncio
async def test_omit_messages_resume_reads_no_transcript(tmp_path, monkeypatch):
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    _seed_compacted(authority.persistence, physical)
    full = await _resume(authority, actor, ref)
    assert any(row.get('archived') for row in full['messages'])  # the default resume still shows archived turns
    from gateway import session_local_recovery

    def lineage_read(*_args):
        raise AssertionError('an omit_messages resume read the transcript lineage')
    # The reply alone cannot tell a skipped read from one whose rows were discarded afterwards.
    monkeypatch.setattr(session_local_recovery, 'display_rows', lineage_read)
    omitted = await _resume(authority, actor, ref, omit_messages=True)
    assert omitted['messages'] == [] and omitted['messages_omitted'] is True
    assert omitted['message_count'] == session_row(authority.persistence, physical)['message_count']
    # Both paths report the same plan: the omitted one reads it from the store's bounded todo lookup.
    assert omitted['todo_state'] == full['todo_state'] and omitted['todo_state']


@pytest.mark.asyncio
async def test_omitted_resume_keeps_an_undone_plan_gone_after_a_rotated_compression(tmp_path, monkeypatch):
    """The rewind removes the child's copy of the Todo turn; the parent's copy must not restore its plan."""
    from gateway.session_mutations import mutate_session
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    db = authority.persistence
    call = {'id': 'todo-1', 'type': 'function', 'function': {'name': 'todo', 'arguments': '{}'}}
    tail = [{'role': 'user', 'content': 'TAIL_Q'}, {'role': 'assistant', 'content': None, 'tool_calls': [call]},
            {'role': 'tool', 'content': json.dumps(TODOS), 'tool_name': 'todo', 'tool_call_id': 'todo-1'},
            {'role': 'assistant', 'content': 'TAIL_A'}]
    seed_messages(db, physical, {'role': 'user', 'content': 'OLD_Q'}, {'role': 'assistant', 'content': 'OLD_A'}, *tail)
    assert db.coordination.try_acquire_compression_lock(session_id=physical, holder='fixture')
    try:
        db.lineage.publish_compression_child(physical, CompressionChild(session_id='child', source='gui'), [
            NewMessage.from_live(m) for m in [{'role': 'user', 'content': SUMMARY_PREFIX + '\nsummary'}, *tail]],
                                             compression_lock_holder='fixture')
    finally:
        db.coordination.release_compression_lock(session_id=physical, holder='fixture')
    handle = authority._handle(ref)
    await mutate_session(authority, actor, ref, {
        'session_id': ref.session_id, 'request_id': 'undo', 'operation': 'rewind',
        'expected_revision': handle.revision, 'expected_generation': handle.execution_generation,
        'payload': {'target_message_id': next(m['id'] for m in transcript_rows(db, 'child') if m['content'] == 'TAIL_Q')}})
    full = await _resume(authority, actor, ref)
    assert full['todo_state'] is None
    omitted = await _resume(authority, actor, ref, omit_messages=True)
    assert omitted['todo_state'] == full['todo_state']


@pytest.mark.asyncio
async def test_omitted_resume_reply_passes_the_wire_contract(tmp_path, monkeypatch):
    """Through ``dispatch``, which checks the reply against ``CanonicalSnapshot`` (strict under tests)."""
    from gateway.session_controls import AuthorityConnection
    from gateway import session_local
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    monkeypatch.setattr(session_local, 'local_session_info', lambda *a: {'desktop_protocol': 'hermes-gateway-v1'})
    _seed_compacted(authority.persistence, physical)
    conn = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    conn.actor = actor
    authority.events[actor.transport_id] = object()
    reply = await conn.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'session.resume',
                                 'params': {'session_id': ref.session_id, 'omit_messages': True}})
    assert 'error' not in reply, reply
    assert reply['result']['messages'] == [] and reply['result']['messages_omitted'] is True


@pytest.mark.asyncio
async def test_omitted_resume_racing_a_delete_answers_not_found(tmp_path, monkeypatch):
    """A delete committing after the attach, before the omitted reply's count/plan read, is ``not_found``."""
    from gateway.session_controls import AuthorityConnection
    from gateway.session_ingress_media import native_inputs_root
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    _seed_compacted(authority.persistence, physical)
    attach = authority.attach

    async def attach_then_delete(*args, **kwargs):
        snapshot = await attach(*args, **kwargs)
        for sid in (physical, ref.session_id):
            authority.persistence.sessions.delete(sid, media_root=str(native_inputs_root()))
        return snapshot
    monkeypatch.setattr(authority, 'attach', attach_then_delete)
    conn = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    conn.actor = actor
    authority.events[actor.transport_id] = object()
    reply = await conn.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'session.resume',
                                 'params': {'session_id': ref.session_id, 'omit_messages': True}})
    assert reply['error']['data']['reason'] == 'not_found', reply
