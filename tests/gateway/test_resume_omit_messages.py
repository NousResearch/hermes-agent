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

TODOS = {'todos': [{'id': '1', 'content': 'ship it', 'status': 'in_progress'}]}


async def _local_session(tmp_path, monkeypatch):
    from gateway import run, session_local
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from hermes_state_local import local_receipt
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    monkeypatch.setattr(session_local, 'local_session_info', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda *a, **k: None, _evict_cached_agent=lambda *a, **k: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'sock')
    ref = session_local.create_local_session(authority, actor, dict(
        request_id='resume', source='gui', cwd=str(tmp_path), model='frozen', toolsets=[]))
    return authority, actor, ref, local_receipt(authority.db, ref.session_id)['entry']['session_id']


async def _resume(authority, actor, ref, **params):
    from gateway.session_controls import AuthorityConnection
    conn = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    conn.actor = actor
    authority.events[actor.transport_id] = object()
    return await conn.resume(ref, {'session_id': ref.session_id, **params})


def _seed_compacted(db, physical):
    for n in range(3):
        db.append_message(physical, 'user', f'OLD_Q{n}')
        db.append_message(physical, 'assistant', f'OLD_A{n}')
    db.append_message(physical, 'assistant', None, tool_calls=[
        {'id': 'todo-1', 'type': 'function', 'function': {'name': 'todo', 'arguments': '{}'}}])
    db.append_message(physical, 'tool', json.dumps(TODOS), tool_name='todo', tool_call_id='todo-1')
    db.append_message(physical, 'user', 'TAIL_Q')
    db.append_message(physical, 'assistant', 'TAIL_A')
    db.archive_and_compact(physical, [{'role': 'user', 'content': SUMMARY_PREFIX + '\nsummary'},
                                      {'role': 'user', 'content': 'TAIL_Q'},
                                      {'role': 'assistant', 'content': 'TAIL_A'}], tail_count=2)


@pytest.mark.asyncio
async def test_omit_messages_resume_reads_no_transcript(tmp_path, monkeypatch):
    authority, actor, ref, physical = await _local_session(tmp_path, monkeypatch)
    _seed_compacted(authority.db, physical)
    full = await _resume(authority, actor, ref)
    assert any(row.get('archived') for row in full['messages'])  # the default resume still shows archived turns
    omitted = await _resume(authority, actor, ref, omit_messages=True)
    assert omitted['messages'] == [] and omitted['messages_omitted'] is True
    assert omitted['message_count'] == authority.db.get_session(physical)['message_count']
    # Both paths report the same plan: the omitted one reads it from the store's bounded todo lookup.
    assert omitted['todo_state'] == full['todo_state'] and omitted['todo_state']
