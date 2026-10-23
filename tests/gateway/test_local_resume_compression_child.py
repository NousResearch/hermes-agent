"""``session.resume`` by a compressed local conversation's physical transcript id (LC-12, #139306).

Session lists and the history picker can hand a client the physical id of a compression or reset
segment. The creation receipt lives on the logical owner only, so the resume must map the segment to
that owner (same conversation, same owner checks) instead of restoring the segment as if it were an
owner and answering ``storage_unavailable``. Real authority, real SQLite, real compression publish.
"""
import json
from types import SimpleNamespace

import pytest


def _runner(tmp_path):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return runner


def _compress(db, parent, child, source):
    history = [{'role': 'user', 'content': 'retained'}, {'role': 'assistant', 'content': 'history'}]
    assert db.try_acquire_compression_lock(parent, 'fixture')
    try:
        db.publish_compression_child(parent_session_id=parent, child_session_id=child,
                                     source=source, messages=history, compression_lock_holder='fixture')
    finally:
        db.release_compression_lock(parent, 'fixture')


async def _resume(authority, user_id, sid):
    from gateway.session_controls import AuthorityConnection
    # The private native bootstrap identity: the subject is the bare principal the session was made by.
    connection = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': user_id, 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:control']})
    return await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})


async def _compressed(tmp_path, monkeypatch):
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    authority = await initialize_session_authority(_runner(tmp_path), profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create', 'session:read'}), 'socket')
    ref = create_local_session(authority, actor, dict(request_id='lc12', source='gui', cwd=str(tmp_path),
                                                      model='frozen', toolsets=[]))
    live = authority.sessions[ref.session_id]
    store = authority.runner.session_store
    for parent, child in ((ref.session_id, 'child-1'), ('child-1', 'child-2')):
        # The gateway turn path: the turn re-stamps its row's routing peer (run_turn_persistence),
        # the compression split publishes the child, and the split sync re-stamps the child
        # (run_turn_runner), which stores the canonical row as ``source='local'``.
        store._record_gateway_session_peer(parent, live.route, live.source)
        _compress(authority.db, parent, child, authority.db.get_session(parent)['source'])
        store._record_gateway_session_peer(child, live.route, live.source)
    source = authority.db.get_session('child-2')['source']
    return SimpleNamespace(tmp_path=tmp_path, owner=ref.session_id, source=source, warm=authority)


async def _cold(compressed):
    from gateway.session_authority import initialize_session_authority
    return await initialize_session_authority(_runner(compressed.tmp_path), profile_id='fixture',
                                              instance_id='second')


@pytest.mark.asyncio
@pytest.mark.parametrize('cold', [False, True])
@pytest.mark.parametrize('segment', ['child-1', 'child-2'])
async def test_resume_by_a_compression_segment_opens_the_logical_owner(tmp_path, monkeypatch, cold, segment):
    compressed = await _compressed(tmp_path, monkeypatch)
    assert compressed.source == 'local'
    authority = await _cold(compressed) if cold else compressed.warm
    reply = await _resume(authority, 'owner', segment)
    assert 'result' in reply, reply
    assert reply['result']['session_id'] == compressed.owner
    assert [m['content'] for m in reply['result']['messages']] == ['retained', 'history']


@pytest.mark.asyncio
async def test_a_segment_of_another_owner_stays_refused(tmp_path, monkeypatch):
    compressed = await _compressed(tmp_path, monkeypatch)
    authority = await _cold(compressed)
    reply = await _resume(authority, 'intruder', 'child-2')
    assert reply['error']['message'] == 'permission_denied', reply


@pytest.mark.asyncio
async def test_a_fork_that_copied_the_owner_chat_id_is_no_alias(tmp_path, monkeypatch):
    compressed = await _compressed(tmp_path, monkeypatch)
    """A row outside the owner's receipt lineage that copied its ``chat_id`` is refused by name."""
    authority = await _cold(compressed)
    tip = authority.db.get_session('child-2')
    authority.db.create_session('stray', source='local', parent_session_id='child-2',
                                model_config={'_branched_from': 'child-2'}, chat_id=tip['chat_id'],
                                user_id=tip['user_id'], session_key=tip['session_key'])
    reply = await _resume(authority, 'owner', 'stray')
    assert reply['error']['message'] == 'storage_unavailable', reply
    assert json.dumps(reply)  # a named refusal, never an internal error
