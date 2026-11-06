"""``session.transcript`` and ``session.compactions``: a stored conversation read without attaching (#150091)."""
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_mutation_model_busy import _owner


async def _pages(connection, session_id, limit, between=None):
    """Every page of *session_id*'s transcript; *between* runs after the first page."""
    texts, cursor = [], None
    while True:
        params = {'session_id': session_id, 'limit': limit, **({'cursor': cursor} if cursor else {})}
        reply = await connection.dispatch({'id': 1, 'method': 'session.transcript', 'params': params})
        assert 'result' in reply, reply
        texts += [row['text'] for row in reply['result']['rows']]
        cursor = reply['result']['next_cursor']
        if between is not None:
            between()
            between = None
        if cursor is None:
            return texts


@pytest.mark.asyncio
async def test_transcript_pages_survive_a_compaction_and_never_attach(tmp_path, monkeypatch):
    from agent.context_compressor import SUMMARY_PREFIX
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local_recovery import transcript_target

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        target = transcript_target(authority, ref)
        for i in (1, 2, 3):
            db.append_message(target, 'user', f'question {i}')
            db.append_message(target, 'assistant', f'answer {i}')

        def compact():
            tail = [{'role': 'user', 'content': 'question 3'}, {'role': 'assistant', 'content': 'answer 3'}]
            db.archive_and_compact(target, [{'role': 'user', 'content': f'{SUMMARY_PREFIX}\n## Goal\nship it'}, *tail],
                                   watermark=db.get_active_message_watermark(target), tail_count=len(tail))

        # The first page is read before the compaction, the rest after it: each message once, in order.
        assert await _pages(owner, ref.session_id, 2, between=compact) == [
            'question 1', 'answer 1', 'question 2', 'answer 2', 'question 3', 'answer 3']
        reply = await owner.dispatch({'id': 2, 'method': 'session.compactions',
                                      'params': {'session_id': ref.session_id}})
        assert [c['summary'] for c in reply['result']['compactions']] == ['## Goal\nship it'], reply
        assert authority.sessions[ref.session_id].subscribers == {}
        assert ref.session_id not in owner.subscriptions

        other = AuthorityConnection(authority, object(), {'user_id': 'someone-else'})
        refused = await other.dispatch({'id': 3, 'method': 'session.transcript',
                                        'params': {'session_id': ref.session_id}})
        assert refused['error']['data']['reason'] == 'permission_denied', refused
        bad = await owner.dispatch({'id': 4, 'method': 'session.transcript',
                                    'params': {'session_id': ref.session_id, 'cursor': 'not-a-cursor'}})
        assert bad['error']['data']['reason'] == 'invalid_params', bad
        await other.close()
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_old_compressed_transcript_reads_across_its_lineage_without_adoption(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    db.create_session('old-tui', source='tui', model='m', cwd=str(tmp_path))
    db.append_message('old-tui', 'user', 'before compression')
    db.append_message('old-tui', 'assistant', 'old answer')
    assert db.try_acquire_compression_lock('old-tui', 'fixture')
    try:
        db.publish_compression_child(parent_session_id='old-tui', child_session_id='tui-tip', source='tui',
            model='m', messages=[{'role': 'user', 'content': 'after compression'},
                                 {'role': 'assistant', 'content': 'new answer'}],
            compression_lock_holder='fixture')
    finally:
        db.release_compression_lock('old-tui', 'fixture')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})
    remote = AuthorityConnection(authority, object(), {'provider': 'remote', 'user_id': 'remote',
        'profile_id': str(tmp_path), 'instance_id': 'owner'}, operator=True)
    try:
        listed = await native.dispatch({'id': 1, 'method': 'session.list', 'params': {'scope': 'stored'}})
        [row] = listed['result']['sessions']
        before = db._conn.total_changes
        assert await _pages(native, row['session_id'], 3) == [
            'before compression', 'old answer', 'after compression', 'new answer']
        assert db._conn.total_changes == before  # nothing adopted, nothing written
        assert authority.sessions == {}
        refused = await remote.dispatch({'id': 2, 'method': 'session.transcript',
                                         'params': {'session_id': row['session_id']}})
        assert refused['error']['data']['reason'] == 'not_found', refused
    finally:
        await native.close()
        await remote.close()
        store.close_all_db_handles()
