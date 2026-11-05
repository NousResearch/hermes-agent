"""Rename and archive of a cold compressed local conversation by its physical id (LC-22).

Session lists hand Desktop the listed row's physical id, which after a compression is a child
segment. ``_owner_ref`` remapped only ``delete`` to the lineage owner, so a rename or archive of a
cold child restored the CHILD as if it were an owner and answered ``storage_unavailable`` (same
class as LC-12 on the resume door). Real authority, real SQLite, real compression publish.
"""
import pytest

from tests.gateway.test_local_resume_compression_child import _cold, _compressed


async def _mutate(authority, sid, request_id, operation, payload):
    from gateway.session_controls import AuthorityConnection
    connection = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:control']})
    try:
        row = authority.db.get_session(sid)
        return await connection.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': request_id, 'expected_revision': row['runtime_revision'],
            'operation': operation, 'payload': payload}})
    finally:
        await connection.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('cold', [False, True])
async def test_rename_and_archive_of_a_compression_segment(tmp_path, monkeypatch, cold):
    compressed = await _compressed(tmp_path, monkeypatch)
    authority = await _cold(compressed) if cold else compressed.warm
    renamed = await _mutate(authority, 'child-2', 'rename-1', 'rename', {'title': 'Renamed'})
    assert 'result' in renamed, renamed
    archived = await _mutate(authority, 'child-2', 'archive-1', 'archive', {'archived': True})
    assert 'result' in archived, archived
    # The write lands on the listed row (the sidebar reads the tip's title back), not on the owner.
    assert authority.db.get_session('child-2').get('title') == 'Renamed'


@pytest.mark.asyncio
@pytest.mark.parametrize('addressed', ['owner', 'segment'])
async def test_a_segment_rewind_keeps_the_owner_busy_gate(tmp_path, monkeypatch, addressed):
    """Owner authorization is for metadata only: a rewind by a compression segment's id must not
    run while the conversation's FIFO holds queued input (the ledger keys admissions by owner id),
    so it is refused exactly like the same rewind by the conversation id (review of #140971)."""
    import hermes_state_runtime as rt
    from gateway.session_controls import AuthorityConnection
    compressed = await _compressed(tmp_path, monkeypatch)
    authority = compressed.warm
    evicted = []
    authority.runner._evict_cached_agent = lambda route, *a, **k: evicted.append(route)
    db = authority.db
    db.append_message('child-2', role='user', content='LIVE_Q')
    db.append_message('child-2', role='assistant', content='LIVE_A')
    target = next(m['id'] for m in db.get_messages('child-2') if m['content'] == 'LIVE_Q')
    rt.admit_session_input(db, epoch=authority.epoch, principal_id='owner', session_id=compressed.owner,
                           request_id='queued-1', payload={'text': 'queued after LIVE_A'})
    sid = compressed.owner if addressed == 'owner' else 'child-2'
    row = db.get_session(sid)
    connection = AuthorityConnection(authority, object(), {
        'provider': 'local', 'user_id': 'owner', 'profile_id': 'fixture', 'instance_id': authority.instance_id,
        'capabilities': ['session:create', 'session:read', 'session:submit', 'session:control']})
    try:
        reply = await connection.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': f'rewind-{addressed}', 'expected_revision': row['runtime_revision'],
            'expected_generation': row['runtime_generation'], 'operation': 'rewind',
            'payload': {'target_message_id': target}}})
    finally:
        await connection.close()
    assert reply.get('error', {}).get('message') in {'session_busy', 'storage_unavailable'}, reply
    assert [m['content'] for m in db.get_messages('child-2')][-2:] == ['LIVE_Q', 'LIVE_A']
