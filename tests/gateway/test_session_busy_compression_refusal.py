"""A rewind refused because another writer holds the conversation's compression lock says so.

Main's busy text told the user to wait for a running ``/compress`` instead of pressing Stop
(``tui_gateway/user_messages.py::busy_message``); the shared owner's refusal keeps its ``session_busy``
reason and names the lock in ``field`` so a client can pick that wording.
"""
import pytest

from tests.gateway.test_session_mutation_model_busy import _owner


async def _rewind(db, owner, ref, target):
    row = db.get_session(ref.session_id)
    first = db.get_messages(target)[0]['id']
    return await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': ref.session_id, 'request_id': 'rewind', 'expected_revision': row['runtime_revision'],
        'expected_generation': row['runtime_generation'], 'operation': 'rewind',
        'payload': {'target_message_id': first}}})


@pytest.mark.asyncio
async def test_rewind_under_a_held_compression_lock_names_the_lock(tmp_path, monkeypatch, caplog):
    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        db.append_message(target, 'user', 'keep')
        assert db.try_acquire_compression_lock(target, 'another-writer')
        import logging
        with caplog.at_level(logging.DEBUG, logger='gateway.session_controls'):
            reply = await _rewind(db, owner, ref, target)
        assert reply['error']['message'] == 'session_busy', reply
        assert reply['error']['data'] == {'reason': 'session_busy', 'field': 'compression'}
        # A retrying client (Desktop's edit retries busy for 6 s) must not flood the log.
        refusals = [r for r in caplog.records if 'session.mutate refused' in r.getMessage()]
        assert refusals and all(r.levelno == logging.DEBUG for r in refusals), refusals
        assert [m['content'] for m in db.get_messages(target)] == ['keep']
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_rewind_with_an_admitted_turn_stays_a_plain_busy_refusal(tmp_path, monkeypatch):
    import hermes_state_runtime as rt

    async def handle(event):
        return 'ok'
    store, db, authority, owner, ref = _owner(tmp_path, monkeypatch, handle)
    try:
        from gateway.session_local_recovery import transcript_target
        target = transcript_target(authority, ref)
        db.append_message(target, 'user', 'keep')
        rt.admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id=ref.session_id,
                               request_id='input', payload={'text': 'go'})
        reply = await _rewind(db, owner, ref, target)
        assert reply['error']['data'] == {'reason': 'session_busy'}, reply
    finally:
        await owner.close()
        store.close_all_db_handles()


def test_authorize_reads_a_null_profile_as_this_owner():
    from types import SimpleNamespace
    from gateway.session_busy_controls import authorize
    from hermes_state_runtime import RuntimeStoreError

    seen = []
    authority = SimpleNamespace(profile_id='/nonexistent/owner-home', authorize=lambda *a: seen.append(a))
    connection = SimpleNamespace(authority=authority, actor='actor')
    authorize(connection, 'ref', {'profile': None}, 'session:read')
    assert seen == [('actor', 'ref', 'session:read')]
    with pytest.raises(RuntimeStoreError) as refused:
        authorize(connection, 'ref', {'profile': 7}, 'session:read')
    assert refused.value.reason == 'invalid_params'
