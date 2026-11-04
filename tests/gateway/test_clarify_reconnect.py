"""Reconnectable canonical batch-clarify snapshots, receipts and identity fences."""
import asyncio

import pytest

from tests.gateway.test_clarify_lock_owner import QUESTIONS, _start_batch, _wait_for
from tests.gateway.test_surface_admission import _authority


async def _resume(connection):
    return await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {
        'session_id': 's', 'omit_messages': True, 'defer_history': True}})


async def _reconcile(connection, request_id, **identity):
    return await connection.dispatch({'id': 20, 'method': 'clarify.reconcile', 'params': {
        'request_id': request_id, **identity}})


async def _lock(connection, request_id, question_id, answer, **identity):
    return await connection.dispatch({'id': 30, 'method': 'clarify.lock', 'params': {
        'request_id': request_id, 'question_id': question_id, 'answer': answer, **identity}})


def _request_id(peer):
    return next(frame['id'] for frame in peer.frames if frame.get('method') == 'clarify')


def _stop_batch(authority, worker):
    from gateway.session_server_requests import withdraw
    if worker.is_alive():
        withdraw(authority.sessions['s'].event_stream, 's', 'clarify')
    worker.join(5)
    assert not worker.is_alive()


@pytest.mark.asyncio
async def test_resume_replays_partial_batch_without_transcript(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'

    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    fresh = None
    worker = None
    try:
        initial = await _resume(owner)
        assert initial['result']['messages_omitted'] is True
        peer, worker, _out = _start_batch(authority)
        await _wait_for(lambda: any(frame.get('method') == 'clarify' for frame in peer.frames))
        request_id = _request_id(peer)
        before_lock = await _resume(owner)
        before_entry = next(item for item in before_lock['result']['open_requests'] if item['id'] == request_id)
        assert before_entry['params']['answers'] == {}
        locked = await _lock(owner, request_id, 'q0', 'blue')
        assert locked['result'] == {'status': 'ok', 'remaining': ['q1']}

        fresh = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        resumed = await _resume(fresh)
        described = await fresh.dispatch({'jsonrpc': '2.0', 'id': 'describe',
                                          'method': 'runtime.describe', 'params': {}})
        assert 'batch-clarify-reconnect-v1' in described['result']['capabilities']
        result = resumed['result']
        assert result['messages'] == []
        assert result['messages_omitted'] is True
        entry = next(item for item in result['open_requests'] if item['id'] == request_id)
        assert entry['method'] == 'clarify'
        assert entry['params']['answers'] == {'q0': 'blue'}
        assert [q['qid'] for q in entry['params']['questions']] == ['q0', 'q1']
        assert entry['params']['session_id'] == 's'
        assert 'blue' not in repr(result['messages'])
    finally:
        if worker is not None:
            _stop_batch(authority, worker)
        if fresh is not None:
            await fresh.close()
        await owner.close()


@pytest.mark.asyncio
async def test_terminal_replay_uses_competing_response_answers(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'

    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    fresh = None
    worker = None
    try:
        await _resume(owner)
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(frame.get('method') == 'clarify' for frame in peer.frames))
        request_id = _request_id(peer)
        await _lock(owner, request_id, 'q0', 'monitor partial')
        final = {'q0': 'rival free', 'q1': '["blue"]'}
        await owner.dispatch({'jsonrpc': '2.0', 'id': request_id,
                              'result': {'answers': final, 'outcome': 'submitted'}})
        await asyncio.to_thread(worker.join, 5)
        assert out['answers'] == final
        fresh = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        restored = await _resume(fresh)
        entry = next(item for item in restored['result']['request_receipts'] if item['id'] == request_id)
        assert entry['params']['answers'] == final
        assert (await _reconcile(fresh, request_id))['result']['answers'] == final
    finally:
        if worker is not None:
            _stop_batch(authority, worker)
        if fresh is not None:
            await fresh.close()
        await owner.close()


@pytest.mark.asyncio
async def test_reconcile_reads_open_and_submitted_receipts_without_replaying_locks(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'

    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    outsider = AuthorityConnection(authority, object(), {'user_id': 'someone-else'})
    worker = None
    try:
        resumed = await _resume(owner)
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(frame.get('method') == 'clarify' for frame in peer.frames))
        request_id = _request_id(peer)
        opened = await _reconcile(owner, request_id)
        assert 'result' in opened, opened
        assert opened['result']['status'] == 'open'
        assert opened['result']['answers'] == {}
        assert opened['result']['remaining'] == ['q0', 'q1']
        assert opened['result']['authority_epoch'] == resumed['result']['authority_epoch']
        assert opened['result']['execution_generation'] >= resumed['result']['execution_generation']
        # A fresh viewer learns the active turn's fences from the read-only receipt, then locks against them.
        identity = {key: opened['result'][key] for key in ('authority_epoch', 'execution_generation')}
        denied = await _reconcile(outsider, request_id, **identity)
        assert denied['result'] == {'status': 'expired', 'request_id': request_id}

        assert (await _lock(owner, request_id, 'q0', 'blue'))['result']['remaining'] == ['q1']
        partial = await _reconcile(owner, request_id, **identity)
        assert partial['result']['answers'] == {'q0': 'blue'}
        assert partial['result']['remaining'] == ['q1']
        assert out == {}

        assert (await _lock(owner, request_id, 'q1', None))['result']['remaining'] == []
        await asyncio.to_thread(worker.join, 5)
        assert out['outcome'] == 'submitted'
        receipt = await _reconcile(owner, request_id, **identity)
        assert receipt['result']['status'] == 'submitted'
        assert receipt['result']['answers'] == {'q0': 'blue', 'q1': None}
        assert receipt['result']['remaining'] is None
        repeated = await _reconcile(owner, request_id, **identity)
        assert repeated == receipt
        assert out['answers'] == {'q0': 'blue', 'q1': None}
        fresh = AuthorityConnection(authority, object(), {'user_id': 'owner'})
        try:
            restored = await _resume(fresh)
            assert request_id not in {entry['id'] for entry in restored['result']['open_requests']}
            receipt_entry = next(
                entry for entry in restored['result']['request_receipts'] if entry['id'] == request_id
            )
            assert receipt_entry['params']['answers'] == {'q0': 'blue', 'q1': None}
            assert restored['result']['messages'] == []
        finally:
            await fresh.close()

        expired = await _reconcile(owner, 'srq-does-not-exist', **identity)
        assert expired['result'] == {'status': 'expired', 'request_id': 'srq-does-not-exist'}
        late = await _lock(owner, request_id, 'q1', 'late')
        assert late['result'] == {'status': 'expired'}
        denied_final = await _reconcile(outsider, request_id, **identity)
        assert denied_final['result'] == {'status': 'expired', 'request_id': request_id}
    finally:
        if worker is not None:
            _stop_batch(authority, worker)
        await outsider.close()
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['cancelled', 'timed_out'])
async def test_reconcile_retains_partial_answers_for_non_success_outcomes(tmp_path, monkeypatch, outcome):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'

    if outcome == 'timed_out':
        monkeypatch.setattr('tools.clarify_gateway.get_clarify_timeout', lambda: 1)
    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    worker = None
    try:
        await _resume(owner)
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(frame.get('method') == 'clarify' for frame in peer.frames))
        request_id = _request_id(peer)
        await _lock(owner, request_id, 'q0', 'kept')
        if outcome == 'cancelled':
            await owner.dispatch({'jsonrpc': '2.0', 'id': request_id, 'result': {}})
        await asyncio.to_thread(worker.join, 5)
        assert out['outcome'] == outcome

        receipt = await _reconcile(owner, request_id)
        assert receipt['result']['status'] == outcome
        assert receipt['result']['answers'] == {'q0': 'kept'}
        assert receipt['result']['status'] != 'submitted'
        late = await _lock(owner, request_id, 'q1', 'must-not-apply')
        assert late['result'] == {'status': 'expired'}
        assert (await _reconcile(owner, request_id))['result']['answers'] == {'q0': 'kept'}
    finally:
        if worker is not None:
            _stop_batch(authority, worker)
        await owner.close()


@pytest.mark.asyncio
async def test_lock_and_reconcile_fences_refuse_stale_identity_without_mutation(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'

    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    worker = None
    try:
        await _resume(owner)
        peer, worker, _out = _start_batch(authority)
        await _wait_for(lambda: any(frame.get('method') == 'clarify' for frame in peer.frames))
        request_id = _request_id(peer)
        current = await _reconcile(owner, request_id)
        identity = {key: current['result'][key] for key in ('authority_epoch', 'execution_generation')}

        stale_epoch = await _lock(owner, request_id, 'q0', 'bad', **{
            **identity, 'authority_epoch': identity['authority_epoch'] + 1})
        assert 'error' in stale_epoch, stale_epoch
        assert stale_epoch['error']['data']['reason'] == 'stale_epoch'
        stale_generation = await _lock(owner, request_id, 'q0', 'bad', **{
            **identity, 'execution_generation': identity['execution_generation'] + 1})
        assert stale_generation['error']['data']['reason'] == 'stale_generation'
        stale_read = await _reconcile(owner, request_id, **{
            **identity, 'execution_generation': identity['execution_generation'] + 1})
        assert stale_read['error']['data']['reason'] == 'stale_generation'
        unchanged = await _reconcile(owner, request_id, **identity)
        assert unchanged['result']['answers'] == {}
        assert unchanged['result']['remaining'] == ['q0', 'q1']

        accepted = await _lock(owner, request_id, 'q0', 'blue', **identity)
        assert accepted['result'] == {'status': 'ok', 'remaining': ['q1']}
        replayed_read = await _reconcile(owner, request_id, **identity)
        assert replayed_read['result']['answers'] == {'q0': 'blue'}
    finally:
        if worker is not None:
            _stop_batch(authority, worker)
        await owner.close()


@pytest.mark.asyncio
async def test_reconcile_checks_identity_fences_on_terminal_receipts(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'

    authority = await _authority(tmp_path, monkeypatch, answer)
    owner = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    worker = None
    try:
        await _resume(owner)
        peer, worker, out = _start_batch(authority)
        await _wait_for(lambda: any(frame.get('method') == 'clarify' for frame in peer.frames))
        request_id = _request_id(peer)
        opened = await _reconcile(owner, request_id)
        identity = {key: opened['result'][key] for key in ('authority_epoch', 'execution_generation')}
        await _lock(owner, request_id, 'q0', 'kept', **identity)
        await owner.dispatch({'jsonrpc': '2.0', 'id': request_id, 'result': {}})
        await asyncio.to_thread(worker.join, 5)
        assert out['outcome'] == 'cancelled'

        assert (await _reconcile(owner, request_id, **identity))['result']['status'] == 'cancelled'
        for field, reason in (('authority_epoch', 'stale_epoch'), ('execution_generation', 'stale_generation')):
            stale = dict(identity)
            stale[field] += 1
            rejected = await _reconcile(owner, request_id, **stale)
            assert rejected['error']['data']['reason'] == reason
            # A rejected read neither evicts nor changes the authoritative receipt.
            receipt = await _reconcile(owner, request_id, **identity)
            assert receipt['result']['status'] == 'cancelled'
            assert receipt['result']['answers'] == {'q0': 'kept'}
    finally:
        if worker is not None:
            _stop_batch(authority, worker)
        await owner.close()
