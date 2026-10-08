"""An owner that is still alive must expose a failed settlement as recoverable uncertainty."""
import asyncio

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import _authority, ACTOR, REF


@pytest.mark.asyncio
@pytest.mark.parametrize('retry_recovery_write', [False, True])
async def test_failed_commit_retains_result_and_releases_fifo_for_explicit_discard(tmp_path, monkeypatch, retry_recovery_write):
    from gateway import session_results
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    calls = []
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    async def execute(owner, ref, row):
        calls.append(row['request_id'])
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': row['request_id'] + ' done', 'completed': True}, 'usage': {}}
        return row['request_id'] + ' done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    original = session_results.finish_result
    def fail_once(*args, **kwargs):
        monkeypatch.setattr(session_results, 'finish_result', original)
        raise OSError('temporary settlement write failure')
    monkeypatch.setattr(session_results, 'finish_result', fail_once)
    if retry_recovery_write:
        from gateway import session_settlement_recovery
        mark = session_settlement_recovery._mark_unknown
        def unavailable_once(*args):
            monkeypatch.setattr(session_settlement_recovery, '_mark_unknown', mark)
            raise OSError('recovery write temporarily unavailable')
        monkeypatch.setattr(session_settlement_recovery, '_mark_unknown', unavailable_once)
    with db:
        head = await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        follower = await authority.submit(ACTOR, Submission('follower', REF, {'text': 'follower'}, 'queue'))
        waiting = [authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
                   for receipt in (head, follower)]
        await asyncio.wait_for(authority._drain(REF), 5)
        row = get_session_admission(db, admission_id=head.admission_id)
        assert row['status'] == 'unknown' and row['owner_epoch'] == authority.epoch
        assert authority.pending_results[head.admission_id]['result']['final_response'] == 'head done'
        for waiter in waiting:
            with pytest.raises(RuntimeStoreError, match='unknown_execution'):
                await waiter
        assert not any(frame['params']['type'] == 'message.complete' for frame in frames)
        await authority.resolve_unknown(ACTOR, REF, head.admission_id, row['generation'])
        assert head.admission_id not in authority.pending_results
        await asyncio.wait_for(authority._drain(REF), 5)
        assert calls == ['head', 'follower']
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'terminal'


@pytest.mark.asyncio
async def test_commit_then_error_publishes_the_exact_committed_result(tmp_path, monkeypatch):
    from gateway import session_results
    db, authority = _authority(tmp_path, monkeypatch)
    frames = []
    authority.sessions['s'].event_stream.observers.add(frames.append)
    async def execute(owner, ref, row):
        owner.pending_results[row['admission_id']] = {
            'result': {'final_response': 'done', 'completed': True, 'response_reused': True}, 'usage': {}}
        return 'done'
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    original = session_results.finish_result
    def fail_after_commit(*args, **kwargs):
        monkeypatch.setattr(session_results, 'finish_result', original)
        original(*args, **kwargs)
        raise OSError('commit response lost')
    monkeypatch.setattr(session_results, 'finish_result', fail_after_commit)
    with db:
        receipt = await authority.submit(ACTOR, Submission('committed', REF, {'text': 'work'}, 'queue'))
        await asyncio.wait_for(authority.sessions['s'].task, 5)
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert row['status'] == 'terminal' and row['outcome'] == 'completed'
        complete, = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert complete['text'] == 'done' and complete['response_reused'] is True
        assert receipt.admission_id not in authority.pending_results
