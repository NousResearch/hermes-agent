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
    from gateway import session_settlement_recovery as recovery
    # Settlement retries a transient failure a bounded number of times; storage that stays down
    # across every attempt is what leaves the claim as recoverable uncertainty.
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01, 0.01), raising=False)
    def unavailable(*args, **kwargs):
        raise OSError('settlement storage unavailable')
    monkeypatch.setattr(session_results, 'finish_result', unavailable)
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
        monkeypatch.setattr(session_results, 'finish_result', original)
        row = get_session_admission(db, admission_id=head.admission_id)
        assert row['status'] == 'unknown' and row['owner_epoch'] == authority.epoch
        assert authority.pending_results[head.admission_id]['result']['final_response'] == 'head done'
        for waiter in waiting:
            with pytest.raises(RuntimeStoreError, match='unknown_execution'):
                await waiter
        assert not any(frame['params']['type'] == 'message.complete' for frame in frames)
        await authority.resolve_unknown(ACTOR, REF, head.admission_id, row['generation'])
        assert head.admission_id not in authority.pending_results
        # The answer was produced in this owner; only its receipt write failed. Resolution
        # commits that exact result instead of discarding it, and viewers get its completion.
        from gateway.session_results import admission_result
        resolved = get_session_admission(db, admission_id=head.admission_id)
        assert (resolved['status'], resolved['outcome']) == ('terminal', 'completed')
        assert admission_result(db, head.admission_id)['result']['final_response'] == 'head done'
        complete, = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
        assert (complete['admission_id'], complete['text']) == (head.admission_id, 'head done')
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


@pytest.mark.asyncio
async def test_transient_settlement_failure_commits_the_completed_result(tmp_path, monkeypatch):
    """One locked/full write must not turn a finished answer into an ``unknown`` turn whose result
    lives only in memory: settlement retries, commits the exact result once, and the FIFO moves on."""
    from gateway import session_results
    from gateway import session_settlement_recovery as recovery
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01, 0.01), raising=False)
    calls, frames = [], []
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
    with db:
        head = await authority.submit(ACTOR, Submission('head', REF, {'text': 'head'}, 'queue'))
        follower = await authority.submit(ACTOR, Submission('follower', REF, {'text': 'follower'}, 'queue'))
        waiter = authority.waiters.setdefault(head.admission_id, asyncio.get_running_loop().create_future())
        await asyncio.wait_for(authority._drain(REF), 5)
        assert await waiter == 'head done'
        row = get_session_admission(db, admission_id=head.admission_id)
        assert (row['status'], row['outcome']) == ('terminal', 'completed')
        assert head.admission_id not in authority.pending_results
        assert get_session_admission(db, admission_id=follower.admission_id)['status'] == 'terminal'
    assert calls == ['head', 'follower'], 'a settlement retry must never re-run inference'
    completions = [f['params']['payload'] for f in frames if f['params']['type'] == 'message.complete']
    assert [c['text'] for c in completions] == ['head done', 'follower done']


@pytest.mark.asyncio
@pytest.mark.parametrize('storage_down', [False, True])
async def test_recovered_native_reply_is_sent_only_after_its_outcome_commits(tmp_path, monkeypatch, storage_down):
    """A recovered turn with no live delivery waiter is answered by the drain itself. The platform
    send must follow the terminal commit: if settlement cannot commit, the admission is fenced
    unknown and the user is never told an answer the FIFO lost. Inference runs exactly once."""
    from gateway import session_ingress, session_results
    from gateway import session_settlement_recovery as recovery
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_authority import LiveSession
    from gateway.session_contract import Principal, SessionRef
    from tests.gateway.test_prompt_attachments import _authority as _store_authority

    calls, sent = [], []
    async def answer(event):
        calls.append(event.text)
        return 'model reply'
    authority = await _store_authority(tmp_path, monkeypatch, answer)
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    authority.runner._adapter_for_source = lambda source: object()
    async def deliver(adapter, event, session_key, response):
        sent.append((response, get_session_admission(authority.db, admission_id=event.message_id)['status']))
    monkeypatch.setattr(session_ingress, 'deliver_response', deliver)
    if storage_down:
        monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01,), raising=False)
        def unavailable(*args, **kwargs):
            raise OSError('settlement storage unavailable')
        monkeypatch.setattr(session_results, 'finish_result', unavailable)
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
    receipt = await authority.submit(actor, Submission('recovered', SessionRef('p', 's'), {'text': 'hi'}, 'queue'))
    await asyncio.wait_for(authority.sessions['s'].task, 10)
    status = get_session_admission(authority.db, admission_id=receipt.admission_id)['status']
    assert calls == ['hi']
    if storage_down:
        assert status == 'unknown' and sent == [], 'an unsettled admission was answered externally'
    else:
        assert status == 'terminal' and sent == [('model reply', 'terminal')]
    assert receipt.admission_id not in authority.pending_deliveries
