"""Commit-then-error must not turn redacted result equality into voice authority."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import session_results, session_settlement_recovery as recovery
from gateway.session_runtime_workers import mutation_tasks
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import REF, _authority, _submit
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.parametrize('different_execution', [False, True])
@pytest.mark.parametrize('retry', [False, True])
async def test_commit_then_error_never_speaks_unproven_streamed_candidate(
        tmp_path, monkeypatch, different_execution, retry):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0,) if retry else ())
    candidate = 'answer with private token'
    execution = {'final_response': candidate}
    event = SimpleNamespace(_deferred_voice_response=candidate,
                            _deferred_voice_result=execution,
                            _deferred_voice_messages=[])
    route = 'original-route'
    authority.runner._should_send_voice_reply = lambda *args, **kwargs: True
    authority.runner._send_voice_reply = AsyncMock()
    seen_routes = []

    async def execute(owner, ref, row):
        event.message_id = row['admission_id']
        owner.pending_results[row['admission_id']] = {'result': execution, 'usage': {}}
        owner.pending_deliveries[row['admission_id']] = (object(), event, route, None, None)

    finish = session_results.finish_result

    def commit_then_error(*args, **kwargs):
        if different_execution:
            # An independently committed result can have the same persisted text. Such
            # equality is not evidence that the voice candidate belongs to that result.
            outcome, stored = kwargs['prepared']
            stored = {**stored, 'result': {**stored['result'], 'other_execution': True}}
            kwargs['prepared'] = (outcome, stored)
        finish(*args, **kwargs)
        raise OSError('commit succeeded but acknowledgement was lost')

    from gateway import session_ingress
    original_deliver = session_ingress.deliver_settled

    async def observe_delivery(owner, admission_id):
        seen_routes.append(owner.pending_deliveries[admission_id][2])
        await original_deliver(owner, admission_id)

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(session_results, 'finish_result', commit_then_error)
    monkeypatch.setattr(session_ingress, 'deliver_settled', observe_delivery)
    monkeypatch.setattr(recovery, '_release_media', lambda *args: None)
    with db:
        receipt = await _submit(authority, 'voice')
        await authority._drain(REF)
        await async_wait_until(lambda: not mutation_tasks(authority), timeout=5, what='media cleanup')
        assert get_session_admission(db, admission_id=receipt.admission_id)['status'] == 'terminal'
        assert seen_routes == [route]
        authority.runner._send_voice_reply.assert_not_awaited()
        assert not getattr(event, '_voice_result_committed', False)
