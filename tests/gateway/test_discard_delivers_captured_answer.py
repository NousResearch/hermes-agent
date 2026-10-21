"""Discard of a turn whose settlement failed commits its captured answer AND delivers it to the chat
the answer was owed to, exactly once; nothing is sent while the turn is unknown."""
import asyncio

import pytest

from gateway.session_contract import Submission
from hermes_state_runtime import RuntimeStoreError, get_session_admission


@pytest.mark.asyncio
@pytest.mark.parametrize('live_waiter', [True, False])
async def test_discard_of_a_captured_answer_delivers_it_once(tmp_path, monkeypatch, live_waiter):
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
        sent.append((response, event.source.chat_id,
                     get_session_admission(authority.db, admission_id=event.message_id)['status']))
    monkeypatch.setattr(session_ingress, 'deliver_response', deliver)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', (0.01,), raising=False)
    original = session_results.finish_result
    def unavailable(*args, **kwargs):
        raise OSError('database is locked')
    monkeypatch.setattr(session_results, 'finish_result', unavailable)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    actor = Principal('human', 'p', frozenset({'session:submit', 'session:control'}), 't')
    ref = SessionRef('p', 's')
    receipt = await authority.submit(actor, Submission('owed', ref, {'text': 'hi'}, 'queue'))
    waiter = None
    if live_waiter:  # the native ingress delivery waiter (admit_message) for this admission
        authority.native_waiters.add(receipt.admission_id)
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
    await asyncio.wait_for(authority._drain(ref), 10)
    monkeypatch.setattr(session_results, 'finish_result', original)
    row = get_session_admission(authority.db, admission_id=receipt.admission_id)
    assert row['status'] == 'unknown' and sent == [], 'an unsettled admission was answered externally'
    if waiter is not None:
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            await waiter  # the chat got the paused notice instead
    await authority.resolve_unknown(actor, ref, receipt.admission_id, row['generation'])
    resolved = get_session_admission(authority.db, admission_id=receipt.admission_id)
    assert (resolved['status'], resolved['outcome']) == ('terminal', 'completed')
    assert sent == [('model reply', 'c', 'terminal')], 'Discard committed the answer but never delivered it'
    await authority.resolve_unknown(actor, ref, receipt.admission_id, row['generation'])
    assert len(sent) == 1, 'a repeated resolution sent the answer twice'
    assert calls == ['hi'], 'inference must never re-run'
