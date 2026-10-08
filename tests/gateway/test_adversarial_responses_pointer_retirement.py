"""A lost response pointer must remain recoverable after the accepted transcript is retired."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_pointer_commit_loss_then_retirement_cannot_reexecute_moved_conversation(api, owner, monkeypatch):
    from gateway.session_contract import SessionRef
    from gateway.session_results import execution_result

    calls = []

    async def handle(event):
        calls.append(event.text)
        execution_result.get()['result'] = {'final_response': 'answer', 'messages': []}
        return 'answer'

    owner.runner._handle_message = handle
    store = api._current_response_store()
    bind = store.bind_request_admission
    def lose_pointer(*args):
        raise RuntimeError('simulated process loss after admission commit')
    monkeypatch.setattr(store, 'bind_request_admission', lose_pointer)
    app = web.Application()
    app.router.add_post('/v1/responses', api._handle_responses)
    body = {'input': 'original-once', 'conversation': 'moving'}
    headers = {'Idempotency-Key': 'lost-pointer'}
    async with TestClient(TestServer(app)) as client:
        failed = await client.post('/v1/responses', json=body, headers=headers)
        assert failed.status == 500
        monkeypatch.setattr(store, 'bind_request_admission', bind)
        admitted = owner.db._read_one('SELECT * FROM session_admissions')
        assert admitted['status'] == 'queued'
        # Owner recovery executes the committed request even though no HTTP response was saved.
        ref = SessionRef(owner.profile_id, admitted['target_session_id'])
        owner._schedule(ref)
        await asyncio.wait_for(owner.sessions[ref.session_id].task, 10)
        assert calls == ['original-once']
        assert owner.db.delete_session(ref.session_id)
        # A later request advances the name to an independent live target.
        advanced = await client.post('/v1/responses', json={'input': 'new-target', 'conversation': 'moving'},
                                     headers={'Idempotency-Key': 'successor'})
        assert advanced.status == 200
        retry = await client.post('/v1/responses', json=body, headers=headers)
        await retry.read()
    assert calls.count('original-once') == 1, 'retired tombstone was missed and the retry executed again'
