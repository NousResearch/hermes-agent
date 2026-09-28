"""Runs retain caller-selected history and exact terminal status."""
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_cutover_contract import api, owner


@pytest.mark.asyncio
async def test_runs_explicit_history_takes_precedence(api, owner, monkeypatch):
    from gateway.platforms import api_server_runs
    from hermes_state_runtime import list_session_admissions
    async def no_execution(*args, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    history = [{'role': 'user', 'content': 'CALLER'}, {'role': 'assistant', 'content': 'REPLY'}]
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/v1/runs', json={
            'session_id': 'history', 'input': 'FOLLOW', 'conversation_history': history})
        assert response.status == 202, await response.text()
    row, = list_session_admissions(owner.db, session_id='history')
    assert row['payload']['api_turn_v1']['history'] == history
    assert row['payload']['api_turn_v1']['settings']['session_history_delivery'] == ''


@pytest.mark.parametrize('outcome,result,expected', [
    ('failed', None, 'failed'),
    ('completed', {'final_response': 'diagnostic', 'interrupted': True}, 'cancelled'),
    ('completed', {'final_response': 'diagnostic', 'failed': True}, 'failed'),
])
def test_terminal_failure_and_cancel_match_result_and_polling(api, owner, outcome, result, expected):
    from gateway.session_api_turn import admit_api_turn
    from gateway.session_results import finish_result, admission_result
    from gateway.platforms.api_server_authority_runs import run_projection
    from hermes_state_runtime import claim_session_input
    _, ref, row = admit_api_turn(api, session_id='result', active_run_id='run',
                                user_message='x', conversation_history=[])
    row = claim_session_input(owner.db, epoch=owner.epoch, session_id=ref.session_id)
    settled, _ = finish_result(owner.db, epoch=owner.epoch, row=row, outcome=outcome,
        response='diagnostic', result={'result': result, 'usage': {}} if result else None)
    saved = admission_result(owner.db, row['admission_id'])['result']
    assert saved.get('failed' if expected == 'failed' else 'interrupted') is True
    assert settled['outcome'] == ('interrupted' if expected == 'cancelled' else expected)
    assert run_projection(api, 'run')['status'] == expected
