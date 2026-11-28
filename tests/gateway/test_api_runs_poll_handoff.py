"""Poll the final response after a turn exhausts its iteration budget."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_runs
from gateway.session_authority import admission_row
from gateway.session_persistence import authority_persistence
from gateway.session_results import finish_result


@pytest.mark.asyncio
async def test_polling_reports_the_reason_for_a_max_iteration_handoff(api, owner, monkeypatch):
    async def no_execution(*args, **kwargs):
        pass

    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', api._handle_get_run)
    async with TestClient(TestServer(app)) as client:
        accepted = await client.post('/v1/runs', json={'input': 'Summarize the finished work'})
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        url = f'/v1/runs/{run_id}'
        queued = await (await client.get(url)).json()
        row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id=queued['session_id']))
        handoff = {
            'final_response': 'Completed the first part of the task.',
            'completed': False,
            'partial': False,
            'interrupted': False,
            'turn_exit_reason': 'max_iterations_reached(8/8)',
        }
        finish_result(authority_persistence(owner), epoch=owner.epoch, row=row, outcome='completed',
                      response=handoff['final_response'], result={'result': handoff, 'usage': {}})
        response = await client.get(url)
        assert response.status == 200, await response.text()
        polled = await response.json()

    assert polled['status'] == 'failed'
    assert polled['output'] == handoff['final_response']
    assert polled['completed'] is False
    assert polled['partial'] is False
    assert polled['interrupted'] is False
    assert polled['turn_exit_reason'] == handoff['turn_exit_reason']
