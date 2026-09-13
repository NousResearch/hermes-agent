"""Inert F9 cleanup evidence; prepare case adapted from #103343 b565ecc."""
import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms import api_server_runs
from tests.gateway.test_api_server_runs import _claim_run, _create_runs_app, _make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['running', 'completed', 'interrupted'])
@pytest.mark.parametrize('other_reader', [False, True])
async def test_prepare_failure_cleans_only_its_reader(tmp_path, monkeypatch, status, other_reader):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    adapter = _make_adapter()
    run_id = 'prepare-fixture'
    stream = api_server_runs._RunStream()
    adapter._run_streams[run_id] = stream
    adapter._set_run_status(run_id, status)
    _claim_run(adapter, run_id)
    existing = stream.subscribe() if other_reader else None
    request = MagicMock()
    request.headers = {}
    request.match_info = {'run_id': run_id}
    request.path = f'/v1/runs/{run_id}/events'
    request.method = 'GET'
    response = MagicMock()
    response.prepare = AsyncMock(side_effect=ConnectionResetError('fixture disconnected'))
    try:
        with patch('gateway.platforms.api_server_runs.web.StreamResponse', return_value=response):
            with pytest.raises(ConnectionResetError):
                await adapter._handle_run_events(request)
        assert stream.subscribers == ({existing} if other_reader else set())
        if other_reader or status == 'running':
            assert adapter._run_streams[run_id] is stream
        else:
            assert run_id not in adapter._run_streams
    finally:
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_last_disconnect_keeps_active_replay_until_terminal_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    adapter = _make_adapter(api_key='fixture-sse-key')
    launches, handlers, requests = [], [], []

    async def no_execution(owner, run, **kwargs):
        launches.append(run)

    @web.middleware
    async def capture(request, handler):
        if request.path.endswith('/events'):
            requests.append(request)
            handlers.append(asyncio.current_task())
        return await handler(request)

    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    no_agent = Mock(side_effect=AssertionError('transport-only test'))
    monkeypatch.setattr(adapter, '_create_agent', no_agent)
    app = _create_runs_app(adapter)
    app.middlewares.append(capture)
    headers = {'Authorization': 'Bearer fixture-sse-key'}
    responses = []
    try:
        async with TestClient(TestServer(app)) as client:
            created = await client.post('/v1/runs', json={'input': 'fixture'}, headers=headers)
            assert created.status == 202, await created.text()
            run_id = (await created.json())['run_id']
            await asyncio.sleep(0)
            run = launches[0]
            adapter._set_run_status(run_id, 'running')
            first = await client.get(f'/v1/runs/{run_id}/events', headers=headers)
            responses.append(first)
            assert first.status == 200
            transport = requests[0].transport
            first.close()
            for _ in range(200):
                if transport.is_closing():
                    break
                await asyncio.sleep(0.01)
            assert transport.is_closing()
            run.put_event(api_server_runs._run_event(run_id, 'fixture.delta', text='retained fixture'))
            with suppress(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(handlers[0]), 2)
            assert adapter._run_streams.get(run_id) is run.queue
            assert not run.queue.subscribers
            # Reconnect to the same retained F9 stream, then finish without executing a Run.
            second = await client.get(f'/v1/runs/{run_id}/events', headers=headers)
            responses.append(second)
            assert second.status == 200
            adapter._set_run_status(run_id, 'completed', output='fixture result')
            run.put_event(api_server_runs._run_event(run_id, 'run.completed', output='fixture result'))
            run.put_event(None)
            body = await asyncio.wait_for(second.text(), 2)
            assert body.index('retained fixture') < body.index('run.completed') < body.index(': stream closed')
            assert not run.queue.subscribers
            assert run_id not in adapter._run_streams
            status = await client.get(f'/v1/runs/{run_id}', headers=headers)
            assert status.status == 200
            assert (await status.json())['output'] == 'fixture result'
            no_agent.assert_not_called()
    finally:
        for response in responses:
            response.close()
        adapter._run_idempotency_store.close()
