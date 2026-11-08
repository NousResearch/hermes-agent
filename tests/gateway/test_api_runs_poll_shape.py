"""GET /v1/runs/{id} keeps the documented poll shape when the canonical admission answers."""
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer


@pytest.mark.asyncio
async def test_canonical_run_poll_keeps_object_model_and_created_at(api, owner, monkeypatch):
    from gateway.platforms import api_server_runs

    async def no_execution(*args, **kwargs):
        pass
    monkeypatch.setattr(api_server_runs, '_execute_run', no_execution)
    app = web.Application()
    app.router.add_post('/v1/runs', api._handle_runs)
    app.router.add_get('/v1/runs/{run_id}', api._handle_get_run)
    async with TestClient(TestServer(app)) as client:
        accepted = await client.post('/v1/runs', json={'input': 'hi', 'model': 'hermes-agent'})
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        polled = await (await client.get(f'/v1/runs/{run_id}')).json()
    assert polled['admission_id'] and polled['status'] == 'queued'
    assert polled['object'] == 'hermes.run' and polled['model'] == 'hermes-agent', polled
    assert isinstance(polled['created_at'], float)
