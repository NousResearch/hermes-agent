"""API storage selection borrows the runtime's store, never another profile's."""
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session_authority import SessionAuthority
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.fixture
def owned_api(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    runner = SimpleNamespace(_draining=False)
    runner.session_authority = SessionAuthority(
        runner, profile_id='default', instance_id='api-test', db=db,
        epoch=begin_runtime_epoch(db, instance_id='api-test'))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'owned-api-test-secret'}))
    adapter.gateway_runner = runner
    yield adapter, db
    adapter._close_cached_session_dbs()
    adapter._response_store.close()
    adapter._run_idempotency_store.close()
    db.close()


@pytest.mark.asyncio
async def test_bound_api_borrows_selected_authority_store(owned_api):
    adapter, db = owned_api
    db.create_session('owned-session', source='api_server')
    assert adapter._ensure_session_db() is db
    assert await adapter._ensure_session_db_async() is db
    app = web.Application()
    app.router.add_get('/api/sessions/{session_id}', adapter._handle_get_session)
    async with TestClient(TestServer(app)) as client:
        denied = await client.get('/api/sessions/owned-session')
        assert denied.status == 401
        reply = await client.get('/api/sessions/owned-session',
                                 headers={'Authorization': 'Bearer owned-api-test-secret'})
        assert reply.status == 200, await reply.text()
        assert (await reply.json())['session']['id'] == 'owned-session'
    adapter._close_cached_session_dbs()
    assert db.get_session('owned-session')['id'] == 'owned-session'


@pytest.mark.asyncio
async def test_bound_api_never_opens_unserved_home(owned_api, tmp_path, monkeypatch):
    adapter, db = owned_api
    other = tmp_path / 'unserved'
    monkeypatch.setenv('HERMES_HOME', str(other))
    # Even an explicit legacy override cannot bypass runtime profile selection.
    adapter._session_db = db
    assert adapter._ensure_session_db() is None
    assert await adapter._ensure_session_db_async() is None
    assert not (other / 'state.db').exists()
    adapter.gateway_runner.session_authority = None
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    assert adapter._ensure_session_db() is None
    assert await adapter._ensure_session_db_async() is None
