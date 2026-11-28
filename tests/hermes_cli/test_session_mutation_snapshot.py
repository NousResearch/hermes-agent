"""HTTP mutation preparation reads the owner's exact persisted counters."""
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from gateway.session_authority import SessionAuthority
from hermes_cli.web_routers.sessions import manage_router
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_snapshot_reports_real_counters_and_absent_import_anchor(tmp_path, monkeypatch):
    from hermes_cli import web_server
    monkeypatch.setattr(web_server, '_SESSION_TOKEN', 'snapshot-token')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    store = storage_handle(tmp_path)
    seed_session(store, 'snapshot-session', 'cli')
    raw_holder(store)._execute_write(lambda c: c.execute('UPDATE sessions SET runtime_revision=7, runtime_generation=3'))
    app.state.session_authority = SessionAuthority(SimpleNamespace(_draining=False),
        profile_id=str(tmp_path), instance_id='snapshot-owner',
        epoch=store.runtime.begin_epoch(instance_id='snapshot-owner'), persistence=store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
        path = '/api/sessions/snapshot-session/mutation-snapshot'
        assert (await client.get(path)).status_code == 401
        client.headers['Authorization'] = 'Bearer snapshot-token'
        result = await client.get(path)
        assert result.status_code == 200, result.text
        assert result.json() == {'session_id': 'snapshot-session', 'exists': True,
            'runtime_revision': 7, 'runtime_generation': 3}
        absent = await client.get('/api/sessions/new-import/mutation-snapshot')
        assert absent.json() == {'session_id': 'new-import', 'exists': False,
            'runtime_revision': 0, 'runtime_generation': None}
