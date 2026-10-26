"""HTTP mutation preparation reads the owner's exact persisted counters."""
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from gateway.session_authority import SessionAuthority
from hermes_cli.web_routers.sessions import manage_router
from hermes_state import SessionDB
from hermes_state_runtime import begin_runtime_epoch


@pytest.mark.asyncio
async def test_snapshot_reports_real_counters_and_absent_import_anchor(tmp_path, monkeypatch):
    from hermes_cli import web_server
    monkeypatch.setattr(web_server, '_SESSION_TOKEN', 'snapshot-token')
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    app = FastAPI()
    app.include_router(manage_router)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('snapshot-session', source='cli')
        db._execute_write(lambda c: c.execute('UPDATE sessions SET runtime_revision=7, runtime_generation=3'))
        app.state.session_authority = SessionAuthority(SimpleNamespace(_draining=False),
            profile_id=str(tmp_path), instance_id='snapshot-owner', db=db,
            epoch=begin_runtime_epoch(db, instance_id='snapshot-owner'))
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
