"""Legacy bulk maintenance must exclude the real gateway owner reservation."""
import httpx
import pytest
from fastapi import FastAPI

from gateway.runtime_ownership import ProfileOwnership
from hermes_cli.web_routers.sessions import manage_router
from hermes_state import SessionDB


@pytest.mark.asyncio
async def test_maintenance_routes_refuse_live_owner_without_changing_rows(tmp_path, monkeypatch):
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('ended-empty', source='cli')
        db.end_session('ended-empty', 'complete')
        before = db.get_session('ended-empty')
        owner = ProfileOwnership()
        owner.reserve([tmp_path])
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
                for method, path, body in [('POST', 'bulk-delete', {'ids': ['ended-empty']}),
                        ('DELETE', 'empty', None), ('POST', 'owner-backfill', {}),
                        ('POST', 'prune', {'older_than_days': 1})]:
                    response = await client.request(method, '/api/sessions/' + path, json=body)
                    assert response.status_code == 409, (path, response.text)
                    assert 'maintenance' in response.json()['detail'].lower()
                    assert db.get_session('ended-empty') == before
        finally:
            owner.close()


@pytest.mark.asyncio
async def test_exclusive_offline_maintenance_preserves_bulk_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('parent', source='cli')
        db.create_session('child', source='cli', parent_session_id='parent')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            result = await client.post('/api/sessions/bulk-delete', json={'ids': ['parent', 'absent']})
            assert result.status_code == 200, result.text
            assert result.json() == {'ok': True, 'deleted': 1}
            assert db.get_session('parent') is None
            assert db.get_session('child')['parent_session_id'] is None
