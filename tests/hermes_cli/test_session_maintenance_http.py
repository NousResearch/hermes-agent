"""Legacy bulk maintenance must exclude the real gateway owner reservation."""
import httpx
import pytest
from fastapi import FastAPI

from gateway.runtime_ownership import ProfileOwnership
from hermes_cli.web_routers.sessions import manage_router
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_session


@pytest.mark.asyncio
async def test_maintenance_routes_refuse_live_owner_without_changing_rows(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, 'ended-empty', 'cli', end_reason='complete')
        before = session_row(store, 'ended-empty')
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
                    assert session_row(store, 'ended-empty') == before
        finally:
            owner.close()


@pytest.mark.asyncio
async def test_exclusive_offline_maintenance_preserves_bulk_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    app = FastAPI()
    app.include_router(manage_router)
    with open_persistence(profile_scope(home=tmp_path)) as store:
        seed_session(store, 'parent', 'cli')
        seed_session(store, 'child', 'cli', parent_session_id='parent')
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
            result = await client.post('/api/sessions/bulk-delete', json={'ids': ['parent', 'absent']})
            assert result.status_code == 200, result.text
            assert result.json() == {'ok': True, 'deleted': 1, 'skipped_active': []}
            assert session_row(store, 'parent') is None
            assert session_row(store, 'child')['parent_session_id'] is None
