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
            assert result.json() == {'ok': True, 'deleted': 1, 'skipped_active': []}
            assert db.get_session('parent') is None
            assert db.get_session('child')['parent_session_id'] is None


@pytest.mark.asyncio
async def test_owner_backfill_skips_exclusive_maintenance_when_no_unowned_rows_exist(tmp_path, monkeypatch):
    """When a live gateway control socket owns the home and all sessions already have
    profile_name stamped, standalone serve must return 200 via read_only precheck instead
    of failing 409 on every Desktop boot; if an unowned NULL row exists, it must still refuse 409."""
    monkeypatch.setattr('hermes_state._default_db_path', lambda: tmp_path / 'state.db')
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(
        'hermes_cli.gateway_runtime_discovery.query_identify',
        lambda home, timeout=1.5: {'state': 'starting', 'runtime_protocol': 1},
    )
    app = FastAPI()
    app.include_router(manage_router)
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('already-owned', source='cli')
        owner = ProfileOwnership()
        owner.reserve([tmp_path])
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://localhost') as client:
                # Polarity 1 (0 NULL rows): read_only precheck short-circuits with 200
                response = await client.post('/api/sessions/owner-backfill', json={})
                assert response.status_code == 200, response.text
                assert response.json() == {'ok': True, 'stamped': 0, 'profile': 'default'}

                # Polarity 2 (legacy NULL row present): must not falsely report stamped=0; hits write lock -> 409
                db._execute_write(
                    lambda conn: conn.execute("UPDATE sessions SET profile_name = NULL WHERE id = 'already-owned'")
                )
                refused = await client.post('/api/sessions/owner-backfill', json={})
                assert refused.status_code == 409, refused.text
        finally:
            owner.close()
