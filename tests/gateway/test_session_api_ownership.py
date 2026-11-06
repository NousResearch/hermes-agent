"""Real HTTP session controls keep cross-profile ownership and one PATCH atomic."""
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB
from tests.gateway.test_session_api import _create_session_app, _register_session_model_route


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    profile = tmp_path / 'profiles' / 'science'
    profile.mkdir(parents=True)
    with SessionDB(tmp_path / 'state.db') as local, SessionDB(profile / 'state.db') as sibling:
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._session_db = local
        yield local, sibling, _create_session_app(adapter)


def foreign(db, sid):
    db._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES (?,'cli',1,'science')", (sid,))
    db._conn.commit()


def snapshot(db):
    return [tuple(row) for row in db._conn.execute('SELECT * FROM sessions ORDER BY id')]


@pytest.mark.asyncio
async def test_patch_foreign_requested_id_returns_conflict_without_write(stores):
    local, sibling, app = stores
    local.create_session('contested', 'cli')
    foreign(sibling, 'contested')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.patch('/api/sessions/contested', json={'title': 'Must refuse'})
        assert response.status == 409, await response.text()
        assert (await response.json())['error']['code'] == 'session_lineage_ambiguous'
    assert (snapshot(local), snapshot(sibling)) == before


@pytest.mark.asyncio
async def test_patch_fields_commit_together_or_not_at_all(stores):
    local, sibling, app = stores
    local._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                        "VALUES ('root','cli',1,2,'compression')")
    local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id) "
                        "VALUES ('tip','cli',3,'root')")
    local._conn.commit()
    foreign(sibling, 'root')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.patch('/api/sessions/tip', json={'title': 'Must remain atomic', 'pinned': True})
        assert (snapshot(local), snapshot(sibling)) == before, {
            'status': response.status, 'tip_title': local.get_session('tip')['title']}
        assert response.status == 409, await response.text()
        assert (await response.json())['error']['code'] == 'session_id_owned_elsewhere'
    assert (snapshot(local), snapshot(sibling)) == before


@pytest.mark.asyncio
async def test_delete_foreign_requested_id_returns_conflict_without_write(stores):
    local, sibling, app = stores
    local.create_session('contested', 'cli')
    foreign(sibling, 'contested')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.delete('/api/sessions/contested')
        assert response.status == 409, await response.text()
        assert (await response.json())['error']['code'] == 'session_lineage_ambiguous'
    assert (snapshot(local), snapshot(sibling)) == before


@pytest.mark.asyncio
async def test_patch_uncontested_fields_commit_as_one_request(stores):
    local, sibling, app = stores
    local.create_session('owned', 'api_server')
    untouched_foreign = snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.patch('/api/sessions/owned', json={
            'title': 'Project', 'archived': True, 'hidden': True,
            'pinned': True, 'unread': True, 'end_reason': 'session_reset'})
        assert response.status == 200, await response.text()
    row = local.get_session('owned')
    assert (row['title'], row['archived'], row['hidden'], row['pinned'], row['last_read_at'],
            row['end_reason']) == ('Project', 1, 0, 1, 0.0, 'session_reset')
    assert snapshot(sibling) == untouched_foreign


@pytest.mark.asyncio
@pytest.mark.parametrize('method', ['patch', 'delete'])
async def test_http_control_refuses_unreadable_profile_inventory(stores, tmp_path, method):
    local, sibling, app = stores
    local.create_session('owned', 'cli')
    broken = tmp_path / 'profiles' / 'broken'
    broken.mkdir()
    database = broken / 'state.db'
    database.write_bytes(b'not a sqlite database')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = (await client.patch('/api/sessions/owned', json={'title': 'Must refuse'})
                    if method == 'patch' else await client.delete('/api/sessions/owned'))
        assert response.status == 503, await response.text()
        body = await response.json()
        assert body['error']['code'] == 'session_lineage_unverifiable'
        assert str(tmp_path) not in str(body)
    assert (snapshot(local), snapshot(sibling)) == before
    assert database.read_bytes() == b'not a sqlite database'


@pytest.mark.asyncio
async def test_fork_refuses_foreign_parent_without_creating_child(stores):
    local, sibling, app = stores
    local.create_session('parent', 'cli')
    local.append_message('parent', 'user', 'Preserved history')
    foreign(sibling, 'parent')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/api/sessions/parent/fork', json={'id': 'child'})
        assert (snapshot(local), snapshot(sibling)) == before, response.status
        assert response.status == 409, await response.text()
        assert (await response.json())['error']['code'] == 'session_lineage_ambiguous'


@pytest.mark.asyncio
async def test_fork_title_conflict_rolls_back_child_and_parent_end(stores):
    local, sibling, app = stores
    local.create_session('parent', 'cli')
    local.append_message('parent', 'user', 'Preserved history')
    local.create_session('named', 'cli')
    local.set_session_title('named', 'Already used')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/api/sessions/parent/fork',
                                     json={'id': 'child', 'title': 'Already used'})
        assert (snapshot(local), snapshot(sibling)) == before, response.status
        assert response.status == 400, await response.text()
        assert (await response.json())['error']['code'] == 'invalid_title'


@pytest.mark.asyncio
async def test_fork_uncontested_copies_history_and_ends_parent(stores):
    local, sibling, app = stores
    local.create_session('parent', 'cli')
    local.append_message('parent', 'user', 'Preserved history')
    before_foreign = snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/api/sessions/parent/fork', json={'id': 'child', 'title': 'Child title'})
        assert response.status == 201, await response.text()
    assert local.get_session('parent')['end_reason'] == 'branched'
    assert local.get_session('child')['parent_session_id'] == 'parent'
    assert local.get_session('child')['title'] == 'Child title'
    assert [row['content'] for row in local.get_messages('child')] == ['Preserved history']
    assert snapshot(sibling) == before_foreign


@pytest.mark.asyncio
async def test_create_title_transfer_fences_ghost_owner(stores):
    local, sibling, app = stores
    local.create_session('ghost', 'cli')
    local._conn.execute("UPDATE sessions SET title='Wanted',ended_at=2 WHERE id='ghost'")
    local._conn.commit()
    foreign(sibling, 'ghost')
    before = snapshot(local), snapshot(sibling)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/api/sessions', json={'id': 'new', 'title': 'Wanted'})
        assert (snapshot(local), snapshot(sibling)) == before, response.status
        assert response.status == 409, await response.text()
        assert (await response.json())['error']['code'] == 'session_id_owned_elsewhere'


@pytest.mark.asyncio
async def test_model_lock_reports_ownership_refusal(stores):
    local, sibling, _app = stores
    local.create_session('contested', 'api_server')
    foreign(sibling, 'contested')
    before = snapshot(local), snapshot(sibling)
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = local
    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, '_resolve_route', return_value=None):
        async with TestClient(TestServer(app)) as client:
            response = await client.post('/api/sessions/contested/model', json={
                'provider': 'nous', 'model': 'x-ai/grok-4.5', 'require_model_lock': True})
            assert response.status == 409, await response.text()
            assert (await response.json())['error']['code'] == 'session_lineage_ambiguous'
    assert (snapshot(local), snapshot(sibling)) == before
