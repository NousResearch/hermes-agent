"""Inventory safety regressions: no writes, no raw persistence fields, exact ownership."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / '.hermes'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    from hermes_cli.web_routers import profiles
    from tui_gateway import server
    monkeypatch.setattr(server, '_sessions', {})
    monkeypatch.setattr(server, '_hermes_home', tmp_path / '.hermes')
    app = FastAPI()
    app.include_router(profiles.sessions_router)
    return tmp_path, monkeypatch, profiles, server, TestClient(app)

def targets(env, homes):
    _, monkeypatch, profiles, _, client = env
    for _, home in homes:
        home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(profiles, '_profile_targets', lambda *a, **kw: homes)
    return client

@pytest.mark.parametrize('enumeration_fails', [False, True])
@pytest.mark.parametrize('live_only', [False, True])
def test_profile_enumeration_failure_is_not_successful_empty_coverage(env, enumeration_fails, live_only):
    tmp, monkeypatch, profiles_router, _, client = env
    home = tmp / '.hermes'
    profile_root = home / 'profiles'
    profile_root.mkdir(parents=True)
    original_iterdir = Path.iterdir
    before = sorted(path.name for path in original_iterdir(home))

    def iterdir(path):
        if enumeration_fails and path == profile_root:
            raise PermissionError('PRIVATE_PROFILE_PATH credentials=PRIVATE_TOKEN')
        return original_iterdir(path)

    monkeypatch.setattr(Path, 'iterdir', iterdir)
    response = client.get('/api/profiles/agent-overview', params={'live_only': live_only})
    assert response.status_code == 200
    body = response.json()
    expected_errors = [{'error': 'profile-inventory-unavailable'}] if enumeration_fails else []
    assert body['errors'] == expected_errors
    assert body['sessions'] == []
    assert body['total'] == 0
    assert 'PRIVATE_' not in json.dumps(body)
    # Existing sidebar callers keep their best-effort fallback contract.
    assert profiles_router._profile_targets('test legacy caller', lightweight=True) == [('default', home)]
    assert sorted(path.name for path in original_iterdir(home)) == before
    assert list(original_iterdir(profile_root)) == []


def test_empty_existing_db_is_not_bootstrapped_by_get(env):
    tmp, _, _, _, _ = env
    home = tmp / 'empty'
    client = targets(env, [('empty', home)])
    path = home / 'state.db'
    path.touch()
    before = path.read_bytes()
    response = client.get('/api/profiles/agent-overview')
    assert response.status_code == 200
    assert path.read_bytes() == before, f'GET grew empty db from {len(before)} to {path.stat().st_size} bytes'

def test_primary_profile_live_row_is_not_dropped(env):
    tmp, monkeypatch, _, server, _ = env
    home = tmp / '.hermes'
    client = targets(env, [('default', home)])
    monkeypatch.setattr(server, '_sessions', {'runtime': {'profile_home': None, 'session_key': 'durable', 'running': True, 'agent': SimpleNamespace(model='m', provider='p')}})
    body = client.get('/api/profiles/agent-overview').json()
    assert [(r['profile'], r['id'], r['status']) for r in body['live']] == [('default', 'durable', 'working')], body

def test_live_title_read_does_not_create_database(env):
    tmp, monkeypatch, _, server, _ = env
    home = tmp / 'worker'
    client = targets(env, [('worker', home)])
    monkeypatch.setattr(server, '_sessions', {'runtime': {'profile_home': str(home), 'session_key': 'durable', 'running': True, 'agent': SimpleNamespace(model='m', provider='p')}})
    assert not (home / 'state.db').exists()
    response = client.get('/api/profiles/agent-overview')
    assert response.status_code == 200
    assert not (home / 'state.db').exists(), 'live_snapshot -> _session_live_item -> _session_live_title acquired writable DB'

def test_private_persistence_fields_do_not_reach_inventory(env):
    from hermes_state import SessionDB
    tmp, _, _, _, _ = env
    home = tmp / 'worker'
    client = targets(env, [('worker', home)])
    with SessionDB(home / 'state.db') as db:
        db.create_session(session_id='private', source='desktop', model='m', system_prompt='PRIVATE_SYSTEM_PROMPT', model_config={'api_key': 'PRIVATE_MODEL_CONFIG'})
        db.set_session_title('private', 'Bot Chat')
        db._conn.execute('UPDATE sessions SET billing_base_url=?, compression_failure_error=?, origin_json=? WHERE id=?', ('https://user:PRIVATE_PASSWORD@example.invalid/v1?api_key=PRIVATE_QUERY', 'PRIVATE_ERROR_BODY', json.dumps({'user_id': 'PRIVATE_ORIGIN'}), 'private'))
        db._conn.commit()
    body = client.get('/api/profiles/agent-overview').json()
    assert 'PRIVATE_SYSTEM_PROMPT' not in json.dumps(body)
    assert 'PRIVATE_MODEL_CONFIG' not in json.dumps(body)
    leaked = {key for group in ('sessions', 'canonical') for row in body[group] for key in ('billing_base_url', 'compression_failure_error', 'origin_json') if row.get(key)}
    assert not leaked, f'Raw private persistence fields exposed: {sorted(leaked)}'

def test_canonical_activity_projection_does_not_write(env):
    """The canonical row now hydrates activity/preview/unread through the rich projection; that
    read must not flush, stamp, or repair anything in the store."""
    from hermes_state import SessionDB
    tmp, _, _, _, _ = env
    home = tmp / 'worker'
    client = targets(env, [('worker', home)])
    with SessionDB(home / 'state.db') as db:
        db.create_session(session_id='bot', source='desktop', model='m')
        db.set_session_title('bot', 'Bot Chat')
        db.set_session_hidden('bot', True)
        db.append_message(session_id='bot', role='user', content='hello')
        db.set_session_read('bot', read=False)
    path = home / 'state.db'
    before = path.read_bytes()
    body = client.get('/api/profiles/agent-overview').json()
    assert body['canonical'][0]['unread'] is True
    assert body['canonical'][0]['preview'] == 'hello'
    assert path.read_bytes() == before
    # A WAL-mode reader may create the -shm index; committed writes would land as WAL frames.
    wal = home / 'state.db-wal'
    assert not wal.exists() or wal.stat().st_size == 0, 'GET wrote WAL frames'

def test_multi_profile_pagination_child_filter_and_compression(env):
    from hermes_state import SessionDB
    tmp, _, _, _, _ = env
    homes = [('a', tmp / 'a'), ('b', tmp / 'b')]
    client = targets(env, homes)
    for name, home in homes:
        with SessionDB(home / 'state.db') as db:
            db.create_session(session_id='same', source='desktop', model='m')
            db.create_session(session_id='root', source='desktop', model='m')
            db.end_session('root', 'compression')
            db.create_session(session_id='tip', source='desktop', model='m', parent_session_id='root')
            db.create_session(session_id='delegate', source='tool', model='m', parent_session_id='same', model_config={'_delegate_from': 'same'})
            db.create_session(session_id='hidden', source='desktop', model='m')
            db.set_session_hidden('hidden', True)
            db.create_session(session_id='archived', source='desktop', model='m')
            db._conn.execute('UPDATE sessions SET archived=1 WHERE id=?', ('archived',))
            db._conn.commit()
    pages = [client.get(f'/api/profiles/agent-overview?limit=1&offset={i}').json() for i in range(5)]
    keys = [(row['profile'], row['id']) for page in pages for row in page['sessions']]
    assert len(keys) == len(set(keys)) == pages[0]['total'] == 4
    assert set(keys) == {('a', 'same'), ('a', 'tip'), ('b', 'same'), ('b', 'tip')}
