"""Canonical ``session.list`` ``title_prefix`` constrains the stored page's query, not its result."""
import time
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_stored_title_prefix_reaches_matches_beyond_the_page(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'platform_toolsets': {'cli': []}})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    base = time.time() - 10_000
    titled = {'task-old': 'Task: oldest', 'task-new': 'Task: newer', 'lower': 'task: lower case'}
    for n, sid in enumerate(['task-old', 'task-new', 'lower', *(f'noise{i}' for i in range(4))]):
        db.create_session(sid, source='tui', model='m', cwd=str(tmp_path))
        db.append_message(sid, 'user', sid + ' question')
        db._conn.execute('UPDATE sessions SET started_at = ? WHERE id = ?', (base + n, sid))
        db._conn.execute('UPDATE messages SET timestamp = ? WHERE session_id = ?', (base + n, sid))
        db.end_session(sid, 'cli_exit')
        if sid in titled:
            db.set_session_title(sid, titled[sid])
    db._conn.commit()
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='owner')
    native = AuthorityConnection(authority, object(), {'provider': 'local', 'user_id': 'native',
        'profile_id': str(tmp_path), 'instance_id': 'owner'})

    async def stored_ids(**params):
        result = await native.dispatch({'id': 1, 'method': 'session.list',
                                        'params': {'scope': 'stored', 'limit': 2, **params}})
        assert 'result' in result, result
        return [row['id'] for row in result['result']['sessions']]

    assert not {'task-old', 'task-new'} & set(await stored_ids())
    assert await stored_ids(title_prefix='Task: ') == ['task-new', 'task-old']
