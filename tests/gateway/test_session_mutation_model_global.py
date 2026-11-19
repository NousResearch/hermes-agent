"""Global TUI selections persist only after the canonical receipt commits."""
import pytest
import hermes_yaml as yaml

from tests.gateway.test_session_mutation_model_route import (
    _bind_resolver, _configure, _pick,
)
from tests.gateway.test_session_follow_profile_config import _create, _owner, _route

pytest_plugins = ('tests.gateway.test_session_mutation_model_route',)


@pytest.mark.asyncio
@pytest.mark.parametrize('global_scope', [False, True])
async def test_model_scope_persists_profile_default(home, endpoints, global_scope, tmp_path, monkeypatch):
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        row = authority.db.get_session(sid)
        params = dict(session_id=sid, request_id='global-pick', operation='model',
                      expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'],
                      payload={'model': 'picked', **({'scope': 'global'} if global_scope else {})})
        ambient_home = tmp_path / 'launch-profile'
        ambient_home.mkdir()
        _configure(ambient_home, 'original', url, 'launch-key')
        untouched = (ambient_home / 'config.yaml').read_bytes()
        with monkeypatch.context() as ambient:
            ambient.setenv('HERMES_HOME', str(ambient_home))
            response = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': params})
        assert (ambient_home / 'config.yaml').read_bytes() == untouched
        assert 'result' in response, response
        saved = yaml.safe_load((home / 'config.yaml').read_text())
        assert saved['model']['default'] == ('picked' if global_scope else 'original')
        assert saved['model']['base_url'] == url
        assert saved['model']['api_key'] == 'key-a'
        assert _route(runner, live) == ('picked', url, 'key-a')
        # A new session reads the actual persisted default, not the first session's override.
        created = await _create(owner, authority, home, 'new-follow')
        assert _route(runner, created)[0] == ('picked' if global_scope else 'original')
        if global_scope:
            other_sid = next(key for key, value in authority.sessions.items() if value is created)
            other = await _pick(owner, authority, other_sid,
                                {'model': 'current', 'scope': 'global'}, request_id='other-global')
            assert 'result' in other, other
            await owner.dispatch({'id': 'old-retry', 'method': 'session.mutate', 'params': params})
            assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'current'
            saved = yaml.safe_load((home / 'config.yaml').read_text())
        newer = await _pick(owner, authority, sid, {'model': 'current'}, request_id='newer')
        assert 'result' in newer, newer
        replay = await owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': params})
        assert replay['result'] == response['result']
        assert yaml.safe_load((home / 'config.yaml').read_text()) == saved
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_global_pick_retry_repairs_failed_config_write(home, endpoints, monkeypatch):
    from hermes_cli import model_switch
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        row = authority.db.get_session(sid)
        params = dict(session_id=sid, request_id='repair', operation='model',
                      expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'],
                      payload={'model': 'picked', 'scope': 'global'})
        persist = model_switch.persist_model_selection
        def fail(*args, **kwargs):
            raise OSError('config write failed')
        monkeypatch.setattr(model_switch, 'persist_model_selection', fail)
        with pytest.raises(OSError, match='config write failed'):
            await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': params})
        committed_generation = authority.db.get_session(sid)['runtime_generation']
        assert committed_generation == row['runtime_generation'] + 1
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'original'
        monkeypatch.setattr(model_switch, 'persist_model_selection', persist)
        retry = await owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': params})
        assert 'result' in retry, retry
        assert authority.db.get_session(sid)['runtime_generation'] == committed_generation
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'picked'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_stale_global_pick_does_not_write_profile(home, endpoints):
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        row = authority.db.get_session(sid)
        before = (home / 'config.yaml').read_bytes()
        response = await owner.dispatch({'id': 'stale', 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': 'stale', 'operation': 'model',
            'payload': {'model': 'picked', 'scope': 'global'},
            'expected_revision': row['runtime_revision'] + 1,
            'expected_generation': row['runtime_generation']}})
        assert 'error' in response, response
        assert (home / 'config.yaml').read_bytes() == before
    finally:
        await owner.close()
        store.close_all_db_handles()

@pytest.mark.asyncio
@pytest.mark.parametrize('follow_profile', [False, True])
async def test_retry_repairs_global_write_after_newer_session_pick(home, endpoints, monkeypatch, follow_profile):
    """A local pick cannot prevent repair of the latest committed global default."""
    from hermes_cli import model_switch
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'follow', 'source': 'gui', 'cwd': str(home), 'toolsets': [],
            'follow_profile_config': follow_profile}})
        assert 'result' in created, created
        sid = created['result']['session_id']
        live = authority.sessions[sid]
        row = authority.db.get_session(sid)
        params = dict(session_id=sid, request_id='global-write-failed', operation='model',
                      expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'],
                      payload={'model': 'picked', 'scope': 'global'})
        persist = model_switch.persist_model_selection

        def fail(*args, **kwargs):
            raise OSError('config write failed')

        monkeypatch.setattr(model_switch, 'persist_model_selection', fail)
        with pytest.raises(OSError, match='config write failed'):
            await owner.dispatch({'id': 'first', 'method': 'session.mutate', 'params': params})
        monkeypatch.setattr(model_switch, 'persist_model_selection', persist)

        local = await _pick(owner, authority, sid, {'model': 'current'}, request_id='local-after-failure')
        assert 'result' in local, local
        assert _route(runner, live) == ('current', url, 'key-a')
        later = authority.db.get_session(sid)
        assert later['runtime_generation'] == row['runtime_generation'] + 2
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'original'
        watermark = live.event_stream.watermark()
        evictions = []
        runner._evict_cached_agent = evictions.append

        replay = await owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': params})
        assert 'result' in replay, replay
        assert replay['result']['execution_generation'] == row['runtime_generation'] + 1
        assert authority.db.get_session(sid) == later
        assert live.event_stream.since(*watermark)['events'] == []
        assert evictions == []
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'picked'
        # Following sessions observe profile edits; frozen sessions retain their later local pick.
        assert _route(runner, live) == ('picked' if follow_profile else 'current', url, 'key-a')
    finally:
        await owner.close()
        store.close_all_db_handles()
