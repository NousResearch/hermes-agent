"""Global TUI selections persist only after the canonical receipt commits."""
import asyncio
import json
import threading

import pytest
import hermes_yaml as yaml

from tests.gateway.test_session_mutation_model_route import (
    _bind_resolver, _configure, _pick,
)
from tests.gateway.test_session_follow_profile_config import _create, _owner, _route
from tests.fixtures.storage_read import session_row

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')

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
        row = session_row(authority.persistence, sid)
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
        row = session_row(authority.persistence, sid)
        params = dict(session_id=sid, request_id='repair', operation='model',
                      expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'],
                      payload={'model': 'picked', 'scope': 'global'})
        persist = model_switch.persist_model_selection
        def fail(*args, **kwargs):
            raise OSError('config write failed')
        monkeypatch.setattr(model_switch, 'persist_model_selection', fail)
        with pytest.raises(OSError, match='config write failed'):
            await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': params})
        committed_generation = session_row(authority.persistence, sid)['runtime_generation']
        assert committed_generation == row['runtime_generation'] + 1
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'original'
        monkeypatch.setattr(model_switch, 'persist_model_selection', persist)
        retry = await owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': params})
        assert 'result' in retry, retry
        assert session_row(authority.persistence, sid)['runtime_generation'] == committed_generation
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'picked'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('fence,reason', [('revision', 'revision_conflict'), ('generation', 'stale_generation')])
async def test_stale_global_pick_does_not_write_profile(home, endpoints, fence, reason):
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'follow')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        row = session_row(authority.persistence, sid)
        before = (home / 'config.yaml').read_bytes()
        response = await owner.dispatch({'id': 'stale', 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': 'stale', 'operation': 'model',
            'payload': {'model': 'picked', 'scope': 'global'},
            'expected_revision': row['runtime_revision'] + (fence == 'revision'),
            'expected_generation': row['runtime_generation'] + (fence == 'generation')}})
        assert response.get('error', {}).get('message') == reason, response
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
        row = session_row(authority.persistence, sid)
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
        later = session_row(authority.persistence, sid)
        assert later['runtime_generation'] == row['runtime_generation'] + 2
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'original'
        watermark = live.event_stream.watermark()
        evictions = []
        runner._evict_cached_agent = evictions.append

        replay = await owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': params})
        assert 'result' in replay, replay
        assert replay['result']['execution_generation'] == row['runtime_generation'] + 1
        assert session_row(authority.persistence, sid) == later
        assert live.event_stream.since(*watermark)['events'] == []
        assert evictions == []
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'picked'
        # Following sessions observe profile edits; frozen sessions retain their later local pick.
        assert _route(runner, live) == ('picked' if follow_profile else 'current', url, 'key-a')
    finally:
        await owner.close()
        store.close_all_db_handles()

@pytest.mark.asyncio
async def test_global_pick_persists_the_selected_provider_endpoint_and_api_mode(home, endpoints):
    (original, url_a), (selected, url_b) = endpoints
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url_a, 'api_key': 'old-key',
                  'api_mode': 'responses'},
        'providers': {'loopback': {'name': 'Loopback', 'base_url': url_b, 'api_key': 'new-key',
                                   'api_mode': 'chat_completions', 'models': ['picked']}}}), encoding='utf-8')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'global-route')
        sid = next(key for key, value in authority.sessions.items() if value is live)
        original.received.clear()
        selected.received.clear()
        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'loopback', 'scope': 'global'})
        assert 'result' in response, response
        model = yaml.safe_load((home / 'config.yaml').read_text())['model']
        assert {key: model[key] for key in ('default', 'provider', 'base_url', 'api_mode')} == {
            'default': 'picked', 'provider': 'loopback', 'base_url': url_b, 'api_mode': 'chat_completions'}
        assert 'api_key' not in model  # the old endpoint's inline key is not carried to the new route
        assert _route(runner, live) == ('picked', url_b, 'new-key')
        new = await _create(owner, authority, home, 'after-global-route')
        assert _route(runner, new) == ('picked', url_b, 'new-key')
        assert original.received == []
        assert 'old-key' not in json.dumps(response)
        assert 'new-key' not in json.dumps(response)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_concurrent_global_picks_wait_for_the_previous_profiles_config_publication(home, endpoints, monkeypatch):
    from hermes_cli import model_switch
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    publishing = asyncio.Event()
    contended = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    persist = model_switch.persist_model_selection
    tasks = []

    class ObservedLock(asyncio.Lock):
        # A real lock with a contention observation point: no sleeps or scheduler-speed assumptions.
        async def acquire(self):
            if self.locked():
                contended.set()
            return await super().acquire()

    authority._global_model_selection_lock = ObservedLock()

    def publish(result, path):
        if result.new_model == 'picked':
            loop.call_soon_threadsafe(publishing.set)
            if not release.wait(10):
                raise AssertionError('first publication was never released')
        return persist(result, path)

    monkeypatch.setattr(model_switch, 'persist_model_selection', publish)
    try:
        first = await _create(owner, authority, home, 'first-global-session')
        second = await _create(owner, authority, home, 'second-global-session')
        first_sid, second_sid = (next(key for key, value in authority.sessions.items() if value is live)
                                 for live in (first, second))
        tasks.append(asyncio.create_task(_pick(owner, authority, first_sid,
                     {'model': 'picked', 'scope': 'global'}, request_id='first-global')))
        await asyncio.wait_for(publishing.wait(), 10)
        tasks.append(asyncio.create_task(_pick(owner, authority, second_sid,
                     {'model': 'current', 'scope': 'global'}, request_id='second-global')))
        await asyncio.wait_for(contended.wait(), 10)
        assert authority.persistence.sessions.get(second_sid).runtime_generation == 0
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'original'
        release.set()
        responses = await asyncio.gather(*tasks)
        assert all('result' in response for response in responses), responses
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == 'current'
        assert not authority.persistence.runtime.is_latest_global_model_selection(session_id=first_sid, generation=1)
        assert authority.persistence.runtime.is_latest_global_model_selection(session_id=second_sid, generation=1)
        assert _route(runner, first)[0] == 'current'
        assert _route(runner, second)[0] == 'current'
    finally:
        release.set()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await owner.close()
        store.close_all_db_handles()
