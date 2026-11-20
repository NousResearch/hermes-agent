"""A deferred global pick's original receipt repairs publication without applying the pick twice."""
import asyncio
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest
import pytest_asyncio

from gateway.session_contract import SessionRef, Submission
from tests.fixtures.storage_read import session_row
from tests.gateway.test_session_follow_profile_config import _owner, _route
from tests.gateway.test_session_mutation_model_route import _bind_resolver, _configure, _pick

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')
pytest_plugins = ('tests.gateway.test_session_mutation_model_route',)


@pytest_asyncio.fixture
async def deferred_global(home, endpoints, monkeypatch, request):
    from gateway import session_mutations
    from hermes_cli import model_switch
    failure = getattr(request, 'param', 'before-write')
    (_, url), _ = endpoints
    _configure(home, 'original', url, 'key-a')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    started, release = asyncio.Event(), asyncio.Event()
    events, evictions, attempts = [], [], []
    failing = {'enabled': True}
    persist = model_switch.persist_model_selection

    async def handle(event):
        started.set()
        await release.wait()
        return 'ok'

    def publish(result, path):
        attempts.append(result.new_model)
        if failure == 'after-write':
            persist(result, path)
        if failing['enabled']:
            raise OSError('deferred config write failed')
        if failure != 'after-write':
            return persist(result, path)

    project = session_mutations._project_result

    async def interrupt_projection(authority_, ref_, live_, params_, *args, **kwargs):
        if failing['enabled'] and params_['request_id'] == 'deferred-pick:deferred-global':
            raise RuntimeError('owner stopped after commit, before projection')
        return await project(authority_, ref_, live_, params_, *args, **kwargs)

    if failure == 'before-projection':
        monkeypatch.setattr(session_mutations, '_project_result', interrupt_projection)
    runner._handle_message = handle
    runner._evict_cached_agent = evictions.append
    monkeypatch.setattr(model_switch, 'persist_model_selection', publish)
    try:
        response = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'frozen', 'source': 'gui', 'cwd': str(home), 'toolsets': [],
            'follow_profile_config': False}})
        assert 'result' in response, response
        sid = response['result']['session_id']
        live = authority.sessions[sid]
        ref = SessionRef(authority.profile_id, sid)
        stream_publish = live.event_stream.publish
        monkeypatch.setattr(live.event_stream, 'publish', lambda session_id, payload, **kwargs: (
            events.append((kwargs.get('event_type'), payload)), stream_publish(session_id, payload, **kwargs))[1])
        await authority.submit(owner.actor, Submission('running', ref, {'text': 'RUNNING'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        row = session_row(authority.persistence, sid)
        assert row is not None
        params = dict(session_id=sid, request_id='deferred-global', operation='model',
                      expected_revision=row['runtime_revision'], expected_generation=row['runtime_generation'],
                      payload={'model': 'picked', 'scope': 'global'})
        accepted = await owner.dispatch({'id': 'pick', 'method': 'session.mutate', 'params': params})
        assert accepted.get('result', {}).get('status') == 'deferred', accepted
        release.set()
        await asyncio.wait_for(asyncio.shield(live.task), 10)
        assert attempts == ([] if failure == 'before-projection' else ['picked'])
        committed = session_row(authority.persistence, sid)
        assert committed is not None and committed['runtime_generation'] == row['runtime_generation'] + 1
        assert yaml.safe_load((home / 'config.yaml').read_text())['model']['default'] == (
            'picked' if failure == 'after-write' else 'original')
        # Isolate user-triggered repair from the next automatic drain; restart uses a
        # fresh authority with its real scheduler, so attach recovery remains exercised.
        monkeypatch.setattr(authority, '_schedule', lambda ref_: None)
        yield SimpleNamespace(home=home, url=url, store=store, runner=runner, authority=authority, owner=owner,
                              sid=sid, ref=ref, live=live, params=params, accepted=accepted['result'],
                              attempts=attempts, failing=failing, events=events, evictions=evictions)
    finally:
        release.set()
        await owner.close()
        authority.persistence.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
@pytest.mark.parametrize('later', [None, 'local', 'global', 'other-global'])
async def test_original_deferred_global_retry_repairs_only_the_unsuperseded_publication(deferred_global, later):
    state = deferred_global
    if later is None:
        before = session_row(state.authority.persistence, state.sid)
        watermark = state.live.event_stream.watermark()
        evictions = len(state.evictions)
        # An unsuccessful user retry must leave the repair pending without replaying
        # any session projection, just like the eventual successful retry below.
        with pytest.raises(OSError, match='deferred config write failed'):
            await state.owner.dispatch({'id': 'still-failing', 'method': 'session.mutate', 'params': state.params})
        assert session_row(state.authority.persistence, state.sid) == before
        assert state.live.event_stream.since(*watermark)['events'] == []
        assert len(state.evictions) == evictions
    state.failing['enabled'] = False
    if later:
        # A later local pick cannot revoke an accepted global default; another global pick can.
        sid = state.sid
        if later == 'other-global':
            created = await state.owner.dispatch({'id': 'other', 'method': 'session.create', 'params': {
                'request_id': 'other', 'source': 'gui', 'cwd': str(state.home), 'toolsets': []}})
            assert 'result' in created, created
            sid = created['result']['session_id']
        response = await _pick(state.owner, state.authority, sid,
                               {'model': 'current', **({'scope': 'global'} if 'global' in later else {})},
                               request_id='later-' + later)
        assert 'result' in response, response
    committed = session_row(state.authority.persistence, state.sid)
    watermark = state.live.event_stream.watermark()
    evictions = len(state.evictions)
    retry = await state.owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': state.params})
    assert retry.get('result') == state.accepted, retry
    saved = yaml.safe_load((state.home / 'config.yaml').read_text())['model']
    assert saved['default'] == ('current' if later in {'global', 'other-global'} else 'picked')
    assert saved['provider'] == 'custom' and saved['base_url'] == state.url
    assert session_row(state.authority.persistence, state.sid) == committed
    assert _route(state.runner, state.live) == (
        'current' if later in {'local', 'global'} else 'picked', state.url, 'key-a')
    assert state.live.event_stream.since(*watermark)['events'] == []
    assert len(state.evictions) == evictions
    assert not [payload for kind, payload in state.events if payload.get('reason') == 'model_resolution_failed']
    updated = [payload for kind, payload in state.events if kind == 'session.updated' and payload.get('model') == 'picked']
    assert len(updated) == 1
    assert state.authority.persistence.runtime.deferred_pick(state.sid) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('deferred_global', ['before-write', 'after-write', 'before-projection'], indirect=True)
async def test_deferred_global_publication_survives_an_owner_restart_without_a_pending_pick(deferred_global):
    state = deferred_global
    committed = session_row(state.authority.persistence, state.sid)
    # Close every owner/connection handle, then mount the durable policy with a fresh authority epoch.
    await state.owner.close()
    state.authority.persistence.close()
    state.store.close_all_db_handles()
    state.failing['enabled'] = False
    store, runner, authority, owner = _owner(state.home)
    _bind_resolver(runner)
    assert authority.epoch > state.authority.epoch
    try:
        resumed = await owner.dispatch({'id': 'resume', 'method': 'session.resume', 'params': {'session_id': state.sid}})
        assert 'result' in resumed, resumed
        live = authority.sessions[state.sid]
        # Attaching to the restarted owner schedules recovery even though applying the
        # selection already consumed its pending slot in the receipt transaction.
        assert live.task is not None
        await asyncio.wait_for(asyncio.shield(live.task), 10)
        saved = yaml.safe_load((state.home / 'config.yaml').read_text())['model']
        assert saved['default'] == 'picked'
        assert _route(runner, live) == ('picked', state.url, 'key-a')
        assert session_row(authority.persistence, state.sid) == committed
        watermark = live.event_stream.watermark()
        retry = await owner.dispatch({'id': 'retry', 'method': 'session.mutate', 'params': state.params})
        assert retry.get('result') == state.accepted, retry
        assert live.event_stream.since(*watermark)['events'] == []
        assert session_row(authority.persistence, state.sid) == committed
        assert authority.persistence.runtime.deferred_pick(state.sid) is None
    finally:
        await owner.close()
        authority.persistence.close()
        store.close_all_db_handles()
