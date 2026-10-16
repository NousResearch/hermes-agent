"""A model switch's reply carries the new route's effort, so a client paints model and effort together."""
import pytest


def _owner(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_switch
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='switched', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    return store, owner, ref


async def _switch(owner, ref, db, payload, request_id='switch'):
    snap = db.get_session(ref.session_id)
    return await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
        'session_id': ref.session_id, 'request_id': request_id, 'expected_revision': snap['runtime_revision'],
        'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': payload}})


@pytest.mark.asyncio
async def test_model_switch_reply_carries_the_picked_effort(tmp_path, monkeypatch):
    store, owner, ref = _owner(tmp_path, monkeypatch)
    try:
        response = await _switch(owner, ref, store._db, {'model': 'switched', 'reasoning': 'high'})
        result = response['result']
        assert result['model'] == 'switched', response
        assert result['route_info']['reasoning_effort'] == 'high', result
        assert set(result['route_info']) <= {'reasoning_effort', 'reasoning_effort_wire', 'service_tier', 'fast'}
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_route_info_read_failure_never_fails_a_committed_switch(tmp_path, monkeypatch):
    from gateway import session_local
    store, owner, ref = _owner(tmp_path, monkeypatch)

    def broken(*a, **k):
        raise RuntimeError('info read failed')
    monkeypatch.setattr(session_local, 'local_session_info', broken)
    try:
        response = await _switch(owner, ref, store._db, {'model': 'switched'})
        assert response['result']['model'] == 'switched', response
        assert 'route_info' not in response['result']
    finally:
        await owner.close()
        store.close_all_db_handles()
