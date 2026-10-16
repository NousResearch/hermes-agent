"""Bot Chat's ``follow_profile_config`` on the shared owner (B23 / C-hx-4).

Main's follow row rebuilt its route from the profile's CURRENT config on every turn; an explicit pick
held only while the profile still targeted what it targeted at pick time. The owner freezes a policy,
so the frozen request says "follow" and each turn re-reads the profile's model sections (credentials
included). Real SessionStore/SessionAuthority and a real config.yaml; ``_load_gateway_config`` is not
patched.
"""
from types import SimpleNamespace

import pytest
import json


def _profile(home, model, port, key):
    (home / "config.yaml").write_text(json.dumps({'model': {
        'provider': 'custom', 'base_url': f'http://127.0.0.1:{port}/v1', 'default': model, 'api_key': key}}))


def _owner(home):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _handle_message=None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    runner.session_authority = authority
    return store, runner, authority, AuthorityConnection(authority, object(), {'user_id': 'human'})


async def _create(owner, authority, home, request_id, **extra):
    response = await owner.dispatch({'id': request_id, 'method': 'session.create', 'params': dict(
        request_id=request_id, source='gui', cwd=str(home), toolsets=[],
        follow_profile_config=True, **extra)})
    assert 'result' in response, response
    return authority.sessions[response['result']['session_id']]


def _route(runner, live):
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin
    model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=live.source)
    return model, runtime.get('base_url'), runtime.get('api_key')


@pytest.mark.asyncio
async def test_follow_profile_session_runs_the_profiles_current_route_and_a_pick_until_superseded(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    # gateway.run binds its home at first import; another test module may have imported it under
    # a different HERMES_HOME. The loader itself is real.
    monkeypatch.setattr(run, '_hermes_home', home)
    _profile(home, 'old', 1, 'k-old')
    store, runner, authority, owner = _owner(home)
    try:
        assert authority.db.db_path.parent.resolve() == home.resolve()
        follow = await _create(owner, authority, home, 'follow')
        pick = await _create(owner, authority, home, 'pick', model='pick')
        assert _route(runner, follow) == ('old', 'http://127.0.0.1:1/v1', 'k-old')
        assert _route(runner, pick)[0] == 'pick'
        # The profile moves to another endpoint with another key: the follow session's next turn runs
        # it (the old key never reaches the new host), and the pick is superseded, as on main.
        _profile(home, 'new', 2, 'k-new')
        assert _route(runner, follow) == ('new', 'http://127.0.0.1:2/v1', 'k-new')
        assert _route(runner, pick) == ('new', 'http://127.0.0.1:2/v1', 'k-new')
    finally:
        await owner.close()
        store.close_all_db_handles()
