"""Desktop create parity on the shared owner: speed tier and cwd provenance (B09 / C-hx-1).

Main's session.create honoured an explicit ``service_tier`` (Fast pins priority, Normal pins normal over
a profile tier, an unknown word is refused) and ``cwd_explicit`` (an inherited client cwd yields to a
NAMED profile's own ``terminal.cwd``). The owner freezes both into the LocalSessionPolicy the turn runs.
"""
import json
from types import SimpleNamespace

import pytest


def _owner(home, monkeypatch, config):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps(config))
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(home / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _handle_message=None, _peek_session_state=lambda key: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    runner.session_authority = authority
    return store, runner, authority, AuthorityConnection(authority, object(), {'user_id': 'human'})


async def _create(owner, authority, request_id, cwd, **extra):
    from gateway.session_policy import policy_for_source
    response = await owner.dispatch({'id': request_id, 'method': 'session.create', 'params': dict(
        request_id=request_id, source='gui', cwd=str(cwd), toolsets=[], model='m', **extra)})
    if 'error' in response:
        return response['error']['data']['reason']
    live = authority.sessions[response['result']['session_id']]
    return live, policy_for_source(authority.runner, live.source)


@pytest.mark.asyncio
async def test_create_speed_tier_is_the_turns_tier_and_normal_pins_over_the_profile(tmp_path, monkeypatch):
    store, runner, authority, owner = _owner(tmp_path / 'home', monkeypatch, {
        'model': {'default': 'm'}, 'agent': {'service_tier': 'fast'}})
    try:
        _, fast = await _create(owner, authority, 'fast', tmp_path, service_tier='fast')
        _, ultra = await _create(owner, authority, 'ultra', tmp_path, service_tier='ultrafast')
        live, normal = await _create(owner, authority, 'normal', tmp_path, service_tier='normal')
        _, inherited = await _create(owner, authority, 'inherit', tmp_path)
        assert fast.current_service_tier(runner, None) == 'priority'
        assert ultra.current_service_tier(runner, None) == 'ultrafast'
        assert normal.current_service_tier(runner, None) is None
        assert inherited.current_service_tier(runner, None) == 'priority'
        # A typed session /fast on this chat still beats the frozen pick.
        runner._peek_session_state = lambda key: SimpleNamespace(conversation=SimpleNamespace(
            service_tier_override='priority')) if key == live.route else None
        assert normal.current_service_tier(runner, live.route) == 'priority'
        assert await _create(owner, authority, 'bad', tmp_path, service_tier='warp') == 'invalid_params'
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_inherited_cwd_yields_to_a_named_profiles_workspace_and_a_pick_does_not(tmp_path, monkeypatch):
    workspace, client = tmp_path / 'profile-workspace', tmp_path / 'client'
    workspace.mkdir()
    client.mkdir()
    store, _runner, authority, owner = _owner(tmp_path / 'root' / 'profiles' / 'work', monkeypatch, {
        'model': {'default': 'm'}, 'terminal': {'cwd': str(workspace)}})
    try:
        _, inherited = await _create(owner, authority, 'inherited', client, cwd_explicit=False)
        _, picked = await _create(owner, authority, 'picked', client, cwd_explicit=True)
        assert inherited.cwd == str(workspace)
        assert json.loads(inherited.terminal_json)['TERMINAL_CWD'] == str(workspace)
        assert picked.cwd == str(client)
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_desktop_project_create_uses_backend_root_and_rejects_conflicting_explicit_cwd(tmp_path, monkeypatch):
    from pathlib import Path
    from hermes_cli import projects_db

    project_root, client_cwd = tmp_path / 'host-project', tmp_path / 'client-cwd'
    project_root.mkdir()
    client_cwd.mkdir()
    store, _runner, authority, owner = _owner(tmp_path / 'root' / 'profiles' / 'work', monkeypatch, {
        'model': {'default': 'm'}})
    try:
        database = Path(authority.db.db_path).resolve().parent / 'projects.db'
        with projects_db.connect_closing(db_path=database) as connection:
            project_id = projects_db.create_project(
                connection, name='Host project', primary_path=str(project_root))

        _, policy = await _create(
            owner, authority, 'project-root', client_cwd,
            desktop_project_id=project_id, cwd_explicit=False)
        request = json.loads(policy.request_json)
        assert policy.cwd == str(project_root.resolve())
        assert request['desktop_project_id'] == project_id

        refused = await _create(
            owner, authority, 'project-conflict', client_cwd,
            desktop_project_id=project_id, cwd_explicit=True)
        assert refused == 'invalid_params'
    finally:
        await owner.close()
        store.close_all_db_handles()
