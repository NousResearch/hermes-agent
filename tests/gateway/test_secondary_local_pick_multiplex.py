"""A secondary profile's local model pick reaches its next turn under multiplexing (#139060).

A LOCAL source has no receiving bot, so identity resolution infers the primary profile as its
transport. Once the primary profile has had a local session of its own, its LocalSessionAdapter
exists, and resolving the secondary chat's policy through the inferred transport found that
adapter instead of the secondary's: the pick committed but every turn ran the profile default.
"""
from pathlib import Path

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.run_runtime import initialize_gateway_runtime
from gateway.runtime_ownership import process_ownership


def _reserve_homes(tmp_path, monkeypatch, names=('beta',)):
    root = tmp_path / '.hermes'
    root.mkdir(mode=0o700)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(root))
    import hermes_state
    monkeypatch.setattr(hermes_state, 'DEFAULT_DB_PATH', hermes_state._IMPORT_DEFAULT_DB_PATH)
    homes = [('default', root)]
    for name in names:
        home = root / 'profiles' / name
        home.mkdir(parents=True, mode=0o700)
        (home / 'config.yaml').write_text('{}')
        homes.append((name, home))
    return root, homes


@pytest.mark.asyncio
async def test_secondary_pick_reaches_next_turn_after_primary_local_session(tmp_path, monkeypatch):
    from gateway import run
    from gateway.session_authorities import owner_scope
    from gateway.session_controls import AuthorityConnection
    from gateway.session_policy import policy_for_source
    from gateway.session_local import LocalSessionAdapter, create_local_session
    from hermes_cli import model_switch

    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {
        'platform_toolsets': {'cli': [], 'tui': []},
        'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'api_key': 'loopback-only'}})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    _root, homes = _reserve_homes(tmp_path, monkeypatch)
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    connections = []
    try:
        await initialize_gateway_runtime(runner)
        registry = runner.session_authorities
        primary, secondary = registry.for_home(homes[0][1]), registry.for_home(homes[1][1])

        def connect(authority):
            connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
            connections.append(connection)
            return connection

        # The primary profile runs a local session first: its LocalSessionAdapter now exists.
        with owner_scope(primary):
            create_local_session(primary, connect(primary).actor, dict(
                request_id='primary', source='cli', cwd=str(tmp_path), model='primary-model', toolsets=[]))
        assert isinstance(runner.adapters.get(Platform.LOCAL), LocalSessionAdapter)

        owner = connect(secondary)
        with owner_scope(secondary):
            ref = create_local_session(secondary, owner.actor, dict(
                request_id='desktop', source='tui', cwd=str(tmp_path), model='default-a', toolsets=[]))
        row = secondary.db.get_session(ref.session_id)
        response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'pick', 'operation': 'model',
            'payload': {'model': 'picked-b'}, 'expected_revision': row['runtime_revision'],
            'expected_generation': row['runtime_generation']}})
        assert response.get('result', {}).get('model') == 'picked-b', response

        # The next turn: inbound canonicalizes the source, then resolves the runtime.
        source = secondary.sessions[ref.session_id].source
        runner._canonicalize(source)
        assert policy_for_source(runner, source) is not None, 'the pick was read through the primary adapter'
        with owner_scope(secondary):
            model, _runtime = runner._resolve_session_agent_runtime(source=source)
        assert model == 'picked-b'
    finally:
        for connection in connections:
            await connection.close()
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
