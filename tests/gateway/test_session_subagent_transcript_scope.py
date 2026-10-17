"""``subagent.transcript`` for a served secondary profile reads that profile's store, in its scope.

The read runs off the owner loop with no turn on the stack: it must bind the owning profile
explicitly and never fall back to the launch profile's home or ``state.db``.
"""
import os
from pathlib import Path

import pytest

from gateway.config import GatewayConfig
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


def _seed_child(db, parent, text):
    db.create_session('child', source='subagent', parent_session_id=parent,
                      model_config={'_delegate_from': parent, '_subagent_id': 'sa-0-scoped'})
    db.append_message('child', role='user', content=text)


@pytest.mark.asyncio
async def test_secondary_profile_child_transcript_reads_its_own_store_in_its_scope(tmp_path, monkeypatch):
    from gateway import run, session_local_recovery
    from gateway.session_ancillary import physical_target
    from gateway.session_authorities import owner_scope
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {
        'platform_toolsets': {'cli': [], 'tui': []},
        'model': {'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1', 'api_key': 'loopback-only'}})
    _root, homes = _reserve_homes(tmp_path, monkeypatch)
    config = GatewayConfig()
    config.multiplex_profiles = True
    config._runtime_profile_homes = tuple(homes)
    runner = GatewayRunner(config)
    process_ownership.reserve([home for _, home in homes])
    seen_homes = []
    stored_history = session_local_recovery.stored_history

    def observed(db, session_id):
        seen_homes.append(Path(get_hermes_home()).resolve())
        return stored_history(db, session_id)

    monkeypatch.setattr(session_local_recovery, 'stored_history', observed)
    connection = None
    try:
        await initialize_gateway_runtime(runner)
        registry = runner.session_authorities
        primary, secondary = registry.for_home(homes[0][1]), registry.for_home(homes[1][1])
        connection = AuthorityConnection(secondary, object(), {'user_id': 'human'})
        with owner_scope(secondary):
            ref = create_local_session(secondary, connection.actor, dict(
                request_id='desktop', source='tui', cwd=str(tmp_path), model='default-a', toolsets=[]))
        target = physical_target(secondary, ref)
        _seed_child(secondary.db, target, 'BETA_CHILD')
        # The launch profile's store holds a same-id decoy under the same parent id.
        primary.db.create_session(target, source='tui')
        _seed_child(primary.db, target, 'PRIMARY_DECOY')
        environ = dict(os.environ)
        response = await connection.dispatch({'id': 1, 'method': 'subagent.transcript', 'params': {
            'session_id': ref.session_id, 'subagent_id': 'sa-0-scoped'}})
        result = response.get('result')
        assert result and result['child_session_id'] == 'child', response
        assert [m['content'] for m in result['messages']] == ['BETA_CHILD']
        assert seen_homes == [homes[1][1].resolve()]
        assert dict(os.environ) == environ
    finally:
        if connection is not None:
            await connection.close()
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(home)
