"""N46: a scalar ``model: <id>`` config.yaml (a supported form main's loaders accept) must not crash
``session.create``: Desktop's normal create (no explicit model) and a ``--provider`` launch."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_scalar_model_config_creates_sessions(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text('model: scalar-model\nplatform_toolsets:\n  cli: []\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def create(request_id, **params):
        reply = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': request_id, 'source': 'gui', 'cwd': str(tmp_path), **params}})
        assert 'result' in reply, reply
        return restore_policy(local_receipt(authority.db, reply['result']['session_id'])['policy'])

    desktop = await create('desktop')
    assert desktop.model == 'scalar-model' and desktop.config()['model'] == {'default': 'scalar-model'}
    pinned = await create('pinned', provider='custom', base_url='http://127.0.0.1:9/v1')
    assert (pinned.provider, pinned.base_url) == ('custom', 'http://127.0.0.1:9/v1')
    assert pinned.config()['model']['default'] == 'scalar-model'
