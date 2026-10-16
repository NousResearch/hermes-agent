"""Config credentials borrow only their frozen profile source across restart."""
import json
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest


def test_config_credentials_recover_only_exact_source(tmp_path, monkeypatch):
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_runtime import RuntimeStoreError
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'model': {'provider': 'custom', 'api_key': 'owned-config-token',
                     'base_url': 'http://127.0.0.1:1234/v1'}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    def owner(path=home, profile='owned', instance='new'):
        return SimpleNamespace(db=SimpleNamespace(db_path=path / 'state.db'),
                               profile_id=profile, instance_id=instance, epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    policy = bind_launch_key(owner(instance='old'), 'session', policy, None, config_secrets=private)
    assert policy.config(owner())['model']['api_key'] == cfg['model']['api_key']
    assert cfg['model']['api_key'] not in json.dumps(asdict(policy))
    for wrong in (owner(profile='wrong'), owner(path=tmp_path)):
        with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
            policy.config(wrong)
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        replace(policy, model='other-policy').config(owner())
    cfg['model']['api_key'] = 'changed-unrelated-token'
    (home / 'config.yaml').write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        policy.config(owner())



@pytest.mark.asyncio
async def test_config_creation_retry_does_not_rebind_current_credentials(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Principal
    from gateway.session_local import create_local_session
    from gateway import run
    config = {'model': {'api_key': 'original-private'}, 'platform_toolsets': {'cli': []}}
    monkeypatch.setattr(run, '_load_gateway_config', lambda: config)
    def runner():
        store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        return SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False)
    first = await initialize_session_authority(runner(), profile_id='fixture', instance_id='first')
    actor = Principal('owner', 'fixture', frozenset({'session:create'}), 'socket')
    params = dict(request_id='retry', cwd=str(tmp_path), model='frozen')
    ref = create_local_session(first, actor, params)
    original = first.runner.adapters[Platform.LOCAL].policies[ref.session_id]
    cold = await initialize_session_authority(runner(), profile_id='fixture', instance_id='cold')
    config['model']['api_key'] = 'changed-private'
    assert create_local_session(cold, actor, params) == ref
    assert cold.runner.adapters[Platform.LOCAL].policies[ref.session_id] == original
    assert not getattr(cold, '_local_config_secrets', {})
    assert not getattr(cold, '_local_launch_keys', {})


def test_dropped_top_level_setting_does_not_wedge_the_session(tmp_path, monkeypatch):
    """A profile that stops defining a frozen top-level setting (an env-routed key the user moved to
    .env, where ``hermes config set`` writes it) leaves the session running on what its frozen tree
    holds instead of failing every read of the chat with ``launch_credentials_unavailable``. A
    ROUTE credential that disappeared still refuses, and a settings-only read never needs either."""
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_runtime import RuntimeStoreError
    home = tmp_path / 'profile'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'API_SERVER_KEY': 'owned-api-server-token',
           'model': {'provider': 'custom', 'api_key': 'route-token', 'base_url': 'http://127.0.0.1:1234/v1'}}
    config_path = home / 'config.yaml'
    config_path.write_text(json.dumps(cfg))

    def owner():
        # A fresh authority per read, like a restarted owner: no in-memory secret cache.
        return SimpleNamespace(db=SimpleNamespace(db_path=home / 'state.db'),
                               profile_id='owned', instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    assert set(private) == {('API_SERVER_KEY',), ('model', 'api_key')}
    policy = bind_launch_key(owner(), 'session', policy, None, config_secrets=private)
    assert cfg['API_SERVER_KEY'] not in json.dumps(asdict(policy))
    assert policy.config(owner())['API_SERVER_KEY'] == cfg['API_SERVER_KEY']

    del cfg['API_SERVER_KEY']  # the user moves the env-routed key to .env
    config_path.write_text(json.dumps(cfg))
    config = policy.config(owner())
    assert config['model']['api_key'] == 'route-token'
    assert config.get('API_SERVER_KEY') is None

    del cfg['model']['api_key']  # the session's own route credential is gone: still refused
    config_path.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable'):
        policy.config(owner())
    settings = policy.config(owner(), secrets=False)  # a display/model-option read needs no secret
    assert settings['model']['provider'] == 'custom' and settings['model']['api_key'] is None
