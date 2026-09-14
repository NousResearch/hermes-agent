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
