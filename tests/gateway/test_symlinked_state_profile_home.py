"""A relocated database file does not relocate its owning profile or credentials."""
import json
from types import SimpleNamespace

import pytest


def _owner(home, storage):
    home.mkdir()
    path = home / 'state.db'
    path.symlink_to(storage)
    return SimpleNamespace(db=SimpleNamespace(db_path=path), profile_id='owned', instance_id='test', epoch=1)


@pytest.mark.asyncio
async def test_cron_and_authority_lookup_keep_database_link_home(tmp_path):
    from gateway import session_cron
    from gateway.session_authorities import authority_for_home
    storage = tmp_path / 'storage.db'
    storage.touch()
    home = tmp_path / 'profile'
    authority = _owner(home, storage)
    session_cron.bind_owner(authority)
    try:
        assert session_cron.owner_for_home(home)[0] is authority
        assert session_cron.owner_for_home(storage.parent) is None
        assert authority_for_home(SimpleNamespace(session_authority=authority), home) is authority
    finally:
        session_cron.unbind_owner(authority)
    assert session_cron.owner_for_home(home) is None


def test_profile_configuration_is_not_read_beside_relocated_database(tmp_path, monkeypatch):
    from gateway.session_policy import profile_config
    storage = tmp_path / 'storage.db'
    storage.touch()
    home = tmp_path / 'profile'
    authority = _owner(home, storage)
    monkeypatch.setenv('HERMES_HOME', str(home))
    (home / 'config.yaml').write_text('model:\n  default: profile-model\n')
    (tmp_path / 'config.yaml').write_text('model:\n  default: unrelated-model\n')
    assert profile_config(authority)['model']['default'] == 'profile-model'


def test_config_credentials_do_not_cross_profiles_sharing_database_target(tmp_path, monkeypatch):
    from gateway.session_policy import build_policy, bind_launch_key
    from hermes_state_runtime import RuntimeStoreError
    storage = tmp_path / 'storage.db'
    storage.touch()
    home = tmp_path / 'profile'
    authority = _owner(home, storage)
    other = _owner(tmp_path / 'other-profile', storage)
    monkeypatch.setenv('HERMES_HOME', str(home))
    cfg = {'model': {'provider': 'custom', 'api_key': 'owned-test-token',
                     'base_url': 'http://127.0.0.1:1234/v1'}}
    (home / 'config.yaml').write_text(json.dumps(cfg))
    (other.db.db_path.parent / 'config.yaml').write_text(json.dumps(cfg))
    private = {}
    policy = build_policy({'cwd': str(home), 'model': 'frozen'}, cfg, private_secrets=private)
    policy = bind_launch_key(authority, 'session', policy, None, config_secrets=private)
    cold = SimpleNamespace(db=authority.db, profile_id='owned', instance_id='new', epoch=2)
    assert policy.config(cold)['model']['api_key'] == 'owned-test-token'
    with pytest.raises(RuntimeStoreError, match='launch_credentials_unavailable') as refused:
        policy.config(other)
    assert refused.value.field == 'home'
