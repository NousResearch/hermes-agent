"""A home moved on the same host (the old path left as a symlink to the new one) still owns its chats'
frozen config credentials: the reference names the old spelling of the same directory. A reference from
a genuinely different directory keeps refusing, and a rotated secret still refuses after the move."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from persistence_assembly import profile_scope


def _stamped(tmp_path, monkeypatch, cfg):
    from gateway.run import _load_gateway_config
    from gateway.session_policy import bind_launch_key, build_policy
    real = tmp_path / 'new-user' / 'profiles' / 'p'
    real.mkdir(parents=True)
    (tmp_path / 'old-user').symlink_to(tmp_path / 'new-user')  # the old home, left as a symlink
    old = tmp_path / 'old-user' / 'profiles' / 'p'
    monkeypatch.setenv('HERMES_HOME', str(real))
    (real / 'config.yaml').write_text(json.dumps(cfg))
    loaded = _load_gateway_config(real / 'config.yaml')

    def owner(home=real):
        return SimpleNamespace(db=SimpleNamespace(db_path=home / 'state.db'),
                               persistence=SimpleNamespace(scope=profile_scope(home=home)),
                               profile_id=str(real.resolve()), instance_id='new', epoch=2)
    private = {}
    policy = build_policy({'cwd': str(real), 'model': 'frozen'}, loaded, private_secrets=private)
    policy = bind_launch_key(owner(), 'session', policy, None, config_secrets=private)
    ref = json.loads(policy.config_secret_ref[len('profile-config-v1:'):])
    # As the old host stamped it: both owner names spell the old path.
    ref['home'] = ref['profile'] = str(old)
    return replace(policy, config_secret_ref='profile-config-v1:' + json.dumps(ref, sort_keys=True)), owner, real


CFG = {'model': {'provider': 'custom', 'api_key': 'route-token', 'base_url': 'http://127.0.0.1:1234/v1'}}


def test_a_home_moved_behind_a_symlink_still_recovers_its_frozen_secrets(tmp_path, monkeypatch):
    from gateway.session_policy_credentials import recover_config_secrets
    policy, owner, _ = _stamped(tmp_path, monkeypatch, json.loads(json.dumps(CFG)))
    assert recover_config_secrets(owner(), policy)[('model', 'api_key')] == 'route-token'


def test_a_rotated_secret_still_refuses_after_the_move(tmp_path, monkeypatch):
    from gateway.session_policy_credentials import recover_config_secrets
    from persistence.core.errors import AuthorityRefusal
    cfg = json.loads(json.dumps(CFG))
    policy, owner, real = _stamped(tmp_path, monkeypatch, cfg)
    cfg['model']['api_key'] = 'rotated-token'
    (real / 'config.yaml').write_text(json.dumps(cfg))
    with pytest.raises(AuthorityRefusal) as refused:
        recover_config_secrets(owner(), policy)
    assert refused.value.field == 'model.api_key'


def test_another_directory_keeps_refusing(tmp_path, monkeypatch):
    from gateway.session_policy_credentials import recover_config_secrets
    from persistence.core.errors import AuthorityRefusal
    policy, owner, _ = _stamped(tmp_path, monkeypatch, json.loads(json.dumps(CFG)))
    other = tmp_path / 'elsewhere' / 'profiles' / 'p'
    other.mkdir(parents=True)
    ref = json.loads(policy.config_secret_ref[len('profile-config-v1:'):])
    ref['home'] = ref['profile'] = str(other)
    foreign = replace(policy, config_secret_ref='profile-config-v1:' + json.dumps(ref, sort_keys=True))
    with pytest.raises(AuthorityRefusal) as refused:
        recover_config_secrets(owner(), foreign)
    assert refused.value.field == 'home'
