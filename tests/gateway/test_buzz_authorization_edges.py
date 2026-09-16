"""Independent bounded checks; no installed runtime execution."""
from types import SimpleNamespace
import pytest
from tests.gateway.test_buzz_authorization_owner_env import clean
from tests.gateway.test_buzz_authorization_contract import (_register_runtime_policy, _Runner, _Adapter,
    _source, PLATFORM_NAME, ALLOWED_USERS_ENV, ALLOW_ALL_ENV)
from gateway.config import Platform

@pytest.mark.parametrize('policy',[None,{}])
def test_resolver_absence_does_not_resurrect_extra_snapshot(policy):
    _register_runtime_policy(resolver=lambda _:policy)
    r=_Runner(); a=_Adapter(); a.config=SimpleNamespace(extra={'allow_from':['alice']})
    r.adapters={Platform(PLATFORM_NAME):a}
    assert r._is_user_authorized(_source('alice')) is False
    _register_runtime_policy(resolver=None)
    assert r._is_user_authorized(_source('alice')) is True

@pytest.mark.parametrize('failed_field',[ALLOWED_USERS_ENV,ALLOW_ALL_ENV])
def test_both_owner_fields_acquired_before_union_grants(tmp_path,monkeypatch,failed_field):
    from agent import secret_scope
    home=tmp_path/'owner'; home.mkdir()
    monkeypatch.setenv('HERMES_HOME',str(home))
    _register_runtime_policy(resolver=lambda _:{'allow_all_users':True,'allowed_users':['alice']})
    r=_Runner();a=_Adapter();a.authorization_home=home;r.adapters={Platform(PLATFORM_NAME):a}
    r.pairing_store=SimpleNamespace(is_approved=lambda *_:True)
    monkeypatch.setenv('GATEWAY_ALLOWED_USERS','*')
    class FailingScope(dict):
        def get(self,key,default=None):
            if key==failed_field: raise RuntimeError('synthetic acquisition failure')
            return super().get(key,default)
    token=secret_scope.set_secret_scope(FailingScope({ALLOWED_USERS_ENV:'alice',ALLOW_ALL_ENV:'true'}))
    try: assert r._is_user_authorized(_source('alice')) is False
    finally: secret_scope.reset_secret_scope(token)

@pytest.mark.parametrize('managed_env,owner_env,policy',[
    ('RUNTIME_AUTH_TEST_ALLOWED_USERS=\n','RUNTIME_AUTH_TEST_ALLOWED_USERS=alice\n',{'allowed_users':['alice']}),
    ('RUNTIME_AUTH_TEST_ALLOW_ALL_USERS=false\n','RUNTIME_AUTH_TEST_ALLOW_ALL_USERS=true\n',{'allow_all_users':True}),
])
def test_managed_presence_beats_owner_env_and_raw_policy(tmp_path,monkeypatch,managed_env,owner_env,policy):
    home=tmp_path/'owner';home.mkdir();(home/'.env').write_text(owner_env)
    managed=tmp_path/'managed';managed.mkdir();(managed/'.env').write_text(managed_env)
    monkeypatch.setenv('HERMES_HOME',str(home));monkeypatch.setenv('HERMES_MANAGED_DIR',str(managed))
    _register_runtime_policy(resolver=lambda _:policy)
    r=_Runner();a=_Adapter();a.authorization_home=home;r.adapters={Platform(PLATFORM_NAME):a}
    assert r._is_user_authorized(_source('alice')) is False
