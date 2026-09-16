"""Owner-scoped raw hooks plus actual registered Buzz composition."""
import weakref
import pytest
from tests.gateway.test_buzz_authorization_contract import (_Runner, _Adapter, _source,
    _register_runtime_policy, PLATFORM_NAME, ALLOWED_USERS_ENV, ALLOW_ALL_ENV)
from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from tests.gateway.test_live_buzz_authorization import registered_buzz, atomic_policy, source

@pytest.fixture(autouse=True)
def clean(monkeypatch):
 for name in (ALLOWED_USERS_ENV, ALLOW_ALL_ENV, 'BUZZ_ALLOWED_USERS', 'BUZZ_ALLOW_ALL_USERS', 'GATEWAY_ALLOWED_USERS', 'GATEWAY_ALLOW_ALL_USERS'):
  monkeypatch.delenv(name, raising=False)
 scope=platform_registry.current_scope_key()
 yield
 platform_registry.unregister(PLATFORM_NAME,scope=scope)

@pytest.mark.parametrize('owner_env,policy,user,expected',[
 ('RUNTIME_AUTH_TEST_ALLOWED_USERS=env\n', {'allowed_users':['yaml']},'env',True),
 ('RUNTIME_AUTH_TEST_ALLOWED_USERS=\n', {'allowed_users':['yaml']},'yaml',False),
 ('RUNTIME_AUTH_TEST_ALLOW_ALL_USERS=false\n', {'allow_all_users':True},'any',False),
 ('RUNTIME_AUTH_TEST_ALLOW_ALL_USERS=true\n', {'allow_all_users':False},'any',True),
 ('', {'allowed_users':['yaml']},'routed',False),
])
def test_raw_owner_env_not_routed_registry_or_scope(tmp_path, monkeypatch, owner_env, policy, user, expected):
 from hermes_constants import set_hermes_home_override, reset_hermes_home_override
 from agent import secret_scope
 home=tmp_path/'owner'; home.mkdir(); (home/'.env').write_text(owner_env)
 other=tmp_path/'other'/'profiles'/'same'; other.mkdir(parents=True)
 monkeypatch.setenv('HERMES_HOME',str(home))
 calls=[]
 _register_runtime_policy(resolver=lambda p: calls.append(p) or policy)
 r=_Runner(); adapter=_Adapter(); adapter.authorization_home=home
 r._profile_adapters={'same':{Platform(PLATFORM_NAME):adapter}}
 event=_source(user,profile='routed');event._transport_adapter_ref=weakref.ref(adapter)
 platform_registry.register(PlatformEntry(name=PLATFORM_NAME,label='wrong home',adapter_factory=lambda _:None,check_fn=lambda:True,allowed_users_env='WRONG_ALLOWED_USERS',allow_all_env='WRONG_ALLOW_ALL_USERS',authorization_config_fn=lambda _: {'allow_all_users':True}),scope=str(other))
 token=set_hermes_home_override(other)
 st=secret_scope.set_secret_scope({ALLOWED_USERS_ENV:'routed',ALLOW_ALL_ENV:'true','WRONG_ALLOW_ALL_USERS':'true'})
 try:
  assert r._is_user_authorized(event) is expected
  assert calls==['same']
 finally:
  secret_scope.reset_secret_scope(st);reset_hermes_home_override(token)
  platform_registry.unregister(PLATFORM_NAME,scope=str(other));platform_registry.unregister(PLATFORM_NAME,scope=str(home))

@pytest.mark.parametrize('broken',['directory','malformed'])
def test_owner_env_acquisition_failure_denies_pairing(tmp_path,monkeypatch,broken):
 home=tmp_path/'owner';home.mkdir();monkeypatch.setenv('HERMES_HOME',str(home))
 if broken=='directory':(home/'.env').mkdir()
 else:(home/'.env').write_text('NOT AN ENV ASSIGNMENT')
 _register_runtime_policy(resolver=lambda _: {'allow_all_users':True})
 r=_Runner();a=_Adapter();a.authorization_home=home;r.adapters={Platform(PLATFORM_NAME):a}
 from types import SimpleNamespace
 r.pairing_store=SimpleNamespace(is_approved=lambda *_:True)
 assert r._is_user_authorized(_source('any')) is False

@pytest.mark.parametrize('env,expected_a,expected_b',[
 ('BUZZ_ALLOWED_USERS='+ 'b'*64+'\n',False,True),
 ('BUZZ_ALLOWED_USERS=\n',False,False),
 ('BUZZ_ALLOW_ALL_USERS=false\n',True,False),
 ('BUZZ_ALLOW_ALL_USERS=true\n',True,True),
 ('',True,False),
])
def test_registered_buzz_owner_env_survives_wrong_home(registered_buzz,tmp_path,env,expected_a,expected_b):
 from hermes_constants import set_hermes_home_override, reset_hermes_home_override
 from agent import secret_scope
 home,r,adapter=registered_buzz
 (home/'.env').write_text(env)
 other=tmp_path/'other'/'profiles'/'same';other.mkdir(parents=True)
 (other/'.env').write_text('BUZZ_ALLOW_ALL_USERS=true\n')
 atomic_policy(other,{'allow_all_users':True})
 token=set_hermes_home_override(other)
 st=secret_scope.set_secret_scope({'BUZZ_ALLOW_ALL_USERS':'true','BUZZ_ALLOWED_USERS':'c'*64})
 def check(user):
  event=source(user,platform='buzz',profile='routed');event._transport_adapter_ref=weakref.ref(adapter)
  return r._is_user_authorized(event)
 try:
  assert check('a'*64) is expected_a
  assert check('b'*64) is expected_b
  (home/'config.yaml').unlink()
  assert check('a'*64) is (env=='BUZZ_ALLOW_ALL_USERS=true\n')
 finally:
  secret_scope.reset_secret_scope(st);reset_hermes_home_override(token)


@pytest.mark.parametrize('multiplex', [False, True])
def test_pinned_same_owner_explicit_scope_revokes_raw_policy(tmp_path, monkeypatch, multiplex):
 from agent import secret_scope
 home=tmp_path/'owner';home.mkdir();(home/'.env').write_text('')
 monkeypatch.setenv('HERMES_HOME',str(home))
 _register_runtime_policy(resolver=lambda _: {'allow_all_users':True, 'allowed_users':['alice']})
 r=_Runner();a=_Adapter();a.authorization_home=home
 r._profile_adapters={'same':{Platform(PLATFORM_NAME):a}}
 event=_source('alice');event._transport_adapter_ref=weakref.ref(a)
 token=secret_scope.set_secret_scope({ALLOWED_USERS_ENV:'',ALLOW_ALL_ENV:'false'})
 old=secret_scope.is_multiplex_active();secret_scope.set_multiplex_active(multiplex)
 try:
  assert r._is_user_authorized(event) is False
 finally:
  secret_scope.reset_secret_scope(token);secret_scope.set_multiplex_active(old)


def test_strict_raw_schema_rejects_tuple(monkeypatch):
 _register_runtime_policy(resolver=lambda _: {'allowed_users':('alice',)})
 monkeypatch.setenv('GATEWAY_ALLOW_ALL_USERS','true')
 assert _Runner()._is_user_authorized(_source('alice')) is False
