from tests.gateway.test_buzz_authorization_contract import _clean_registry_and_env, _register_runtime_policy, _Runner, _source, ALLOWED_USERS_ENV, ALLOW_ALL_ENV
import pytest
@pytest.mark.parametrize('policy,env,user,expected',[
 ({'allowed_users':['yaml']},{ALLOWED_USERS_ENV:'env'},'yaml',False),
 ({'allowed_users':['yaml']},{ALLOWED_USERS_ENV:'env'},'env',True),
 ({'allowed_users':['yaml']},{ALLOWED_USERS_ENV:''},'yaml',False),
 ({'allow_all_users':True},{ALLOW_ALL_ENV:''},'any',False),
 ({'allow_all_users':True},{ALLOW_ALL_ENV:'false'},'any',False),
 ({'allow_all_users':False},{ALLOW_ALL_ENV:'true'},'any',True),
 ({'allowed_users':[]},{ALLOWED_USERS_ENV:'env'},'env',True),
 ({'allowed_users':[], 'allow_all_users':False},{'GATEWAY_ALLOWED_USERS':'*'},'any',True),
])
def test_preserving_precedence(policy,env,user,expected,monkeypatch):
 _register_runtime_policy(resolver=lambda _:policy)
 for k,v in env.items(): monkeypatch.setenv(k,v)
 observed=_Runner()._is_user_authorized(_source(user))
 assert observed is expected, {'expected':expected,'observed':observed,'policy':policy,'env':env}
