"""Follow-profile sessions advertise the same route and account as their next turn."""
import json
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.session_contract import SessionRef
from hermes_agent.gateway.session_policy import policy_for_source
from hermes_constants import get_hermes_home
from tests.gateway.test_session_create_tier_cwd import _owner


@pytest.mark.asyncio
@pytest.mark.parametrize('initially_supported', [False, True])
@pytest.mark.parametrize('retained', [False, True])
async def test_service_tier_choices_follow_profile_model_provider_and_endpoint_together(
    tmp_path, monkeypatch, initially_supported, retained,
):
    monkeypatch.chdir(tmp_path)
    home = get_hermes_home()
    supported = {'provider': 'openai', 'default': 'gpt-6.1-sol', 'base_url': 'https://api.openai.com/v1'}
    unsupported = {'provider': 'custom', 'default': 'gpt-6.1-sol', 'base_url': 'http://127.0.0.1:12345/v1'}
    before, after = (supported, unsupported) if initially_supported else (unsupported, supported)
    store, runner, authority, owner = _owner(home, monkeypatch, {'model': before})
    try:
        assert authority.db.db_path.parent.resolve() == home.resolve()
        created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'follow-route', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
            'follow_profile_config': True,
        }})
        sid = created['result']['session_id']
        previous = SimpleNamespace(model=before['default'], provider=before['provider'],
                                   base_url=before['base_url'], api_key='previous-route-key') if retained else None
        monkeypatch.setattr(runner, '_cached_agent_for', lambda route: previous)
        (home / 'config.yaml').write_text(json.dumps({'model': after}))
        assert authority.agent(SessionRef(authority.profile_id, sid)) is previous
        policy = policy_for_source(authority.runner, authority.sessions[sid].source)
        assert policy is not None
        config, model = policy.effective(authority, secrets=False)
        assert (model, config['model']['provider'], config['model']['base_url']) == (
            after['default'], after['provider'], after['base_url'])
        reply = await owner.dispatch({'id': 'choices', 'method': 'config.get', 'params': {
            'session_id': sid, 'key': 'service_tier',
        }})
        assert 'result' in reply, reply
        expected = [] if initially_supported else ['normal', 'fast', 'auto', 'cold', 'ultrafast']
        assert reply['result']['options'] == expected
    finally:
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_followed_account_eligibility_does_not_use_the_retained_agents_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    home = get_hermes_home()
    before = {'provider': 'custom', 'default': 'gpt-6.1-sol', 'base_url': 'http://127.0.0.1:12345/v1'}
    after = {'provider': 'openai-codex', 'default': 'gpt-6.1-sol',
             'base_url': 'https://chatgpt.com/backend-api/codex', 'api_key': 'next-route-key'}
    observed = []

    def ultrafast_offered(model, **route):
        observed.append(route)
        return route['access_token'] == 'next-route-key'

    monkeypatch.setattr('agent.codex_service_tiers.ultrafast_offered', ultrafast_offered)
    store, runner, _, owner = _owner(home, monkeypatch, {'model': before})
    try:
        created = await owner.dispatch({'id': 'create', 'method': 'session.create', 'params': {
            'request_id': 'follow-account', 'source': 'gui', 'cwd': str(tmp_path), 'toolsets': [],
            'follow_profile_config': True,
        }})
        sid = created['result']['session_id']
        previous = SimpleNamespace(model=before['default'], provider=before['provider'],
                                   base_url=before['base_url'], api_key='previous-route-key')
        monkeypatch.setattr(runner, '_cached_agent_for', lambda route: previous)
        (home / 'config.yaml').write_text(json.dumps({'model': after}))
        reply = await owner.dispatch({'id': 'choices', 'method': 'config.get', 'params': {
            'session_id': sid, 'key': 'service_tier',
        }})
        assert 'ultrafast' in reply['result']['options']
        assert observed == [{'provider': after['provider'], 'base_url': after['base_url'],
                             'access_token': 'next-route-key'}]
    finally:
        await owner.close()
        store.close_all_db_handles()
