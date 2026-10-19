"""A model pick on a profile-following session that carries frozen config secrets.

A Desktop Bot Chat is created with ``follow_profile_config``; when the profile holds a secret
outside the followed sections (an MCP server's env token) its policy also carries a
``config_secret_ref`` whose recovery is non-empty. The composer pick on that chat must commit,
not crash session.mutate, and the unrelated secret stays bound to the session.
Real SessionStore/SessionAuthority, a real config.yaml and a loopback catalog endpoint.
"""
import json

import pytest

from tests.gateway.test_session_follow_profile_config import _create, _owner, _route
from tests.gateway.test_session_mutation_model_route import _bind_resolver, _pick

pytest_plugins = ['tests.gateway.test_session_mutation_model_route']


@pytest.mark.asyncio
async def test_a_pick_on_a_following_chat_with_config_secrets_commits(home, endpoints):
    from gateway.session_policy import policy_for_source
    (_, url), _ = endpoints
    (home / 'config.yaml').write_text(json.dumps({
        'model': {'provider': 'custom', 'default': 'original', 'base_url': url, 'api_key': 'key-a'},
        'mcp_servers': {'side': {'command': 'true', 'env': {'SIDE_TOKEN': 'side-secret'}}},
    }), encoding='utf-8')
    store, runner, authority, owner = _owner(home)
    _bind_resolver(runner)
    try:
        live = await _create(owner, authority, home, 'bot-chat', title='Bot Chat')
        sid = next(sid for sid, value in authority.sessions.items() if value is live)
        policy = policy_for_source(runner, live.source)
        assert policy.follows_profile
        assert policy.config_secret_ref, 'precondition: the chat holds frozen config secrets'

        response = await _pick(owner, authority, sid, {'model': 'picked', 'provider': 'custom'})

        assert 'result' in response, response
        assert _route(runner, live) == ('picked', url, 'key-a')
        config = policy_for_source(runner, live.source).config(authority)
        assert config['mcp_servers']['side']['env']['SIDE_TOKEN'] == 'side-secret'
    finally:
        await owner.close()
        store.close_all_db_handles()
