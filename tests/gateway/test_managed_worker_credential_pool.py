"""A managed worker (every Kanban run) builds its agent with the route's credential pool.

Without the pool the worker's agent cannot refresh a pooled OAuth token: when the token
expires mid-turn the 401 recovery has nothing to refresh, so the run dies with exit 78 at
the token's expiry while a manual ``hermes auth refresh`` would have fixed it.
"""
import json
import time

import agent.anthropic_credentials as ac
from agent.error_classifier import FailoverReason
from run_agent import AIAgent

from tests.gateway.test_managed_turn_contract import assignment
from tests.gateway.test_managed_worker_route_key import _execute

MODEL = 'claude-opus-5-5'


def _anthropic_frame(tmp_path, monkeypatch, expires_in_s=3600):
    from hermes_constants import get_hermes_home
    for var in ('ANTHROPIC_TOKEN', 'ANTHROPIC_API_KEY', 'CLAUDE_CODE_OAUTH_TOKEN'):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'no-claude'))
    row = {'id': 'oauth1', 'label': 'acct', 'auth_type': 'oauth', 'priority': 0, 'source': 'manual:hermes_pkce',
           'access_token': 'sk-ant-oat01-launch', 'refresh_token': 'sk-ant-ort01-launch',
           'base_url': 'https://api.anthropic.com', 'expires_at_ms': int((time.time() + expires_in_s) * 1000)}
    (get_hermes_home() / 'auth.json').write_text(json.dumps(
        {'version': 1, 'providers': {}, 'credential_pool': {'anthropic': [row]}}))
    frame, _ = assignment(tmp_path, monkeypatch)
    config = {**json.loads(frame['policy']['config_json']), 'model': {'default': MODEL, 'provider': 'anthropic'}}
    request = {k: v for k, v in json.loads(frame['policy']['request_json']).items() if k not in ('provider', 'base_url')}
    policy = {**frame['policy'], 'model': MODEL,
              'config_json': json.dumps(config), 'request_json': json.dumps(request)}
    return {**frame, 'policy': policy, 'api_key': None}


def test_keyless_worker_gets_the_routes_credential_pool(tmp_path, monkeypatch):
    built = _execute(_anthropic_frame(tmp_path, monkeypatch), monkeypatch)
    assert built['api_key'] == 'sk-ant-oat01-launch'
    pool = built.get('credential_pool')
    assert pool is not None and pool.provider == 'anthropic'
    assert [e.id for e in pool.entries()] == ['oauth1']


def test_worker_agent_refreshes_a_token_that_expires_mid_turn(tmp_path, monkeypatch):
    """The real AIAgent the worker builds turns an expired-token 401 into one refresh and a retry."""
    posts = []

    def refresh(refresh_token, *, use_json=False):
        posts.append(refresh_token)
        return {'access_token': 'sk-ant-oat01-rotated', 'refresh_token': 'sk-ant-ort01-rotated',
                'expires_at_ms': int((time.time() + 8 * 3600) * 1000)}

    monkeypatch.setattr(ac, 'refresh_anthropic_oauth_pure', refresh)
    agents = []

    class Recording(AIAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            agents.append(self)

    _execute(_anthropic_frame(tmp_path, monkeypatch), monkeypatch, base=Recording)
    agent = agents[0]
    recovered, _ = agent._recover_with_credential_pool(
        status_code=401, has_retried_429=False, classified_reason=FailoverReason.auth,
        error_context={'reason': 'authentication_error',
                       'message': 'OAuth access token has expired. Re-authenticate to continue.'})
    assert recovered
    assert posts == ['sk-ant-ort01-launch']
    assert agent._anthropic_api_key == 'sk-ant-oat01-rotated'
