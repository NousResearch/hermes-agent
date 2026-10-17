"""Launch-route gates compare what the runtime resolves, not how the provider was spelled."""
import pytest

from gateway.session_route_identity import ROUTE_BOUND_KEYS, resolved_route, same_route

A, B = 'http://127.0.0.1:8001/v1', 'http://127.0.0.1:8002/v1'
CONFIG = {'providers': {'team': {'base_url': A, 'api_key': 'sk-team'}}}


@pytest.mark.parametrize('a, b, config, expected', [
    ({'provider': 'vllm', 'base_url': A}, {'provider': 'custom', 'base_url': A + '/'}, {}, True),
    ({'provider': 'ollama', 'base_url': A}, {'provider': 'custom', 'base_url': A}, {}, True),
    ({'provider': 'team'}, {'provider': 'custom:team'}, CONFIG, True),
    ({'provider': 'team'}, {'provider': 'custom:team', 'base_url': A}, CONFIG, True),
    ({'provider': 'team'}, {'provider': 'custom:team', 'base_url': B}, CONFIG, False),
    ({'provider': 'team'}, {'provider': 'team'},
     {'providers': {'team': {'base_url': B}}}, True),
    ({'provider': 'auto'}, {'provider': 'custom', 'base_url': B}, {}, True),
    ({'provider': 'deepinfra'}, {'provider': 'deep-infra'}, {}, True),
    ({'provider': 'openrouter'}, {'provider': 'anthropic'}, {}, False),
    ({'provider': 'custom', 'base_url': A}, {'provider': 'custom', 'base_url': B}, {}, False),
])
def test_same_route(a, b, config, expected):
    assert same_route(a, b, config) is expected
    assert same_route(b, a, config) is expected


def test_resolved_route_identity_and_url():
    assert resolved_route({'provider': 'custom:team'}, CONFIG) == ('custom:team', A)
    assert resolved_route({'provider': 'TEAM', 'base_url': B + '/'}, CONFIG) == ('custom:team', B)
    assert resolved_route({'provider': 'ollama', 'base_url': A}, {}) == ('custom', A)
    assert resolved_route({'provider': ''}, CONFIG) == ('auto', '')


def test_route_bound_keys_cover_route_wire_window_and_credentials():
    assert isinstance(ROUTE_BOUND_KEYS, tuple)
    for key in ('provider', 'base_url', 'api_mode', 'context_length',
                'api_key', 'api', 'api_key_from', 'key_env', 'api_key_env'):
        assert key in ROUTE_BOUND_KEYS
