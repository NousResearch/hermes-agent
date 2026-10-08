"""Real profile/config/pool resolution must preserve the selected vision billing route."""
import json
import time
from pathlib import Path

import pytest
import hermes_yaml as yaml

from agent import auxiliary_client as aux
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def _profile(root, name, *, pool=False, fallback=False):
    home = root / name
    home.mkdir()
    (home / 'config.yaml').write_text(yaml.safe_dump({
        'model': {'provider': 'openai-codex', 'default': 'gpt-6-sol'},
        'auxiliary': {'vision': {
            'provider': 'openai-codex', 'model': 'gpt-6-sol',
            'fallback_chain': ([{'provider': 'openrouter', 'model': 'openai/gpt-6-sol'}]
                               if fallback else []),
        }},
    }))
    (home / '.env').write_text('OPENROUTER_API_KEY=test-openrouter-key\n')
    if pool:
        (home / 'auth.json').write_text(json.dumps({
            'version': 1, 'providers': {},
            'credential_pool': {'openai-codex': [{
                'id': name, 'label': name, 'auth_type': 'api_key', 'priority': 0,
                'source': 'manual', 'access_token': 'test-codex-' + name,
                'base_url': 'https://chatgpt.com/backend-api/codex',
                'model_cooldowns': {'grok-4.7': time.time() + 3600},
            }]},
        }))
    return home


def _vision(async_mode):
    return aux._resolve_call_client(
        'vision', provider=None, model=None, base_url=None, api_key=None,
        resolved_provider='openai-codex', resolved_model='gpt-6-sol',
        resolved_base_url=None, resolved_api_key=None, resolved_api_mode=None,
        main_runtime={'provider': 'openai-codex', 'model': 'gpt-6-sol'},
        async_mode=async_mode,
    )


@pytest.mark.parametrize('async_mode', [False, True])
def test_vision_pool_selection_uses_requested_model_across_profiles(tmp_path, monkeypatch, async_mode):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    homes = [_profile(tmp_path, name, pool=True) for name in ['a', 'b']]
    for home in [*homes, homes[0]]:
        monkeypatch.setenv('HERMES_HOME', str(home))
        scope = set_hermes_home_override(home)
        try:
            route = _vision(async_mode)
            assert (route.effective_provider, route.final_model) == ('openai-codex', 'gpt-6-sol')
            from agent.credential_pool import load_pool
            assert load_pool('openai-codex').select(model='grok-4.7') is None
            assert route.client.api_key == 'test-codex-' + home.name
        finally:
            reset_hermes_home_override(scope)


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('fallback', [False, True])
def test_unavailable_explicit_vision_only_uses_declared_fallback(tmp_path, monkeypatch, async_mode, fallback):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home = _profile(tmp_path, 'unavailable', fallback=fallback)
    monkeypatch.setenv('HERMES_HOME', str(home))
    scope = set_hermes_home_override(home)
    try:
        if fallback:
            route = _vision(async_mode)
            assert route.final_model == 'openai/gpt-6-sol'
            assert route.effective_provider == 'fallback_chain[0](openrouter)'
        else:
            with pytest.raises(aux.AuxiliaryClientUnavailable):
                _vision(async_mode)
    finally:
        reset_hermes_home_override(scope)
