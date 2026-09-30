"""An Entra-only fallback entry must not borrow the primary model's auth mode."""
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI


@pytest.mark.parametrize("stage", ["client", "runtime"])
def test_fallback_entry_entra_survives_non_azure_primary(monkeypatch, stage):
    from hermes_constants import get_hermes_home
    import agent.azure_identity_adapter as identity
    from agent.auxiliary_client import resolve_provider_client
    from agent.client_lifecycle import _swap_fallback_clients
    from hermes_cli.fallback_config import get_fallback_chain, resolve_entry_api_key
    from hermes_cli.config import load_config_readonly
    from hermes_cli.runtime_provider import resolve_runtime_provider

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text('''model:
  provider: openrouter
  default: fixture-primary
fallback_providers:
  - provider: azure-foundry
    model: fixture-deployment
    base_url: http://127.0.0.1:9999/v1
    auth_mode: entra_id
    entra:
      scope: https://fixture.invalid/.default
''', encoding='utf8')
    token = ['first-fixture-token']
    seen_scopes = []
    def build(*, config):
        seen_scopes.append(config.scope)
        return lambda: token[0]
    monkeypatch.setattr(identity, 'build_token_provider', build)
    monkeypatch.delenv('AZURE_FOUNDRY_API_KEY', raising=False)
    config = load_config_readonly()
    entry = get_fallback_chain(config)[0]
    key = resolve_entry_api_key(entry)
    if stage == "runtime":
        runtime = resolve_runtime_provider(requested=entry['provider'], target_model=entry['model'],
            explicit_base_url=entry['base_url'], explicit_api_key=key)
        assert runtime['api_key'] is key
        assert runtime['auth_mode'] == 'entra_id'
        key = runtime['api_key']
    client, model = resolve_provider_client(entry['provider'], model=entry['model'],
        explicit_base_url=entry['base_url'], explicit_api_key=key, api_mode='chat_completions')
    assert client is not None, 'Entra fallback must resolve without a static Azure key'
    agent = SimpleNamespace()
    captured = []
    def respond(request):
        captured.append(request.headers['Authorization'])
        return httpx.Response(200, json={'id': 'fixture', 'choices': []})
    try:
        _swap_fallback_clients(agent, client, entry['provider'], model, str(client.base_url), 'chat_completions')
        with OpenAI(**agent._client_kwargs, http_client=httpx.Client(transport=httpx.MockTransport(respond))) as rebuilt:
            for value in ['first-fixture-token', 'rotated-fixture-token']:
                token[0] = value
                rebuilt.chat.completions.create(model=model, messages=[])
        assert captured == ['Bearer first-fixture-token', 'Bearer rotated-fixture-token']
        assert seen_scopes == ['https://fixture.invalid/.default']
    finally:
        client.close()


@pytest.mark.parametrize('override', [{'api_key': 'inline'}, {'key_env': 'FIXTURE_KEY'}, {'api_key_env': 'FIXTURE_KEY'}])
def test_explicit_static_credential_wins_over_entra(monkeypatch, override):
    from hermes_cli.fallback_config import resolve_entry_api_key
    monkeypatch.setenv('FIXTURE_KEY', 'env-key')
    def unexpected(*args, **kwargs):
        pytest.fail('explicit static keys must not construct an Entra provider')
    monkeypatch.setattr('agent.azure_identity_adapter.build_token_provider', unexpected)
    entry = {'provider': 'azure-foundry', 'auth_mode': 'entra_id', **override}
    assert resolve_entry_api_key(entry) == override.get('api_key', 'env-key')


@pytest.mark.parametrize('entry', [None, {}, {'provider': 'openrouter', 'auth_mode': 'entra_id'},
    {'provider': 'azure-foundry'}, {'provider': 'azure-foundry', 'auth_mode': 'api_key'}])
def test_unselected_entra_is_not_acquired(monkeypatch, entry):
    from hermes_cli.fallback_config import resolve_entry_api_key
    def unexpected(*args, **kwargs):
        pytest.fail('unselected Entra must not acquire credentials')
    monkeypatch.setattr('agent.azure_identity_adapter.build_token_provider', unexpected)
    assert resolve_entry_api_key(entry) is None


@pytest.mark.parametrize('error', [ImportError, RuntimeError, ValueError])
def test_failed_entra_construction_keeps_walking_chain(monkeypatch, error):
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_with_fallback
    import hermes_cli.runtime_provider as rp
    def resolve(**kwargs):
        if kwargs['requested'] == 'primary':
            raise AuthError('fixture primary unavailable')
        return {'provider': 'custom', 'api_key': kwargs['explicit_api_key']}
    def unavailable(**kwargs):
        raise error('fixture identity unavailable')
    monkeypatch.setattr(rp, 'resolve_runtime_provider', resolve)
    monkeypatch.setattr('agent.azure_identity_adapter.build_token_provider', unavailable)
    last = {'provider': 'custom', 'model': 'fixture', 'api_key': 'last-key'}
    config = {'fallback_providers': [
        {'provider': 'azure-foundry', 'model': 'fixture', 'auth_mode': 'entra_id'}, last]}
    runtime, entry = resolve_runtime_with_fallback(config, requested='primary')
    assert entry == last
    assert runtime['api_key'] == 'last-key'


@pytest.mark.parametrize('field', ['key_env', 'api_key_env'])
def test_missing_static_override_still_uses_declared_entra(monkeypatch, field):
    from hermes_cli.fallback_config import resolve_entry_api_key
    source = lambda: 'fixture-token'
    monkeypatch.delenv('UNSET_FIXTURE_KEY', raising=False)
    monkeypatch.setattr('agent.azure_identity_adapter.build_token_provider', lambda **kw: source)
    assert resolve_entry_api_key({'provider': 'azure-foundry', 'auth_mode': 'entra_id',
        field: 'UNSET_FIXTURE_KEY'}) is source
