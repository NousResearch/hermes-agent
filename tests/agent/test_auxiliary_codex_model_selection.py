"""Codex auxiliary selection must not apply a sibling model's cooldown."""
import time
from unittest.mock import MagicMock

import pytest

from agent import auxiliary_client as aux
from agent.credential_pool import CredentialPool, PooledCredential


@pytest.mark.parametrize('raw_codex', [False, True])
@pytest.mark.parametrize('blocked_model,available', [
    ('GPT-6-astra-900k', True),
    ('gpt-6-luna-900k', False),
])
def test_codex_selects_for_requested_model(monkeypatch, raw_codex, blocked_model, available):
    entry = PooledCredential(
        provider='openai-codex', id='synthetic', label='test', auth_type='oauth',
        priority=0, source='manual:loopback_pkce', access_token='synthetic-token',
        base_url='https://chatgpt.com/backend-api/codex', last_status='ok',
        model_cooldowns={blocked_model: time.time() + 3600},
    )
    pool = CredentialPool('openai-codex', [entry])
    monkeypatch.setattr(pool, '_entry_needs_refresh', lambda entry: False)
    monkeypatch.setattr(aux, '_load_pool_with_credentials', lambda provider: pool)
    monkeypatch.setattr(aux, '_read_codex_singleton_token', lambda: None)
    monkeypatch.setattr(aux, '_codex_base_url_override', lambda: '')
    factory = MagicMock()
    monkeypatch.setattr(aux, 'OpenAI', factory)

    client, model = aux.resolve_provider_client(
        'openai-codex', 'gpt-6-luna-900k', raw_codex=raw_codex,
    )

    assert (client is not None) is available
    if available:
        assert factory.call_args.kwargs['api_key'] == 'synthetic-token'
        assert factory.call_args.kwargs['base_url'] == 'https://chatgpt.com/backend-api/codex'
        assert model == 'gpt-6-luna-900k'
    else:
        factory.assert_not_called()
    assert pool._entries[0].model_cooldowns == entry.model_cooldowns


def test_unscoped_selection_stays_conservative(monkeypatch):
    entry = PooledCredential(
        provider='openai-codex', id='synthetic', label='test', auth_type='oauth',
        priority=0, source='manual:loopback_pkce', access_token='synthetic-token',
        model_cooldowns={'GPT-6-astra-900k': time.time() + 3600},
    )
    pool = CredentialPool('openai-codex', [entry])
    monkeypatch.setattr(aux, '_load_pool_with_credentials', lambda provider: pool)
    assert aux._select_pool_entry('openai-codex') == (True, None)


def test_codex_cache_tracks_model_selected_account_rotation(monkeypatch):
    state = {'token': 'synthetic-a1'}

    def load_fresh_pool(provider, note=''):
        entries = [PooledCredential(
            provider='openai-codex', id=entry_id, label='test', auth_type='oauth',
            priority=priority, source='manual:loopback_pkce', access_token=token,
            base_url='https://chatgpt.com/backend-api/codex', last_status='ok',
            model_cooldowns=cooldowns,
        ) for entry_id, priority, token, cooldowns in [
            ('a', 0, state['token'], {'GPT-6-astra-900k': time.time() + 3600}),
            ('b', 1, 'synthetic-b1', {}),
        ]]
        pool = CredentialPool(provider, entries)
        monkeypatch.setattr(pool, '_entry_needs_refresh', lambda entry: False)
        return pool

    monkeypatch.setattr(aux, '_load_pool_with_credentials', load_fresh_pool)
    monkeypatch.setattr(aux, '_read_codex_singleton_token', lambda: None)
    monkeypatch.setattr(aux, '_codex_base_url_override', lambda: '')
    factory = MagicMock(side_effect=lambda **kwargs: MagicMock(api_key=kwargs['api_key']))
    monkeypatch.setattr(aux, '_create_openai_client', factory)
    aux.shutdown_cached_clients()
    try:
        first, _ = aux._get_cached_client('openai-codex', 'gpt-6-luna-900k')
        state['token'] = 'synthetic-a2'
        second, _ = aux._get_cached_client('openai-codex', 'gpt-6-luna-900k')
        third, _ = aux._get_cached_client('openai-codex', 'gpt-6-luna-900k')
        assert first is not second
        assert third is second
        assert [call.kwargs['api_key'] for call in factory.call_args_list] == [
            'synthetic-a1', 'synthetic-a2',
        ]
    finally:
        aux.shutdown_cached_clients()
