"""Picker and selection share a disk catalog until explicit refresh or identity change."""
from unittest.mock import Mock

from hermes_cli import models
from hermes_cli.models_cache_policy import manual_catalog_refresh, catalog_refresh_is_manual
from hermes_cli.models_validate import validate_requested_model
from hermes_cli.model_switch_providers import _prefetch_provider_models_parallel


def test_selection_prefetch_and_refresh_share_the_same_catalog(monkeypatch):
    live = Mock(return_value=['cached-model'])
    monkeypatch.setattr(models, 'provider_model_ids', live)
    monkeypatch.setattr(models, '_credential_fingerprint', lambda provider: 'account-a')
    network = Mock(side_effect=AssertionError('selection bypassed disk catalog'))
    monkeypatch.setattr(models, 'fetch_api_models', network)
    monkeypatch.setattr(models, '_spawn_swr_refresh', network)
    with manual_catalog_refresh():
        assert models.cached_provider_model_ids('gemini') == ['cached-model']
        cache = models._load_provider_models_cache()
        cache['gemini']['at'] = 0  # Age cannot trigger background or blocking I/O.
        models._save_provider_models_cache(cache)
        assert models.cached_provider_model_ids('gemini', non_blocking=True) == ['cached-model']
        _prefetch_provider_models_parallel(['gemini'])
        assert validate_requested_model('cached-model', 'gemini')['accepted']
        assert live.call_count == 1
        live.return_value = ['new-model']
        assert models.cached_provider_model_ids('gemini', force_refresh=True) == ['new-model']
        assert validate_requested_model('new-model', 'gemini')['accepted']
        assert not validate_requested_model('cached-model', 'gemini')['accepted']
        assert live.call_count == 2
        monkeypatch.setattr(models, '_credential_fingerprint', lambda provider: 'account-b')
        live.return_value = ['account-b-model']
        assert models.cached_provider_model_ids('gemini') == ['account-b-model']
        assert live.call_count == 3
    assert not catalog_refresh_is_manual()
    network.assert_not_called()


def test_profile_owned_catalog_validation_reuses_manual_snapshot(monkeypatch):
    from providers.base import ProviderProfile

    profile = ProviderProfile(name='catalog-test', models_url='https://example.test/catalog')
    monkeypatch.setattr('providers.get_provider_profile', lambda provider: profile)
    monkeypatch.setattr(models, '_credential_fingerprint', lambda provider: 'account-a')
    live = Mock(return_value=['cached-model'])
    monkeypatch.setattr(models, 'provider_model_ids', live)
    network = Mock(side_effect=AssertionError('validation rediscovered a profile-owned catalog'))
    monkeypatch.setattr(models, 'fetch_api_models', network)
    with manual_catalog_refresh():
        assert models.cached_provider_model_ids(profile.name) == ['cached-model']
        live.side_effect = AssertionError('validation bypassed the picker snapshot')
        assert validate_requested_model('cached-model', profile.name,
                                        api_key='test-key', base_url='https://example.test/v1')['accepted']
    assert live.call_count == 1
    network.assert_not_called()


def test_custom_endpoint_cache_survives_age_but_not_credential_rotation(monkeypatch):
    live = Mock(return_value=['custom-model'])
    monkeypatch.setattr(models, 'fetch_api_models', live)
    with manual_catalog_refresh():
        assert models.cached_fetch_api_models('key-a', 'https://example.test/v1') == ['custom-model']
        cache = models._load_provider_models_cache()
        for entry in cache.values():
            entry['at'] = 0
        models._save_provider_models_cache(cache)
        assert validate_requested_model('custom-model', 'custom', api_key='key-a',
                                        base_url='https://example.test/v1')['accepted']
        assert live.call_count == 1
        models.cached_fetch_api_models('key-a', 'https://example.test/v1', force_refresh=True)
        assert live.call_count == 2
        models.cached_fetch_api_models('key-b', 'https://example.test/v1')
        assert live.call_count == 3
