"""A retained adapter fallback keeps the store's live mapping, not a revoked copy."""

pytest_plugins = ["tests.agent.test_secret_store_routing"]


def test_external_fallback_follows_store_revocation(store, monkeypatch):
    provider, home = store
    from gateway.platforms import _shared
    monkeypatch.setattr(_shared, '_UNSCOPED_PROFILE_SECRETS', None)
    monkeypatch.delenv('BUZZ_ACCESS_KEY', raising=False)
    provider.seed_external(home, {'BUZZ_ACCESS_KEY': 'old-marker'})
    provider.refresh(home)
    assert _shared.get_scoped_secret('BUZZ_ACCESS_KEY', external_fallback=True) == 'old-marker'
    provider._remote(home)['env'] = {}
    provider.refresh(home)
    assert _shared.get_scoped_secret('BUZZ_ACCESS_KEY', external_fallback=True) is None
