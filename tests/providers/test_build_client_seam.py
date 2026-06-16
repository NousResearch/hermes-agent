"""Tests for the ProviderProfile.build_client seam.

Covers the generic, provider-agnostic client-construction hook:
  * ProviderProfile.build_client default returns None (stock OpenAI() path).
  * build_custom_client() asks each registered profile and returns the first
    non-None, else None.
  * async_mode is forwarded; the owning profile decides the client form.
  * a profile raising in build_client is skipped, not fatal.

These assert the dispatch contract only — no real network/client construction.
"""

import pytest

import providers as _pkg
from providers import build_custom_client, register_provider
from providers.base import ProviderProfile


_OWNED_URL = "http://owned.local/v1"


class _DummyClient:
    """Stand-in for a provider's custom client (e.g. a native transport)."""

    _hermes_native = True

    def __init__(self, base_url):
        self.base_url = base_url

    def as_async_client(self):
        return _DummyAsyncClient(self)


class _DummyAsyncClient:
    _hermes_native = True

    def __init__(self, sync_client):
        self._sync = sync_client


class _DummyProfile(ProviderProfile):
    """Owns exactly _OWNED_URL; supplies a custom client for it, None otherwise."""

    def build_client(self, *, base_url, api_key=None, async_mode=False, **context):
        if base_url != _OWNED_URL:
            return None
        client = _DummyClient(base_url)
        return client.as_async_client() if async_mode else client


class _RaisingProfile(ProviderProfile):
    def build_client(self, *, base_url, api_key=None, async_mode=False, **context):
        raise RuntimeError("boom")


@pytest.fixture
def isolated_registry():
    """Swap in a registry we fully control; restore the real one afterward."""
    saved_registry = dict(_pkg._REGISTRY)
    saved_aliases = dict(_pkg._ALIASES)
    saved_discovered = _pkg._discovered
    _pkg._REGISTRY.clear()
    _pkg._ALIASES.clear()
    _pkg._discovered = True  # skip filesystem discovery; we register by hand
    try:
        yield
    finally:
        _pkg._REGISTRY.clear()
        _pkg._REGISTRY.update(saved_registry)
        _pkg._ALIASES.clear()
        _pkg._ALIASES.update(saved_aliases)
        _pkg._discovered = saved_discovered


class TestBuildClientHook:
    def test_default_returns_none(self):
        """The base hook must default to None so existing providers are unchanged."""
        profile = ProviderProfile(name="plain")
        assert profile.build_client(base_url="http://anything/v1") is None
        assert profile.build_client(base_url="http://anything/v1", async_mode=True) is None


class TestBuildCustomClientDispatch:
    def test_routes_to_owning_profile(self, isolated_registry):
        register_provider(_DummyProfile(name="dummy-native"))
        client = build_custom_client(base_url=_OWNED_URL, api_key="EMPTY")
        assert client is not None
        assert isinstance(client, _DummyClient)
        assert getattr(client, "_hermes_native", False) is True

    def test_returns_none_for_unowned_url(self, isolated_registry):
        register_provider(_DummyProfile(name="dummy-native"))
        assert build_custom_client(base_url="http://somewhere-else/v1") is None

    def test_returns_none_with_no_profiles(self, isolated_registry):
        assert build_custom_client(base_url=_OWNED_URL) is None

    def test_async_mode_forwarded(self, isolated_registry):
        register_provider(_DummyProfile(name="dummy-native"))
        client = build_custom_client(base_url=_OWNED_URL, async_mode=True)
        assert isinstance(client, _DummyAsyncClient)

    def test_first_non_none_wins(self, isolated_registry):
        # A non-owning profile is consulted first and returns None; the owning
        # one is reached next. Both present → the one that claims the URL wins.
        register_provider(ProviderProfile(name="declarative-only"))  # build_client → None
        register_provider(_DummyProfile(name="dummy-native"))
        client = build_custom_client(base_url=_OWNED_URL)
        assert isinstance(client, _DummyClient)

    def test_raising_profile_is_skipped(self, isolated_registry):
        # A profile that raises must not abort dispatch; a later owner still wins.
        register_provider(_RaisingProfile(name="raises"))
        register_provider(_DummyProfile(name="dummy-native"))
        client = build_custom_client(base_url=_OWNED_URL)
        assert isinstance(client, _DummyClient)
