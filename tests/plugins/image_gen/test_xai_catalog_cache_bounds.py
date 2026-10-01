"""Regression test: the credential-scoped xAI image-model catalog must not grow per key rotation.

Under a profile home override the live catalog is keyed by ``(base_url, api-key fingerprint)`` so
one profile is never handed another's catalog. That scoping is correct, but because the key embeds a
fingerprint, every rotated key minted a new entry — and the 300s TTL gates only the READ, so a key
that is never presented again kept its catalog entry for the life of the process (#130261).

The contract: a long run of rotating keys stays bounded, a key retired past the TTL is released, and
the credential in use is still served from the cache.

Fixes #130261.
"""

import pytest

_ROTATIONS = 200

_LIVE = {"grok-imagine-image": {"input_modalities": ["text"], "aliases": []}}


class _ScopedHost:
    """A multiplexed host: an ACTIVE profile-home override is what selects the credential-scoped
    branch, and the api key it resolves to is what a rotation changes."""

    def __init__(self, monkeypatch):
        import hermes_constants
        from plugins.image_gen import xai as xai_mod

        self._hermes_constants = hermes_constants
        self._xai = xai_mod
        self._api_key = "cred-0"
        self.fetches = 0

        def _fetch(creds=None):
            self.fetches += 1
            return dict(_LIVE)

        monkeypatch.setattr(xai_mod, "_fetch_live_models", _fetch)
        # The plugin binds this name at import, so patching tools.xai_http would miss the read.
        monkeypatch.setattr(xai_mod, "resolve_xai_http_credentials", lambda: {
            "api_key": self._api_key, "base_url": "https://api.x.ai",
        })
        self._token = hermes_constants.set_hermes_home_override(
            str(hermes_constants.get_hermes_home()))

    def rotate(self, api_key: str) -> None:
        self._api_key = api_key

    def cache(self) -> dict:
        return self._xai._LIVE_CACHE_BY_CREDENTIAL

    def close(self) -> None:
        self._hermes_constants.reset_hermes_home_override(self._token)


@pytest.fixture(autouse=True)
def _clear_catalog_caches():
    from plugins.image_gen import xai as xai_mod

    xai_mod._LIVE_CACHE = None
    xai_mod._LIVE_CACHE_BY_CREDENTIAL.clear()
    yield
    xai_mod._LIVE_CACHE = None
    xai_mod._LIVE_CACHE_BY_CREDENTIAL.clear()


@pytest.fixture
def scoped_host(monkeypatch):
    host = _ScopedHost(monkeypatch)
    yield host
    host.close()


def test_live_catalog_is_bounded_across_key_rotations(scoped_host):
    """Each rotation presents a key the process will never see again. The catalog must not grow
    with the rotation count, and the key in use must still be served from it."""
    from plugins.image_gen import xai as xai_mod

    for rotation in range(_ROTATIONS):
        scoped_host.rotate(f"cred-{rotation}")
        xai_mod._live_models()

    cache = scoped_host.cache()
    assert len(cache) < _ROTATIONS, "kept one entry per key rotation"

    # The key in use must still be a cache HIT, not a refetch — so the bound did not come from
    # dropping entries a live credential still needs.
    fetches_before = scoped_host.fetches
    assert xai_mod._live_models() == _LIVE, "the in-use key is served from the cache"
    assert scoped_host.fetches == fetches_before, "the in-use key was refetched instead of cached"


def test_live_catalog_releases_a_key_past_its_ttl(scoped_host):
    """A key retired mid-window is past the TTL and must be released on the next fetch, while the
    credential in use is still served from the cache."""
    from plugins.image_gen import xai as xai_mod

    scoped_host.rotate("cred-retired")
    xai_mod._live_models()
    (retired_key,) = scoped_host.cache()

    scoped_host.rotate("cred-live")
    xai_mod._live_models()
    (live_key,) = [key for key in scoped_host.cache() if key != retired_key]
    assert xai_mod._live_models() == _LIVE, "the in-use key is served from the cache"
    assert len(scoped_host.cache()) == 2, "both keys are still inside the TTL"

    # Age the retired key's entry past the TTL, then fetch under a third credential.
    _, fetched_at = scoped_host.cache()[retired_key]
    scoped_host.cache()[retired_key] = (_LIVE, fetched_at - xai_mod._LIVE_CACHE_TTL - 1)
    scoped_host.rotate("cred-fresh")
    xai_mod._live_models()

    assert retired_key not in scoped_host.cache()
    assert live_key in scoped_host.cache(), "the in-use key must stay cached"
