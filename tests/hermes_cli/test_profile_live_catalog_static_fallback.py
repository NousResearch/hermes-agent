"""The curated catalog must lead over a provider's stale ``fallback_models``.

``_profile_live_catalog`` falls back to ``profile.fallback_models`` when the live endpoint yields
nothing (no credentials, offline, or a fetch past the timeout). That tuple exists for plugin
providers with no curated entry, and it goes stale silently: ``zai``'s still said
glm-5.2/glm-5/glm-4-9b long after glm-5.3 shipped, so one failed fetch hid every current GLM from
the picker — and because the tuple is non-empty, ``cached_provider_model_ids`` then served it as a
real catalog for its full TTL.

The maintained ``_PROVIDER_MODELS`` entry is authoritative and correctly ordered, so it leads; the
profile tuple stays the last resort for slugs the curated map does not know.
"""

from unittest.mock import patch

from hermes_cli.models import _PROVIDER_MODELS, _profile_live_catalog


def _dead_endpoint():
    """Force the 'live fetch returned nothing' branch: no credentials resolve."""
    return patch("hermes_cli.models._api_key_credentials", return_value=(None, None))


def test_curated_catalog_leads_when_no_live_catalog_is_available():
    curated = _PROVIDER_MODELS["zai"]
    assert curated and len(curated) > 3, "fixture drift: zai needs a curated entry to be meaningful"

    with _dead_endpoint():
        models = _profile_live_catalog("zai")

    assert models == list(curated)
    assert models[0] == curated[0], "curated order is authoritative — its newest model leads"


def test_provider_without_a_curated_entry_still_uses_its_profile_fallback():
    class _Profile:
        auth_type = "api_key"
        base_url = "https://fallback.example/v1"
        fallback_models = ("plugin-a", "plugin-b")

    with (
        _dead_endpoint(),
        patch("providers.get_provider_profile", return_value=_Profile()),
    ):
        models = _profile_live_catalog("not-in-the-curated-map")

    assert models == ["plugin-a", "plugin-b"]


def test_no_catalog_at_all_returns_none():
    class _Profile:
        auth_type = "api_key"
        base_url = "https://empty.example/v1"
        fallback_models = ()

    with (
        _dead_endpoint(),
        patch("providers.get_provider_profile", return_value=_Profile()),
    ):
        assert _profile_live_catalog("not-in-the-curated-map") is None
