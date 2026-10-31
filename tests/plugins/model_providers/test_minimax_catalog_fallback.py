"""The MiniMax picker must offer the current MiniMax models, and a degraded catalog
fetch must never be pinned in the disk cache as the account's own catalog.

MiniMax publishes its model catalog at ``<base>/v1/models`` — the Anthropic-compatible
route answers there, with ``x-api-key`` or with Bearer — while the generic
``<base>/models`` probe the framework issues does not answer, and the failure is
swallowed, so every picker surface silently degraded to the curated floor and a newly
released model never appeared (#142648).

Declaring the catalog on the profile as ``fallback_models`` fixes both halves:

* the picker offers the current models when the live probe does not answer, and
* ``merge_profile_catalog`` wraps the fallback in ``CuratedFallbackModels``, which the
  disk cache treats as a PLACEHOLDER — it expires on the short fallback TTL, is never
  stale-served, and never replaces the account's real catalog. Without the marker the
  degraded list was indistinguishable from the account's live catalog: cached for a full
  hour and stale-served for the whole stale-while-revalidate window.

Verified 2026-10-26 against the live endpoint: ``GET https://api.minimax.io/anthropic/v1/models``
returns HTTP 200 and nine ids, ``MiniMax-M3.1-Flash`` first. The China route serves the
same catalog, and models.dev lists ``MiniMax-M3.1-Flash`` for both ``minimax`` and
``minimax-cn``.

Refs:
  - Issue #142648: MiniMax new models never surface in the picker
  - PR #141412: the curated catalog entry for ``MiniMax-M3.1-Flash``
  - PR #142892 / #144993: the live-catalog fetch (complementary — this covers the DEGRADED path)
  - PR #120052: "a curated fallback never pins itself in the provider models cache"
"""

from __future__ import annotations

import time

import pytest

# The catalog MiniMax serves, in the endpoint's own order. The ``-highspeed`` variants
# are not here: they are the 2x-priced tier the curated catalog deliberately leaves
# out, and the live catalog contributes them whenever it is reachable.
EXPECTED_CATALOG = (
    "MiniMax-M3.1-Flash", "MiniMax-M3",
    "MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-M2.1", "MiniMax-M2",
)


@pytest.fixture(params=["minimax", "minimax-cn"])
def minimax_profile(request):
    """Resolve the registered MiniMax profile for each API-key route.

    Going through ``providers.get_provider_profile`` keeps the test honest — if the
    registration is replaced by a plain ``ProviderProfile`` the assertions below collapse.
    """
    import providers

    providers._discover_providers()
    profile = providers.get_provider_profile(request.param)
    assert profile is not None, f"{request.param} provider profile must be registered"
    return profile, request.param


class TestMinimaxFallbackCatalog:
    """The profile declares the catalog the endpoint actually serves."""

    def test_fallback_catalog_is_the_current_catalog(self, minimax_profile):
        profile, provider_id = minimax_profile
        assert profile.fallback_models == EXPECTED_CATALOG, (
            f"{provider_id} fallback_models drifted from the catalog MiniMax serves — the picker "
            f"would offer {profile.fallback_models} instead"
        )

    def test_fallback_leads_with_the_current_flagship(self, minimax_profile):
        profile, provider_id = minimax_profile
        assert profile.fallback_models[0] == "MiniMax-M3.1-Flash", (
            f"{provider_id} must lead with the current flagship — a stale floor hides it"
        )

    def test_fallback_covers_the_curated_row(self, minimax_profile):
        """The fallback is a superset of the curated row, so the degraded path loses nothing."""
        from hermes_cli.models import static_catalog

        profile, provider_id = minimax_profile
        curated = set(static_catalog(provider_id))
        assert curated, f"{provider_id} must have a curated row for this comparison to mean anything"
        assert curated <= set(profile.fallback_models), (
            f"{provider_id} curated ids {sorted(curated - set(profile.fallback_models))} are missing "
            "from the fallback — the degraded path would drop them"
        )


class TestDegradedCatalogIsAPlaceholder:
    """A failed catalog fetch must not be cached as the account's own catalog."""

    def test_degraded_picker_serves_the_catalog_marked_as_a_placeholder(self, minimax_profile):
        from hermes_cli.models import merge_profile_catalog
        from hermes_cli.models_catalog_static import CuratedFallbackModels

        profile, provider_id = minimax_profile
        merged = merge_profile_catalog(provider_id, profile, None)

        assert merged is not None, f"{provider_id} picker must not be empty when the probe declines"
        assert list(merged) == list(EXPECTED_CATALOG)
        assert isinstance(merged, CuratedFallbackModels), (
            "the degraded list must be a CuratedFallbackModels — the disk cache treats any other "
            "list as the account's live catalog"
        )

    def test_degraded_catalog_is_not_pinned_in_the_disk_cache(self, minimax_profile):
        """The stored row must carry the fallback marker, so it expires instead of being served."""
        import hermes_cli.models as mod

        profile, provider_id = minimax_profile
        degraded = mod.merge_profile_catalog(provider_id, profile, None)
        assert degraded is not None

        row = mod._live_result_entry("fp", degraded, None)
        assert row is not None
        assert row.get("fallback") is True, (
            "a curated fallback stored without the fallback marker is served for the full "
            "catalog TTL and is stale-served for the whole stale-while-revalidate window"
        )

        now = time.time()
        aged = {**row, "at": now - 3600}
        assert mod._disk_serve_tier(aged, "fp", now, is_ollama=False) is None, (
            "a fallback row must never be served from disk"
        )

    def test_degraded_catalog_does_not_overwrite_a_live_row(self, minimax_profile):
        """The account's real catalog survives a degraded fetch."""
        import hermes_cli.models as mod

        profile, provider_id = minimax_profile
        degraded = mod.merge_profile_catalog(provider_id, profile, None)
        assert degraded is not None
        live = ["MiniMax-M3.1-Flash"]

        existing = mod._cache_entry("fp", live, time.time() - 60)
        assert mod._live_result_entry("fp", degraded, existing) is None, (
            "a curated fallback must never replace the account's live catalog for the same "
            "credentials"
        )
        assert isinstance(degraded, list)
