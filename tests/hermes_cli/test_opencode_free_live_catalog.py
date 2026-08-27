"""Regression tests for #95914 — keyless opencode-free catalog live revalidation.

The opencode-free (keyless) model catalog used to be served exclusively from a
hardcoded in-repo snapshot (_PROVIDER_MODELS["opencode-free"]). When the OpenCode
Zen relay delisted a free model (e.g. x-preview-f-free, 2026-08-26), the picker
kept offering it and selecting it failed with a non-retryable HTTP 401
("Model x-preview-f-free is not supported"). The SWR disk cache only refreshed
AUTHED providers (its entries were keyed by a credential fingerprint, which
keyless providers have none of), so the keyless catalog never revalidated.

The fix makes provider_model_ids("opencode-free") revalidate LIVE against
GET /zen/v1/models (anonymous, filtered to the free tier) and gives the keyless
provider a stable disk-cache fingerprint so the picker's SWR path serves stale
immediately while refreshing off-thread — the same behavior authed providers
already get.

These tests PROVE the fix: reverting the live-fetch wiring makes the
delisted-model / newly-live-model assertions fail (the catalog reverts to the
static snapshot only).
"""

from unittest.mock import patch

from hermes_cli.models import (
    _KEYLESS_STABLE_CACHE_PROVIDERS,
    _PROVIDER_MODELS,
    cached_provider_model_ids,
    provider_model_ids,
)

# Static floor (may lag the live relay by design — it is only the offline fallback).
_STATIC_FLOOR = list(_PROVIDER_MODELS["opencode-free"])

# The live relay's current free tier. x-preview-f-free was DELISTED 2026-08-26;
# deepseek-v4-flash-free + mimo-v2.5-free are back on the live list.
_LIVE_FREE_MODELS = [
    "deepseek-v4-flash-free",
    "hy3-free",
    "mimo-v2.5-free",
    "laguna-s-2.1-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "muse-spark-1.2-contributor-free",
]

# The raw live /zen/v1/models dump also lists paid/subscription + KEYED-free IDs
# (e.g. ox-alpha-free is a Go-subscription model despite the suffix). The filter
# must keep only anonymous-servable free models.
_LIVE_RAW_IDS = _LIVE_FREE_MODELS + [
    "claude-sonnet-5",          # paid
    "gpt-5.6-sol",              # paid
    "ox-alpha-free",            # KEYED Go-subscription (suffix looks free)
]


class TestProviderModelIdsOpencodeFree:
    def test_live_catalog_revalidation_excludes_delisted(self):
        """The delisted model must NOT appear when the live relay no longer lists it."""
        with patch(
            "hermes_cli.models._fetch_opencode_free_models",
            return_value=list(_LIVE_FREE_MODELS),
        ):
            result = provider_model_ids("opencode-free")

        assert "x-preview-f-free" not in result  # delisted — REVERT-PROOF
        assert "deepseek-v4-flash-free" in result  # newly-live — REVERT-PROOF
        assert "mimo-v2.5-free" in result  # newly-live — REVERT-PROOF

    def test_live_catalog_filters_out_keyed_free_suffix_model(self):
        """ox-alpha-free (KEYED Go-subscription) must never enter the keyless picker."""
        with patch(
            "hermes_cli.models._fetch_opencode_free_models",
            return_value=list(_LIVE_FREE_MODELS),
        ):
            result = provider_model_ids("opencode-free")
        assert "ox-alpha-free" not in result

    def test_falls_back_to_static_floor_when_live_fetch_fails(self):
        """On live-fetch failure/empty, the static floor keeps the picker populated."""
        with patch("hermes_cli.models._fetch_opencode_free_models", return_value=None):
            result = provider_model_ids("opencode-free")
        assert result == _STATIC_FLOOR
        assert result  # never empty on a transient outage

    def test_empty_live_result_falls_back_to_static_floor(self):
        """An empty live result (no free models) is not trusted over the floor."""
        with patch("hermes_cli.models._fetch_opencode_free_models", return_value=[]):
            result = provider_model_ids("opencode-free")
        assert result == _STATIC_FLOOR


class TestOpencodeFreeCacheFingerprint:
    def test_keyless_provider_has_stable_fingerprint(self):
        """opencode-free is in the stable-fingerprint set (no credential to rotate)."""
        assert "opencode-free" in _KEYLESS_STABLE_CACHE_PROVIDERS

        import hermes_cli.models as mod

        fp1 = mod._credential_fingerprint("opencode-free")
        fp2 = mod._credential_fingerprint("opencode-free")
        assert fp1 == fp2
        assert fp1.startswith("keyless:opencode-free")

    def test_cached_picker_path_revalidates_live(self):
        """cached_provider_model_ids('opencode-free') serves the live catalog and
        persists it under a stable fingerprint (SWR cache path the picker uses)."""
        import time

        import hermes_cli.models as mod

        with (
            patch.object(mod, "_load_provider_models_cache", return_value={}),
            patch.object(
                mod,
                "_fetch_opencode_free_models",
                return_value=list(_LIVE_FREE_MODELS),
            ) as fetch,
            patch.object(mod, "_save_provider_models_cache") as save,
        ):
            out = cached_provider_model_ids("opencode-free")

        assert out == list(_LIVE_FREE_MODELS)
        fetch.assert_called_once()
        # The persisted entry carries the stable keyless fingerprint so future
        # SWR lookups match and don't re-fetch on unrelated auth changes.
        written = save.call_args[0][0]
        entry = written["opencode-free"]
        assert entry["fp"].startswith("keyless:opencode-free")
        assert isinstance(entry["at"], float) and not isinstance(entry["at"], bool)
