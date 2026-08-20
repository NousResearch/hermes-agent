"""Regression tests for Gemini/Google live model discovery (issue #73825,
corrected per review of #73952, then per review of #75306).

provider_model_ids("gemini") flows through the SHARED generic profile-based
merge in hermes_cli.models (get_provider_profile(...).fetch_models(...)),
the same mechanism every other simple api-key provider uses -- not a
separate, early-returning gemini-specific branch. GeminiProfile.fetch_models
(plugins/model-providers/gemini/__init__.py) points at Gemini's OpenAI-
compat /v1beta/openai subpath and strips the "models/" prefix its response
IDs carry (by calling the base class's HTTP-fetch mechanics via super());
the generic merge in hermes_cli.models then handles curated-list
preservation the same way it does for every other provider.

Note: an earlier revision of this fix added a separate gemini-specific
branch in hermes_cli.models that returned the live result directly,
bypassing the shared merge entirely -- so a partial/stale live catalog
would drop curated entries the merge is specifically designed to preserve.
That branch has been removed; the fix now lives entirely in the provider
profile's fetch_models() override, letting the existing shared machinery
do the rest.

An earlier revision before that also incorrectly claimed Gemini 3.x model
IDs (gemini-3.1-pro-preview, gemini-3-pro-preview, gemini-3.6-flash,
gemini-3.1-flash-lite-preview) were "fictional"/OpenRouter-only and removed
them from the curated list. That was wrong: plugins/model-providers/gemini/
__init__.py's own default_aux_model is gemini-3.6-flash, and
website/docs/guides/google-gemini.md documents these as genuine native
Gemini IDs. The curated list is untouched.
"""
from __future__ import annotations

from unittest.mock import patch

from providers import get_provider_profile
from providers.base import ProviderProfile

from hermes_cli.models import provider_model_ids, _PROVIDER_MODELS


class TestGeminiLiveModelDiscovery:
    def _mock_credentials(self, monkeypatch, api_key="AIzaFakeKey"):
        monkeypatch.setattr(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            lambda provider_id: {
                "provider": provider_id,
                "api_key": api_key,
                "base_url": "",
                "source": "GEMINI_API_KEY" if api_key else "",
            },
        )

    def _patch_base_fetch(self, raw_ids):
        """Patch the BASE ProviderProfile.fetch_models (what
        GeminiProfile.fetch_models's super() call invokes for the actual
        HTTP request), so GeminiProfile's own override logic -- the
        endpoint URL adjustment and the "models/" prefix stripping under
        test -- genuinely executes rather than being bypassed. Returns
        the mock itself so callers can assert on its call_args (e.g. the
        rewritten base_url), not just the return value flowing through."""
        return patch.object(ProviderProfile, "fetch_models", return_value=raw_ids)

    def test_live_fetch_strips_models_prefix(self, monkeypatch):
        """Regression: Gemini's OpenAI-compat endpoint returns IDs
        prefixed with 'models/' (e.g. 'models/gemini-2.5-pro'), matching
        the same normalization already applied at the existing Gemini
        validation call site (#12532). GeminiProfile.fetch_models must
        strip it before the picker ever sees the result."""
        self._mock_credentials(monkeypatch)

        with self._patch_base_fetch(
            ["models/gemini-2.5-pro", "models/gemini-2.5-flash"]
        ) as mock_fetch:
            result = provider_model_ids("gemini")

        assert "gemini-2.5-pro" in result
        assert "gemini-2.5-flash" in result
        assert not any(m.startswith("models/") for m in result), (
            f"Live-fetched IDs must have the 'models/' prefix stripped: {result}"
        )
        # Verify the endpoint rewrite independently of the return value:
        # GeminiProfile.fetch_models() must have called the base
        # implementation with the OpenAI-compat subpath (the surface
        # whose {"data": [{"id": ...}]} response shape the base
        # implementation's own parsing expects), not the native
        # /v1beta root the profile's base_url otherwise resolves to.
        assert mock_fetch.called, "GeminiProfile.fetch_models must call super().fetch_models()"
        called_base_url = mock_fetch.call_args.kwargs.get("base_url")
        assert called_base_url == "https://generativelanguage.googleapis.com/v1beta/openai", (
            f"GeminiProfile.fetch_models must rewrite base_url to the "
            f"OpenAI-compat subpath before calling super(): {called_base_url!r}"
        )

    def test_partial_live_catalog_preserves_curated_entries(self, monkeypatch):
        """Regression (review of #75306): the shared generic merge
        preserves curated entries when the live endpoint returns a
        PARTIAL catalog (stale cache, incomplete rollout) -- this is the
        existing contract every other provider gets, and Gemini must not
        bypass it via a separate early-returning branch. A live response
        containing just one model must not cause the curated 3.x entries
        to disappear from the picker."""
        self._mock_credentials(monkeypatch)

        with self._patch_base_fetch(["models/gemini-2.5-pro"]):
            result = provider_model_ids("gemini")

        curated = _PROVIDER_MODELS.get("gemini", [])
        assert curated, "expected a non-empty curated gemini list to test against"
        for model_id in curated:
            assert model_id in result, (
                f"Curated entry {model_id!r} must survive a partial live "
                f"catalog merge, not be dropped: {result}"
            )
        assert "gemini-2.5-pro" in result

    def test_falls_back_when_no_api_key(self, monkeypatch):
        """No credentials configured: falls through to the existing
        static/models.dev fallback chain."""
        self._mock_credentials(monkeypatch, api_key="")

        result = provider_model_ids("gemini")

        assert isinstance(result, list)
        assert result
        assert not any(isinstance(m, str) and m.startswith("models/") for m in result)

    def test_google_alias_normalizes_to_gemini_live_fetch(self, monkeypatch):
        """normalize_provider() canonicalizes 'google' to 'gemini' before
        provider_model_ids() runs, so passing either must hit the same
        path and receive normalized IDs."""
        self._mock_credentials(monkeypatch)

        with self._patch_base_fetch(["models/gemini-2.5-flash"]):
            result = provider_model_ids("google")

        assert "gemini-2.5-flash" in result
        assert not any(m.startswith("models/") for m in result)

    def test_no_source_ever_leaks_models_prefix(self, monkeypatch):
        """Picker/native-ID compatibility invariant: no matter which
        source provider_model_ids("gemini") ends up returning from (live
        fetch, static list, or models.dev merge), every ID in the result
        must be in the bare, native form the rest of the codebase
        expects -- never 'models/'-prefixed."""
        for api_key, fetch_result in (
            ("AIzaFakeKey", ["models/gemini-2.5-pro"]),
            ("", None),
        ):
            self._mock_credentials(monkeypatch, api_key=api_key)
            with self._patch_base_fetch(fetch_result):
                result = provider_model_ids("gemini")
            assert all(
                isinstance(m, str) and not m.startswith("models/") for m in result
            ), f"picker output leaked 'models/' prefix: {result}"

    def test_gemini_profile_still_registered_and_reachable(self):
        """Sanity: the generic get_provider_profile("gemini") lookup this
        whole mechanism depends on actually resolves to GeminiProfile."""
        profile = get_provider_profile("gemini")
        assert profile is not None
        assert profile.name == "gemini"
        assert profile.auth_type == "api_key"
