"""The picker must not collapse to a profile's own stub when the live probe declines.

A plugin provider can carry BOTH a hand-curated ``_PROVIDER_MODELS`` row (the curated floor
upstream maintains) and a smaller ``fallback_models`` catalogue of its own. When the live probe yields
nothing — no credential, a 401, a timeout — the old code returned the profile's catalogue and
never reached the floor, so the picker showed 2-3 ids for providers that serve dozens. The
profile catalogue also shadowed the models.dev merge that ``_static_catalog`` performs for
``_MODELS_DEV_PREFERRED`` providers.

Neither curated catalogue may shadow the other: the floor leads, the profile's own catalogue
completes it, and nothing that was on offer before is dropped.

Regression coverage for that collapse.
"""

from unittest.mock import patch

from hermes_cli.chat_catalog import without_generation_models
from hermes_cli.models import provider_model_ids
from hermes_cli.models_catalog_static import _PROVIDER_MODELS
from providers import get_provider_profile


class TestCuratedCatalogPrecedence:
    def _no_credential(self):
        return patch(
            "hermes_cli.models._api_key_credentials", return_value=(None, None)
        )

    def test_premise_stub_is_smaller_than_curated(self):
        """zai carries a 3-id profile catalogue next to a much larger curated floor."""
        profile = get_provider_profile("zai")
        assert len(profile.fallback_models) < len(_PROVIDER_MODELS["zai"])

    def test_curated_floor_wins_over_profile_catalogue(self):
        with self._no_credential():
            out = provider_model_ids("zai")
        assert len(out) > len(get_provider_profile("zai").fallback_models)
        expected = without_generation_models(list(_PROVIDER_MODELS["zai"]))
        assert set(expected).issubset(set(out))

    def test_models_dev_merge_is_not_shadowed(self):
        """The static path merges models.dev; the profile catalogue used to bypass it entirely.

        The merge helper is patched rather than ``agent.models_dev.list_agentic_models``:
        importing ``agent.models_dev`` has a network side effect on some revisions, and this
        test is about WHICH path the picker takes, not about the registry itself.
        """
        with (
            self._no_credential(),
            patch(
                "hermes_cli.models._merge_with_models_dev",
                return_value=["glm-9.9-mdev"],
            ),
        ):
            out = provider_model_ids("zai")
        assert "glm-9.9-mdev" in out

    def test_failed_probe_with_credential_still_serves_curated(self):
        profile = get_provider_profile("zai")
        with (
            patch(
                "hermes_cli.models._api_key_credentials", return_value=("sk-test", None)
            ),
            patch.object(
                type(profile), "fetch_models", side_effect=RuntimeError("boom")
            ),
        ):
            out = provider_model_ids("zai")
        assert len(out) > len(profile.fallback_models)

    def test_profile_catalogue_still_served_when_no_static_row(self):
        """A provider with only its own catalogue keeps it — nothing better exists."""
        profile = get_provider_profile("upstage")
        assert "upstage" not in _PROVIDER_MODELS
        with self._no_credential():
            out = provider_model_ids("upstage")
        assert list(out) == list(profile.fallback_models)

    def test_nothing_offered_before_is_dropped(self):
        """The profile's own catalogue completes the floor; it never replaces it."""
        for provider in ("zai", "nvidia", "huggingface"):
            profile = get_provider_profile(provider)
            floor = list(_PROVIDER_MODELS[provider])
            own = list(profile.fallback_models or ())
            with self._no_credential():
                out = provider_model_ids(provider)
            for model_id in own:
                if model_id not in floor:
                    assert model_id in out, f"{provider}: {model_id} was dropped"
