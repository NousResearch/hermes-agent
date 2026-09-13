"""``discover_models: false`` must narrow CANONICAL_PROVIDERS picker rows (section 2b) to the
declared ``models:`` list instead of merely prepending it to the live/curated catalog (#107106)."""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.model_switch_providers import _lap_canonical_rows, _PickerBuild
from hermes_cli.models_catalog_static import ProviderEntry


def _run(user_providers):
    b = _PickerBuild(
        current_provider="", current_base_url="", current_model="", max_models=None,
        for_picker=True, force_fresh_nous_tier=False, probe_custom_providers=False,
        probe_current_custom_provider=False, refresh=False, excluded=set(), curated={})
    fake_cp_config = SimpleNamespace(api_key_env_vars=["FAKE_PROVIDER_API_KEY"], auth_type="api_key")
    with (
        patch("hermes_cli.models.CANONICAL_PROVIDERS", [ProviderEntry("fake-provider", "Fake Provider", "")]),
        patch("hermes_cli.auth.PROVIDER_REGISTRY", {"fake-provider": fake_cp_config}),
        patch("hermes_cli.model_switch_providers._live_or_curated_ids", return_value=["live-a", "live-b", "live-c"]),
        patch.dict("os.environ", {"FAKE_PROVIDER_API_KEY": "test-key"}),
    ):
        _lap_canonical_rows(b, user_providers)
    return next(row for row in b.results if row["slug"] == "fake-provider")


def test_discover_models_false_narrows_canonical_row_to_configured_models():
    row = _run({"fake-provider": {"discover_models": False, "models": {"curated-x": {}, "curated-y": {}}}})

    assert row["models"] == ["curated-x", "curated-y"]
    assert row["total_models"] == 2


def test_discover_models_true_leaves_live_catalog_unchanged():
    """Unaffected by the fix: with discovery enabled (the default), the row still reflects the
    live catalog, unlike section 1's builtin rows which prepend declared models."""
    row = _run({"fake-provider": {"models": {"curated-x": {}}}})

    assert row["models"] == ["live-a", "live-b", "live-c"]
    assert row["total_models"] == 3
