"""Test that /model picker exposes MiniMax highspeed variants on direct API (#29142)."""

import os
from unittest.mock import patch

from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import provider_model_ids


_MINIMAX_REQUIRED = {
    "MiniMax-M2.7-highspeed",
    "MiniMax-M2.7",
    "MiniMax-M2.5-highspeed",
    "MiniMax-M2.5",
}


def test_minimax_curated_includes_highspeed_variants():
    """Both highspeed variants must be in the static curated list."""
    models = set(provider_model_ids("minimax"))
    missing = _MINIMAX_REQUIRED - models
    assert not missing, (
        f"minimax curated should include highspeed variants; missing: {sorted(missing)}. "
        f"Got: {provider_model_ids('minimax')}"
    )


@patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}, clear=False)
def test_minimax_picker_lists_highspeed_when_api_key_set():
    """When MINIMAX_API_KEY is set, the /model picker must list the highspeed variants."""
    # Mock models.dev so the test doesn't depend on network reachability; the
    # section-1 loop in list_authenticated_providers skips a provider when
    # ``data.get(mdev_id)`` isn't a dict, so an unreachable registry on CI
    # would otherwise drop minimax even when MINIMAX_API_KEY is set.
    fake_registry = {"minimax": {"name": "MiniMax", "env": ["MINIMAX_API_KEY"], "models": {}}}
    # ``max_models`` is set well beyond the curated list size so the assertion
    # never trips on truncation if the curated catalog grows.
    with patch("agent.models_dev.fetch_models_dev", return_value=fake_registry):
        providers = list_authenticated_providers(current_provider="openrouter", max_models=1000)

    minimax = next((p for p in providers if p["slug"] == "minimax"), None)
    assert minimax is not None, "minimax should appear when MINIMAX_API_KEY is set"

    present = set(minimax["models"])
    missing = _MINIMAX_REQUIRED - present
    assert not missing, (
        f"/model picker for minimax should include highspeed variants; "
        f"missing: {sorted(missing)}. Got: {minimax['models']}"
    )
