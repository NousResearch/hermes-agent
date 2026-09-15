"""Follow-up S: entitlement is independent of recorded availability."""
from unittest.mock import Mock

import pytest

from hermes_cli import inventory


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("tier", [None, True, False])
@pytest.mark.parametrize("priced", [False, True])
@pytest.mark.parametrize("status", ["exhausted", "dead"])
def test_recorded_nous_entitlement(monkeypatch, refresh, tier, priced, status):
    from hermes_cli import models, models_pricing
    from hermes_cli.picker_state import recorded_pool_state
    from hermes_cli import auth
    monkeypatch.setattr(auth, "read_credential_pool", lambda slug: [{
        "access_token": "synthetic", "last_status": status,
        "last_error_reset_at": 4102444800}])
    warning_state = recorded_pool_state("nous")
    assert warning_state["warning"]
    row = {"slug": "nous", "models": ["free-model", "paid-model"], **warning_state}
    monkeypatch.setattr(models, "get_cached_nous_free_tier", lambda: tier)
    live = Mock(side_effect=AssertionError("no live entitlement"))
    monkeypatch.setattr(models, "check_nous_free_tier", live)
    def cached_prices(slug, **kwargs):
        assert slug == "nous"
        assert kwargs == {"cached_only": True}
        return {"free-model": {"prompt": "0", "completion": "0"},
                "paid-model": {"prompt": "0.000001", "completion": "0.000002"}} if priced else {}
    prices = Mock(side_effect=cached_prices)
    monkeypatch.setattr(models_pricing, "get_pricing_for_provider", prices)
    inventory._apply_pricing([row], cached_only=not refresh, force_fresh_nous_tier=refresh)
    live.assert_not_called()
    prices.assert_called_once()
    assert row["warning"] == warning_state["warning"]
    assert row.get("free_tier_pending", False) is (tier is None)
    expected = row["models"] if tier is None or (tier and not priced) else (["paid-model"] if tier else [])
    assert row["unavailable_models"] == expected
    if tier is not None:
        assert row["free_tier"] is tier
