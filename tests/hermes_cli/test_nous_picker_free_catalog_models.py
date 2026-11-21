"""Free-tier Nous pickers surface every zero-priced chat model the gateway serves, not only
the Portal's recommendations.

The Portal's ``freeRecommendedModels`` is a curation: verified against the live gateway it
recommended 7 of the 9 zero-priced rows ``GET /v1/models`` publishes. A model served at $0 but
not recommended (``inclusionai/ling-3.1-flash``) is absent from the curated manifest AND from the
recommendation union, so it never entered the candidate set — and ``partition_nous_models_by_tier``
only reorders ids already present, so no downstream filter could reintroduce it.
"""

from __future__ import annotations

import hermes_cli.models_pricing as mp
from hermes_cli import model_switch_providers as msp
from hermes_cli.models import union_with_nous_free_catalog_models

_PAID = {"prompt": "0.000002", "completion": "0.00001"}
_ZERO = {"prompt": "0", "completion": "0"}


def test_appends_zero_priced_models_missing_from_list():
    pricing = {"curated/a": dict(_PAID), "vendor/free": dict(_ZERO), "vendor/paid": dict(_PAID)}

    ids = union_with_nous_free_catalog_models(["curated/a"], pricing)

    assert ids == ["curated/a", "vendor/free"]


def test_curated_order_preserved_and_free_rows_not_duplicated():
    pricing = {mid: dict(_ZERO) for mid in ("vendor/b", "vendor/a", "curated/a")}

    ids = union_with_nous_free_catalog_models(["curated/a"], pricing)

    assert ids[0] == "curated/a"
    assert ids.count("curated/a") == 1
    assert ids[1:] == ["vendor/b", "vendor/a"]  # catalog order, not re-sorted


def test_only_zero_in_both_directions_counts_as_free():
    """A model free to prompt but paid to complete still costs money; neither may leak in."""
    pricing = {
        "half/free": {"prompt": "0", "completion": "0.000002"},
        "free/half": {"prompt": "0.000002", "completion": "0"},
        "malformed/blank": {"prompt": "", "completion": ""},
        "truly/free": dict(_ZERO),
    }
    assert union_with_nous_free_catalog_models([], pricing) == ["truly/free"]


def test_skips_tool_less_and_generation_rows():
    """Same exclusion rule as the on-sale union: Hermes is tool-calling-first and chat-only."""
    pricing = {
        "free/no-tools": {**_ZERO, "tools": False},
        "free/typed": {**_ZERO, "generation": True},
        "free/chat": dict(_ZERO),
    }
    assert union_with_nous_free_catalog_models([], pricing) == ["free/chat"]


def test_subscription_billed_row_is_free_for_a_free_account():
    """``_is_model_free`` treats a subscription-billed row as free — same rule the tier split uses."""
    pricing = {"vendor/plan": {**_ZERO, "billing_mode": "subscription"}}
    assert union_with_nous_free_catalog_models([], pricing) == ["vendor/plan"]


def test_no_pricing_leaves_list_unchanged():
    """A cold cache must never invent models."""
    assert union_with_nous_free_catalog_models(["a", "b"], {}) == ["a", "b"]
    assert union_with_nous_free_catalog_models(["a", "b"], None) == ["a", "b"]


def _stub_portal(monkeypatch, *, free_tier: bool, pricing: dict, allowed=None) -> None:
    monkeypatch.setattr(mp, "get_pricing_for_provider", lambda provider, **kw: pricing)
    monkeypatch.setattr("hermes_cli.models.check_nous_free_tier", lambda **kw: free_tier)
    monkeypatch.setattr("hermes_cli.models.fetch_nous_recommended_models", lambda *a, **kw: None)
    monkeypatch.setattr(mp, "nous_policy_allowed_ids", lambda **kw: allowed)


def test_picker_free_tier_lists_unrecommended_zero_priced_model(monkeypatch):
    """The reported bug: a $0 model the Portal does not recommend was unreachable in the picker."""
    pricing = {"curated/only": dict(_PAID), "vendor/free": dict(_ZERO)}
    _stub_portal(monkeypatch, free_tier=True, pricing=pricing)
    assert msp._nous_picker_model_ids({"nous": ["curated/only"]}, False) == ["curated/only", "vendor/free"]


def test_picker_free_tier_still_hides_paid_models(monkeypatch):
    pricing = {"curated/only": dict(_PAID), "vendor/paid": dict(_PAID)}
    _stub_portal(monkeypatch, free_tier=True, pricing=pricing)
    assert "vendor/paid" not in msp._nous_picker_model_ids({"nous": ["curated/only"]}, False)


def test_picker_org_policy_still_narrows_free_catalog_models(monkeypatch):
    """The gateway catalog is not a policy bypass — org policy still wins."""
    pricing = {"curated/only": dict(_ZERO), "vendor/blocked": dict(_ZERO)}
    _stub_portal(monkeypatch, free_tier=True, pricing=pricing, allowed={"curated/only"})
    assert msp._nous_picker_model_ids({"nous": ["curated/only"]}, False) == ["curated/only"]


def test_picker_paid_tier_unchanged(monkeypatch):
    """The paid branch keys off on-sale rows, not zero-priced ones; leave it alone."""
    pricing = {"curated/only": dict(_PAID), "vendor/paid": dict(_PAID)}
    _stub_portal(monkeypatch, free_tier=False, pricing=pricing)
    assert "vendor/paid" not in msp._nous_picker_model_ids({"nous": ["curated/only"]}, False)
