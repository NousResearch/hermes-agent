"""Tests for inventory._apply_pricing — the pricing/tier enrichment that

feeds the desktop GUI model picker (and onboarding) so it can show $/Mtok
columns + Free/Pro badges and gate paid models on free Nous accounts, the
same way the `hermes model` CLI picker does.
"""

import hermes_cli.inventory as inv
import hermes_cli.models as models_mod


def _patch_pricing(monkeypatch, *, free_tier, pricing, unavailable=None):
    monkeypatch.setattr(models_mod, "get_pricing_for_provider", lambda slug, **kw: pricing.get(slug, {}))
    monkeypatch.setattr(models_mod, "check_nous_free_tier", lambda *, force_fresh=False: free_tier)
    monkeypatch.setattr(
        models_mod, "partition_nous_models_by_tier",
        lambda ids, pr, free_tier: (
            [m for m in ids if m not in (unavailable or [])],
            list(unavailable or []),
        ),
    )


def test_apply_pricing_formats_per_model_prices(monkeypatch):
    """Each model gets formatted input/output/cache + a free flag."""
    _patch_pricing(
        monkeypatch,
        free_tier=False,
        pricing={
            "openrouter": {
                "a/paid": {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003"},
                "b/free": {"prompt": "0", "completion": "0"},
            }
        },
    )
    rows = [{"slug": "openrouter", "models": ["a/paid", "b/free"]}]
    inv._apply_pricing(rows)

    pricing = rows[0]["pricing"]
    assert pricing["a/paid"] == {"input": "$3.00", "output": "$15.00", "cache": "$0.30", "free": False}
    assert pricing["b/free"]["free"] is True
    assert pricing["b/free"]["input"] == "free"


def test_apply_pricing_nous_free_tier_gates_paid_models(monkeypatch):
    """A free-tier Nous account marks paid models unavailable and sets the flag."""
    _patch_pricing(
        monkeypatch,
        free_tier=True,
        pricing={
            "nous": {
                "free/model": {"prompt": "0", "completion": "0"},
                "paid/model": {"prompt": "0.000005", "completion": "0.00001"},
            }
        },
        unavailable=["paid/model"],
    )
    rows = [{"slug": "nous", "models": ["free/model", "paid/model"]}]
    inv._apply_pricing(rows)

    assert rows[0]["free_tier"] is True
    assert rows[0]["unavailable_models"] == ["paid/model"]
    assert rows[0]["pricing"]["free/model"]["free"] is True


def test_apply_pricing_nous_paid_tier_no_gating(monkeypatch):
    """A paid Nous account gates nothing."""
    _patch_pricing(
        monkeypatch,
        free_tier=False,
        pricing={"nous": {"x/model": {"prompt": "0.000001", "completion": "0.000002"}}},
    )
    rows = [{"slug": "nous", "models": ["x/model"]}]
    inv._apply_pricing(rows)

    assert rows[0]["free_tier"] is False
    assert rows[0]["unavailable_models"] == []


def test_apply_pricing_skips_providers_without_pricing(monkeypatch):
    """A provider with no live pricing simply gets no pricing key."""
    _patch_pricing(monkeypatch, free_tier=False, pricing={})
    rows = [{"slug": "anthropic", "models": ["claude-x"]}]
    inv._apply_pricing(rows)

    assert "pricing" not in rows[0]


def test_apply_pricing_failure_is_swallowed(monkeypatch):
    """A pricing fetch that raises must not break the whole payload."""
    def boom(slug, **kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(models_mod, "get_pricing_for_provider", boom)
    rows = [{"slug": "openrouter", "models": ["a/b"]}]
    inv._apply_pricing(rows)  # must not raise

    assert "pricing" not in rows[0]
