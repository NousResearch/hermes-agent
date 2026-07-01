"""M2 route-purity-gated reprice pass (SPEC §5C / INV-3,4,5,8,9).

Every case runs against the REAL store + REAL pricing engine under a temp
HERMES_HOME — no mocking the pricing path (AGENTS.md: E2E over green mocks).
"""
import importlib

import pytest


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import plugins.blackbox.store as store_mod
    importlib.reload(store_mod)
    return store_mod


@pytest.fixture
def real_pricing_fn():
    from plugins.blackbox.cost import compute_turn_cost

    def _fn(model, provider, tok):
        return compute_turn_cost(
            model,
            provider,
            None,
            [
                dict(
                    input_tokens=tok["input_tokens"],
                    output_tokens=tok["output_tokens"],
                    cache_read_tokens=tok["cache_read_tokens"],
                    cache_write_tokens=tok["cache_write_tokens"],
                )
            ],
        )

    return _fn


def _seed(store, turn_id, provider, model, i=0, o=0, cr=0, cw=0, **cost_cols):
    from plugins.blackbox.record import TurnRecord

    rec = TurnRecord(
        turn_id=turn_id,
        provider=provider,
        model=model,
        input_tokens=i,
        output_tokens=o,
        cache_read_tokens=cr,
        cache_write_tokens=cw,
        cost_usd=cost_cols.get("cost_usd"),
        cost_uncached_usd=cost_cols.get("cost_uncached_usd"),
        cost_cache_read_usd=cost_cols.get("cost_cache_read_usd"),
        cost_cache_write_usd=cost_cols.get("cost_cache_write_usd"),
        cost_output_usd=cost_cols.get("cost_output_usd"),
    )
    store.insert_turn(rec)


def test_real_token_now_priceable_row_is_repriced(store, real_pricing_fn):
    _seed(store, "t1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    res = store.reprice_unpriced(real_pricing_fn, apply=True)
    assert res == {"scanned": 1, "repriced": 1, "zeroed": 0, "still_unknown": 0}
    row = store.get_turn("t1")
    assert row["cost_usd"] is not None and row["cost_usd"] > 0
    assert row["cost_status"] == "estimated"


def test_zero_token_row_becomes_priced_zero_regardless_of_route(store, real_pricing_fn):
    # copilot-acp resolves to billing_mode 'unknown' — but a 0-token turn is $0
    # regardless of route, so it heals to priced_zero, never still_unknown.
    _seed(store, "z1", "copilot-acp", "claude-opus-4-8")
    res = store.reprice_unpriced(real_pricing_fn, apply=True)
    assert res == {"scanned": 1, "repriced": 0, "zeroed": 1, "still_unknown": 0}
    row = store.get_turn("z1")
    assert row["cost_usd"] == 0.0 and row["cost_status"] == "priced_zero"


def test_genuinely_unknown_real_token_row_stays_null(store, real_pricing_fn):
    _seed(store, "u1", "totally-unknown", "frobnicate-9", i=100, o=50)
    res = store.reprice_unpriced(real_pricing_fn, apply=True)
    assert res["still_unknown"] == 1 and res["repriced"] == 0
    assert store.get_turn("u1")["cost_usd"] is None


def test_live_catalog_route_left_null_route_purity(store, real_pricing_fn):
    # INV-9: an official_models_api (live-catalog) route must NOT be repriced
    # offline from summed tokens.
    _seed(store, "or1", "openrouter", "anthropic/claude-4.8-opus", i=1000, o=500)
    res = store.reprice_unpriced(real_pricing_fn, apply=True)
    assert res["still_unknown"] == 1 and res["repriced"] == 0
    assert store.get_turn("or1")["cost_usd"] is None


def test_nonlinear_request_cost_route_left_null(store, real_pricing_fn, monkeypatch):
    # INV-9/RC-B: a resolved entry carrying request_cost != None can't be
    # reconstructed from summed tokens → leave NULL.
    import plugins.blackbox.store as store_mod
    from agent.usage_pricing import PricingEntry
    from decimal import Decimal

    real_get = store_mod.get_pricing_entry

    def fake_get(model, provider=None, base_url=None, api_key=None):
        entry = real_get(model, provider=provider, base_url=base_url)
        if entry is None:
            return None
        # clone with a non-linear per-request charge
        return PricingEntry(
            input_cost_per_million=entry.input_cost_per_million,
            output_cost_per_million=entry.output_cost_per_million,
            cache_read_cost_per_million=entry.cache_read_cost_per_million,
            cache_write_cost_per_million=entry.cache_write_cost_per_million,
            request_cost=Decimal("0.01"),
            source=entry.source,
            pricing_version=entry.pricing_version,
        )

    monkeypatch.setattr(store_mod, "get_pricing_entry", fake_get)
    _seed(store, "nl1", "openai", "claude-opus-4-8", i=1000, o=500)
    res = store.reprice_unpriced(real_pricing_fn, apply=True)
    assert res["still_unknown"] == 1 and res["repriced"] == 0
    assert store.get_turn("nl1")["cost_usd"] is None


def test_priced_row_never_touched(store, real_pricing_fn):
    _seed(store, "p1", "anthropic", "claude-opus-4-8", i=100, o=50, cost_usd=0.42)
    before = store.get_turn("p1")["cost_usd"]
    store.reprice_unpriced(real_pricing_fn, apply=True)
    assert store.get_turn("p1")["cost_usd"] == before


@pytest.mark.parametrize(
    "col",
    ["cost_uncached_usd", "cost_cache_read_usd", "cost_cache_write_usd", "cost_output_usd"],
)
def test_partial_perclass_row_never_clobbered(store, real_pricing_fn, col):
    # RC-C: cost_usd NULL but ANY one per-class col set → skip (each independently).
    _seed(store, f"pc_{col}", "openai", "claude-opus-4-8", i=100, o=50, **{col: 0.01})
    store.reprice_unpriced(real_pricing_fn, apply=True)
    assert store.get_turn(f"pc_{col}")["cost_usd"] is None


def test_dry_run_writes_nothing(store, real_pricing_fn):
    _seed(store, "d1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    res = store.reprice_unpriced(real_pricing_fn, apply=False)
    assert res["repriced"] == 1  # would-be count
    assert store.get_turn("d1")["cost_usd"] is None  # but nothing written


def test_idempotent_second_run_zero(store, real_pricing_fn):
    _seed(store, "i1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    store.reprice_unpriced(real_pricing_fn, apply=True)
    res2 = store.reprice_unpriced(real_pricing_fn, apply=True)
    assert res2 == {"scanned": 0, "repriced": 0, "zeroed": 0, "still_unknown": 0}


def test_reprice_never_flips_alerted(store, real_pricing_fn):
    _seed(store, "a1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    store.reprice_unpriced(real_pricing_fn, apply=True)
    assert store.get_turn("a1")["alerted"] in (0, False)


def test_apply_writes_rollback_manifest(store, real_pricing_fn, tmp_path):
    import json
    from pathlib import Path

    _seed(store, "m1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    store.reprice_unpriced(real_pricing_fn, apply=True)
    manifests = list((Path(tmp_path) / "blackbox").glob("reprice-run-*.json"))
    assert manifests, "expected a reprice-run manifest"
    ids = json.loads(manifests[0].read_text())
    assert "m1" in ids


def test_no_manifest_when_nothing_changed(store, real_pricing_fn, tmp_path):
    # A dry-run, or an apply with no eligible rows, must NOT drop a manifest
    # (manifest is written only after rows actually commit).
    from pathlib import Path

    _seed(store, "d1", "openai", "claude-opus-4-8", i=1826, o=548, cr=282825)
    store.reprice_unpriced(real_pricing_fn, apply=False)  # dry-run
    store.reprice_unpriced(real_pricing_fn, apply=True)   # applies d1 → 1 manifest
    store.reprice_unpriced(real_pricing_fn, apply=True)   # nothing left → no new manifest
    manifests = list((Path(tmp_path) / "blackbox").glob("reprice-run-*.json"))
    assert len(manifests) == 1, f"expected exactly one manifest, got {len(manifests)}"
