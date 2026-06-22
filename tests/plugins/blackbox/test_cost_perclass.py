"""SPEC-C Phase 2 — compute_turn_cost returns a per-class $ breakdown.

The plugin sums the engine's per-class terms across calls and returns a third
element: a dict {uncached, cache_read, cache_write, output} of floats (or None).
partial/unknown turns get an all-None dict (D-9 / INV-5) — the split is
deliberately withheld so the hover never presents a known-calls-only sum as the
whole truth.
"""
from __future__ import annotations

import plugins.blackbox.cost as cost_mod
from agent.usage_pricing import CostResult
from decimal import Decimal
from plugins.blackbox.cost import compute_turn_cost


_PERCLASS_KEYS = {"uncached", "cache_read", "cache_write", "output"}


def _opus_call(**tok):
    base = dict(input_tokens=0, output_tokens=0,
                cache_read_tokens=0, cache_write_tokens=0, reasoning_tokens=0)
    base.update(tok)
    return base


def test_returns_three_tuple_with_perclass_dict():
    calls = [_opus_call(output_tokens=130_000, cache_read_tokens=109_700_000,
                        cache_write_tokens=646_000)]
    total, status, perclass = compute_turn_cost(
        "claude-opus-4-8", "claude-api-proxy", None, calls)
    assert status == "estimated"
    assert isinstance(perclass, dict)
    assert set(perclass) == _PERCLASS_KEYS
    s = sum(perclass[k] for k in _PERCLASS_KEYS)
    assert abs(s - total) < 1e-9


def test_perclass_sums_across_multiple_calls():
    calls = [
        _opus_call(input_tokens=1000, output_tokens=500,
                   cache_read_tokens=2_000_000, cache_write_tokens=10_000),
        _opus_call(input_tokens=3000, output_tokens=20_000,
                   cache_read_tokens=500_000),
    ]
    total, status, perclass = compute_turn_cost(
        "claude-opus-4-8", "claude-api-proxy", None, calls)
    assert status == "estimated"
    s = sum(perclass[k] for k in _PERCLASS_KEYS)
    assert abs(s - total) < 1e-9
    assert perclass["cache_read"] > perclass["output"]


def test_fully_unknown_turn_returns_none_split():
    total, status, perclass = compute_turn_cost(
        "totally-unknown-xyz", "mystery", None,
        [_opus_call(input_tokens=5000, output_tokens=2000)])
    assert status == "unknown"
    assert total is None
    assert set(perclass) == _PERCLASS_KEYS
    assert all(perclass[k] is None for k in _PERCLASS_KEYS)


def test_partial_turn_withholds_split(monkeypatch):
    # Force a genuine partial: first call prices, second returns unknown. The
    # turn total is the known-calls sum (real, non-None) but the per-class split
    # is deliberately all-None — a split that omits the unknown call would be a
    # hover that lies (D-9 / INV-2).
    priced = CostResult(
        amount_usd=Decimal("1.50"), status="estimated", source="official_docs_snapshot",
        label="~$1.50", cost_input_usd=Decimal("0.50"), cost_output_usd=Decimal("0.50"),
        cost_cache_read_usd=Decimal("0.30"), cost_cache_write_usd=Decimal("0.20"))
    unknown = CostResult(amount_usd=None, status="unknown", source="none", label="n/a")
    seq = iter([priced, unknown])
    monkeypatch.setattr(cost_mod, "estimate_usage_cost", lambda *a, **k: next(seq))

    total, status, perclass = compute_turn_cost(
        "claude-opus-4-8", "claude-api-proxy", None,
        [_opus_call(output_tokens=1), _opus_call(output_tokens=1)])
    assert status == "partial"
    assert total == 1.50  # known-calls-only total, untouched
    assert set(perclass) == _PERCLASS_KEYS
    assert all(perclass[k] is None for k in _PERCLASS_KEYS)


def test_empty_calls_returns_included_with_none_split():
    total, status, perclass = compute_turn_cost("m", "p", None, [])
    assert (total, status) == (0.0, "included")
    assert set(perclass) == _PERCLASS_KEYS
    assert all(perclass[k] is None for k in _PERCLASS_KEYS)
