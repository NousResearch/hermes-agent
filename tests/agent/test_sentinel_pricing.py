"""A negative or non-finite catalog rate is not a price.

OpenRouter's router sentinels are quoted as ``"-1"`` per token in the models catalog
(``-1,000,000`` per million after the catalog's own scale). Multiplying that as money
produced a negative estimate with ``status='estimated'`` — a fabricated number presented
as a price. A negative rate at any scale now refuses the WHOLE entry, so the route reads
``unknown`` instead; a genuine ``$0`` request fee and a missing cache key still price.

Synthetic catalogs only — no network, no provider I/O.
"""

from __future__ import annotations

import pytest

from agent.usage_pricing import (
    CanonicalUsage, _pricing_entry_from_metadata, estimate_usage_cost,
)

_SOURCE = dict(source_url="x", pricing_version="openrouter-models-api")


def _entry(pricing):
    meta = {"m": {"pricing": pricing}}
    return _pricing_entry_from_metadata(meta, "m", **_SOURCE)


@pytest.mark.parametrize(
    "pricing",
    [
        {"prompt": "-1", "completion": "-1"},          # openrouter/auto's real catalog shape
        {"prompt": "-0.000001", "completion": "0.000002"},
        {"prompt": "0.000001", "completion": "-1"},
        {"prompt": "0.000001", "request": "-0.01"},     # raw dollars, not per-million
        {"prompt": "0.000001", "cache_read": "-1"},
        {"prompt": "0.000001", "cache_write": "-1"},
        {"prompt": "0.000001", "cached_prompt": "-1"},  # alias chain
    ],
)
def test_negative_rate_refuses_the_whole_entry(pricing):
    assert _entry(pricing) is None


@pytest.mark.parametrize(
    "pricing",
    [
        {"prompt": "0", "completion": "0.000002", "request": "0"},  # legitimate $0 request fee
        {"prompt": "0.000001", "completion": "0.000002"},           # missing cache keys
        {"prompt": "0.000001", "cache_read": "0", "cache_write": "0"},
    ],
)
def test_zero_rate_and_missing_keys_still_price(pricing):
    entry = _entry(pricing)
    assert entry is not None
    assert entry.input_cost_per_million is not None
    assert float(entry.input_cost_per_million) == pytest.approx(float(pricing["prompt"]) * 1_000_000)


def test_sentinel_route_yields_no_negative_amount_and_no_estimate(monkeypatch):
    """The reproduced bug: ``estimate_usage_cost('openrouter/auto', ...)`` returned a
    negative dollar amount with ``status='estimated'``."""
    from agent import usage_pricing as up

    catalog = {"openrouter/auto": {"pricing": {"prompt": "-1", "completion": "-1"}}}
    monkeypatch.setattr(up, "fetch_model_metadata", lambda *a, **k: catalog)

    result = estimate_usage_cost(
        "openrouter/auto", CanonicalUsage(input_tokens=1000, output_tokens=100), provider="openrouter",
    )
    assert result.amount_usd is None
    assert result.status != "estimated"
    assert result.status == "unknown"


def test_priced_route_through_the_same_seam_still_estimates(monkeypatch):
    """Control: the refusal must not disarm estimation for routes that do carry a price."""
    from agent import usage_pricing as up

    catalog = {"openrouter/auto": {"pricing": {"prompt": "0.000001", "completion": "0.000002"}}}
    monkeypatch.setattr(up, "fetch_model_metadata", lambda *a, **k: catalog)

    result = estimate_usage_cost(
        "openrouter/auto", CanonicalUsage(input_tokens=1000, output_tokens=100), provider="openrouter",
    )
    assert result.amount_usd is not None
    assert result.amount_usd > 0
    assert result.status == "estimated"


def test_non_finite_rates_refuse_the_entry():
    """A NaN or infinite catalog rate is not a price either.

    A NaN rate used to survive into ``estimate_usage_cost`` and raise
    ``InvalidOperation`` out of ``format_cost_label`` into the turn; an
    ``-Infinity`` rate produced a ``~$-Infinity`` label. Both must read as
    'unknown', the same as the negative sentinels.
    """
    import agent.usage_pricing as up

    for rate in ("NaN", "Infinity", "-Infinity", "sNaN"):
        entry = up._pricing_entry_from_metadata(
            {"m": {"pricing": {"prompt": rate}}}, "m",
            source_url="test", pricing_version="test",
        )
        assert entry is None, f"{rate} rate must refuse the entry"


def test_zero_and_negative_zero_rates_still_price():
    """Zero is a real price; -0.0 is the same zero and must not be refused."""
    import agent.usage_pricing as up

    for rate in ("0", "0.0", "-0.0"):
        entry = up._pricing_entry_from_metadata(
            {"m": {"pricing": {"prompt": rate}}}, "m",
            source_url="test", pricing_version="test",
        )
        assert entry is not None, f"{rate} is a zero price, not a sentinel"
        assert entry.input_cost_per_million == 0
