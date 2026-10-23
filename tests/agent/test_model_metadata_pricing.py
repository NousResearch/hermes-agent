"""Endpoint pricing extraction for OpenAI-compatible ``/models`` catalogs
(``agent/model_metadata_pricing.py``): per-token / per-million quoting styles
and provider-specific field families."""

import pytest


def test_endpoint_pricing_requesty_field_family_resolved():
    """#136583: Requesty-style routing gateways quote USD per token under input_price/output_price/cached_price
    with no pricing dict at all. The generic alias map must resolve that family, otherwise every model on the
    route reports cost_status='unknown' and insights underreports spend to $0 while the gateway invoices it."""
    from agent import model_metadata as mm
    from agent import usage_pricing as up

    model = {
        "id": "deepseek/deepseek-v4.1-flash",
        "input_price": 0.0000006, "output_price": 0.0000022, "cached_price": 0.00000006,
    }
    meta = {"deepseek/deepseek-v4.1-flash": mm._endpoint_model_entry(model, "deepseek/deepseek-v4.1-flash", None)}
    entry = up._pricing_entry_from_metadata(
        meta, "deepseek/deepseek-v4.1-flash", source_url="x", pricing_version="openai-compatible-models-api")
    assert float(entry.input_cost_per_million) == pytest.approx(0.6)
    assert float(entry.output_cost_per_million) == pytest.approx(2.2)
    assert float(entry.cache_read_cost_per_million) == pytest.approx(0.06)


def test_endpoint_pricing_requesty_per_million_quotes_not_inflated():
    """Control: a gateway quoting input_price/output_price in $/M rides the existing unit heuristic
    and reaches usage_pricing as per-token rates, the same way the pricing-dict branch does."""
    from agent import model_metadata as mm
    from agent import usage_pricing as up

    model = {"id": "m", "input_price": 0.6, "output_price": 1.2}
    meta = {"m": mm._endpoint_model_entry(model, "m", None)}
    entry = up._pricing_entry_from_metadata(meta, "m", source_url="x", pricing_version="openai-compatible-models-api")
    assert float(entry.input_cost_per_million) == pytest.approx(0.6)
    assert float(entry.output_cost_per_million) == pytest.approx(1.2)


def test_endpoint_pricing_novita_family_still_resolved_after_split():
    """Regression: the Novita branch (input_token_price_per_m, 1/10_000-$ per M tokens) must keep its
    exact scaling after the extractor moved to agent/model_metadata_pricing.py."""
    from agent import model_metadata as mm
    from agent import usage_pricing as up

    model = {"id": "m", "input_token_price_per_m": 60.0, "output_token_price_per_m": 120.0}
    meta = {"m": mm._endpoint_model_entry(model, "m", None)}
    entry = up._pricing_entry_from_metadata(meta, "m", source_url="x", pricing_version="openai-compatible-models-api")
    assert float(entry.input_cost_per_million) == pytest.approx(0.006)
    assert float(entry.output_cost_per_million) == pytest.approx(0.012)
