"""Endpoint ``/models`` pricing conventions, end to end.

An OpenAI-compatible ``/models`` document may quote token rates dollars-per-token
(OpenRouter-style) or dollars-per-million (flat, nested under ``global``, or with an
explicit ``unit``). The contract is that the pair

    ``_parse_models_payload`` -> ``_pricing_entry_from_metadata``

hands back the quoted dollars-per-million either way, applying the unit once.

Both halves are asserted through that real pair rather than a hand-built metadata
dict: ``_extract_pricing`` keeps only the keys whose spelling its alias map knows,
so a rate spelled differently is dropped before the pricing layer ever sees it.
``completions`` was such a spelling (#108642): the input rate priced, the output
rate came back unknown.
"""

from decimal import Decimal
from typing import Optional

import pytest

from agent.model_metadata import _parse_models_payload
from agent.usage_pricing import _pricing_entry_from_metadata

# Endpoint quotes round-trip through the per-token string contract as floats, so a
# quoted 2.90 comes back as 2.8999999999999998; the tolerance is for that, not for scale.
_RATE_TOLERANCE = 1e-9


def _rate(value: Optional[Decimal]) -> float:
    assert value is not None, "rate was not priced"
    return float(value)


def _entry(pricing):
    metadata = _parse_models_payload({"data": [{"id": "model-x", "pricing": pricing}]})
    return _pricing_entry_from_metadata(
        metadata, "model-x",
        source_url="https://g.example/v1/models", pricing_version="test",
    )


@pytest.mark.parametrize(
    "pricing, in_per_million, out_per_million",
    [
        # OpenRouter-style: the value is dollars per token.
        ({"prompt": "0.00000075", "completion": "0.00000375"}, 0.75, 3.75),
        # Dollars per million, flat.
        ({"prompt": 0.75, "completion": 3.00}, 0.75, 3.00),
        # Dollars per million nested under `global`, output under the `completions`
        # spelling - the shape reported in #108642.
        ({"global": {"prompt": 0.75, "completions": 3.00}}, 0.75, 3.00),
        # Explicit units are applied exactly once: a per-million quote must not be
        # re-scaled by the per-token consumer, nor left at 1e-6 of its quote.
        ({"prompt": "2.90", "completion": "10.00", "unit": "per_1m_tokens"}, 2.90, 10.00),
        ({"prompt": "0.002", "completion": "0.004", "unit": "per_1m_tokens"}, 0.002, 0.004),
        ({"prompt": "0.75", "completion": "3.00", "unit": "per_1k_tokens"}, 750.0, 3000.0),
        # An explicit unit outranks the magnitude fallback: these values are per-token,
        # and read as a per-million quote they would be $0.005/M instead of $5,000/M.
        ({"prompt": "0.005", "completion": "0.008", "unit": "per_token"}, 5000.0, 8000.0),
        # Nesting and an explicit unit together.
        ({"global": {"prompt": 0.75, "completions": 3.00, "unit": "per_1k_tokens"}}, 750.0, 3000.0),
    ],
)
def test_quoted_rate_survives_the_probe(pricing, in_per_million, out_per_million):
    entry = _entry(pricing)
    assert entry is not None
    assert _rate(entry.input_cost_per_million) == pytest.approx(in_per_million, rel=_RATE_TOLERANCE)
    assert _rate(entry.output_cost_per_million) == pytest.approx(out_per_million, rel=_RATE_TOLERANCE)


def test_nested_cache_rate_and_request_fee_survive_the_probe():
    """The nest is unwrapped for every rate field, and a flat per-request fee is not scaled."""
    entry = _entry({
        "global": {"prompt": 0.75, "completions": 3.75, "input_cache_read": 0.075},
        "regional_increase_percent": 0.1,
    })
    assert entry is not None
    assert _rate(entry.cache_read_cost_per_million) == pytest.approx(0.075, rel=_RATE_TOLERANCE)

    entry = _entry({"prompt": 0.75, "completion": 3.00, "request": 0.005})
    assert entry is not None
    assert _rate(entry.request_cost) == pytest.approx(0.005, rel=_RATE_TOLERANCE)


@pytest.mark.parametrize(
    "pricing",
    [
        {"prompt": "0.00000075", "completion": "0.00000375", "completions": "0.00000999"},
        {"completions": "0.00000999", "completion": "0.00000375", "prompt": "0.00000075"},
    ],
)
def test_completion_spelling_wins_over_completions(pricing):
    """`completions` is a fallback spelling; an endpoint sending both means `completion`."""
    entry = _entry(pricing)
    assert entry is not None
    assert _rate(entry.output_cost_per_million) == pytest.approx(3.75, rel=_RATE_TOLERANCE)
