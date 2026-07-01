"""INV-6 / RC-2: the pricing engine is flat-token LINEAR, so repricing M2's
SUMMED token columns == the original per-call pricing. This guard proves that
property today; if a future rate table adds a per-call minimum/tier/threshold,
it goes RED — and M2's route-purity gate must already refuse a route whose
entry carries a non-linear ``request_cost`` (see test_reprice_unpriced.py::
test_nonlinear_request_cost_route_left_null).
"""
from decimal import Decimal

from agent.usage_pricing import CanonicalUsage, estimate_usage_cost


def _price(usage):
    return estimate_usage_cost("claude-opus-4-8", usage, provider="anthropic").amount_usd


def test_sum_of_percall_equals_aggregate_for_flat_rates():
    total = CanonicalUsage(
        input_tokens=300000,
        output_tokens=60000,
        cache_read_tokens=280000,
        cache_write_tokens=0,
    )
    agg = _price(total)
    chunks = [
        CanonicalUsage(input_tokens=100000, output_tokens=20000, cache_read_tokens=90000, cache_write_tokens=0),
        CanonicalUsage(input_tokens=150000, output_tokens=25000, cache_read_tokens=100000, cache_write_tokens=0),
        CanonicalUsage(input_tokens=50000, output_tokens=15000, cache_read_tokens=90000, cache_write_tokens=0),
    ]
    per_call_sum = sum((_price(c) for c in chunks), Decimal("0"))
    assert abs(agg - per_call_sum) < Decimal("0.000001")


def test_single_call_equals_itself_trivially():
    u = CanonicalUsage(input_tokens=1826, output_tokens=548, cache_read_tokens=282825, cache_write_tokens=0)
    assert _price(u) == _price(u)
