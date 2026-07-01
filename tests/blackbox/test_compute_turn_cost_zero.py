"""M3 zero-token turns are costless, not unpriced (SPEC §5D / INV-7)."""

from plugins.blackbox.cost import compute_turn_cost


def test_all_zero_token_call_is_priced_zero():
    calls = [
        dict(
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
    ]
    cost, status, perclass = compute_turn_cost("claude-opus-4-8", "openai", None, calls)
    assert cost == 0.0
    assert status == "priced_zero"
    assert perclass == {"uncached": 0.0, "cache_read": 0.0, "cache_write": 0.0, "output": 0.0}


def test_one_output_token_unknown_route_stays_unknown():
    # INV-7: any nonzero billed class disqualifies the zero-token shortcut; an
    # unpriceable route must stay honestly unknown.
    calls = [
        dict(
            input_tokens=0,
            output_tokens=1,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
    ]
    cost, status, perclass = compute_turn_cost("frobnicate-9", "totally-unknown", None, calls)
    assert cost is None
    assert status == "unknown"
    assert perclass == {"uncached": None, "cache_read": None, "cache_write": None, "output": None}


def test_no_calls_still_included():
    cost, status, perclass = compute_turn_cost("anything", "anything", None, [])
    assert cost == 0.0
    assert status == "included"
    assert perclass == {"uncached": None, "cache_read": None, "cache_write": None, "output": None}
