from agent.usage_pricing import CanonicalUsage


def test_consume_reference_details():
    from agent.moa_loop import MoAChatCompletions
    from agent.moa_loop import _RefAccounting

    facade = MoAChatCompletions("closed")

    acct1 = _RefAccounting(
        provider="p1", model="m1", usage=CanonicalUsage(input_tokens=10), cost_usd=0.1
    )
    acct2 = _RefAccounting(
        provider="p2", model="m2", usage=CanonicalUsage(input_tokens=20), cost_usd=0.2
    )

    # Simulate a run
    facade._pending_reference_usage = CanonicalUsage(input_tokens=30)
    facade._pending_reference_cost = 0.3
    facade._pending_reference_details = [
        {
            "role": "reference",
            "slot_index": 0,
            "provider": "p1",
            "model": "m1",
            "usage": acct1.usage,
            "cost_usd": 0.1,
        },
        {
            "role": "reference",
            "slot_index": 1,
            "provider": "p2",
            "model": "m2",
            "usage": acct2.usage,
            "cost_usd": 0.2,
        },
    ]

    usage, cost, details = facade.consume_reference_details()
    assert usage.input_tokens == 30
    assert cost == 0.3
    assert len(details) == 2
    assert details[0]["provider"] == "p1"
    assert details[0]["slot_index"] == 0

    # State should be cleared
    assert facade._pending_reference_usage.input_tokens == 0
    assert facade._pending_reference_cost is None
    assert facade._pending_reference_details == []


def test_moaclient_consume_reference_details_delegates_and_preserves_preset():
    from agent.moa_loop import MoAClient

    client = MoAClient("code-hard")
    client.chat.completions._pending_reference_usage = CanonicalUsage(input_tokens=12)
    client.chat.completions._pending_reference_cost = 0.04
    client.chat.completions._pending_reference_details = [
        {
            "role": "reference",
            "slot_index": 0,
            "provider": "openrouter",
            "model": "anthropic/claude-opus",
            "usage": CanonicalUsage(input_tokens=12),
            "cost_usd": 0.04,
        }
    ]

    usage, cost, details = client.consume_reference_details()

    assert client.preset_name == "code-hard"
    assert usage.input_tokens == 12
    assert cost == 0.04
    assert details[0]["slot_index"] == 0


def test_legacy_usage_consumer_also_clears_pending_details():
    from agent.moa_loop import MoAChatCompletions

    facade = MoAChatCompletions("closed")
    facade._pending_reference_details = [{"slot_index": 0}]

    facade.consume_reference_usage()

    assert facade._pending_reference_details == []
