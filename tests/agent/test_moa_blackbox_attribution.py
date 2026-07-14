"""Physical-call attribution passed from the MoA loop to Blackbox."""

from agent.usage_pricing import CanonicalUsage


def test_build_moa_pricing_calls_appends_real_aggregator_after_advisors():
    from agent.conversation_loop import _build_moa_pricing_calls

    advisor_calls = [{
        "model": "gpt-5.5",
        "provider": "openai-codex",
        "base_url": "https://codex.invalid",
        "input_tokens": 1000,
        "output_tokens": 100,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
    }]
    aggregator_usage = CanonicalUsage(
        input_tokens=200,
        output_tokens=50,
        cache_read_tokens=300,
        cache_write_tokens=40,
        reasoning_tokens=10,
    )

    calls = _build_moa_pricing_calls(
        advisor_calls,
        aggregator_usage,
        aggregator_model="anthropic/claude-opus-4.8",
        aggregator_provider="openrouter",
        aggregator_base_url="https://openrouter.ai/api/v1",
    )

    assert calls[0] == advisor_calls[0]
    assert calls[1] == {
        "model": "anthropic/claude-opus-4.8",
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "input_tokens": 200,
        "output_tokens": 50,
        "cache_read_tokens": 300,
        "cache_write_tokens": 40,
        "reasoning_tokens": 10,
    }
    # The helper must not mutate the facade-owned pending list.
    assert len(advisor_calls) == 1
