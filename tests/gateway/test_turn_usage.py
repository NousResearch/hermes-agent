from types import SimpleNamespace

from gateway.turn_usage import (
    agent_usage_snapshot,
    budget_status,
    format_compact_usage,
    should_show_usage_receipt,
    turn_usage_from_result,
    usage_delta,
)


def test_usage_delta_nonnegative_and_formats_compact_line():
    before = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_read_tokens": 30,
        "cache_write_tokens": 0,
        "total_tokens": 150,
        "estimated_cost_usd": 0.10,
        "cost_status": "estimated",
        "model": "gpt-5.5",
        "provider": "openai-codex",
    }
    after = {
        "input_tokens": 1100,
        "output_tokens": 120,
        "cache_read_tokens": 230,
        "cache_write_tokens": 10,
        "total_tokens": 1460,
        "estimated_cost_usd": 0.25,
        "cost_status": "estimated",
        "model": "gpt-5.5",
        "provider": "openai-codex",
    }

    delta = usage_delta(after, before)
    delta["api_calls"] = 3
    usage = turn_usage_from_result({"api_calls": 3, "turn_usage": delta}, elapsed_seconds=62)

    assert usage.input_tokens == 1000
    assert usage.output_tokens == 100
    assert usage.cache_read_tokens == 200
    assert usage.cache_write_tokens == 10
    assert usage.total_tokens == 1310
    assert usage.estimated_cost_usd == 0.15
    assert format_compact_usage(usage) == "usage: 3 calls · 1,000 in · 200 cache-r · 10 cache-w · 100 out · ~$0.1500 · 1.0m"


def test_agent_usage_snapshot_reads_context_and_counters():
    agent = SimpleNamespace(
        session_input_tokens=10,
        session_output_tokens=2,
        session_cache_read_tokens=3,
        session_cache_write_tokens=4,
        session_prompt_tokens=17,
        session_completion_tokens=2,
        session_total_tokens=19,
        session_estimated_cost_usd=0.01,
        session_cost_status="estimated",
        model="gpt-5.5",
        provider="openai-codex",
        context_compressor=SimpleNamespace(last_prompt_tokens=123, context_length=1000),
    )

    snap = agent_usage_snapshot(agent)

    assert snap["total_tokens"] == 19
    assert snap["context_tokens"] == 123
    assert snap["context_length"] == 1000


def test_receipt_and_budget_thresholds():
    usage = turn_usage_from_result(
        {
            "api_calls": 8,
            "turn_usage": {"input_tokens": 100_000, "output_tokens": 1_000, "total_tokens": 101_000},
        },
        elapsed_seconds=10,
    )

    assert should_show_usage_receipt(usage, min_api_calls=2, min_tokens=25_000, min_seconds=30)
    assert budget_status(usage, warn_api_calls=8, warn_tokens=100_000, hard_api_calls=30, hard_tokens=350_000) == "warn"

    hard = turn_usage_from_result({"api_calls": 30, "turn_usage": {"total_tokens": 10}}, elapsed_seconds=1)
    assert budget_status(hard, warn_api_calls=8, warn_tokens=100_000, hard_api_calls=30, hard_tokens=350_000) == "hard"
