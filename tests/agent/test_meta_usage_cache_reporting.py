"""Usage cache reporting for Meta Responses path."""

from types import SimpleNamespace

from agent.usage_pricing import normalize_usage


def test_meta_cached_tokens_flows_to_cache_read():
    usage = SimpleNamespace(
        input_tokens=4000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=3920, cache_creation_tokens=0),
        output_tokens_details=None,
        prompt_tokens=None,
        completion_tokens=None,
    )
    cu = normalize_usage(usage, api_mode="codex_responses")
    assert cu.cache_read_tokens == 3920
    assert cu.input_tokens == 80  # 4000 - 3920
    # rendered cache line would be cache=3920/4000 (98%)
    total = cu.prompt_tokens
    assert total == 4000
    pct = (cu.cache_read_tokens / total * 100) if total else 0
    assert 97 <= pct <= 99
    rendered = f"cache={cu.cache_read_tokens}/{total} ({pct:.0f}%)"
    assert rendered == "cache=3920/4000 (98%)"


def test_meta_cache_write_tokens():
    first = SimpleNamespace(
        input_tokens=4000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=0, cache_creation_tokens=4000),
    )
    cu1 = normalize_usage(first, api_mode="codex_responses")
    assert cu1.cache_write_tokens == 4000
    assert cu1.cache_read_tokens == 0

    second = SimpleNamespace(
        input_tokens=4000,
        output_tokens=100,
        input_tokens_details=SimpleNamespace(cached_tokens=3920, cache_creation_tokens=0),
    )
    cu2 = normalize_usage(second, api_mode="codex_responses")
    assert cu2.cache_read_tokens == 3920
    assert cu2.cache_write_tokens == 0
