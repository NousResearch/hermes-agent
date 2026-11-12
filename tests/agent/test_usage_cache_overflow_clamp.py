"""An impossible cache counter must never reach cost math.

Some providers report prompt_tokens_details.cached_tokens as a cumulative
counter rather than a per-call count (observed with ollama-cloud: a
monotonically growing, 128-quantized value that reached ~147x the prompt).
Unclamped, normalize_usage produced input_tokens=0 and a cache read larger
than the entire prompt, which then flowed into cost estimation and cache
analytics.
"""

from types import SimpleNamespace

from agent.usage_pricing import normalize_usage


def _chat_usage(prompt, cached, write=0):
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=50,
        total_tokens=prompt + 50,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached, cache_write_tokens=write),
        completion_tokens_details=None,
    )


def test_cumulative_cached_counter_is_clamped_to_prompt():
    cu = normalize_usage(_chat_usage(807, 118656), provider="ollama-cloud")
    assert cu.prompt_tokens == 807
    assert cu.cache_read_tokens == 807  # clamped, never above the prompt
    assert cu.input_tokens == 0
    assert cu.total_tokens >= cu.prompt_tokens  # accounting stays monotonic


def test_cache_write_and_read_cannot_together_exceed_prompt():
    # Existing behaviour is preserved: an inconsistent cache_write is not clamped
    # and the prompt total still includes it (see
    # test_normalize_usage_clamps_inconsistent_cache_total in test_usage_pricing.py).
    cu = normalize_usage(_chat_usage(100, 80, write=50), provider="openrouter")
    assert cu.cache_read_tokens == 80
    assert cu.cache_write_tokens == 50
    assert cu.prompt_tokens == 130


def test_impossible_counter_does_not_change_plausible_calls():
    cu = normalize_usage(_chat_usage(4000, 3920), provider="openrouter")
    assert cu.cache_read_tokens == 3920
    assert cu.input_tokens == 80


def test_anthropic_shape_is_exempt():
    # Anthropic reports UNCACHED input in input_tokens, so a cache read larger
    # than it is legitimate. Clamping here would corrupt every Claude call.
    usage = SimpleNamespace(
        input_tokens=200,
        output_tokens=50,
        cache_read_input_tokens=8000,
        cache_creation_input_tokens=0,
    )
    cu = normalize_usage(usage, provider="anthropic", api_mode="anthropic_messages")
    assert cu.input_tokens == 200
    assert cu.cache_read_tokens == 8000


def test_codex_shape_is_clamped_too():
    usage = SimpleNamespace(
        input_tokens=1000,
        output_tokens=50,
        input_tokens_details=SimpleNamespace(cached_tokens=5000, cache_creation_tokens=0),
    )
    cu = normalize_usage(usage, api_mode="codex_responses")
    assert cu.cache_read_tokens == 1000
    assert cu.input_tokens == 0
