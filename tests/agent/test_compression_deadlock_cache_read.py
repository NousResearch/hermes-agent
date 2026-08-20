"""Regression: last_total_tokens must exclude cache_read_tokens.

When an API response includes cache_read_tokens, total_tokens =
prompt_tokens + completion_tokens includes the cache-read portion.
If last_total_tokens keeps cache_read, it stays above threshold after
compression and re-triggers compaction — the \"compression deadlock\"
(issue #40803).

Both last_prompt_tokens and last_total_tokens must exclude cache_read
so the compressor correctly sees that compression reduced the context.
"""

from agent.context_compressor import ContextCompressor


def test_total_tokens_excludes_cache_read():
    """last_total_tokens must subtract cache_read_tokens, matching last_prompt_tokens."""
    c = ContextCompressor(model="test", quiet_mode=True, config_context_length=200000)

    # Scenario: 200k model, cache_read=80k, input=40k, output=10k
    # prompt_tokens = 40 + 80 + 10(cache_write, assume 0) = 120
    # total_tokens  = 120 + 10 = 130
    # last_prompt_tokens = 120 - 80 = 40  (excludes cache_read)
    # last_total_tokens  must also be ~50 = 40(input) + 10(output)  (excludes cache_read)
    c.update_from_response({
        "prompt_tokens": 120,
        "completion_tokens": 10,
        "total_tokens": 130,
        "cache_read_tokens": 80,
    })
    assert c.last_prompt_tokens == 40
    assert c.last_total_tokens == 50, (
        f"Expected last_total_tokens=50 (excludes cache_read), got {c.last_total_tokens}"
    )


def test_total_tokens_no_cache_unchanged():
    """When cache_read_tokens is 0 or absent, total should equal API total."""
    c = ContextCompressor(model="test", quiet_mode=True, config_context_length=200000)

    c.update_from_response({
        "prompt_tokens": 100,
        "completion_tokens": 30,
        "total_tokens": 130,
    })
    assert c.last_total_tokens == 130

    c.update_from_response({
        "prompt_tokens": 100,
        "completion_tokens": 30,
        "total_tokens": 130,
        "cache_read_tokens": 0,
    })
    assert c.last_total_tokens == 130
