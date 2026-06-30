"""Tests for concurrent @-reference expansion in context_references.

RED before the refactor: test_refs_expand_concurrently asserts that N URL refs
(each a ~0.2s fetch) complete in roughly one fetch-time, not N×. On the serial
`for ref in refs: await` loop this FAILS (takes ~N×0.2s); after switching to
asyncio.gather it passes. The output-contract test guards that concurrency does
NOT change ordering, warnings, blocks, or token accounting.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from agent.context_references import preprocess_context_references_async


async def _slow_fetcher(url: str) -> str:
    # Simulate a per-URL network fetch (web_extract round trip).
    await asyncio.sleep(0.2)
    return f"CONTENT[{url}]"


@pytest.mark.asyncio
async def test_refs_expand_concurrently(tmp_path):
    # Three independent URL refs in one message.
    msg = "see @url:https://a.example/x @url:https://b.example/y @url:https://c.example/z please"
    t0 = time.perf_counter()
    res = await preprocess_context_references_async(
        msg, cwd=tmp_path, context_length=100_000, url_fetcher=_slow_fetcher,
    )
    elapsed = time.perf_counter() - t0
    # Serial would be ~0.6s (3×0.2). Concurrent ~0.2s. Assert well under 2× one fetch.
    assert elapsed < 0.4, f"expected concurrent (~0.2s), got {elapsed:.2f}s (serial?)"
    # All three blocks present, in order.
    assert res.expanded
    body = res.message
    assert body.index("a.example") < body.index("b.example") < body.index("c.example"), \
        "reference blocks must stay in original order"


@pytest.mark.asyncio
async def test_concurrent_preserves_output_contract(tmp_path):
    """Concurrency must not change which blocks/warnings appear or their order."""
    msg = "@url:https://one.example/p @url:https://two.example/q"
    res = await preprocess_context_references_async(
        msg, cwd=tmp_path, context_length=100_000, url_fetcher=_slow_fetcher,
    )
    assert "CONTENT[https://one.example/p]" in res.message
    assert "CONTENT[https://two.example/q]" in res.message
    assert res.message.index("one.example") < res.message.index("two.example")
    assert res.injected_tokens > 0
