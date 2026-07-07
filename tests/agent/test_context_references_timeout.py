"""Regression tests for the per-message context-reference preprocessing
timeout (original bug: an unbounded blocking bridge on the gateway worker
path let a slow/hanging @reference fetch block the worker thread forever,
eventually exhausting the turn pool and stopping ALL message processing).

These tests patch ``preprocess_context_references_async`` with a coroutine
that sleeps far longer than the timeout, then assert the sync entrypoint
returns promptly with a non-lossy degraded result instead of hanging.
"""

import asyncio

import pytest

import agent.context_references as cr


def _make_hanging_coro(sleep_s: float):
    async def _hang():
        await asyncio.sleep(sleep_s)
        return cr.ContextReferenceResult(message="x", original_message="x")

    return _hang


def test_no_loop_branch_times_out(monkeypatch):
    """Without a running loop the bridge must still respect the timeout and
    fall back to the raw message instead of blocking forever."""
    monkeypatch.setattr(
        cr, "PREPROCESS_CONTEXT_TIMEOUT_SECONDS", 0.2
    )
    monkeypatch.setattr(
        cr, "preprocess_context_references_async", lambda *a, **k: _make_hanging_coro(5.0)()
    )

    import time

    start = time.monotonic()
    result = cr.preprocess_context_references(
        "@file:foo.txt", cwd="/tmp", context_length=100000
    )
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"preprocessing hung for {elapsed:.1f}s"
    assert result.message == "@file:foo.txt"
    assert any("timed out" in w for w in result.warnings)


def test_loop_branch_times_out(monkeypatch):
    """With a running event loop (the gateway worker path) the bridge must
    respect the timeout via the thread-pool + .result(timeout) path."""
    monkeypatch.setattr(
        cr, "PREPROCESS_CONTEXT_TIMEOUT_SECONDS", 0.2
    )
    monkeypatch.setattr(
        cr, "preprocess_context_references_async", lambda *a, **k: _make_hanging_coro(5.0)()
    )

    import time

    async def _call():
        return cr.preprocess_context_references(
            "@url:https://example.com", cwd="/tmp", context_length=100000
        )

    start = time.monotonic()
    result = asyncio.run(_call())
    elapsed = time.monotonic() - start

    assert elapsed < 2.0, f"preprocessing hung for {elapsed:.1f}s"
    assert result.message == "@url:https://example.com"
    assert any("timed out" in w for w in result.warnings)
