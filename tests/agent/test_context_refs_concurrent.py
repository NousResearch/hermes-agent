"""Tests for concurrent @-reference expansion in context_references.

test_refs_expand_concurrently asserts that N URL refs are fetched CONCURRENTLY.
It proves this with an asyncio.Barrier rendezvous rather than a stopwatch: all
N fetches must be in flight at the same instant before any is allowed to
return. On the serial `for ref in refs: await` loop the first fetch waits for
partners that never arrive and the test fails; with asyncio.gather it passes.
The output-contract test guards that concurrency does NOT change ordering,
warnings, blocks, or token accounting.
"""
from __future__ import annotations

import asyncio

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

    # Concurrency is proven by construction, not by measuring elapsed time.
    #
    # The old form asserted `elapsed < 0.4` ("well under 2x one 0.2s fetch").
    # That makes the event-loop scheduler and any fixed setup part of the
    # assertion: under a loaded CI box the inequality can flip with nothing
    # wrong in the code under test, and the margin shrinks silently if setup
    # cost is ever added ahead of dispatch.
    #
    # A barrier asserts the invariant directly: all THREE fetches must be
    # inside the fetcher AT THE SAME TIME before any is allowed to return. If
    # expansion ever goes serial the first fetch blocks waiting for partners
    # that will not arrive, the barrier times out, and the test fails with an
    # explicit message. No wall-clock constant, no load sensitivity.
    N_REFS = 3
    rendezvous = asyncio.Barrier(N_REFS)
    entered: list[str] = []
    overlapped = asyncio.Event()

    async def barrier_fetcher(url: str) -> str:
        entered.append(url)
        # Generous relative to real scheduling latency (a rendezvous between
        # already-dispatched coroutines needs milliseconds), but finite so a
        # serial regression fails fast instead of hanging the suite.
        async with asyncio.timeout(10):
            await rendezvous.wait()
        overlapped.set()
        return f"CONTENT[{url}]"

    try:
        res = await preprocess_context_references_async(
            msg, cwd=tmp_path, context_length=100_000, url_fetcher=barrier_fetcher,
        )
    except (asyncio.BrokenBarrierError, TimeoutError):  # pragma: no cover - serial regression
        pytest.fail(
            "references did not expand concurrently: a fetch reached the "
            f"rendezvous alone, so expansion never had {N_REFS} fetches in "
            f"flight at once (entered: {entered})"
        )

    # The barrier only clears when all three fetches are in flight together.
    assert overlapped.is_set(), f"references never overlapped (entered: {entered})"
    assert len(entered) == N_REFS, f"expected {N_REFS} fetches, got {entered}"
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


@pytest.mark.asyncio
async def test_git_ref_expands_off_the_event_loop(tmp_path, monkeypatch):
    """The git expander must not run on the caller's event loop.

    ``_expand_git_reference`` shells out to ``git`` with a 30s timeout. Called
    inline it holds whatever loop runs the coroutine — on the gateway path
    (``preprocess_context_references_async`` from gateway/run.py) that is the
    shared loop serving every other platform, so one ``@diff`` from one user
    stalls the whole gateway.
    """
    import threading

    from agent import context_references as cr

    main_thread = threading.current_thread()
    seen: dict = {}

    def _fake_git(ref, cwd, args, label):
        seen["thread"] = threading.current_thread()
        return None, f"GIT[{label}]"

    monkeypatch.setattr(cr, "_expand_git_reference", _fake_git)

    res = await preprocess_context_references_async(
        "look at @diff please", cwd=tmp_path, context_length=100_000,
    )

    assert "GIT[git diff]" in res.message
    assert seen.get("thread") is not None, "git expander never ran"
    assert seen["thread"] is not main_thread, (
        "git expansion ran on the event-loop thread; it must be dispatched via "
        "asyncio.to_thread so a 30s git subprocess can't stall the gateway loop"
    )


@pytest.mark.asyncio
async def test_sync_refs_expand_concurrently(tmp_path, monkeypatch):
    """The gather must deliver real concurrency for the *sync* ref kinds too.

    The existing concurrency coverage uses ``@url:`` refs, which are natively
    awaitable — so it passed even while file/folder/git refs ran inline and
    serialised. Blocking calls cannot interleave, so N such refs cost the SUM
    of their runtimes rather than the max.
    """
    from agent import context_references as cr

    def _slow_file(ref, cwd, *, allowed_root=None):
        time.sleep(0.2)
        return None, f"FILE[{ref.target}]"

    monkeypatch.setattr(cr, "_expand_file_reference", _slow_file)

    msg = "@file:a.txt @file:b.txt @file:c.txt"
    t0 = time.perf_counter()
    res = await preprocess_context_references_async(
        msg, cwd=tmp_path, context_length=100_000,
    )
    elapsed = time.perf_counter() - t0

    assert "FILE[a.txt]" in res.message
    assert elapsed < 0.4, f"expected concurrent (~0.2s), got {elapsed:.2f}s (serial?)"
