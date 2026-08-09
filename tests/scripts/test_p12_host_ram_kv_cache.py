"""Tests for scripts/p12_host_ram_kv_cache.py (P12 host-RAM KV offload path).

Covers the acceptance criteria in kanban task t_ec875f9f:

- Requests >128K tokens (default threshold) use host-RAM KV-cache storage
  with a minimal GPU working set.
- Host<->GPU transfers are asynchronous and overlapped with compute so
  decode latency stays bounded.
- The offload path is never chosen for contexts at or below the threshold
  unless explicitly forced by config/override.
- Memory accounting covers both host-RAM usage and GPU working-set usage.

Test-plan mapping:

- Functional: synthetic 192K-context input; pages host-resident, GPU working
  set stays below the configurable cap.
- Concurrency: multiple long-context requests; serving loop responsive,
  no deadlocks.
- Boundary: 128K+1 triggers offload, 128K stays GPU-resident (policy gate).
- Transfer overlap: async copies overlap; decode waits stay bounded.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "p12_host_ram_kv_cache.py"
)

DEFAULT_THRESHOLD = 131072  # 128K — must match scripts/kv_cache_policy.py
DEFAULT_MAX = 262144        # 256K

# Small page size keeps tests fast and makes working-set math easy.
TOKENS_PER_PAGE = 2048
BYTES_PER_TOKEN = 256
PAGE_BYTES = TOKENS_PER_PAGE * BYTES_PER_TOKEN  # 512 KiB


def load_module():
    spec = importlib.util.spec_from_file_location(
        "p12_host_ram_kv_cache", MODULE_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules so dataclass machinery can resolve string
    # annotations (from __future__ import annotations) against the module.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_cache(module, ctx, **kwargs):
    """Construct a cache with a tiny page size so tests are fast."""
    kwargs.setdefault("tokens_per_page", TOKENS_PER_PAGE)
    kwargs.setdefault("bytes_per_token", BYTES_PER_TOKEN)
    return module.HostRamKVCache(ctx, **kwargs)


# ---------------------------------------------------------------------------
# Policy is the gate — the offload path is never chosen at/below threshold
# ---------------------------------------------------------------------------


def test_offload_path_refused_at_exactly_128k():
    module = load_module()
    with pytest.raises(module.OffloadPolicyError):
        make_cache(module, DEFAULT_THRESHOLD)


def test_offload_path_refused_below_threshold():
    module = load_module()
    for ctx in (0, 4096, 65536, 131071):
        with pytest.raises(module.OffloadPolicyError):
            make_cache(module, ctx)


def test_offload_path_allows_128k_plus_1():
    module = load_module()
    cache = make_cache(module, DEFAULT_THRESHOLD + 1)
    assert cache.context_len == DEFAULT_THRESHOLD + 1


def test_offload_path_allows_256k():
    module = load_module()
    cache = make_cache(module, DEFAULT_MAX)
    assert cache.context_len == DEFAULT_MAX


def test_force_override_bypasses_policy_for_testing():
    module = load_module()
    cache = make_cache(module, DEFAULT_THRESHOLD, force=True)
    assert cache.context_len == DEFAULT_THRESHOLD


def test_force_override_on_tiny_context():
    module = load_module()
    cache = make_cache(module, 1000, force=True)
    assert cache.context_len == 1000


def test_context_above_max_rejected_by_policy():
    module = load_module()
    with pytest.raises(module.ContextOutOfRangeError):
        make_cache(module, DEFAULT_MAX + 1)


def test_negative_context_rejected():
    module = load_module()
    with pytest.raises(ValueError):
        make_cache(module, -5)


def test_explicit_decision_object_respected():
    """A caller that already made a policy decision can pass it in."""
    module = load_module()
    # Use the real policy module (the same one p12_host_ram_kv_cache imports
    # at module scope) so enum identity matches.
    from scripts.kv_cache_policy import decide_tier  # noqa: E402

    decision = decide_tier(200000)
    cache = module.HostRamKVCache(200000, decision=decision)
    assert cache.decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD


# ---------------------------------------------------------------------------
# Host residency + minimal GPU working set (functional, 192K synthetic)
# ---------------------------------------------------------------------------

CTX_192K = 192 * 1024  # 196608 tokens
PAGES_192K = (CTX_192K + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE  # 96


@pytest.mark.asyncio
async def test_192k_pages_are_host_resident():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=4)
    assert cache.num_pages == PAGES_192K

    async with cache:
        await cache.prefetch([0, 1, 2])
        # Host side: every prefetched page is host-resident.
        host_indices = cache.host_resident_indices()
        assert 0 in host_indices and 1 in host_indices and 2 in host_indices
        # GPU working set: only the prefetched pages are staged, bounded by cap.
        assert cache.gpu_resident_bytes() <= cache.gpu_cap_bytes
        assert len(cache.gpu_resident_indices()) <= 4

        await cache.fetch(0)
        assert cache.gpu_resident_indices() == [0, 1, 2]

    # After close: everything written back to host; GPU working set empty.
    assert cache.gpu_resident_indices() == []
    assert cache.gpu_resident_bytes() == 0
    assert cache.host_resident_bytes() > 0


@pytest.mark.asyncio
async def test_gpu_working_set_never_exceeds_cap():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=3)

    async with cache:
        # Touch 8 distinct pages; the working set must stay <= 3 pages.
        for i in range(8):
            await cache.fetch(i)
        assert len(cache.gpu_resident_indices()) <= 3
        assert cache.gpu_resident_bytes() <= cache.gpu_cap_bytes

    # Working set empty after close; host has all touched pages.
    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_gpu_working_set_lru_eviction_writes_back():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    async with cache:
        await cache.fetch(0)
        await cache.fetch(1)
        assert sorted(cache.gpu_resident_indices()) == [0, 1]
        await cache.fetch(2)  # evicts LRU (0)
        assert 2 in cache.gpu_resident_indices()
        assert cache.stats.working_set_evictions >= 1
        # The evicted page is still host-resident (write-back target exists).
        assert 0 in cache.host_resident_indices()

    assert cache.gpu_resident_indices() == []


# ---------------------------------------------------------------------------
# Async overlap: transfers overlap, decode waits stay bounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_transfers_overlap_with_compute():
    module = load_module()
    # Non-zero latency so multiple copies are genuinely in flight at once.
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.05)
    cache = make_cache(
        module, CTX_192K, transfer=transfer,
        gpu_working_set_pages=8, max_inflight_transfers=8,
    )

    async with cache:
        await cache.prefetch([0, 1, 2, 3, 4])
        # Let the scheduled copies start (they run in the background).
        await asyncio.sleep(0.02)
        # All 5 copies were issued; they overlap because issue is async and
        # the first is still in flight while the rest start.
        assert transfer.count_transfers("h2g") == 5
        # Peak in-flight must exceed 1: copies for different pages overlap.
        assert transfer.max_overlap() > 1

        # A fetch for an already-prefetched page does not re-transfer.
        await cache.fetch(0)
        assert transfer.count_transfers("h2g") == 5

    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_prefetch_returns_before_transfers_complete():
    """The async-overlap contract: prefetch must return while copies are
    still running, so compute can proceed in parallel with the copies."""
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.2)
    cache = make_cache(
        module, CTX_192K, transfer=transfer,
        gpu_working_set_pages=8, max_inflight_transfers=8,
    )

    async with cache:
        t0 = asyncio.get_event_loop().time()
        await cache.prefetch([0, 1, 2])
        dt = asyncio.get_event_loop().time() - t0
        # prefetch issued 3 copies each taking 0.2s; it must NOT have waited
        # for any of them (that would take >= 0.2s).
        assert dt < 0.1
        # The copies run in the background: give the loop a tick to start
        # them, then all three must be in flight concurrently.
        await asyncio.sleep(0.02)
        assert transfer.count_transfers("h2g") == 3
        assert transfer.in_flight == 3

    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_fetch_waits_are_bounded_and_prefetch_wins_race():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.05)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=4)

    async with cache:
        await cache.prefetch([0])
        # Wait for the prefetch copy to land, then fetch: zero extra wait.
        await asyncio.sleep(0.06)
        t0 = asyncio.get_event_loop().time()
        await cache.fetch(0)
        dt = asyncio.get_event_loop().time() - t0
        assert dt < 0.02  # already GPU-resident: no transfer wait
        assert cache.stats.fetch_wait_s < 0.02
        assert cache.stats.prefetch_hits >= 0

    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_cold_fetch_waits_but_only_for_the_transfer():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.05)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=4)

    async with cache:
        t0 = asyncio.get_event_loop().time()
        await cache.fetch(0)  # cold miss: must wait for one copy
        dt = asyncio.get_event_loop().time() - t0
        # Waited roughly one transfer latency, not unbounded.
        assert dt >= 0.04 and dt < 0.2
        assert cache.stats.fetch_wait_s >= 0.04
        assert transfer.count_transfers("h2g") == 1

    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_decode_loop_never_blocks_on_prefetched_pages():
    """Simulate a decode loop that prefetches ahead and fetches as it goes:
    total time must be bounded by (transfers / overlap) * latency, with
    no serialization."""
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.05)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=4)

    async with cache:
        # Prefetch a window of 4 pages, then decode them one at a time.
        t0 = asyncio.get_event_loop().time()
        await cache.prefetch([0, 1, 2, 3])
        for i in range(4):
            await cache.fetch(i)
        dt = asyncio.get_event_loop().time() - t0
        # If copies serialized one-at-a-time this would be ~4*0.05 = 0.2s;
        # overlapping keeps it well below that.
        assert dt < 0.16
        assert transfer.max_overlap() > 1

    assert cache.gpu_resident_indices() == []


# ---------------------------------------------------------------------------
# Concurrency: multiple long-context requests, no deadlock, responsive loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_sessions_no_deadlock():
    """Multiple long-context requests served concurrently through the pool:
    all complete, no deadlock, serving loop responsive."""
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.01)
    pool = module.OffloadSessionPool(
        max_sessions=4,
        transfer=transfer,
        tokens_per_page=TOKENS_PER_PAGE,
        bytes_per_token=BYTES_PER_TOKEN,
        gpu_working_set_pages=2,
    )

    async def worker(n):
        cache = await pool.acquire(CTX_192K)
        try:
            await cache.prefetch([n % 4])
            await cache.fetch(n % 4)
            await asyncio.sleep(0.01)
            return n
        finally:
            await pool.release(cache)

    results = await asyncio.gather(*[worker(i) for i in range(8)])
    assert sorted(results) == list(range(8))
    # Serving loop remained responsive: all workers completed.
    assert pool.active_sessions == 0


@pytest.mark.asyncio
async def test_session_is_single_use():
    """A HostRamKVCache is a single-session object: after close, re-entering
    raises instead of deadlocking on the (released) session lock."""
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.01)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    async with cache:
        await cache.fetch(0)

    with pytest.raises(RuntimeError):
        async with cache:
            pass  # re-entry after close is a hard error


@pytest.mark.asyncio
async def test_reentrant_enter_same_task_raises():
    """Entering the same cache twice in one task is a programming error."""
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.01)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    await cache.__aenter__()
    try:
        with pytest.raises(RuntimeError):
            await cache.__aenter__()
    finally:
        await cache.close()


@pytest.mark.asyncio
async def test_pool_bounds_concurrent_sessions():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    pool = module.OffloadSessionPool(
        max_sessions=2,
        transfer=transfer,
        tokens_per_page=TOKENS_PER_PAGE,
        bytes_per_token=BYTES_PER_TOKEN,
    )

    async def worker(n):
        cache = await pool.acquire(CTX_192K)
        try:
            await asyncio.sleep(0.05)
            return n
        finally:
            await pool.release(cache)

    results = await asyncio.gather(*[worker(i) for i in range(6)])
    assert sorted(results) == list(range(6))
    assert pool.peak_active_sessions <= 2
    assert pool.active_sessions == 0


@pytest.mark.asyncio
async def test_pool_times_out_when_saturated():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    pool = module.OffloadSessionPool(
        max_sessions=1,
        acquire_timeout_s=0.05,
        transfer=transfer,
        tokens_per_page=TOKENS_PER_PAGE,
        bytes_per_token=BYTES_PER_TOKEN,
    )

    first = await pool.acquire(CTX_192K)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await pool.acquire(CTX_192K)
    finally:
        await pool.release(first)
    assert pool.active_sessions == 0


@pytest.mark.asyncio
async def test_pool_release_is_idempotent():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    pool = module.OffloadSessionPool(
        max_sessions=2,
        transfer=transfer,
        tokens_per_page=TOKENS_PER_PAGE,
        bytes_per_token=BYTES_PER_TOKEN,
    )
    cache = await pool.acquire(CTX_192K)
    await pool.release(cache)
    await pool.release(cache)  # no-op
    assert pool.active_sessions == 0


# ---------------------------------------------------------------------------
# Memory accounting (host + GPU)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_host_accounting_counts_allocated_pages():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=4)

    assert cache.host_resident_bytes() == 0
    async with cache:
        await cache.prefetch([0, 1])
        assert cache.host_resident_bytes() == 2 * PAGE_BYTES
        await cache.prefetch([2, 3])
        assert cache.host_resident_bytes() == 4 * PAGE_BYTES

    # Close keeps host pages (they are the steady-state residence).
    assert cache.host_resident_bytes() == 4 * PAGE_BYTES
    assert cache.stats.host_peak_bytes >= 4 * PAGE_BYTES


@pytest.mark.asyncio
async def test_gpu_accounting_counts_staged_pages():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=3)

    async with cache:
        await cache.fetch(0)
        await cache.fetch(1)
        assert cache.gpu_resident_bytes() == 2 * PAGE_BYTES
        assert cache.stats.gpu_peak_bytes >= 2 * PAGE_BYTES
        await cache.fetch(2)
        assert cache.gpu_resident_bytes() == 3 * PAGE_BYTES
        assert cache.gpu_resident_bytes() <= cache.gpu_cap_bytes

    assert cache.gpu_resident_bytes() == 0


@pytest.mark.asyncio
async def test_stats_report_both_memories():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    async with cache:
        await cache.prefetch([0, 1, 2])
        await cache.fetch(0)
        await cache.fetch(1)
        await cache.fetch(2)

    stats = cache.stats.to_dict()
    assert stats["host_resident_bytes"] > 0
    assert stats["gpu_resident_bytes"] == 0  # after close
    assert stats["gpu_peak_bytes"] > 0
    assert stats["host_peak_bytes"] > 0
    assert stats["h2g_transfers"] > 0
    assert stats["g2h_transfers"] > 0


@pytest.mark.asyncio
async def test_host_cap_enforced():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(
        module,
        CTX_192K,
        transfer=transfer,
        gpu_working_set_pages=4,
        host_cap_bytes=2 * PAGE_BYTES,
    )

    async with cache:
        await cache.prefetch([0, 1])
        with pytest.raises(module.HostRamCapacityError):
            await cache.prefetch([2])  # would exceed 2-page host cap


@pytest.mark.asyncio
async def test_gpu_cap_enforced_after_eviction_room():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(
        module,
        CTX_192K,
        transfer=transfer,
        gpu_working_set_pages=2,
        gpu_cap_bytes=2 * PAGE_BYTES,
    )

    async with cache:
        await cache.fetch(0)
        await cache.fetch(1)
        # Working set full; fetching 2 evicts LRU then fits.
        await cache.fetch(2)
        assert len(cache.gpu_resident_indices()) <= 2
        assert cache.gpu_resident_bytes() <= cache.gpu_cap_bytes


# ---------------------------------------------------------------------------
# Boundary (mirrors policy tests, through the runtime path)
# ---------------------------------------------------------------------------


def test_boundary_128k_gpu_resident_128k_plus_1_offload():
    module = load_module()
    # 128K exactly: policy routes GPU -> offload path refused.
    with pytest.raises(module.OffloadPolicyError):
        make_cache(module, DEFAULT_THRESHOLD)
    # 128K+1: policy routes offload -> constructed.
    cache = make_cache(module, DEFAULT_THRESHOLD + 1)
    assert cache.decision.tier is module.KVCacheTier.HOST_RAM_OFFLOAD
    assert cache.context_len == DEFAULT_THRESHOLD + 1


def test_num_pages_matches_context():
    module = load_module()
    cache = make_cache(module, DEFAULT_THRESHOLD + 1)
    assert cache.num_pages == (DEFAULT_THRESHOLD + 1 + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE


# ---------------------------------------------------------------------------
# Dedup: never copy the same page twice concurrently
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefetch_dedupes_inflight_pages():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.05)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=8)

    async with cache:
        await cache.prefetch([0, 0, 0])  # same page three times
        await asyncio.sleep(0.02)  # let the scheduled copy start
        assert transfer.count_transfers("h2g") == 1
        await cache.prefetch([1, 1])     # still one copy per page
        await asyncio.sleep(0.02)
        assert transfer.count_transfers("h2g") == 2

    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_fetch_after_release_re_stages():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=4)

    async with cache:
        await cache.fetch(0)
        await cache.release(0)
        assert 0 not in cache.gpu_resident_indices()
        # Host still holds it.
        assert 0 in cache.host_resident_indices()
        await cache.fetch(0)
        assert 0 in cache.gpu_resident_indices()

    assert cache.gpu_resident_indices() == []


@pytest.mark.asyncio
async def test_max_inflight_transfers_bounds_concurrency():
    """The runtime never exceeds ``max_inflight_transfers`` concurrent
    host<->GPU copies, even when many pages are prefetched at once."""
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0.1)
    cache = make_cache(
        module, CTX_192K, transfer=transfer,
        gpu_working_set_pages=16, max_inflight_transfers=2,
    )

    async with cache:
        await cache.prefetch([0, 1, 2, 3, 4])
        await asyncio.sleep(0.05)  # let the copies start
        # Backend peak in-flight must never exceed the cap of 2.
        assert transfer.peak_in_flight <= 2
        # All copies still complete.
        await asyncio.sleep(0.4)
        assert transfer.count_transfers("h2g") == 5

    assert cache.gpu_resident_indices() == []


# ---------------------------------------------------------------------------
# Close semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close_is_idempotent():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    async with cache:
        await cache.fetch(0)
    await cache.close()  # second close is a no-op
    assert cache.closed is True


@pytest.mark.asyncio
async def test_operations_after_close_raise():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    async with cache:
        pass
    with pytest.raises(RuntimeError):
        await cache.prefetch([0])
    with pytest.raises(RuntimeError):
        await cache.fetch(0)
    with pytest.raises(RuntimeError):
        await cache.release(0)


# ---------------------------------------------------------------------------
# Page index validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_out_of_range_page_index_raises():
    module = load_module()
    transfer = module.SimulatedAsyncTransferBackend(transfer_latency_s=0)
    cache = make_cache(module, CTX_192K, transfer=transfer, gpu_working_set_pages=2)

    async with cache:
        with pytest.raises(IndexError):
            await cache.prefetch([cache.num_pages])
        with pytest.raises(IndexError):
            await cache.fetch(-1)
        with pytest.raises(IndexError):
            await cache.release(cache.num_pages)
