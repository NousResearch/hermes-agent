#!/usr/bin/env python3
"""P12 host-RAM KV-cache offload path (runtime serving mode).

Implements the *runtime* side of docs/adr/0012-p12-256k-memory-policy.md and
the tier decision in scripts/kv_cache_policy.py:

- Ordinary contexts (<= KV_CACHE_OFFLOAD_THRESHOLD, default 128K tokens) use
  the fast GPU-resident KV cache. This module is never used for them; the
  placement policy is the gate.
- Rare long contexts (> threshold, up to 256K tokens) use this host-RAM
  KV-cache path. The cache lives primarily in host RAM; only a small,
  bounded page window is staged on the GPU as the decode working set.
- Host<->GPU transfers are asynchronous and overlapped with compute so a
  long decode loop never blocks on a page fetch that could have been issued
  earlier.
- Memory accounting covers both host-RAM usage and the GPU working-set
  usage, and both stay within configurable caps.

Relationship to the other P12 modules:

- ``scripts/kv_cache_policy.py`` is the single source of truth for *which*
  tier a request uses. This module consults ``decide_tier()`` on
  construction and refuses to build an offload cache for a context the
  policy routes to the GPU fast path (unless an explicit override forces
  it). It never re-implements the boundary.
- ``scripts/p12_offload_gate.py`` decides *launch-time* argv (whether the
  server boots with ``--no-kv-offload``). This module is the *in-process*
  runtime that actually stages pages between host RAM and the GPU working
  set during inference. The two are complementary; a real deployment uses
  both (gate at spawn, this at decode time).
- ``scripts/p12_weight_placement.py`` guarantees weights stay GPU0-only;
  this module never moves weights and never references GPU1 (ADR 0012).

The module is deliberately stdlib-only: ``asyncio`` for the async transfer
layer, no torch/cuda imports at module scope. The actual host<->GPU copy is
a pluggable ``TransferBackend`` so tests use a deterministic simulated
backend and production supplies a real (e.g. ``cudaMemcpyAsync`` /
``torch``) one. This keeps the concurrency/accounting/policy logic fully
testable in CI without a GPU.

Usage (serving path):

    from scripts.kv_cache_policy import decide_tier, KVCacheTier
    from scripts.p12_host_ram_kv_cache import HostRamKVCache

    decision = decide_tier(context_len=ctx)
    if decision.tier is KVCacheTier.HOST_RAM_OFFLOAD:
        cache = HostRamKVCache(
            context_len=ctx,
            transfer=backend,           # TransferBackend
            gpu_working_set_pages=8,    # GPU-side working set cap
        )
        async with cache:
            await cache.prefetch(pages)      # overlap with compute
            page = await cache.fetch(page)   # GPU-side working set
            ...
    # else: GPU-resident fast path — this module is never constructed.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Make `python scripts/p12_host_ram_kv_cache.py` resolve the repo-root
# `scripts` package the same way `python -m scripts.p12_host_ram_kv_cache`
# does. Without this the direct CLI form would hit an ImportError.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.kv_cache_policy import (  # noqa: E402
    DEFAULT_MAX_CONTEXT,
    DEFAULT_OFFLOAD_THRESHOLD,
    ContextOutOfRangeError,
    KVCacheTier,
    PlacementDecision,
    decide_tier,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

# Default GPU working-set cap in pages. Kept small: the whole point of the
# offload path is that the GPU holds only a small paged slice of the KV
# cache while the bulk lives in host RAM.
DEFAULT_GPU_WORKING_SET_PAGES = 8

# Default maximum number of concurrent host<->GPU transfers. Bounding this
# keeps PCIe saturation bounded and gives the concurrency test a hard cap to
# assert against.
DEFAULT_MAX_INFLIGHT_TRANSFERS = 4

# Default cap on concurrent offload sessions. A handful of simultaneous
# long-context requests is supported; unbounded concurrency would let host
# RAM (and the shared GPU) be exhausted by a flood of 256K contexts.
DEFAULT_MAX_CONCURRENT_SESSIONS = 4

# A page covers this many tokens of the KV cache. Page size is a storage
# granularity, not a memory guarantee: the host RAM footprint is
# tokens_per_page * bytes_per_token (plus bookkeeping).
DEFAULT_TOKENS_PER_PAGE = 2048

# Approximate KV bytes per token (f16, per layer scaled). This is a sizing
# constant for accounting only — the real production number comes from the
# model config (see docs/kv-cache-serving-mode.md §4.1). It is overridable
# at construction so the accounting tests are not tied to a hardcoded
# silicon number.
DEFAULT_BYTES_PER_TOKEN = 256


class OffloadPolicyError(ValueError):
    """Raised when the offload path is used for a context the placement
    policy routes to the GPU fast path (policy violation)."""


class HostRamCapacityError(MemoryError):
    """Raised when a host-RAM allocation would exceed the configured host
    cap, or a GPU working-set staging would exceed the GPU cap."""


class KVPageState(enum.Enum):
    """Lifecycle of one KV-cache page in the offload path."""

    COLD = "cold"          # never touched; nothing allocated anywhere
    HOST = "host"          # resident in host RAM (the default residence)
    STAGING = "staging"    # host->GPU transfer in flight
    GPU = "gpu"            # resident in the GPU working set
    EVICTING = "evicting"  # GPU->host transfer in flight (write-back)


# ---------------------------------------------------------------------------
# Transfer backend (pluggable)
# ---------------------------------------------------------------------------


class TransferBackend:
    """Interface for asynchronous host<->GPU page copies.

    Production implementations wrap the platform's async copy (e.g.
    ``cudaMemcpyAsync`` on a side stream, or a torch ``non_blocking=True``
    copy). The reference implementation in this module is the simulated
    backend used by tests: it records transfer timing and concurrency so the
    overlap and bounded-concurrency contracts can be asserted deterministically.

    Backends must be safe to call from multiple concurrent tasks: each
    ``copy_host_to_gpu`` / ``copy_gpu_to_host`` call returns an awaitable and
    the runtime awaits them concurrently (up to ``max_inflight_transfers``).
    """

    async def copy_host_to_gpu(self, page: "KVPage") -> None:
        raise NotImplementedError

    async def copy_gpu_to_host(self, page: "KVPage") -> None:
        raise NotImplementedError


class SimulatedAsyncTransferBackend(TransferBackend):
    """Deterministic fake for tests.

    ``transfer_latency_s`` is how long each copy takes (wall-clock). While a
    transfer is in flight the backend counts it, so a test can assert the
    runtime never exceeds ``max_inflight_transfers`` concurrent copies, and
    that copies for different pages overlap (proving async, non-blocking
    issue) rather than serialize.
    """

    def __init__(self, transfer_latency_s: float = 0.02, rng_seed: int = 0):
        self.transfer_latency_s = transfer_latency_s
        self.in_flight: int = 0
        self.peak_in_flight: int = 0
        self.transfers: List[Tuple[str, int, float]] = []  # (dir, page, started)
        self.lock = asyncio.Lock()

    async def _copy(self, direction: str, page: "KVPage") -> None:
        async with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
            self.transfers.append((direction, page.index, time.monotonic()))
        try:
            if self.transfer_latency_s > 0:
                await asyncio.sleep(self.transfer_latency_s)
        finally:
            async with self.lock:
                self.in_flight -= 1

    async def copy_host_to_gpu(self, page: "KVPage") -> None:
        await self._copy("h2g", page)

    async def copy_gpu_to_host(self, page: "KVPage") -> None:
        await self._copy("g2h", page)

    def count_transfers(self, direction: Optional[str] = None) -> int:
        if direction is None:
            return len(self.transfers)
        return sum(1 for d, _, _ in self.transfers if d == direction)

    def max_overlap(self) -> int:
        """Return the peak number of concurrently in-flight transfers. >1
        proves copies for different pages overlap (async issue)."""
        return self.peak_in_flight


# ---------------------------------------------------------------------------
# Page bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class KVPage:
    """One fixed-size page of the KV cache.

    The page object itself is a tiny bookkeeping record; the payload bytes
    live in host RAM (``host_bytes``) and, transiently, in the GPU working
    set (``gpu_bytes``). Transfers move the payload between the two, never
    through an extra copy.
    """

    index: int
    tokens_per_page: int
    bytes_per_page: int
    state: KVPageState = KVPageState.COLD
    host_bytes: Optional[bytes] = None      # None until first host allocation
    gpu_bytes: Optional[bytes] = None       # None until staged to GPU
    last_gpu_access: float = 0.0            # LRU clock for working-set eviction
    host_allocated: bool = False
    gpu_allocated: bool = False

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"KVPage(idx={self.index}, state={self.state.value}, "
            f"host={self.host_allocated}, gpu={self.gpu_allocated})"
        )


# ---------------------------------------------------------------------------
# Runtime offload path
# ---------------------------------------------------------------------------


@dataclass
class OffloadStats:
    """Aggregate memory/transfer accounting for one offload session."""

    context_len: int = 0
    page_size: int = 0
    num_pages: int = 0
    bytes_per_token: int = 0
    host_cap_bytes: int = 0
    gpu_cap_bytes: int = 0
    host_resident_bytes: int = 0
    gpu_resident_bytes: int = 0
    host_peak_bytes: int = 0
    gpu_peak_bytes: int = 0
    h2g_transfers: int = 0
    g2h_transfers: int = 0
    fetch_wait_s: float = 0.0        # wall time decode waited on fetch()
    prefetch_hits: int = 0
    working_set_evictions: int = 0

    def to_dict(self) -> dict:
        return {
            "context_len": self.context_len,
            "page_size": self.page_size,
            "num_pages": self.num_pages,
            "bytes_per_token": self.bytes_per_token,
            "host_cap_bytes": self.host_cap_bytes,
            "gpu_cap_bytes": self.gpu_cap_bytes,
            "host_resident_bytes": self.host_resident_bytes,
            "gpu_resident_bytes": self.gpu_resident_bytes,
            "host_peak_bytes": self.host_peak_bytes,
            "gpu_peak_bytes": self.gpu_peak_bytes,
            "h2g_transfers": self.h2g_transfers,
            "g2h_transfers": self.g2h_transfers,
            "fetch_wait_s": self.fetch_wait_s,
            "prefetch_hits": self.prefetch_hits,
            "working_set_evictions": self.working_set_evictions,
        }


class HostRamKVCache:
    """Paged host-RAM KV cache with a bounded GPU working set.

    The placement policy (``decide_tier``) is the gate: constructing this
    cache for a context the policy routes to the GPU fast path raises
    ``OffloadPolicyError`` unless ``force=True`` (explicit config/override).

    Ownership/lifecycle:

    - ``async with cache`` acquires a session slot (bounded by
      ``max_concurrent_sessions``) and prepares the host-RAM pages.
    - ``prefetch(indices)`` issues asynchronous host->GPU copies for pages
      that are about to be decoded, overlapping them with compute. Returns
      immediately; fetches for the same pages later hit the staged working
      set with zero transfer latency.
    - ``fetch(index)`` returns the page's GPU payload, staging it (and
      evicting LRU pages as needed) if it is not already resident. This is
      the one call that can block the decode loop, and only when a page was
      not prefetched early enough; ``prefetch_hits`` counts how often the
      async issue actually won the race.
    - ``release(index)`` writes a page back to host RAM asynchronously
      (dirty write-back), freeing GPU working-set capacity.
    - ``close()`` (via ``async with`` exit) evicts all pages back to host
      RAM, records final accounting, and releases the session slot.

    All transfers go through ``transfer`` and are deduplicated: a page that
    is already in the GPU working set or already being staged is never
    copied twice concurrently.
    """

    def __init__(
        self,
        context_len: int,
        *,
        transfer: Optional[TransferBackend] = None,
        tokens_per_page: int = DEFAULT_TOKENS_PER_PAGE,
        bytes_per_token: int = DEFAULT_BYTES_PER_TOKEN,
        gpu_working_set_pages: int = DEFAULT_GPU_WORKING_SET_PAGES,
        max_inflight_transfers: int = DEFAULT_MAX_INFLIGHT_TRANSFERS,
        max_concurrent_sessions: int = DEFAULT_MAX_CONCURRENT_SESSIONS,
        host_cap_bytes: Optional[int] = None,
        gpu_cap_bytes: Optional[int] = None,
        force: bool = False,
        session_lock: Optional[asyncio.Lock] = None,
        decision: Optional[PlacementDecision] = None,
    ) -> None:
        if context_len < 0:
            raise ValueError(f"context_len must be >= 0, got {context_len}")

        # The placement policy is the gate: this path exists only for
        # contexts the policy routes to host-RAM offload. ``force=True`` is
        # the explicit override (tests/edge cases) — it never silently
        # bypasses the policy.
        self.decision = decision or decide_tier(context_len)
        if self.decision.tier is not KVCacheTier.HOST_RAM_OFFLOAD and not force:
            raise OffloadPolicyError(
                f"context {context_len} routed to "
                f"{self.decision.tier.value} by placement policy; offload "
                "path refused (policy is the gate). Use force=True only for "
                "explicit config/override."
            )

        self.context_len = int(context_len)
        self.tokens_per_page = max(1, int(tokens_per_page))
        self.bytes_per_token = max(1, int(bytes_per_token))
        self.bytes_per_page = self.tokens_per_page * self.bytes_per_token
        self.num_pages = self._compute_num_pages(self.context_len)
        if self.num_pages < 1:
            self.num_pages = 1

        self.gpu_working_set_pages = max(1, int(gpu_working_set_pages))
        self.max_inflight_transfers = max(1, int(max_inflight_transfers))
        self.max_concurrent_sessions = max(1, int(max_concurrent_sessions))
        # Bounds how many host<->GPU copies may be in flight at once so PCIe
        # saturation stays bounded. Every stage/write-back task acquires
        # this before touching the backend.
        self._transfer_sem = asyncio.Semaphore(self.max_inflight_transfers)

        # Memory caps: default host cap is the full context at bytes/token
        # (the entire KV cache fits); default GPU cap is the working set.
        self.host_cap_bytes = int(host_cap_bytes or (self.bytes_per_page * self.num_pages))
        self.gpu_cap_bytes = int(
            gpu_cap_bytes or (self.bytes_per_page * self.gpu_working_set_pages)
        )

        self.transfer = transfer or SimulatedAsyncTransferBackend()
        self.session_lock = session_lock or asyncio.Lock()
        self._session_acquired = False
        self._session_owner = None

        self.pages: List[KVPage] = [
            KVPage(
                index=i,
                tokens_per_page=self.tokens_per_page,
                bytes_per_page=self.bytes_per_page,
            )
            for i in range(self.num_pages)
        ]
        self._page_by_index = {p.index: p for p in self.pages}

        # GPU working set (LRU) + in-flight transfer dedup.
        self._gpu_lru: List[int] = []  # page indices, MRU first
        self._inflight: Dict[int, asyncio.Task] = {}

        # Accounting.
        self.stats = OffloadStats(
            context_len=self.context_len,
            page_size=self.bytes_per_page,
            num_pages=self.num_pages,
            bytes_per_token=self.bytes_per_token,
            host_cap_bytes=self.host_cap_bytes,
            gpu_cap_bytes=self.gpu_cap_bytes,
        )
        self.closed = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_num_pages(context_len: int, tokens_per_page: int = DEFAULT_TOKENS_PER_PAGE) -> int:
        return max(1, math.ceil(context_len / max(1, tokens_per_page)))

    # ------------------------------------------------------------------
    # Memory accounting
    # ------------------------------------------------------------------

    def host_resident_bytes(self) -> int:
        return sum(p.bytes_per_page for p in self.pages if p.host_allocated)

    def gpu_resident_bytes(self) -> int:
        """Bytes currently committed to the GPU working set. Includes pages
        that are staged (transfer in flight) because the GPU buffer is
        reserved from issue time, not from copy completion."""
        return sum(
            p.bytes_per_page for p in self.pages if p.gpu_allocated
        )

    def _note_accounting(self) -> None:
        host = self.host_resident_bytes()
        gpu = self.gpu_resident_bytes()
        self.stats.host_resident_bytes = host
        self.stats.gpu_resident_bytes = gpu
        self.stats.host_peak_bytes = max(self.stats.host_peak_bytes, host)
        self.stats.gpu_peak_bytes = max(self.stats.gpu_peak_bytes, gpu)

    def _check_host_cap(self, additional_bytes: int) -> None:
        if self.host_resident_bytes() + additional_bytes > self.host_cap_bytes:
            raise HostRamCapacityError(
                f"host-RAM KV allocation of {additional_bytes} bytes would "
                f"exceed cap {self.host_cap_bytes} (resident "
                f"{self.host_resident_bytes()})"
            )

    def _check_gpu_cap(self, additional_bytes: int) -> None:
        if self.gpu_resident_bytes() + additional_bytes > self.gpu_cap_bytes:
            raise HostRamCapacityError(
                f"GPU working-set allocation of {additional_bytes} bytes "
                f"would exceed cap {self.gpu_cap_bytes} (resident "
                f"{self.gpu_resident_bytes()})"
            )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "HostRamKVCache":
        if self.closed:
            raise RuntimeError("cache is closed; a session is single-use")
        task = asyncio.current_task()
        if self._session_acquired and self._session_owner is task:
            raise RuntimeError("session already acquired by this task")
        await self.session_lock.acquire()
        self._session_acquired = True
        self._session_owner = task
        logger.debug(
            "offload session enter ctx=%d pages=%d host_cap=%d gpu_cap=%d",
            self.context_len, self.num_pages, self.host_cap_bytes, self.gpu_cap_bytes,
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Evict everything back to host RAM and release the session slot."""
        if self.closed:
            return
        await self._evict_all_to_host()
        # Cancel any lingering in-flight transfers (should be none after the
        # drain above, but be defensive).
        for task in list(self._inflight.values()):
            if not task.done():
                task.cancel()
        self._inflight.clear()
        self.closed = True
        self._note_accounting()
        if self._session_acquired:
            self.session_lock.release()
            self._session_acquired = False
            self._session_owner = None
        logger.debug("offload session close stats=%s", self.stats.to_dict())

    # ------------------------------------------------------------------
    # Host allocation
    # ------------------------------------------------------------------

    def _allocate_host(self, page: KVPage) -> None:
        """Mark a page host-resident. Called lazily so pages that are never
        touched cost nothing; the full-context host allocation only happens
        when the decode window actually reaches those pages."""
        if page.host_allocated:
            return
        self._check_host_cap(page.bytes_per_page)
        page.host_bytes = b"\x00" * page.bytes_per_page
        page.host_allocated = True
        if page.state is KVPageState.COLD:
            page.state = KVPageState.HOST
        self._note_accounting()

    # ------------------------------------------------------------------
    # GPU working set (bounded, LRU)
    # ------------------------------------------------------------------

    def _touch_gpu_lru(self, index: int) -> None:
        """Move a page to MRU position. O(n) is fine: working set is small."""
        try:
            self._gpu_lru.remove(index)
        except ValueError:
            pass
        self._gpu_lru.insert(0, index)

    async def _evict_lru_if_needed(self) -> None:
        """Evict the LRU GPU page(s) if the working set is at its cap.

        If the LRU victim is mid-stage (host->GPU copy still in flight), we
        must wait for that copy to land before we can evict it — a page
        mid-transfer cannot be written back yet. This is the only case where
        prefetch stalls: it happens when a single prefetch window requests
        more pages than the working set can hold, which is a caller error;
        graceful stall beats over-committing the GPU cap.

        The GPU slot is freed synchronously on eviction; the host write-back
        runs as a background task. Never blocks except in the in-flight
        victim case above."""
        while len(self._gpu_lru) >= self.gpu_working_set_pages:
            victim_index = self._gpu_lru[-1]
            if victim_index in self._inflight:
                # Wait for the in-flight copy to finish (it will become GPU
                # and thus evictable on the next iteration).
                task = self._inflight[victim_index]
                await asyncio.shield(task)
                continue
            victim = self._page_by_index[victim_index]
            self._gpu_lru.pop()
            self._evict_page(victim)
            self.stats.working_set_evictions += 1

    def _evict_page(self, page: KVPage) -> None:
        """Write a GPU page back to host RAM and free its GPU slot.

        The GPU slot is freed *immediately* (synchronously) so the LRU
        replacement can admit a new page even while the write-back copy is
        still in flight; the copy drains the buffer that the working set
        has already released. ``gpu_resident_bytes()`` therefore counts
        only pages the decode loop can actually touch, which is the
        working-set contract."""
        if page.gpu_allocated and page.host_allocated:
            page.gpu_allocated = False  # slot freed now
            page.state = KVPageState.EVICTING
            self._inflight[page.index] = asyncio.ensure_future(
                self._write_back(page)
            )
        else:
            # Nothing to write back; just drop the GPU side.
            page.gpu_bytes = None
            page.gpu_allocated = False
            page.state = KVPageState.HOST if page.host_allocated else KVPageState.COLD

    async def _write_back(self, page: KVPage) -> None:
        """GPU->host copy. Host bytes are always the source of truth after
        this completes."""
        try:
            async with self._transfer_sem:
                await self.transfer.copy_gpu_to_host(page)
        finally:
            page.gpu_bytes = None
            page.gpu_allocated = False
            page.state = KVPageState.HOST if page.host_allocated else KVPageState.COLD
            self.stats.g2h_transfers += 1
            self._note_accounting()
            self._inflight.pop(page.index, None)

    # ------------------------------------------------------------------
    # Prefetch / fetch (the async overlap contract)
    # ------------------------------------------------------------------

    async def prefetch(self, indices: Sequence[int]) -> None:
        """Issue asynchronous host->GPU copies for pages that will be decoded
        soon. Returns immediately; transfers run in the background and are
        deduplicated (a page already staged or already in flight is skipped).
        Decode compute can proceed while these copies overlap."""
        if self.closed:
            raise RuntimeError("cache is closed")
        for raw in indices:
            index = int(raw)
            if index < 0 or index >= self.num_pages:
                raise IndexError(f"page index {index} out of range 0..{self.num_pages - 1}")
            page = self._page_by_index[index]
            if page.state is KVPageState.GPU or page.index in self._inflight:
                continue  # already resident or already being staged
            self._allocate_host(page)
            await self._stage_to_gpu(page)

    async def _stage_to_gpu(self, page: KVPage) -> None:
        """Begin an async host->GPU copy for a page. The GPU slot is
        reserved immediately (bounded by the working-set cap via eviction);
        the actual copy is a background task so issue does not block."""
        if page.index in self._inflight or page.state is KVPageState.GPU:
            return
        # Make room first (evict LRU pages), then verify the cap against the
        # current resident bytes (which excludes this page), then reserve
        # the slot. The slot counts from issue time so the working set can
        # never exceed the cap even while transfers are in flight.
        await self._evict_lru_if_needed()
        self._check_gpu_cap(page.bytes_per_page)
        page.gpu_allocated = True  # reserve GPU buffer immediately
        self._gpu_lru.insert(0, page.index)
        page.state = KVPageState.STAGING
        self._inflight[page.index] = asyncio.ensure_future(
            self._stage_complete(page)
        )

    async def _stage_complete(self, page: KVPage) -> None:
        try:
            async with self._transfer_sem:
                await self.transfer.copy_host_to_gpu(page)
        except Exception:
            # Transfer failed: release the reserved GPU slot and drop back
            # to host residence so the decode loop can retry or surface the
            # error instead of leaving a zombie GPU allocation.
            page.gpu_bytes = None
            page.gpu_allocated = False
            page.state = KVPageState.HOST if page.host_allocated else KVPageState.COLD
            try:
                self._gpu_lru.remove(page.index)
            except ValueError:
                pass
            self._note_accounting()
            self._inflight.pop(page.index, None)
            raise
        else:
            page.state = KVPageState.GPU
            page.gpu_bytes = page.host_bytes  # simulated staging
            page.last_gpu_access = time.monotonic()
            self.stats.h2g_transfers += 1
            self._note_accounting()
            self._inflight.pop(page.index, None)

    async def fetch(self, index: int) -> KVPage:
        """Return the page's GPU payload, staging it if necessary.

        This is the decode-loop touch point. When the page was prefetched
        early enough (or is already resident), it returns without waiting —
        the async overlap did its job. When a page arrives cold (prefetch
        lost the race), the decode loop waits for the host->GPU copy.
        """
        if self.closed:
            raise RuntimeError("cache is closed")
        index = int(index)
        if index < 0 or index >= self.num_pages:
            raise IndexError(f"page index {index} out of range 0..{self.num_pages - 1}")
        page = self._page_by_index[index]

        if page.state is KVPageState.GPU:
            page.last_gpu_access = time.monotonic()
            self._touch_gpu_lru(index)
            return page

        if page.index in self._inflight:
            # Already being staged (or being written back). Wait for the
            # in-flight copy to land, then confirm the page actually made
            # it to the GPU — a write-back completion leaves it host-
            # resident, in which case we must stage it afresh.
            self.stats.prefetch_hits += 1
            started = time.monotonic()
            task = self._inflight[page.index]
            await asyncio.shield(task)
            self.stats.fetch_wait_s += time.monotonic() - started
            if page.state is not KVPageState.GPU:
                # The in-flight copy was a write-back; stage again and wait
                # for the fresh host->GPU copy.
                self._allocate_host(page)
                await self._stage_to_gpu(page)
                if page.index in self._inflight:
                    await asyncio.shield(self._inflight[page.index])
            self._touch_gpu_lru(index)
            return page

        # Cold miss: allocate host side, stage synchronously (via the same
        # async machinery, but we must wait for the copy before returning).
        self._allocate_host(page)
        started = time.monotonic()
        await self._stage_to_gpu(page)
        task = self._inflight[page.index]
        await asyncio.shield(task)
        self.stats.fetch_wait_s += time.monotonic() - started
        self._touch_gpu_lru(index)
        return page

    async def release(self, index: int) -> None:
        """Write a page back to host RAM asynchronously (dirty write-back),
        freeing its GPU working-set slot. The host copy remains the source
        of truth."""
        if self.closed:
            raise RuntimeError("cache is closed")
        index = int(index)
        if index < 0 or index >= self.num_pages:
            raise IndexError(f"page index {index} out of range 0..{self.num_pages - 1}")
        page = self._page_by_index[index]
        if page.index in self._inflight:
            # Let an in-flight copy finish; then it is GPU-resident and the
            # caller can release it on a later pass.
            await asyncio.shield(self._inflight[page.index])
        if page.state is KVPageState.GPU:
            self._evict_page(page)
            try:
                self._gpu_lru.remove(index)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Working set state (test/observability surface)
    # ------------------------------------------------------------------

    def gpu_resident_indices(self) -> List[int]:
        return sorted(p.index for p in self.pages if p.state is KVPageState.GPU)

    def host_resident_indices(self) -> List[int]:
        return sorted(p.index for p in self.pages if p.host_allocated)

    def page_state(self, index: int) -> KVPageState:
        return self._page_by_index[int(index)].state

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    async def _evict_all_to_host(self) -> None:
        """Write back every GPU-resident page and drain in-flight transfers.
        The host side becomes the only resident copy (the steady state for
        a session that is ending)."""
        # Loop until no GPU-resident page remains: pages that finish staging
        # mid-drain transition to GPU and must be written back too.
        while True:
            to_evict = [
                p
                for p in self.pages
                if p.state is KVPageState.GPU and p.index not in self._inflight
            ]
            if not to_evict:
                break
            for page in to_evict:
                self._evict_page(page)
            inflight = list(self._inflight.values())
            if inflight:
                await asyncio.gather(*inflight, return_exceptions=True)
        # Defensive: drain any remaining in-flight transfers (staging that
        # was mid-flight during the loop above).
        inflight = list(self._inflight.values())
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        for page in self.pages:
            page.state = KVPageState.HOST if page.host_allocated else KVPageState.COLD
        self._gpu_lru.clear()
        self._note_accounting()


# ---------------------------------------------------------------------------
# Session pool — bounded concurrency for simultaneous long-context requests
# ---------------------------------------------------------------------------


class OffloadSessionPool:
    """Bounded pool of offload sessions.

    Concurrency is the danger zone for the offload path: each session holds
    a multi-hundred-MB to multi-GB KV cache in host RAM and competes for the
    shared GPU working set. The pool bounds simultaneous sessions
    (``max_sessions``) with an asyncio semaphore so a flood of long-context
    requests cannot exhaust host RAM or starve ordinary traffic. Requests
    beyond the cap wait (or time out) instead of degrading the serving loop.
    """

    def __init__(
        self,
        max_sessions: int = DEFAULT_MAX_CONCURRENT_SESSIONS,
        *,
        acquire_timeout_s: Optional[float] = None,
        transfer: Optional[TransferBackend] = None,
        tokens_per_page: int = DEFAULT_TOKENS_PER_PAGE,
        bytes_per_token: int = DEFAULT_BYTES_PER_TOKEN,
        gpu_working_set_pages: int = DEFAULT_GPU_WORKING_SET_PAGES,
        max_inflight_transfers: int = DEFAULT_MAX_INFLIGHT_TRANSFERS,
        host_cap_bytes: Optional[int] = None,
        gpu_cap_bytes: Optional[int] = None,
        force: bool = False,
        decision: Optional[PlacementDecision] = None,
    ) -> None:
        self.max_sessions = max(1, int(max_sessions))
        self._sem = asyncio.Semaphore(self.max_sessions)
        self.acquire_timeout_s = acquire_timeout_s
        self._base_kwargs: Dict = dict(
            transfer=transfer,
            tokens_per_page=tokens_per_page,
            bytes_per_token=bytes_per_token,
            gpu_working_set_pages=gpu_working_set_pages,
            max_inflight_transfers=max_inflight_transfers,
            host_cap_bytes=host_cap_bytes,
            gpu_cap_bytes=gpu_cap_bytes,
            force=force,
            decision=decision,
        )
        self._active = 0
        self._peak_active = 0
        self._lock = asyncio.Lock()

    @property
    def active_sessions(self) -> int:
        return self._active

    @property
    def peak_active_sessions(self) -> int:
        return self._peak_active

    async def acquire(self, context_len: int, **overrides) -> "HostRamKVCache":
        """Acquire a session for a context. Waits up to ``acquire_timeout_s``
        for a slot; raises ``asyncio.TimeoutError`` when the pool is
        saturated and the wait budget is exceeded. The returned cache is
        already inside its ``async with`` session (slot held); the caller
        must ``await cache.close()`` (or ``async with`` it, which is the
        canonical form) to release it.

        The pool semaphore is held for the *lifetime* of the session and
        only released by ``release()``, so ``max_sessions`` genuinely bounds
        simultaneous offload sessions."""
        if self.acquire_timeout_s is not None:
            try:
                await asyncio.wait_for(
                    self._sem.acquire(), timeout=self.acquire_timeout_s
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"offload session pool saturated "
                    f"({self.max_sessions} sessions); request for context "
                    f"{context_len} timed out after {self.acquire_timeout_s}s"
                )
        else:
            await self._sem.acquire()
        try:
            kwargs = dict(self._base_kwargs)
            kwargs.update(overrides)
            cache = HostRamKVCache(context_len, **kwargs)
            await cache.__aenter__()  # acquire the per-cache session lock slot
            async with self._lock:
                self._active += 1
                self._peak_active = max(self._peak_active, self._active)
            return cache
        except Exception:
            self._sem.release()  # construction failed; give the slot back
            raise

    async def release(self, cache: "HostRamKVCache") -> None:
        """Release a session acquired via ``acquire``. Idempotent."""
        if cache is None:
            return
        was_closed = cache.closed
        await cache.close()
        if not was_closed:
            async with self._lock:
                self._active -= 1
            self._sem.release()  # the session's lifetime is over


def main(argv=None) -> int:  # pragma: no cover - CLI smoke/diagnostic
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="P12 host-RAM KV-cache offload path (runtime)"
    )
    parser.add_argument("--ctx-size", type=int, required=True,
                        help="requested context length in tokens")
    parser.add_argument("--force", action="store_true",
                        help="allow offload path even if policy routes to GPU")
    parser.add_argument("--gpu-pages", type=int,
                        default=DEFAULT_GPU_WORKING_SET_PAGES,
                        help="GPU working-set size in pages")
    parser.add_argument("--tokens-per-page", type=int,
                        default=DEFAULT_TOKENS_PER_PAGE)
    args = parser.parse_args(argv)

    async def run():
        cache = HostRamKVCache(
            args.ctx_size,
            tokens_per_page=args.tokens_per_page,
            gpu_working_set_pages=args.gpu_pages,
            force=args.force,
        )
        async with cache:
            await cache.prefetch(range(0, min(cache.num_pages, args.gpu_pages)))
            if cache.num_pages > 0:
                await cache.fetch(0)
        print(json.dumps(cache.stats.to_dict(), indent=2, sort_keys=True))

    try:
        asyncio.run(run())
        return 0
    except (OffloadPolicyError, HostRamCapacityError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
