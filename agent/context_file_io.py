"""Bounded context-file reads (aiofiles + wait_for + mtime LRU).

``read_text_sync`` offloads to a worker thread when an asyncio loop is already
running so gateway threads avoid unbounded ``Path.read_text`` stalls.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Final

import aiofiles

logger = logging.getLogger(__name__)

_CACHE_MAX: Final[int] = 64
_cache: "OrderedDict[tuple[str, int], str]" = OrderedDict()
_cache_lock = threading.Lock()

# One-off pool for nested "already in asyncio loop" paths (short-lived tasks).
_nested_pool = ThreadPoolExecutor(1, thread_name_prefix="ctxfileio")


def clear_context_file_cache() -> None:
    """Drop all cached context file bodies (tests / hot reload)."""
    with _cache_lock:
        _cache.clear()


def _cache_get(key: tuple[str, int]) -> str | None:
    with _cache_lock:
        val = _cache.get(key)
        if val is not None:
            _cache.move_to_end(key)
        return val


def _cache_put(key: tuple[str, int], text: str) -> None:
    with _cache_lock:
        _cache[key] = text
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)


async def read_text_async(path: Path, *, timeout_s: float = 10.0, encoding: str = "utf-8") -> str:
    """Read *path* with aiofiles under an ``asyncio`` timeout."""
    resolved = path.expanduser()
    try:
        resolved = resolved.resolve(strict=False)
    except OSError as exc:
        logger.debug("context path resolve failed %s: %s", path, exc)
        return ""

    if not resolved.is_file():
        return ""

    try:
        mtime_ns = await asyncio.wait_for(
            asyncio.to_thread(lambda: resolved.stat().st_mtime_ns),
            timeout=min(2.0, timeout_s),
        )
    except (OSError, asyncio.TimeoutError) as exc:
        logger.debug("context file stat failed %s: %s", resolved, exc)
        return ""

    key = (str(resolved), int(mtime_ns))
    hit = _cache_get(key)
    if hit is not None:
        return hit

    async def _body() -> str:
        async with aiofiles.open(resolved, mode="r", encoding=encoding) as handle:
            return await handle.read()

    try:
        text = await asyncio.wait_for(_body(), timeout=timeout_s)
    except (asyncio.TimeoutError, OSError, UnicodeDecodeError) as exc:
        logger.debug("context async read failed %s: %s", resolved, exc)
        return ""

    _cache_put(key, text)
    return text


def read_text_sync(path: Path, *, timeout_s: float = 10.0, encoding: str = "utf-8") -> str:
    """Synchronous API used by ``prompt_builder`` (thread-safe, bounded latency)."""
    resolved = path.expanduser()
    try:
        resolved = resolved.resolve(strict=False)
    except OSError as exc:
        logger.debug("context path resolve failed %s: %s", path, exc)
        return ""

    if not resolved.is_file():
        return ""

    try:
        mtime_ns = resolved.stat().st_mtime_ns
    except OSError as exc:
        logger.debug("context file stat failed %s: %s", resolved, exc)
        return ""

    key = (str(resolved), int(mtime_ns))
    hit = _cache_get(key)
    if hit is not None:
        return hit

    def _fresh_read() -> str:
        return asyncio.run(read_text_async(resolved, timeout_s=timeout_s, encoding=encoding))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        try:
            return _fresh_read()
        except RuntimeError:
            fut = _nested_pool.submit(_fresh_read)
            try:
                return fut.result(timeout=timeout_s + 5.0)
            except Exception as exc:  # noqa: BLE001
                logger.debug("context read fallback failed %s: %s", resolved, exc)
                return ""
    else:
        fut = _nested_pool.submit(_fresh_read)
        try:
            return fut.result(timeout=timeout_s + 5.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("context read (nested thread) failed %s: %s", resolved, exc)
            return ""


def read_many_sync(paths: list[Path], *, timeout_s: float = 10.0, encoding: str = "utf-8") -> list[str]:
    """Read several paths in parallel (each call is independently bounded)."""
    if not paths:
        return []
    workers = min(8, len(paths))
    out: list[str | None] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(read_text_sync, p, timeout_s=timeout_s, encoding=encoding): i for i, p in enumerate(paths)}
        for fut in as_completed(futs):
            idx = futs[fut]
            try:
                out[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.debug("parallel context read failed: %s", exc)
                out[idx] = ""
    return [x if x is not None else "" for x in out]
