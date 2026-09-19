# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------------
# Portions of this file are adapted from BaiLongma
#   Upstream: https://github.com/xiaoyuanda666-ship-it/BaiLongma
#   Original: src/prefetch/runner.js
#             src/db/repositories/prefetch.js (cache schema shape only)
#   Copyright (c) 2026 xiaoyuanda666-ship-it — Licensed under MIT
#   License text: see LICENSES/BaiLongma-MIT.txt
# ---------------------------------------------------------------------------
"""Background prefetch runner.

Runs a set of registered "prefetch tasks" concurrently, each producing
a text payload that gets stored in a caller-supplied cache. Tasks
declare their own TTL and tag set; the cache decides when to expire
them and how to serve them back to future tool calls (or memory
retrieval).

## Why this is a *primitive*, not a set of tasks

BaiLongma's upstream file bundles the framework AND three hardcoded
tasks (Beijing weather, Lufeng weather, HackerNews top). Those are
business logic — they don't belong in a general-purpose agent. This
port keeps only the **framework**:

* :class:`PrefetchTask`      — task descriptor + async fetcher
* :class:`PrefetchRunner`    — register/run/settle-all orchestrator
* :class:`PrefetchCacheStore` — protocol the caller implements

The runner has **zero built-in tasks**. Callers register their own
via :meth:`PrefetchRunner.register` or pass a dynamic-task provider
callable (mirroring BaiLongma's DB-backed ``getEnabledPrefetchTasks``
without hardcoding a DB).

## Design invariants

* **Never blocks the caller.** All tasks run under
  :func:`asyncio.gather` with ``return_exceptions=True`` — a single
  failing task never aborts the batch.
* **Cache-agnostic.** The runner never touches the filesystem, sqlite,
  or HTTP directly. Task fetchers own the "get me content" side;
  the store owns the "persist for later" side.
* **Best-effort logging.** Failures log at WARNING and are returned
  as :class:`PrefetchOutcome` entries with ``error`` populated — no
  exceptions escape :meth:`PrefetchRunner.run`.
* **Stdlib only.** No new deps.

Ported from BaiLongma's ``src/prefetch/runner.js`` (MIT).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


logger = logging.getLogger(__name__)


# ── Types ──────────────────────────────────────────────────────────


PrefetchFetcher = Callable[[], Awaitable[str]]
"""Async callable that returns the text payload for one prefetch pass.

Fetchers should be self-contained — they own their HTTP client, their
timeout, and their parsing. The runner passes them no arguments.
"""


@dataclass
class PrefetchTask:
    """One prefetch task descriptor.

    ``source`` is the primary cache key — pick something stable and
    filter-friendly like ``"weather:Beijing"`` or ``"news:hn"``.
    ``tags`` is optional metadata the cache can use for tag-based
    invalidation / lookup.
    """

    source: str
    fetch: PrefetchFetcher
    ttl_minutes: int = 60
    label: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PrefetchOutcome:
    """Per-task result after a run.

    ``ok=True`` means the fetcher completed and the cache write was
    invoked. Cache-write failures are treated as fatal for the entry
    (the outcome switches to ``ok=False`` with ``error`` populated).
    """

    source: str
    label: str
    ok: bool
    error: Optional[str] = None


@runtime_checkable
class PrefetchCacheStore(Protocol):
    """Contract for the caller's cache backend.

    Two synchronous ops are enough — this deliberately doesn't force
    the caller to hand us an async store; a sync SQLite / dict / file
    store is the common case.
    """

    def save(
        self,
        *,
        source: str,
        content: str,
        ttl_minutes: int,
        tags: Sequence[str],
    ) -> None:
        ...

    def clear_expired(self) -> None:
        ...


# Optional dynamic-task provider — BaiLongma's DB-backed task list.
# In Hermes this is just a callable, so the caller can plug in any
# source (DB, config file, plugin registry) without the runner caring.
DynamicTaskProvider = Callable[[], Iterable[PrefetchTask]]


# ── Runner ─────────────────────────────────────────────────────────


class PrefetchRunner:
    """Register prefetch tasks and run them concurrently.

    Usage::

        cache = MyCacheStore()   # implements PrefetchCacheStore
        runner = PrefetchRunner(cache=cache)
        runner.register(PrefetchTask(
            source="weather:beijing",
            fetch=fetch_beijing_weather,
            ttl_minutes=60,
            label="Beijing weather",
            tags=["weather", "beijing"],
        ))
        outcomes = await runner.run()
    """

    def __init__(
        self,
        *,
        cache: PrefetchCacheStore,
        dynamic_provider: Optional[DynamicTaskProvider] = None,
    ) -> None:
        self._cache = cache
        self._dynamic_provider = dynamic_provider
        self._tasks: dict[str, PrefetchTask] = {}

    # ── Registration ───────────────────────────────────────────────

    def register(self, task: PrefetchTask) -> None:
        """Register (or overwrite) a task by ``source``.

        BaiLongma's upstream uses list-append semantics; we prefer
        keyed replacement because two tasks with the same source
        would race on the cache row.
        """
        if not task.source:
            raise ValueError("PrefetchTask.source must be a non-empty string")
        if not callable(task.fetch):
            raise ValueError("PrefetchTask.fetch must be an async callable")
        self._tasks[task.source] = task

    def unregister(self, source: str) -> bool:
        return self._tasks.pop(source, None) is not None

    def registered_sources(self) -> list[str]:
        return sorted(self._tasks.keys())

    # ── Task assembly ──────────────────────────────────────────────

    def _resolve_dynamic_tasks(self) -> list[PrefetchTask]:
        if self._dynamic_provider is None:
            return []
        try:
            provided = list(self._dynamic_provider())
        except Exception as err:  # noqa: BLE001 — a broken provider
            # must not block the static tasks.
            logger.warning(
                "[prefetch] dynamic task provider raised %s", err
            )
            return []
        # Reject duplicates so the static registry always wins.
        return [t for t in provided if t.source not in self._tasks]

    def _select_tasks(
        self, sources: Optional[Sequence[str]] = None
    ) -> list[PrefetchTask]:
        all_tasks = list(self._tasks.values()) + self._resolve_dynamic_tasks()
        if sources is None:
            return all_tasks
        wanted = set(sources)
        return [t for t in all_tasks if t.source in wanted]

    # ── Execution ─────────────────────────────────────────────────

    async def _run_one(self, task: PrefetchTask) -> PrefetchOutcome:
        try:
            content = await task.fetch()
        except asyncio.CancelledError:
            raise
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "[prefetch] ✗ %s: fetch failed: %s",
                task.label or task.source,
                err,
            )
            return PrefetchOutcome(
                source=task.source,
                label=task.label,
                ok=False,
                error=str(err) or err.__class__.__name__,
            )

        try:
            self._cache.save(
                source=task.source,
                content=content,
                ttl_minutes=task.ttl_minutes,
                tags=list(task.tags),
            )
        except Exception as err:  # noqa: BLE001
            logger.warning(
                "[prefetch] ✗ %s: cache write failed: %s",
                task.label or task.source,
                err,
            )
            return PrefetchOutcome(
                source=task.source,
                label=task.label,
                ok=False,
                error=str(err) or err.__class__.__name__,
            )

        logger.info(
            "[prefetch] ✓ %s (ttl=%dm, tags=%s)",
            task.label or task.source,
            task.ttl_minutes,
            task.tags,
        )
        return PrefetchOutcome(
            source=task.source, label=task.label, ok=True
        )

    async def run(
        self, task_sources: Optional[Sequence[str]] = None
    ) -> list[PrefetchOutcome]:
        """Run all (or a subset of) registered tasks concurrently.

        Expired cache entries are cleaned up first, then tasks run
        under :func:`asyncio.gather`. Individual failures are captured
        in the returned outcome list — this method never raises.
        """
        try:
            self._cache.clear_expired()
        except Exception as err:  # noqa: BLE001 — expiration cleanup
            # is best-effort; failing here would defeat the runner.
            logger.warning(
                "[prefetch] clear_expired failed: %s", err
            )

        tasks = self._select_tasks(task_sources)
        if not tasks:
            logger.info("[prefetch] no matching tasks")
            return []

        outcomes = await asyncio.gather(
            *(self._run_one(task) for task in tasks),
            return_exceptions=False,
        )
        return list(outcomes)


__all__ = [
    "DynamicTaskProvider",
    "PrefetchCacheStore",
    "PrefetchFetcher",
    "PrefetchOutcome",
    "PrefetchRunner",
    "PrefetchTask",
]
