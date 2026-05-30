"""Optional read-model cache for Oryn-facing gateway views."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedReadModel:
    fingerprint: str
    payload: dict[str, Any]
    cache_status: str
    total_ms: float
    cache_read_ms: float = 0.0
    compute_ms: float = 0.0
    store_ms: float = 0.0
    cache_age_ms: Optional[float] = None
    backend: str = "memory"


class ReadModelCache:
    """Small read-through cache with memory default and optional Redis storage."""

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_items: int = 512,
    ) -> None:
        self.backend = (backend or os.getenv("ORYN_READ_MODEL_CACHE", "memory")).strip().lower()
        self.ttl_seconds = max(1, int(ttl_seconds or os.getenv("ORYN_READ_MODEL_CACHE_TTL_SECONDS", "30")))
        self.max_items = max_items
        self._memory: OrderedDict[str, tuple[float, str, dict[str, Any]]] = OrderedDict()
        self._inflight: dict[tuple[str, str], asyncio.Task[dict[str, Any]]] = {}
        self._redis = self._connect_redis() if self.backend == "redis" else None

    @property
    def enabled(self) -> bool:
        return self.backend != "off"

    async def get_or_compute(
        self,
        *,
        key: str,
        fingerprint: str,
        compute: Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]],
    ) -> CachedReadModel:
        started = time.perf_counter()
        if not self.enabled:
            compute_started = time.perf_counter()
            payload = await self._call_compute(compute)
            compute_ms = (time.perf_counter() - compute_started) * 1000.0
            return CachedReadModel(
                fingerprint=fingerprint,
                payload=payload,
                cache_status="bypass",
                total_ms=(time.perf_counter() - started) * 1000.0,
                compute_ms=compute_ms,
                backend=self.backend,
            )

        cache_started = time.perf_counter()
        cached = await self._get(key)
        cache_read_ms = (time.perf_counter() - cache_started) * 1000.0
        if cached and cached[0] == fingerprint:
            return CachedReadModel(
                fingerprint=fingerprint,
                payload=cached[1],
                cache_status=cached[2],
                total_ms=(time.perf_counter() - started) * 1000.0,
                cache_read_ms=cache_read_ms,
                cache_age_ms=cached[3],
                backend=self.backend,
            )

        inflight_key = (key, fingerprint)
        task = self._inflight.get(inflight_key)
        if task is None:
            task = asyncio.create_task(self._compute_and_store(key, fingerprint, compute))
            self._inflight[inflight_key] = task
            task.add_done_callback(lambda done: self._inflight.pop(inflight_key, None))
            cache_status = "stale" if cached else "miss"
        else:
            cache_status = "inflight"

        payload, compute_ms, store_ms = await asyncio.shield(task)
        return CachedReadModel(
            fingerprint=fingerprint,
            payload=payload,
            cache_status=cache_status,
            total_ms=(time.perf_counter() - started) * 1000.0,
            cache_read_ms=cache_read_ms,
            compute_ms=compute_ms,
            store_ms=store_ms,
            backend=self.backend,
        )

    def invalidate_prefix(self, prefix: str) -> None:
        for key in list(self._memory.keys()):
            if key.startswith(prefix):
                self._memory.pop(key, None)
        if self._redis is not None:
            try:
                for raw_key in self._redis.scan_iter(f"oryn:read-model:{prefix}*"):
                    self._redis.delete(raw_key)
            except Exception as exc:
                logger.debug("Redis read-model invalidation failed: %s", exc)

    async def _compute_and_store(
        self,
        key: str,
        fingerprint: str,
        compute: Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]],
    ) -> tuple[dict[str, Any], float, float]:
        compute_started = time.perf_counter()
        payload = await self._call_compute(compute)
        compute_ms = (time.perf_counter() - compute_started) * 1000.0
        store_started = time.perf_counter()
        await self._set(key, fingerprint, payload)
        store_ms = (time.perf_counter() - store_started) * 1000.0
        return payload, compute_ms, store_ms

    async def _call_compute(
        self,
        compute: Callable[[], Awaitable[dict[str, Any]] | dict[str, Any]],
    ) -> dict[str, Any]:
        value = compute()
        if asyncio.iscoroutine(value):
            value = await value
        return value

    async def _get(self, key: str) -> Optional[tuple[str, dict[str, Any], str, Optional[float]]]:
        now = time.time()
        item = self._memory.get(key)
        if item is not None:
            ts, fingerprint, payload = item
            if now - ts <= self.ttl_seconds:
                self._memory.move_to_end(key)
                return fingerprint, payload, "memory_hit", (now - ts) * 1000.0
            self._memory.pop(key, None)

        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._redis_key(key))
            if not raw:
                return None
            decoded = json.loads(raw)
            payload = decoded.get("payload")
            if not isinstance(payload, dict):
                return None
            ts = float(decoded.get("stored_at") or 0)
            age_ms = ((now - ts) * 1000.0) if ts else None
            return str(decoded["fingerprint"]), payload, "redis_hit", age_ms
        except Exception as exc:
            logger.debug("Redis read-model cache get failed: %s", exc)
            return None

    async def _set(self, key: str, fingerprint: str, payload: dict[str, Any]) -> None:
        self._memory[key] = (time.time(), fingerprint, payload)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_items:
            self._memory.popitem(last=False)

        if self._redis is None:
            return
        try:
            raw = json.dumps(
                {"fingerprint": fingerprint, "payload": payload, "stored_at": time.time()},
                ensure_ascii=False,
                default=str,
            )
            self._redis.setex(self._redis_key(key), self.ttl_seconds, raw)
        except Exception as exc:
            logger.debug("Redis read-model cache set failed: %s", exc)

    def _connect_redis(self):
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            logger.warning("ORYN_READ_MODEL_CACHE=redis but REDIS_URL is unset; using memory cache")
            return None
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(redis_url, socket_timeout=0.25, socket_connect_timeout=0.25)
            client.ping()
            return client
        except Exception as exc:
            logger.warning("Redis read-model cache unavailable; using memory cache: %s", exc)
            return None

    @staticmethod
    def _redis_key(key: str) -> str:
        return f"oryn:read-model:{key}"


def read_model_etag(fingerprint: str) -> str:
    return f'"{fingerprint}"'


def request_fingerprint(request: Any) -> Optional[str]:
    raw = request.headers.get("If-None-Match") if request is not None else None
    if not raw:
        return None
    return raw.strip().strip('"')


def read_model_metric_headers(cached: CachedReadModel, *, payload_size_bytes: Optional[int] = None) -> dict[str, str]:
    headers = {
        "X-Oryn-Read-Model-Cache": cached.cache_status,
        "X-Oryn-Read-Model-Backend": cached.backend,
        "X-Oryn-Read-Model-Total-Ms": f"{cached.total_ms:.2f}",
        "X-Oryn-Read-Model-Cache-Read-Ms": f"{cached.cache_read_ms:.2f}",
        "X-Oryn-Read-Model-Compute-Ms": f"{cached.compute_ms:.2f}",
        "X-Oryn-Read-Model-Store-Ms": f"{cached.store_ms:.2f}",
    }
    if cached.cache_age_ms is not None:
        headers["X-Oryn-Read-Model-Cache-Age-Ms"] = f"{cached.cache_age_ms:.2f}"
    if payload_size_bytes is not None:
        headers["X-Oryn-Read-Model-Bytes"] = str(payload_size_bytes)
    return headers
