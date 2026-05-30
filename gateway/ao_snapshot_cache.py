"""Optional Redis-backed snapshots for expensive AO live-state reads."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AOSnapshot:
    project_id: Optional[str]
    sessions: list[dict[str, Any]]
    health_by_id: dict[str, dict[str, Any]]
    captured_at: float
    cache_status: str
    backend: str
    total_ms: float
    age_ms: Optional[float] = None
    refresh_ms: float = 0.0


class AOSnapshotCache:
    """Small read-through cache for AO session list + runtime health snapshots."""

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        requested_backend = backend or os.getenv("ORYN_AO_SNAPSHOT_CACHE") or os.getenv("ORYN_READ_MODEL_CACHE", "memory")
        self.backend = requested_backend.strip().lower()
        self.ttl_seconds = max(1, int(ttl_seconds or os.getenv("ORYN_AO_SNAPSHOT_TTL_SECONDS", "60")))
        self._memory: dict[str, dict[str, Any]] = {}
        self._redis = self._connect_redis() if self.backend == "redis" else None

    @property
    def enabled(self) -> bool:
        return self.backend != "off"

    def get_or_load(
        self,
        *,
        project_id: Optional[str],
        load: Callable[[], dict[str, Any]],
    ) -> AOSnapshot:
        started = time.perf_counter()
        if not self.enabled:
            refresh_started = time.perf_counter()
            record = self._normalize_record(project_id=project_id, value=load())
            return self._snapshot(
                project_id=project_id,
                record=record,
                cache_status="bypass",
                started=started,
                refresh_ms=(time.perf_counter() - refresh_started) * 1000.0,
            )

        key = self._cache_key(project_id)
        cached = self._get(key)
        if cached is not None:
            age_ms = max(0.0, (time.time() - float(cached.get("captured_at") or 0)) * 1000.0)
            return self._snapshot(
                project_id=project_id,
                record=cached,
                cache_status=cached.get("_cache_status") or "memory_hit",
                started=started,
                age_ms=age_ms,
            )

        refresh_started = time.perf_counter()
        record = self._normalize_record(project_id=project_id, value=load())
        refresh_ms = (time.perf_counter() - refresh_started) * 1000.0
        self._set(key, record)
        snapshot = self._snapshot(
            project_id=project_id,
            record=record,
            cache_status="miss",
            started=started,
            refresh_ms=refresh_ms,
        )
        logger.info(
            "oryn ao-snapshot project=%s cache=%s backend=%s total_ms=%.2f refresh_ms=%.2f sessions=%s",
            project_id or "all",
            snapshot.cache_status,
            snapshot.backend,
            snapshot.total_ms,
            snapshot.refresh_ms,
            len(snapshot.sessions),
        )
        return snapshot

    def invalidate(self, project_id: Optional[str] = None) -> None:
        if project_id is None:
            self._memory.clear()
            if self._redis is not None:
                try:
                    for raw_key in self._redis.scan_iter("oryn:ao:snapshot:*"):
                        self._redis.delete(raw_key)
                except Exception as exc:
                    logger.debug("Redis AO snapshot invalidation failed: %s", exc)
            return

        key = self._cache_key(project_id)
        self._memory.pop(key, None)
        if self._redis is not None:
            try:
                self._redis.delete(self._redis_key(key))
            except Exception as exc:
                logger.debug("Redis AO snapshot invalidation failed: %s", exc)

    def _get(self, key: str) -> Optional[dict[str, Any]]:
        now = time.time()
        item = self._memory.get(key)
        if item is not None:
            captured_at = float(item.get("captured_at") or 0)
            if now - captured_at <= self.ttl_seconds:
                item = dict(item)
                item["_cache_status"] = "memory_hit"
                return item
            self._memory.pop(key, None)

        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._redis_key(key))
            if not raw:
                return None
            decoded = json.loads(raw)
            decoded["_cache_status"] = "redis_hit"
            self._memory[key] = {
                "project_id": decoded.get("project_id"),
                "sessions": decoded.get("sessions") or [],
                "health_by_id": decoded.get("health_by_id") or {},
                "captured_at": decoded.get("captured_at") or now,
            }
            return decoded
        except Exception as exc:
            logger.debug("Redis AO snapshot get failed: %s", exc)
            return None

    def _set(self, key: str, record: dict[str, Any]) -> None:
        self._memory[key] = record
        if self._redis is None:
            return
        try:
            self._redis.setex(
                self._redis_key(key),
                self.ttl_seconds,
                json.dumps(record, ensure_ascii=False, default=str),
            )
        except Exception as exc:
            logger.debug("Redis AO snapshot set failed: %s", exc)

    def _connect_redis(self):
        redis_url = os.getenv("REDIS_URL", "").strip()
        if not redis_url:
            logger.warning("ORYN_AO_SNAPSHOT_CACHE=redis but REDIS_URL is unset; using memory snapshots")
            return None
        try:
            import redis  # type: ignore

            client = redis.Redis.from_url(redis_url, socket_timeout=0.25, socket_connect_timeout=0.25)
            client.ping()
            return client
        except Exception as exc:
            logger.warning("Redis AO snapshot cache unavailable; using memory snapshots: %s", exc)
            return None

    def _snapshot(
        self,
        *,
        project_id: Optional[str],
        record: dict[str, Any],
        cache_status: str,
        started: float,
        age_ms: Optional[float] = None,
        refresh_ms: float = 0.0,
    ) -> AOSnapshot:
        return AOSnapshot(
            project_id=project_id,
            sessions=list(record.get("sessions") or []),
            health_by_id=dict(record.get("health_by_id") or {}),
            captured_at=float(record.get("captured_at") or time.time()),
            cache_status=cache_status,
            backend=self.backend,
            total_ms=(time.perf_counter() - started) * 1000.0,
            age_ms=age_ms,
            refresh_ms=refresh_ms,
        )

    @staticmethod
    def _normalize_record(*, project_id: Optional[str], value: dict[str, Any]) -> dict[str, Any]:
        return {
            "project_id": project_id,
            "sessions": list(value.get("sessions") or []),
            "health_by_id": dict(value.get("health_by_id") or {}),
            "captured_at": time.time(),
        }

    @staticmethod
    def _cache_key(project_id: Optional[str]) -> str:
        scope = project_id or "all"
        digest = sha256(scope.encode("utf-8")).hexdigest()
        return f"{scope[:24]}:{digest}"

    @staticmethod
    def _redis_key(key: str) -> str:
        return f"oryn:ao:snapshot:{key}"
