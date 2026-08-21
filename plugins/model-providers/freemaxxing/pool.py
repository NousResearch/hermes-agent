"""Thread-safe backend state, selection, and catalog authority."""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Set

from .policy import (
    _CATALOG_TTL_SECONDS,
    _MAX_CATALOG_BODY_BYTES,
    _PREFERRED_AUTO_MODELS,
    _ROUTER_MODELS,
    AuthError,
    TransientError,
    _accept_catalog_id,
    _hermes_user_agent,
    _is_router_model,
    _open_credentialed,
)

logger = logging.getLogger("freemaxxing.proxy")


class Backend:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        tier: int = 0,
        refresh: Optional[Callable[[], tuple[str, str]]] = None,
        default_model: str = "",
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tier = int(tier)
        self.refresh = refresh
        self.default_model = default_model

        self.cooldown_until = 0.0
        self.last_success = 0.0
        self.last_error_class: Optional[str] = None
        self.cached_models: Optional[List[str]] = None
        self.cached_models_until = 0.0

        self.refresh_lock = threading.Lock()
        self.catalog_lock = threading.Lock()
        self._state_lock = threading.RLock()

    def is_available(self) -> bool:
        with self._state_lock:
            return time.time() >= self.cooldown_until

    def record_success(self) -> None:
        with self._state_lock:
            self.last_success = time.time()
            self.last_error_class = None
            self.cooldown_until = 0.0

    def record_failure(
        self,
        retry_after: float = 30.0,
        error_class: str = "transient",
    ) -> None:
        retry_after = max(0.0, float(retry_after))
        with self._state_lock:
            self.cooldown_until = max(
                self.cooldown_until,
                time.time() + retry_after,
            )
            self.last_error_class = error_class

    def get_cached_models(self) -> List[str]:
        with self._state_lock:
            if (
                self.cached_models is not None
                and time.time() < self.cached_models_until
            ):
                return list(self.cached_models)
        return []

    def set_cached_models(
        self,
        models: List[str],
        ttl: float = _CATALOG_TTL_SECONDS,
    ) -> None:
        with self._state_lock:
            self.cached_models = list(models)
            self.cached_models_until = time.time() + max(0.0, float(ttl))

    def supports_model(self, model: str) -> bool:
        if not model:
            return True
        cached = self.get_cached_models()
        return True if not cached else model in cached

    def credential_snapshot(self) -> tuple[str, str]:
        with self.refresh_lock:
            return self.base_url, self.api_key

    def health(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "name": self.name,
                "tier": self.tier,
                "available": time.time() >= self.cooldown_until,
                "cooldown_until": self.cooldown_until,
                "last_success": self.last_success,
                "last_error_class": self.last_error_class,
                "models_cached": self.cached_models is not None,
                "models_cached_until": self.cached_models_until,
            }


class BackendPool:
    def __init__(self) -> None:
        self.backends: List[Backend] = []
        self._index = 0
        self._lock = threading.RLock()
        self._catalog_refresh_lock = threading.Lock()
        self._catalog_refresh_until = 0.0

    def add(self, backend: Backend) -> None:
        with self._lock:
            self.backends.append(backend)

    def clear(self) -> None:
        with self._lock:
            self.backends.clear()
            self._index = 0
            self._catalog_refresh_until = 0.0

    def count(self) -> int:
        with self._lock:
            return len(self.backends)

    def snapshot(self) -> List[Backend]:
        with self._lock:
            return list(self.backends)

    def _pick_round_robin(
        self,
        candidates: List[Backend],
    ) -> Optional[Backend]:
        if not candidates:
            return None
        candidate_ids = {id(item) for item in candidates}
        count = len(self.backends)
        for _ in range(count):
            backend = self.backends[self._index]
            self._index = (self._index + 1) % count
            if id(backend) in candidate_ids:
                return backend
        return candidates[0]

    def next(
        self,
        requested_model: str = "",
        exclude: Optional[Set[str]] = None,
    ) -> Optional[Backend]:
        """Return one untried backend with strict tier precedence."""
        excluded = exclude or set()
        with self._lock:
            available = [
                backend
                for backend in self.backends
                if backend.name not in excluded and backend.is_available()
            ]
            if not available:
                return None

            if _is_router_model(requested_model):
                candidates = available
            else:
                supporters = [
                    backend
                    for backend in available
                    if backend.supports_model(requested_model)
                ]
                candidates = supporters or available

            minimum_tier = min(backend.tier for backend in candidates)
            same_tier = [
                backend
                for backend in candidates
                if backend.tier == minimum_tier
            ]
            return self._pick_round_robin(same_tier)

    def _refresh_catalogs_if_due(self) -> None:
        with self._lock:
            if time.time() < self._catalog_refresh_until:
                return
        if not self._catalog_refresh_lock.acquire(blocking=False):
            return
        try:
            with self._lock:
                if time.time() < self._catalog_refresh_until:
                    return
                backends = list(self.backends)

            fetched: List[tuple[Backend, List[str]]] = []
            for backend in backends:
                try:
                    fetched.append((backend, self._fetch_models(backend)))
                except Exception as exc:
                    logger.debug(
                        "fremaxxing: catalog refresh failed for %s: %s",
                        backend.name,
                        exc,
                    )

            for backend, models in fetched:
                backend.set_cached_models(models)
            with self._lock:
                self._catalog_refresh_until = (
                    time.time() + _CATALOG_TTL_SECONDS
                )
        finally:
            self._catalog_refresh_lock.release()

    def get_aggregated_models(self) -> List[Dict[str, str]]:
        self._refresh_catalogs_if_due()
        return [
            {
                "id": "freemaxxing",
                "object": "model",
                "owned_by": "freemaxxing",
            }
        ]

    def _fetch_models(self, backend: Backend) -> List[str]:
        base_url, api_key = backend.credential_snapshot()
        if not api_key and backend.refresh is not None:
            _refresh_backend_credentials(backend, require_new=False)
            base_url, api_key = backend.credential_snapshot()
        if not api_key:
            raise AuthError(f"backend {backend.name} has no credential")

        request = urllib.request.Request(
            base_url.rstrip("/") + "/models",
            headers={
                "Accept": "application/json",
                "User-Agent": _hermes_user_agent(),
                "Authorization": f"Bearer {api_key}",
            },
        )
        with _open_credentialed(request, timeout=8.0) as response:
            raw = response.read(_MAX_CATALOG_BODY_BYTES + 1)
        if len(raw) > _MAX_CATALOG_BODY_BYTES:
            raise TransientError(
                f"backend {backend.name} catalog exceeded "
                f"{_MAX_CATALOG_BODY_BYTES} bytes"
            )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TransientError(
                f"backend {backend.name} returned an invalid model catalog"
            ) from exc
        items = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(items, list):
            raise TransientError(
                f"backend {backend.name} model catalog is not a list"
            )
        return [
            str(item["id"])
            for item in items
            if isinstance(item, dict)
            and item.get("id")
            and _accept_catalog_id(backend, str(item["id"]))
        ]

    def exhaustion_detail(self) -> str:
        with self._lock:
            if not self.backends:
                return "no backends configured"
            cooling = [
                f"{backend.name}({backend.last_error_class or 'cooldown'})"
                for backend in self.backends
                if not backend.is_available()
            ]
            if len(cooling) == len(self.backends):
                return "all backends on cooldown: " + ", ".join(cooling)
            return "no eligible backend"

    def health(self) -> Dict[str, Any]:
        return {"backends": [backend.health() for backend in self.snapshot()]}


pool = BackendPool()


def _resolve_auto_model(backend: Backend) -> str:
    cached = backend.get_cached_models()
    if not cached:
        with backend.catalog_lock:
            cached = backend.get_cached_models()
            if not cached:
                try:
                    cached = pool._fetch_models(backend)
                    backend.set_cached_models(cached)
                except Exception as exc:
                    logger.debug(
                        "freemaxxing: auto-model fetch failed for %s: %s",
                        backend.name,
                        exc,
                    )
                    cached = []

    by_lower = {model.lower(): model for model in cached}
    preferred: List[str] = []
    if backend.default_model:
        preferred.append(backend.default_model)
    preferred.extend(_PREFERRED_AUTO_MODELS)
    for candidate in preferred:
        match = by_lower.get(candidate.lower())
        if match:
            return match

    if cached:
        small = [
            model
            for model in cached
            if any(
                marker in model.lower()
                for marker in ("flash", "mini", "nano", "small", "lite")
            )
        ]
        for candidate in small + cached:
            lowered = candidate.lower()
            if (
                lowered not in _ROUTER_MODELS
                and not lowered.endswith(":batch")
                and not lowered.startswith("~")
            ):
                return candidate

    if backend.default_model and _accept_catalog_id(
        backend,
        backend.default_model,
    ):
        return backend.default_model
    return ""


def _refresh_backend_credentials(
    backend: Backend,
    *,
    require_new: bool,
) -> bool:
    if backend.refresh is None:
        return False
    with backend.refresh_lock:
        old_base = backend.base_url
        old_key = backend.api_key
        new_base, new_key = backend.refresh()
        new_base = str(new_base or old_base).rstrip("/")
        new_key = str(new_key or "").strip()
        if not new_key:
            return False
        if require_new and new_key == old_key and new_base == old_base:
            return False
        backend.base_url = new_base
        backend.api_key = new_key
        return True
