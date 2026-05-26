"""Backend identity + short-lived health tracking for fallback routing.

This module keeps a small in-process registry of backend failures keyed by
provider family / API protocol / endpoint, not exact model slug. That lets the
main agent and auxiliary tasks avoid hammering the same broken localhost facade
multiple times in one turn or across back-to-back turns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Lock
from typing import Dict, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class BackendIdentity:
    provider: str
    api_mode: str
    base_url: str
    model_family: Optional[str] = None


@dataclass
class BackendHealth:
    consecutive_failures: int = 0
    first_failure_ts: Optional[float] = None
    last_failure_ts: Optional[float] = None
    down_until_ts: Optional[float] = None
    last_error: str = ""


_REGISTRY: Dict[BackendIdentity, BackendHealth] = {}
_LOCK = Lock()


def _normalize_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = (parsed.scheme or "").lower()
    netloc = (parsed.netloc or "").lower()
    path = (parsed.path or "").rstrip("/")
    return f"{scheme}://{netloc}{path}" if scheme or netloc else raw.rstrip("/").lower()


def _normalize_model_family(model: Optional[str]) -> Optional[str]:
    raw = str(model or "").strip().lower()
    if not raw:
        return None
    if "/" in raw:
        raw = raw.split("/", 1)[-1]
    for sep in ("-", ":"):
        if sep in raw:
            parts = [p for p in raw.split(sep) if p]
            if len(parts) >= 2:
                return f"{parts[0]}-{parts[1]}"
    return raw


def backend_identity_from_runtime(
    *,
    provider: str,
    api_mode: str,
    base_url: str,
    model: Optional[str] = None,
) -> BackendIdentity:
    return BackendIdentity(
        provider=str(provider or "").strip().lower(),
        api_mode=str(api_mode or "").strip().lower(),
        base_url=_normalize_base_url(base_url),
        model_family=_normalize_model_family(model),
    )


def get_backend_health(identity: BackendIdentity) -> BackendHealth:
    with _LOCK:
        current = _REGISTRY.get(identity)
        if current is None:
            return BackendHealth()
        return BackendHealth(**current.__dict__)


def is_backend_temporarily_down(identity: Optional[BackendIdentity], *, now: Optional[float] = None) -> bool:
    if identity is None:
        return False
    ts = time.monotonic() if now is None else now
    with _LOCK:
        current = _REGISTRY.get(identity)
        return bool(current and current.down_until_ts and current.down_until_ts > ts)


def record_backend_success(identity: Optional[BackendIdentity]) -> None:
    if identity is None:
        return
    with _LOCK:
        _REGISTRY.pop(identity, None)


def record_backend_failure(
    identity: Optional[BackendIdentity],
    exc: BaseException,
    *,
    category: str = "server",
    now: Optional[float] = None,
) -> BackendHealth:
    if identity is None:
        return BackendHealth()
    ts = time.monotonic() if now is None else now
    err = str(exc or "").strip()[:500]
    with _LOCK:
        current = _REGISTRY.get(identity)
        if current is None:
            current = BackendHealth()
            _REGISTRY[identity] = current
        if current.last_failure_ts is None or (ts - current.last_failure_ts) > 180:
            current.consecutive_failures = 0
            current.first_failure_ts = ts
        current.consecutive_failures += 1
        current.last_failure_ts = ts
        current.last_error = err
        if current.first_failure_ts is None:
            current.first_failure_ts = ts

        cooldown = 0.0
        if category == "transport":
            if current.consecutive_failures >= 2:
                cooldown = 180.0
        elif category == "server":
            if current.consecutive_failures >= 2:
                cooldown = 300.0
        if "process exited with code" in err.lower() and current.consecutive_failures >= 1:
            cooldown = max(cooldown, 300.0)
        if cooldown > 0:
            current.down_until_ts = ts + cooldown
        return BackendHealth(**current.__dict__)


def summarize_backend_health(identity: Optional[BackendIdentity]) -> str:
    if identity is None:
        return ""
    health = get_backend_health(identity)
    parts = []
    if health.consecutive_failures:
        parts.append(f"failures={health.consecutive_failures}")
    if health.down_until_ts and health.down_until_ts > time.monotonic():
        remaining = max(0, int(health.down_until_ts - time.monotonic()))
        parts.append(f"down_for={remaining}s")
    if health.last_error:
        parts.append(f"last_error={health.last_error[:120]}")
    return ", ".join(parts)


def reset_backend_health_registry() -> None:
    with _LOCK:
        _REGISTRY.clear()
