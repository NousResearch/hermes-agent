# -*- coding: utf-8 -*-
"""Same-provider API key pool with automatic failover.

Manages multiple API keys for a single provider. When one key hits
rate-limit / quota exhaustion (HTTP 429/402), automatically rotates to
the next available key. Exhausted keys enter a configurable cooldown.

Configuration (hermes.toml):
    [key_pool]
    base_url = "https://api.longcat.chat/openai"
    strategy = "fill_first"          # fill_first | round_robin | least_used
    cooldown_429 = 3600              # seconds
    cooldown_default = 3600

    [[key_pool.keys]]
    api_key = "ak_xxx"
    label = "primary"
    priority = 0

    [[key_pool.keys]]
    api_key = "ak_yyy"
    label = "backup"
    priority = 1

Public API:
    router = get_router()                       # singleton from config
    client, model = router.get_client(model=…)  # pick next key
    call_with_failover(router, fn, …)           # auto-retry on exhaustion
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

# ── defaults ────────────────────────────────────────────────────────────────

STATE_DIR = Path(os.path.expanduser("~/.hermes/state"))
STATE_FILE = STATE_DIR / "key_pool.json"

COOLDOWN_429 = 3600
COOLDOWN_DEFAULT = 3600
COOLDOWN_MIN = 60

EXHAUSTION_CODES = frozenset({402, 429})
EXHAUSTION_KEYWORDS = (
    "quota", "exceeded", "insufficient", "credits",
    "billing", "rate limit", "token limit",
)

# ── key entry ───────────────────────────────────────────────────────────────

@dataclass
class KeyEntry:
    """Tracks a single API key's runtime state."""
    api_key: str
    label: str = ""
    priority: int = 0
    status: str = "ok"                      # ok | exhausted
    exhausted_at: float = 0.0
    exhausted_until: float = 0.0
    error_code: Optional[int] = None
    error_message: str = ""
    request_count: int = 0
    error_count: int = 0

    @property
    def short_key(self) -> str:
        if len(self.api_key) <= 12:
            return self.api_key
        return f"{self.api_key[:8]}…{self.api_key[-4:]}"

    @property
    def is_available(self) -> bool:
        if self.status != "exhausted":
            return True
        return time.monotonic() >= self.exhausted_until

    @property
    def remaining_cooldown(self) -> float:
        if self.status != "exhausted":
            return 0.0
        return max(0.0, self.exhausted_until - time.monotonic())

    def mark_exhausted(
        self,
        error_code: int = 0,
        retry_after: Optional[float] = None,
        error_message: str = "",
    ) -> None:
        self.status = "exhausted"
        self.exhausted_at = time.monotonic()
        self.error_code = error_code
        self.error_message = error_message
        self.error_count += 1
        if retry_after and retry_after > 0:
            self.exhausted_until = self.exhausted_at + max(retry_after, COOLDOWN_MIN)
        elif error_code == 429:
            self.exhausted_until = self.exhausted_at + COOLDOWN_429
        else:
            self.exhausted_until = self.exhausted_at + COOLDOWN_DEFAULT
        log.warning(
            "[key_router] %s exhausted (code=%s, cool=%.0fs, msg=%s)",
            self.short_key, error_code,
            self.exhausted_until - self.exhausted_at,
            error_message[:120] if error_message else "n/a",
        )

    def mark_ok(self) -> None:
        prev = self.status
        self.status = "ok"
        self.exhausted_at = 0.0
        self.exhausted_until = 0.0
        self.error_code = None
        self.error_message = ""
        if prev != "ok":
            log.info("[key_router] %s recovered → ok", self.short_key)

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key, "label": self.label,
            "priority": self.priority, "status": self.status,
            "exhausted_at": self.exhausted_at,
            "exhausted_until": self.exhausted_until,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "request_count": self.request_count,
            "error_count": self.error_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> KeyEntry:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ── key router ──────────────────────────────────────────────────────────────

class KeyRouter:
    """Thread-safe API key pool with automatic failover."""

    def __init__(
        self,
        base_url: str,
        *,
        strategy: str = "fill_first",
        cooldown_429: int = COOLDOWN_429,
        cooldown_default: int = COOLDOWN_DEFAULT,
    ):
        self.base_url = base_url.rstrip("/")
        self.strategy = strategy
        self.cooldown_429 = cooldown_429
        self.cooldown_default = cooldown_default
        self._keys: List[KeyEntry] = []
        self._rr_idx = 0
        self._lock = threading.Lock()
        self._load_state()

    # ── persistence ──

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text())
            if data.get("base_url") != self.base_url:
                return
            saved = {k["api_key"]: k for k in data.get("keys", [])}
            with self._lock:
                for entry in self._keys:
                    if entry.api_key in saved:
                        s = saved[entry.api_key]
                        entry.status = s.get("status", "ok")
                        entry.exhausted_until = s.get("exhausted_until", 0.0)
                        entry.error_code = s.get("error_code")
                        entry.request_count = s.get("request_count", 0)
                        entry.error_count = s.get("error_count", 0)
                        if entry.status == "exhausted" and time.monotonic() >= entry.exhausted_until:
                            entry.mark_ok()
        except Exception as exc:
            log.warning("[key_router] state load failed: %s", exc)

    def save_state(self) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "base_url": self.base_url,
            "updated_at": time.time(),
            "keys": [k.to_dict() for k in self._keys],
        }
        try:
            STATE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as exc:
            log.warning("[key_router] state save failed: %s", exc)

    # ── key management ──

    def add_key(self, api_key: str, *, label: str = "", priority: int = 0) -> None:
        with self._lock:
            if any(k.api_key == api_key for k in self._keys):
                return
            self._keys.append(KeyEntry(api_key=api_key, label=label, priority=priority))
            self._keys.sort(key=lambda k: k.priority)

    def available_keys(self) -> List[KeyEntry]:
        with self._lock:
            return [k for k in self._keys if k.is_available]

    def select_key(self) -> Optional[KeyEntry]:
        with self._lock:
            available = sorted(
                (k for k in self._keys if k.is_available),
                key=lambda k: k.priority,
            )
            if not available:
                return None
            if self.strategy == "round_robin":
                idx = self._rr_idx % len(available)
                self._rr_idx += 1
                return available[idx]
            if self.strategy == "least_used":
                return min(available, key=lambda k: k.request_count)
            return available[0]

    # ── client creation ──

    def get_client(self, model: Optional[str] = None, **extra: Any):
        entry = self.select_key()
        if entry is None:
            raise RuntimeError(
                f"[key_router] all keys exhausted for {self.base_url}\n"
                + self.status()
            )
        with self._lock:
            entry.request_count += 1
        self.save_state()
        from openai import OpenAI
        kwargs: Dict[str, Any] = {"api_key": entry.api_key, "base_url": self.base_url, **extra}
        log.info("[key_router] → %s (%s) req=%d", entry.short_key, entry.label or "–", entry.request_count)
        return OpenAI(**kwargs), model

    def get_async_client(self, model: Optional[str] = None, **extra: Any):
        entry = self.select_key()
        if entry is None:
            raise RuntimeError(
                f"[key_router] all keys exhausted for {self.base_url}\n"
                + self.status()
            )
        with self._lock:
            entry.request_count += 1
        self.save_state()
        from openai import AsyncOpenAI
        kwargs: Dict[str, Any] = {"api_key": entry.api_key, "base_url": self.base_url, **extra}
        log.info("[key_router] → %s (%s) req=%d", entry.short_key, entry.label or "–", entry.request_count)
        return AsyncOpenAI(**kwargs), model

    # ── error handling ──

    @staticmethod
    def _is_exhaustion(exc: Exception) -> bool:
        code = getattr(exc, "status_code", None)
        if code in EXHAUSTION_CODES:
            return True
        msg = str(exc).lower()
        return any(kw in msg for kw in EXHAUSTION_KEYWORDS)

    @staticmethod
    def _extract_retry_after(exc: Exception) -> Optional[float]:
        resp = getattr(exc, "response", None)
        if resp is not None:
            ra = getattr(resp, "headers", {}).get("retry-after")
            if ra:
                try:
                    return float(ra)
                except (ValueError, TypeError):
                    pass
        m = re.search(r"retry\s+(?:after|in)\s+(\d+)\s*s", str(exc), re.I)
        if m:
            return float(m.group(1))
        return None

    def mark_exhausted(
        self, api_key: str, *, error_code: int = 0,
        retry_after: Optional[float] = None, error_message: str = "",
    ) -> None:
        with self._lock:
            for entry in self._keys:
                if entry.api_key == api_key:
                    entry.mark_exhausted(
                        error_code=error_code,
                        retry_after=retry_after or self._extract_retry_after(
                            type("E", (), {"__str__": lambda _: error_message})()
                        ),
                        error_message=error_message,
                    )
                    break
        self.save_state()

    def mark_ok(self, api_key: str) -> None:
        with self._lock:
            for entry in self._keys:
                if entry.api_key == api_key:
                    entry.mark_ok()
                    break
        self.save_state()

    # ── status ──

    def status(self) -> str:
        lines = [
            f"Key pool: {self.base_url}  strategy={self.strategy}",
            f"{'Label':<12} {'Status':<10} {'Reqs':>6} {'Errs':>6} {'Cooldown':>8}",
            "-" * 48,
        ]
        with self._lock:
            for k in self._keys:
                cd = f"{k.remaining_cooldown:.0f}s" if k.status == "exhausted" else "–"
                label = k.label or k.short_key
                lines.append(f"{label:<12} {k.status:<10} {k.request_count:>6} {k.error_count:>6} {cd:>8}")
            avail = sum(1 for k in self._keys if k.is_available)
        lines.append(f"Available: {avail}/{len(self._keys)}")
        return "\n".join(lines)


# ── failover wrappers ───────────────────────────────────────────────────────

def call_with_failover(router: KeyRouter, fn: Callable[..., T], *a: Any, **kw: Any) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(len(router._keys)):
        client, _ = router.get_client()
        current_key = client.api_key
        try:
            return fn(client, *a, **kw)
        except Exception as exc:
            last_exc = exc
            if not router._is_exhaustion(exc):
                raise
            router.mark_exhausted(
                current_key, error_code=getattr(exc, "status_code", 0),
                retry_after=router._extract_retry_after(exc),
                error_message=str(exc),
            )
            if attempt < len(router._keys) - 1:
                log.warning("[key_router] attempt %d/%d failed, rotating…", attempt + 1, len(router._keys))
                continue
    raise last_exc  # type: ignore[misc]


async def async_call_with_failover(router: KeyRouter, fn: Callable[..., T], *a: Any, **kw: Any) -> T:
    last_exc: Optional[Exception] = None
    for attempt in range(len(router._keys)):
        client, _ = router.get_async_client()
        current_key = client.api_key
        try:
            return await fn(client, *a, **kw)
        except Exception as exc:
            last_exc = exc
            if not router._is_exhaustion(exc):
                raise
            router.mark_exhausted(
                current_key, error_code=getattr(exc, "status_code", 0),
                retry_after=router._extract_retry_after(exc),
                error_message=str(exc),
            )
            if attempt < len(router._keys) - 1:
                log.warning("[key_router] attempt %d/%d failed, rotating…", attempt + 1, len(router._keys))
                continue
    raise last_exc  # type: ignore[misc]


# ── singleton ───────────────────────────────────────────────────────────────

_router: Optional[KeyRouter] = None
_router_lock = threading.Lock()


def get_router(force_reload: bool = False) -> Optional[KeyRouter]:
    global _router
    if _router is not None and not force_reload:
        return _router
    with _router_lock:
        if _router is not None and not force_reload:
            return _router
        config_path = Path(os.path.expanduser("~/.hermes/config/hermes.toml"))
        if not config_path.exists():
            return None
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            config = tomllib.loads(config_path.read_text())
        except Exception as exc:
            log.warning("[key_router] config parse error: %s", exc)
            return None
        pool_cfg = config.get("key_pool")
        if not pool_cfg:
            return None
        base_url = pool_cfg.get("base_url", "").strip()
        if not base_url:
            return None
        r = KeyRouter(
            base_url=base_url,
            strategy=pool_cfg.get("strategy", "fill_first"),
            cooldown_429=pool_cfg.get("cooldown_429", COOLDOWN_429),
            cooldown_default=pool_cfg.get("cooldown_default", COOLDOWN_DEFAULT),
        )
        for kcfg in pool_cfg.get("keys", []):
            ak = kcfg.get("api_key", "").strip()
            if ak:
                r.add_key(ak, label=kcfg.get("label", ""), priority=kcfg.get("priority", 0))
        if not r._keys:
            return None
        log.info("[key_router] ready: %d keys, strategy=%s", len(r._keys), r.strategy)
        _router = r
        return r
