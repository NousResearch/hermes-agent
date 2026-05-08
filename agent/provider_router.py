# -*- coding: utf-8 -*-
"""Multi-provider API routing with priority-based failover.

Sits on top of KeyRouter: each "provider" wraps its own KeyRouter +
base_url + model mapping. Requests are tried in priority order.

When a provider fails >= N times within a rolling window (default 1 hour),
it enters a cooldown for the remainder of that window — all subsequent
requests skip it until the window resets.

Configuration (hermes.toml):
    [provider_pool]
    fail_threshold = 3           # failures before cooldown
    window_seconds = 3600        # rolling window size (1 hour)
    strategy = "priority"        # priority | round_robin

    [[provider_pool.providers]]
    name = "longcat"
    base_url = "https://api.longcat.chat/openai"
    priority = 0                 # lower = tried first

    [[provider_pool.providers.keys]]
    api_key = "ak_xxx"
    label = "longcat-primary"

    [[provider_pool.providers]]
    name = "openrouter"
    base_url = "https://openrouter.ai/api/v1"
    priority = 1
    model_map = {"default": "anthropic/claude-sonnet-4"}

    [[provider_pool.providers.keys]]
    api_key = "sk-or-xxx"
    label = "openrouter-primary"

Public API:
    pm = get_provider_manager()                    # singleton
    client, provider, model = pm.get_client()      # pick best provider
    result = pm.call_with_failover(fn, *a, **kw)   # auto-failover
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

log = logging.getLogger(__name__)

T = TypeVar("T")

STATE_DIR = Path(os.path.expanduser("~/.hermes/state"))
PROVIDER_STATE_FILE = STATE_DIR / "provider_pool.json"

DEFAULT_FAIL_THRESHOLD = 3
DEFAULT_WINDOW_SECONDS = 3600


@dataclass
class ProviderState:
    name: str
    base_url: str
    priority: int = 0
    model_map: Dict[str, str] = field(default_factory=dict)
    fail_threshold: int = DEFAULT_FAIL_THRESHOLD
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    recent_failures: List[float] = field(default_factory=list)
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    _key_router: Any = field(default=None, repr=False)

    @property
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    @property
    def remaining_cooldown(self) -> float:
        return max(0.0, self.cooldown_until - time.time())

    @property
    def failure_count_in_window(self) -> int:
        self._prune_old_failures()
        return len(self.recent_failures)

    def _prune_old_failures(self) -> None:
        cutoff = time.time() - self.window_seconds
        self.recent_failures = [t for t in self.recent_failures if t > cutoff]

    def record_failure(self) -> bool:
        now = time.time()
        self.total_failures += 1
        self.recent_failures.append(now)
        self._prune_old_failures()
        if len(self.recent_failures) >= self.fail_threshold:
            self.cooldown_until = now + self.window_seconds
            log.warning(
                "provider %s: %d failures in %ds -> cooldown until %s",
                self.name, len(self.recent_failures), self.window_seconds,
                time.strftime("%H:%M:%S", time.localtime(self.cooldown_until)),
            )
            return True
        return False

    def record_success(self) -> None:
        self.total_requests += 1

    def reset_cooldown(self) -> None:
        self.cooldown_until = 0.0
        self.recent_failures.clear()
        log.info("provider %s: cooldown manually reset", self.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "priority": self.priority,
            "model_map": self.model_map,
            "fail_threshold": self.fail_threshold,
            "window_seconds": self.window_seconds,
            "recent_failures": self.recent_failures,
            "cooldown_until": self.cooldown_until,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProviderState":
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__ and k != "_key_router"})


class ProviderManager:
    def __init__(self, providers, *, default_fail_threshold=DEFAULT_FAIL_THRESHOLD,
                 default_window_seconds=DEFAULT_WINDOW_SECONDS):
        self._providers = sorted(providers, key=lambda p: p.priority)
        self._lock = threading.Lock()
        self._default_fail_threshold = default_fail_threshold
        self._default_window_seconds = default_window_seconds
        self._load_state()

    @property
    def providers(self):
        return list(self._providers)

    def select_provider(self):
        with self._lock:
            available = [p for p in self._providers if not p.is_in_cooldown]
            if available:
                return available[0]
            return min(self._providers, key=lambda p: p.cooldown_until) if self._providers else None

    def get_client(self, model=None, **extra):
        provider = self.select_provider()
        if provider is None:
            raise RuntimeError("No API providers configured")
        if provider._key_router is not None:
            client, resolved_model = provider._key_router.get_client(model=model, **extra)
        else:
            from openai import OpenAI
            client = OpenAI(base_url=provider.base_url, api_key=extra.pop("api_key", "placeholder"), **extra)
            resolved_model = model or provider.model_map.get("default", "gpt-4")
        return client, provider.name, resolved_model

    def call_with_failover(self, fn, *args, **kwargs):
        last_exc = None
        attempted = []
        for provider in self._providers:
            if provider.is_in_cooldown:
                log.debug("provider %s in cooldown (%.0fs left), skipping", provider.name, provider.remaining_cooldown)
                continue
            try:
                attempted.append(provider.name)
                result = fn(provider, *args, **kwargs)
                provider.record_success()
                self._save_state()
                return result
            except Exception as exc:
                last_exc = exc
                entered_cooldown = provider.record_failure()
                log.warning("provider %s failed (%s), failures: %d/%d%s",
                           provider.name, str(exc)[:120],
                           provider.failure_count_in_window, provider.fail_threshold,
                           " -> COOLDOWN" if entered_cooldown else "")
                self._save_state()
                continue
        raise RuntimeError(f"All providers failed (tried: {attempted}). Last: {last_exc}") from last_exc

    async def async_call_with_failover(self, fn, *args, **kwargs):
        last_exc = None
        attempted = []
        for provider in self._providers:
            if provider.is_in_cooldown:
                continue
            try:
                attempted.append(provider.name)
                result = await fn(provider, *args, **kwargs)
                provider.record_success()
                self._save_state()
                return result
            except Exception as exc:
                last_exc = exc
                provider.record_failure()
                self._save_state()
                continue
        raise RuntimeError(f"All providers failed (tried: {attempted}). Last: {last_exc}") from last_exc

    def _load_state(self):
        if not PROVIDER_STATE_FILE.exists():
            return
        try:
            data = json.loads(PROVIDER_STATE_FILE.read_text())
            saved = {e["name"]: e for e in data.get("providers", [])}
            for p in self._providers:
                if p.name in saved:
                    s = saved[p.name]
                    p.recent_failures = s.get("recent_failures", [])
                    p.cooldown_until = s.get("cooldown_until", 0.0)
                    p.total_requests = s.get("total_requests", 0)
                    p.total_failures = s.get("total_failures", 0)
        except Exception as exc:
            log.warning("Failed to load provider state: %s", exc)

    def _save_state(self):
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            PROVIDER_STATE_FILE.write_text(json.dumps({
                "providers": [p.to_dict() for p in self._providers],
                "saved_at": time.time(),
            }, indent=2))
        except Exception as exc:
            log.warning("Failed to save provider state: %s", exc)

    def status(self):
        lines = ["Provider Router Status:"]
        for p in self._providers:
            icon = "COOLDOWN" if p.is_in_cooldown else "OK"
            cd = f" (cd {p.remaining_cooldown:.0f}s)" if p.is_in_cooldown else ""
            lines.append(f"  [{icon}] {p.name}  priority={p.priority}  "
                        f"fails={p.failure_count_in_window}/{p.fail_threshold}  "
                        f"total={p.total_requests}r/{p.total_failures}f{cd}")
        return "\n".join(lines)

    def __repr__(self):
        return f"ProviderManager([{', '.join(p.name for p in self._providers)}])"


def _load_provider_manager(force_reload=False):
    global _instance
    if _instance is not None and not force_reload:
        return _instance
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return None
    config_path = Path(os.path.expanduser("~/.hermes/hermes.toml"))
    if not config_path.exists():
        return None
    try:
        cfg = tomllib.loads(config_path.read_text())
    except Exception:
        return None
    pool_cfg = cfg.get("provider_pool")
    if not pool_cfg:
        return None
    default_threshold = pool_cfg.get("fail_threshold", DEFAULT_FAIL_THRESHOLD)
    default_window = pool_cfg.get("window_seconds", DEFAULT_WINDOW_SECONDS)
    providers = []
    for p_cfg in pool_cfg.get("providers", []):
        ps = ProviderState(
            name=p_cfg["name"],
            base_url=p_cfg["base_url"],
            priority=p_cfg.get("priority", len(providers)),
            model_map=p_cfg.get("model_map", {}),
            fail_threshold=p_cfg.get("fail_threshold", default_threshold),
            window_seconds=p_cfg.get("window_seconds", default_window),
        )
        providers.append(ps)
    if not providers:
        return None
    _instance = ProviderManager(providers, default_fail_threshold=default_threshold,
                                default_window_seconds=default_window)
    return _instance


_instance = None

def get_provider_manager(force_reload=False):
    return _load_provider_manager(force_reload)
