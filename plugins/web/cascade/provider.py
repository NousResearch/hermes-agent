"""CascadeSearchProvider — web search with fallback chain, circuit breaker, and retries.

Exposes web_search_tool() compatible dict interface.
Uses strategies module for actual search execution.
Uses circuit_breaker for backend health tracking with per-error-type cooldowns.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agent.web_search_provider import WebSearchProvider
from .circuit_breaker import CircuitBreaker
from .strategies import serial, hedge, hybrid

logger = logging.getLogger(__name__)

# Default config — overridable via config.yaml under web.cascade
_DEFAULT = {
    "strategy": "hybrid",
    "per_backend_timeout": 8,
    "total_timeout": 15,
    "hedge_trigger_after": 3,
}

# Error-type → cooldown (seconds)
_COOLDOWNS = {
    "transient": 60,
    "rate_limit": 120,
    "quota": 86400,
    "auth": 86400,
    "unknown": 300,
}


def _load_config() -> Dict[str, Any]:
    """Load cascade config from Hermes config.yaml."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        web = cfg.get("web", {})
        cascade = web.get("cascade", {})
        return {**_DEFAULT, **cascade}
    except Exception:
        return dict(_DEFAULT)


def _resolve_backend_order(available: List[str], preferred: str) -> List[str]:
    """Order backends: preferred first, then others in availability order."""
    if preferred in available:
        others = [b for b in available if b != preferred]
        return [preferred] + others
    return available


def _get_provider(name: str):
    """Get a search provider by name from the registry."""
    try:
        from agent.web_search_registry import get_provider
        return get_provider(name)
    except Exception:
        pass
    return None


def _attempt_record(name: str, success: bool, latency_ms: float, error_type: str, count: int) -> Dict[str, Any]:
    """Build a human-readable attempt record for the result."""
    if success:
        return {"backend": name, "status": "ok", "latency_ms": round(latency_ms), "attempts": count}
    return {
        "backend": name,
        "status": error_type or "error",
        "latency_ms": round(latency_ms),
        "attempts": count,
    }


class CascadeSearchProvider(WebSearchProvider):
    """Cascade search provider with fallback, circuit breaker, and retries."""

    @property
    def name(self) -> str:
        return "cascade"

    def is_available(self) -> bool:
        """Cascade is always available — it wraps other backends."""
        return True

    def supports_search(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        cfg = _load_config()
        strategy_name = cfg.get("strategy", "hybrid")
        per_timeout = cfg.get("per_backend_timeout", 15)
        total_timeout = cfg.get("total_timeout", 30)
        trigger_after = cfg.get("hedge_trigger_after", 3)

        # Load cooldown config
        cooldowns = cfg.get("cooldowns", _COOLDOWNS)
        cb = CircuitBreaker(cooldowns=cooldowns)

        # Get all known search backends
        all_backends = self._discover_backends()
        if not all_backends:
            return {"success": False, "error": "No search backends available", "query": query}

        # Filter by circuit breaker
        available_names = [n for n, _ in all_backends if cb.is_available(n)]
        if not available_names:
            # All in cooldown — try forcing through with a warning
            available_names = [n for n, _ in all_backends]
            logger.warning("All backends in cooldown, forcing through: %s", available_names)

        # Build ordered backend list
        preferred = cfg.get("preferred", all_backends[0][0] if all_backends else "")
        ordered = _resolve_backend_order(available_names, preferred)
        backends = [(n, p) for n, p in all_backends if n in ordered]

        # Pick strategy
        strategy_fn = {"serial": serial, "hedge": hedge, "hybrid": hybrid}.get(strategy_name, serial)

        start = time.monotonic()
        kwargs = {}
        if strategy_name == "hybrid":
            kwargs["trigger_after"] = trigger_after
        winner_name, result, latency_ms, attempts = strategy_fn(
            backends, query, limit, per_timeout, total_timeout, **kwargs
        )
        total_ms = (time.monotonic() - start) * 1000

        # Update circuit breaker based on results
        for name, success, lat, error_type, count in attempts:
            if success:
                cb.record_success(name, lat)
            else:
                cb.record_failure(name, error_type)

        # Build attempt summary for result
        attempt_records = [_attempt_record(n, s, l, e, c) for n, s, l, e, c in attempts]

        if result.get("success"):
            result["cascade"] = {
                "winner": winner_name,
                "strategy": strategy_name,
                "total_ms": round(total_ms),
                "attempts": attempt_records,
            }
            return result

        # All failed
        return {
            "success": False,
            "error": f"All cascade backends failed ({strategy_name})",
            "query": query,
            "cascade": {
                "winner": None,
                "strategy": strategy_name,
                "total_ms": round(total_ms),
                "attempts": attempt_records,
            },
        }

    def _discover_backends(self) -> List[Tuple[str, Any]]:
        """Discover available search backends from the registry."""
        backends = []

        try:
            from agent.web_search_registry import get_provider, list_providers
            # Try each known backend by name
            for name in ["tavily", "firecrawl", "ddgs", "brave-free", "exa", "serper"]:
                prov = get_provider(name)
                if prov and prov.is_available():
                    backends.append((name, prov))
            # Also pick up any providers registered by plugins (skip cascade itself)
            for prov in list_providers():
                if prov.name != "cascade" and prov.name not in {n for n, _ in backends}:
                    if prov.is_available():
                        backends.append((prov.name, prov))
        except Exception as e:
            logger.warning("Cascade _discover_backends failed: %s", e)

        return backends


# Singleton for registration
cascade_provider = CascadeSearchProvider()


def web_search_tool(query: str, limit: int = 5) -> Dict[str, Any]:
    """web_search_tool compatible entry point."""
    return cascade_provider.search(query, limit)
