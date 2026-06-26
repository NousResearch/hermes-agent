"""Helpers for optional platform-aware per-turn model routing.

This module intentionally keeps the routing decision small and config-driven so
messaging surfaces can choose a lower-cost/default model without mutating the
conversation prompt or tool schema mid-turn.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from utils import is_truthy_value

_COMPLEX_KEYWORDS = {
    "audit",
    "benchmark",
    "build",
    "compare",
    "configure",
    "cron",
    "debug",
    "debugging",
    "delegate",
    "design",
    "docker",
    "error",
    "exception",
    "fix",
    "implement",
    "implementation",
    "investigate",
    "kubernetes",
    "optimize",
    "optimise",
    "patch",
    "plan",
    "planning",
    "refactor",
    "review",
    "shell",
    "stacktrace",
    "subagent",
    "terminal",
    "test",
    "tests",
    "tool",
    "tools",
    "traceback",
    "triad",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _route_config(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    provider = str(raw.get("provider") or "").strip().lower()
    model = str(raw.get("model") or "").strip()
    if not provider or not model:
        return None
    route = dict(raw)
    route["provider"] = provider
    route["model"] = model
    return route


def _primary_route(primary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": primary.get("model"),
        "runtime": {
            "api_key": primary.get("api_key"),
            "base_url": primary.get("base_url"),
            "provider": primary.get("provider"),
            "api_mode": primary.get("api_mode"),
            "command": primary.get("command"),
            "args": list(primary.get("args") or []),
            "credential_pool": primary.get("credential_pool"),
            "max_tokens": primary.get("max_tokens"),
        },
        "label": None,
        "signature": (
            primary.get("model"),
            primary.get("provider"),
            primary.get("base_url"),
            primary.get("api_mode"),
            primary.get("command"),
            tuple(primary.get("args") or ()),
        ),
    }


def _message_is_simple(user_message: str, cfg: Optional[Dict[str, Any]]) -> bool:
    cfg = cfg or {}
    text = (user_message or "").strip()
    if not text:
        return False

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    if len(text) > max_chars:
        return False
    if len(text.split()) > max_words:
        return False
    if text.count("\n") > 1:
        return False
    if "```" in text or "`" in text:
        return False
    if _URL_RE.search(text):
        return False

    keywords = set(_COMPLEX_KEYWORDS)
    extra_keywords = cfg.get("complex_keywords") or []
    if isinstance(extra_keywords, (list, tuple, set)):
        keywords |= {str(token).strip().lower() for token in extra_keywords if str(token).strip()}

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    return not bool(words & keywords)


def _resolve_configured_route(route: Dict[str, Any], label: str) -> Dict[str, Any]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    runtime = resolve_runtime_provider(
        requested=route.get("provider"),
        explicit_api_key=explicit_api_key,
        explicit_base_url=route.get("base_url"),
    )

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
            "max_tokens": runtime.get("max_tokens"),
        },
        "label": label,
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple."""
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    route = _route_config(cfg.get("cheap_model") or {})
    if not route:
        return None
    if not _message_is_simple(user_message, cfg):
        return None

    route["routing_reason"] = "simple_turn"
    return route


def choose_platform_model_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    platform: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Return a platform-specific default/strong/cheap route when configured."""
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None
    platform_key = str(platform or "").strip().lower()
    if not platform_key:
        return None

    routes = cfg.get("platform_routes") or {}
    if not isinstance(routes, dict):
        return None
    pcfg = routes.get(platform_key) or routes.get(platform_key.replace("-", "_"))
    if not isinstance(pcfg, dict):
        return None
    if pcfg.get("enabled") is False:
        return None

    excluded = {"default_model", "default", "model", "strong_model", "complex_model", "cheap_model"}
    merged_cfg = dict(cfg)
    merged_cfg.update({k: v for k, v in pcfg.items() if k not in excluded})
    simple = _message_is_simple(user_message, merged_cfg)

    if simple:
        route = _route_config(pcfg.get("cheap_model") or {})
        if route:
            route["routing_reason"] = f"{platform_key}_simple_turn"
            return route

    if not simple:
        route = _route_config(pcfg.get("strong_model") or pcfg.get("complex_model") or {})
        if route:
            route["routing_reason"] = f"{platform_key}_complex_turn"
            return route

    route = _route_config(pcfg.get("default_model") or pcfg.get("default") or pcfg.get("model") or {})
    if route:
        route["routing_reason"] = f"{platform_key}_default"
        return route
    return None


def resolve_turn_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    primary: Dict[str, Any],
    platform: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.  Any failure in
    the optional routing layer falls back to the primary session route so a bad
    model route never breaks gateway message handling.
    """
    route = choose_platform_model_route(user_message, routing_config, platform)
    if route is None:
        route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return _primary_route(primary)

    try:
        return _resolve_configured_route(
            route,
            f"smart route -> {route.get('model')} ({route.get('provider')}; {route.get('routing_reason')})",
        )
    except Exception:
        return _primary_route(primary)
