"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from utils import is_truthy_value

_NON_SIMPLE_KEYWORDS = {
    "debug",
    "debugging",
    "implement",
    "implementation",
    "refactor",
    "patch",
    "traceback",
    "stacktrace",
    "exception",
    "error",
    "analyze",
    "analysis",
    "investigate",
    "architecture",
    "design",
    "compare",
    "benchmark",
    "optimize",
    "optimise",
    "review",
    "terminal",
    "shell",
    "tool",
    "tools",
    "pytest",
    "test",
    "tests",
    "plan",
    "planning",
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

_DEFAULT_COMPLEX_ESCALATION_KEYWORDS = (
    "research",
    "review",
    "audit",
    "研究",
    "審查",
    "審閱",
)

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_route_config(route_config: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(route_config, dict):
        return None
    provider = str(route_config.get("provider") or "").strip().lower()
    model = str(route_config.get("model") or "").strip()
    if not provider or not model:
        return None
    route = dict(route_config)
    route["provider"] = provider
    route["model"] = model
    return route


def _is_simple_message(text: str, routing_config: Optional[Dict[str, Any]]) -> bool:
    if not text:
        return False

    cfg = routing_config or {}
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

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _NON_SIMPLE_KEYWORDS:
        return False

    return True


def _should_escalate_to_complex_model(text: str, routing_config: Optional[Dict[str, Any]]) -> bool:
    cfg = routing_config or {}
    configured_keywords = cfg.get("complex_keywords")
    if isinstance(configured_keywords, (list, tuple, set)):
        keywords = tuple(str(keyword).strip().lower() for keyword in configured_keywords if str(keyword).strip())
    else:
        keywords = _DEFAULT_COMPLEX_ESCALATION_KEYWORDS

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    for keyword in keywords:
        normalized = keyword.lower()
        if not normalized:
            continue
        if any(ord(ch) > 127 for ch in normalized):
            if normalized in lowered:
                return True
            continue
        if normalized in words:
            return True
    return False


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


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Conservative by design: if the message has signs of code/tool/debugging/
    long-form work, keep the primary model.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    route = _normalize_route_config(cfg.get("cheap_model") or {})
    if not route:
        return None

    text = (user_message or "").strip()
    if _should_escalate_to_complex_model(text, cfg):
        return None
    if not _is_simple_message(text, cfg):
        return None

    route["routing_reason"] = "simple_turn"
    return route


def choose_complex_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured complex-model route when a message is not simple."""
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    route = _normalize_route_config(cfg.get("complex_model") or {})
    if not route:
        return None

    text = (user_message or "").strip()
    if not text:
        return None
    if not _should_escalate_to_complex_model(text, cfg):
        return None

    route["routing_reason"] = "complex_turn"
    return route


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    """
    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        route = choose_complex_model_route(user_message, routing_config)
    if not route:
        return _primary_route(primary)

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return _primary_route(primary)

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
        },
        "label": f"smart route → {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }
