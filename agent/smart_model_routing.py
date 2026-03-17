"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

_CODE_HINTS = {
    "code",
    "coding",
    "bug",
    "debug",
    "debugging",
    "fix",
    "implement",
    "implementation",
    "patch",
    "refactor",
    "traceback",
    "stacktrace",
    "exception",
    "syntax",
    "compile",
    "pytest",
    "test",
    "tests",
}

_TOOL_HINTS = {
    "tool",
    "tools",
    "terminal",
    "shell",
    "command",
    "execute",
    "run",
    "delegate",
    "subagent",
    "browser",
    "docker",
    "cron",
    "git",
    "file",
    "filesystem",
}

_THINKING_HINTS = {
    "analyze",
    "analysis",
    "compare",
    "comparison",
    "design",
    "architecture",
    "plan",
    "planning",
    "reason",
    "reasoning",
    "tradeoff",
    "tradeoffs",
    "thinking",
    "think",
    "evaluate",
    "strategy",
}

_COMPLEX_KEYWORDS = {
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

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _tokenize(text: str) -> set[str]:
    return {
        token.strip(".,:;!?()[]{}\"'`")
        for token in (text or "").lower().split()
        if token.strip(".,:;!?()[]{}\"'`")
    }


def _classify_prompt_intent(user_message: str) -> str:
    """Classify a user turn into a routing intent."""
    text = (user_message or "").strip()
    if not text:
        return "simple"

    lowered = text.lower()
    tokens = _tokenize(lowered)

    if "```" in text or "def " in lowered or "class " in lowered or "import " in lowered:
        return "coding"
    if _URL_RE.search(text):
        return "tool"
    if tokens & _CODE_HINTS:
        return "coding"
    if tokens & _TOOL_HINTS:
        return "tool"
    if tokens & _THINKING_HINTS:
        return "thinking"

    return "simple"


def _resolve_route_runtime(route_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    provider = str(route_config.get("provider") or "").strip().lower()
    model = str(route_config.get("model") or "").strip()
    if not provider or not model:
        return None

    from hermes_cli.runtime_provider import resolve_runtime_provider

    explicit_api_key = None
    api_key_env = str(route_config.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    return resolve_runtime_provider(
        requested=provider,
        explicit_api_key=explicit_api_key,
        explicit_base_url=route_config.get("base_url"),
    )


def _build_routed_turn(route_config: Dict[str, Any], intent: str, route_reason: str) -> Optional[Dict[str, Any]]:
    runtime = _resolve_route_runtime(route_config)
    if not runtime:
        return None

    model = str(route_config.get("model") or "").strip()
    if not model:
        return None

    return {
        "model": model,
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
        },
        "label": f"smart route[{intent}] → {model} ({runtime.get('provider')})",
        "signature": (
            model,
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
        ),
        "routing_reason": route_reason,
        "intent": intent,
    }


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a routed model when a message fits a configured intent.

    Conservative by design: simple turns can go to ``cheap_model``, while
    more specific intents (coding, tool, thinking) can opt into dedicated
    routes if the config defines them.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    intent = _classify_prompt_intent(user_message)
    intent_routes = cfg.get("intent_routes")
    if isinstance(intent_routes, dict):
        route = intent_routes.get(intent)
        if isinstance(route, dict):
            routed = _build_routed_turn(route, intent, f"{intent}_turn")
            if routed:
                return routed

    if intent != "simple":
        return None

    cheap_model = cfg.get("cheap_model") or {}
    if not isinstance(cheap_model, dict):
        return None
    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    text = (user_message or "").strip()
    if not text:
        return None

    max_chars = _coerce_int(cfg.get("max_simple_chars"), 160)
    max_words = _coerce_int(cfg.get("max_simple_words"), 28)

    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 1:
        return None
    if "```" in text or "`" in text:
        return None
    if _URL_RE.search(text):
        return None

    lowered = text.lower()
    words = _tokenize(lowered)
    if words & _COMPLEX_KEYWORDS:
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    route["intent"] = intent
    return route


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Returns a dict with model/runtime/signature/label fields.
    """
    route = choose_cheap_model_route(user_message, routing_config)
    if not route:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
            ),
        }

    runtime = route.get("runtime") if isinstance(route.get("runtime"), dict) else None
    if not runtime:
        try:
            runtime = _resolve_route_runtime(route)
        except Exception:
            runtime = None
    if not runtime:
        return {
            "model": primary.get("model"),
            "runtime": {
                "api_key": primary.get("api_key"),
                "base_url": primary.get("base_url"),
                "provider": primary.get("provider"),
                "api_mode": primary.get("api_mode"),
            },
            "label": None,
            "signature": (
                primary.get("model"),
                primary.get("provider"),
                primary.get("base_url"),
                primary.get("api_mode"),
            ),
        }

    intent = str(route.get("intent") or _classify_prompt_intent(user_message))
    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
        },
        "label": f"smart route[{intent}] → {route.get('model')} ({runtime.get('provider')})",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
        ),
    }
