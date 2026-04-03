"""Helpers for optional cheap-vs-strong model routing."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from utils import is_truthy_value

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
    # Finance analysis keywords (route to strong model)
    "technical",
    "technicals",
    "fundamental",
    "fundamentals",
    "valuation",
    "earnings",
    "quarterly",
    "financials",
    "fii",
    "dii",
    "institutional",
    "flows",
    "sector",
    "sectoral",
    "correlation",
    "rsi",
    "macd",
    "sma",
    "ema",
    "bollinger",
    "portfolio",
    "allocation",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)

import logging as _logging
_routing_logger = _logging.getLogger(__name__)


def _try_classifier_override(
    text: str,
    cheap_model: Dict[str, Any],
    conversation_depth: int = 0,
) -> Optional[Dict[str, Any]]:
    """Ask the learned classifier if a message that static heuristics rejected is actually simple.

    Returns a cheap route dict if the classifier is confident, else None.
    Silently returns None if the classifier is unavailable.
    """
    try:
        from agent.routing_classifier import get_routing_classifier
        from agent.routing_features import extract_features
    except ImportError:
        return None

    classifier = get_routing_classifier()
    if classifier is None:
        return None

    features = extract_features(text, conversation_depth)
    if classifier.should_route_cheap(features):
        route = dict(cheap_model)
        route["routing_reason"] = "classifier"
        _routing_logger.debug("Classifier override: routing to cheap model")
        return route

    return None


def _coerce_bool(value: Any, default: bool = False) -> bool:
    return is_truthy_value(value, default=default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def choose_cheap_model_route(user_message: str, routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the configured cheap-model route when a message looks simple.

    Conservative by design: if the message has signs of code/tool/debugging/
    long-form work, keep the primary model.
    """
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
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

    # Static heuristic checks — if any fail, give the classifier a chance
    static_rejected = False

    if len(text) > max_chars:
        static_rejected = True
    elif len(text.split()) > max_words:
        static_rejected = True
    elif text.count("\n") > 1:
        static_rejected = True
    elif "```" in text or "`" in text:
        static_rejected = True
    elif _URL_RE.search(text):
        static_rejected = True
    else:
        lowered = text.lower()
        words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
        if words & _COMPLEX_KEYWORDS:
            static_rejected = True

    if static_rejected:
        return _try_classifier_override(text, cheap_model, conversation_depth=0)

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
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

    return {
        "model": route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
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
