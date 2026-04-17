"""Unified smart model routing for Hermes.

Version 2: routing_v2 (category-aware + tier-based + task_state) is the
primary path. The legacy cheap-vs-strong heuristic is retained as a
private fallback when no benchmarks are available.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from agent import routing_v2
from agent import task_state as _ts
from agent import benchmark_runner as _br
from utils import is_truthy_value

logger = logging.getLogger(__name__)

_OPUS_KEYWORD_MARKERS = {"[OPUS]", "[HEAVY]", "[CRÍTICO]", "[CRITICO]", "[ARCHITECTURE]"}

_ROUTER_DIR = os.path.expanduser("~/.hermes/router")
_TASK_STATE_PATH = os.path.join(_ROUTER_DIR, "task_state.json")
_BENCHMARK_PATH = os.path.join(_ROUTER_DIR, "benchmarks.json")


def _ensure_router_dir() -> None:
    os.makedirs(_ROUTER_DIR, exist_ok=True)


def _load_benchmarks() -> Dict[str, Dict[str, float]]:
    try:
        if os.path.exists(_BENCHMARK_PATH):
            rep = _br.load_report(_BENCHMARK_PATH)
            return _br.report_to_benchmarks(rep)
    except Exception:
        pass
    return {}


def _load_task_state() -> Dict[str, Any]:
    return _ts.load(_TASK_STATE_PATH)


def _save_task_state(state: Dict[str, Any]) -> None:
    try:
        _ensure_router_dir()
        _ts.save(_TASK_STATE_PATH, state)
    except Exception:
        logger.debug("Failed to persist task_state", exc_info=True)


def _legacy_cheap_route(user_message: str, cheap_model: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Private fallback: simple length/keyword heuristic for cheap routing.

    Only used when v2 has no benchmarks and no tiers configured.
    """
    text = (user_message or "").strip()
    if not text:
        return None

    provider = str(cheap_model.get("provider") or "").strip().lower()
    model = str(cheap_model.get("model") or "").strip()
    if not provider or not model:
        return None

    # OPUS keyword bail-out
    upper = text.upper()
    if any(marker in upper for marker in _OPUS_KEYWORD_MARKERS):
        return None

    # Continuation markers never go cheap
    if _ts.is_continuation(text):
        return None

    max_chars = 160
    max_words = 28
    if len(text) > max_chars:
        return None
    if len(text.split()) > max_words:
        return None
    if text.count("\n") > 1:
        return None
    if "```" in text or "`" in text:
        return None

    import re
    if re.search(r"https?://|www\.", text, re.IGNORECASE):
        return None

    _COMPLEX_KEYWORDS = {
        "debug", "debugging", "implement", "implementation", "refactor",
        "patch", "traceback", "stacktrace", "exception", "error",
        "analyze", "analysis", "investigate", "architecture", "design",
        "compare", "benchmark", "optimize", "optimise", "review",
        "terminal", "shell", "tool", "tools", "pytest", "test", "tests",
        "plan", "planning", "delegate", "subagent", "cron", "docker",
        "kubernetes",
    }
    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return None

    return dict(cheap_model, provider=provider, model=model, routing_reason="simple_turn")


def resolve_turn_route(user_message: str, routing_config: Optional[Dict[str, Any]], primary: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the effective model/runtime for one turn.

    Uses routing_v2 (category-aware, tier-based, task_state sticky) as the
    primary decision path. Falls back to legacy cheap heuristic only when
    v2 has no tiers or benchmarks available.
    """
    cfg = routing_config or {}

    # --- Load persistent state and benchmarks ---
    benches = _load_benchmarks()
    state = _load_task_state()
    tiers = getattr(routing_v2, "DEFAULT_TIERS", None) or []

    # --- Determine if turn is "easy" for state tracking ---
    is_easy = bool(user_message and len(user_message) <= 160 and len(user_message.split()) <= 28)

    # --- Primary path: routing_v2 ---
    if tiers or benches:
        try:
            sel = routing_v2.select_model(user_message, benches, tiers, state)
            model = sel.get("model")
            category = sel.get("category", "unknown")
            tier = sel.get("tier", 0)

            # Update task state
            if sel.get("reason") == "continuation":
                state = _ts.record_turn(state, user_message, was_easy=is_easy)
            else:
                state = _ts.start_task(state, tier=tier, model=model or primary.get("model", ""), category=category)
            _save_task_state(state)

            # Build runtime dict preserving primary connection info
            return {
                "model": model or primary.get("model"),
                "runtime": {
                    "api_key": primary.get("api_key"),
                    "base_url": primary.get("base_url"),
                    "provider": primary.get("provider"),
                    "api_mode": primary.get("api_mode"),
                    "command": primary.get("command"),
                    "args": list(primary.get("args") or []),
                    "credential_pool": primary.get("credential_pool"),
                },
                "label": f"v2 smart route → {model}",
                "signature": (model, primary.get("provider"), primary.get("base_url")),
            }
        except Exception:
            logger.debug("routing_v2 failed, falling back to legacy", exc_info=True)

    # --- Fallback: legacy cheap heuristic ---
    cheap_model = cfg.get("cheap_model") or {}
    cheap_route = _legacy_cheap_route(user_message, cheap_model)

    # Update task state for fallback path
    if cheap_route:
        state = _ts.record_turn(state, user_message, was_easy=True)
    else:
        state = _ts.start_task(state, tier=0, model=primary.get("model", ""), category="default")
    _save_task_state(state)

    if not cheap_route:
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
    api_key_env = str(cheap_route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = resolve_runtime_provider(
            requested=cheap_route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=cheap_route.get("base_url"),
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
        "model": cheap_route.get("model"),
        "runtime": {
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
            "credential_pool": runtime.get("credential_pool"),
        },
        "label": f"smart route → {cheap_route.get('model')} ({runtime.get('provider')})",
        "signature": (
            cheap_route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }


# Backwards-compatible alias — callers that imported choose_cheap_model_route
# will still work, but now they go through the unified router.
choose_cheap_model_route = _legacy_cheap_route


# ─────────────────────────────────────────────────────────────
# Optional telemetry instrumentation (opt-in via env var)
# ─────────────────────────────────────────────────────────────

def instrument_resolve_turn_route(store=None):
    """Instrument resolve_turn_route with telemetry wrapping.

    This function is idempotent: if resolve_turn_route is already
    instrumented (has _routing_instrumented=True attribute), it's a no-op.

    Args:
        store: Optional Path for telemetry store. If None, uses DEFAULT_STORE.
    """
    if getattr(resolve_turn_route, "_routing_instrumented", False):
        return

    try:
        from routing_telemetry import wrap_resolve_turn_route
        wrapped = wrap_resolve_turn_route(store=store)(resolve_turn_route)
        wrapped._routing_instrumented = True
        globals()["resolve_turn_route"] = wrapped
    except ImportError:
        logger.debug("routing_telemetry not available, skipping instrumentation")


# Auto-instrumentation at import time if env var is set
if os.getenv("HERMES_ROUTING_TELEMETRY"):
    if is_truthy_value(os.getenv("HERMES_ROUTING_TELEMETRY")):
        instrument_resolve_turn_route()