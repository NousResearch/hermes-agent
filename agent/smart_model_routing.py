"""Helpers for policy-driven per-turn model routing."""

from __future__ import annotations

import os
import re
import threading
from typing import Any, Dict, Optional, TYPE_CHECKING

from hermes_cli import runtime_provider

if TYPE_CHECKING:
    from agent.tiny_router import RouterOutput

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
    "delegate",
    "subagent",
    "cron",
    "docker",
    "kubernetes",
}

_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_TIER_ORDER = ("low", "medium", "high")
_SESSION_ROUTE_LOCK = threading.Lock()
_SESSION_HIGH_TIER_CALLS: Dict[str, int] = {}


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


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_tier(value: Any, default: str = "medium") -> str:
    raw = str(value or "").strip().lower()
    if raw in _TIER_ORDER:
        return raw
    return default


def _is_simple_turn(user_message: str, routing_config: Optional[Dict[str, Any]]) -> bool:
    """Conservative heuristic gate for short, low-complexity turns."""
    cfg = routing_config or {}
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

    lowered = text.lower()
    words = {token.strip(".,:;!?()[]{}\"'`") for token in lowered.split()}
    if words & _COMPLEX_KEYWORDS:
        return False
    return True


def _high_tier_usage_count(session_id: str) -> int:
    with _SESSION_ROUTE_LOCK:
        return int(_SESSION_HIGH_TIER_CALLS.get(session_id, 0))


def _increment_high_tier_usage_count(session_id: Optional[str]) -> None:
    if not session_id:
        return
    with _SESSION_ROUTE_LOCK:
        current = int(_SESSION_HIGH_TIER_CALLS.get(session_id, 0))
        _SESSION_HIGH_TIER_CALLS[session_id] = current + 1


def _apply_high_tier_budget(
    desired_tier: str,
    routing_config: Optional[Dict[str, Any]],
    session_id: Optional[str],
) -> str:
    if desired_tier != "high":
        return desired_tier
    cfg = routing_config or {}
    cap = _coerce_int(cfg.get("max_high_tier_calls_per_session"), 0)
    if cap <= 0 or not session_id:
        return desired_tier
    if _high_tier_usage_count(session_id) < cap:
        return desired_tier
    return "medium"


def choose_cheap_model_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
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

    if not _is_simple_turn(user_message, cfg):
        return None

    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = "simple_turn"
    return route


def _cheap_route_from_config(routing_config: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
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
    route = dict(cheap_model)
    route["provider"] = provider
    route["model"] = model
    return route


def _route_name_for_tier(
    routing_config: Optional[Dict[str, Any]],
    tier: str,
) -> str:
    cfg = routing_config or {}
    tier_routes = cfg.get("tier_routes") or {}
    if isinstance(tier_routes, dict):
        route_name = tier_routes.get(tier)
        if isinstance(route_name, str) and route_name.strip():
            return route_name.strip()
    if tier == "low":
        return "cheap"
    return "primary"


def _route_from_name(
    routing_config: Optional[Dict[str, Any]],
    route_name: str,
    *,
    reason: str,
) -> Optional[Dict[str, Any]]:
    cfg = routing_config or {}
    if not _coerce_bool(cfg.get("enabled"), False):
        return None

    raw_name = str(route_name or "").strip()
    normalized = raw_name.lower()
    if not normalized or normalized == "primary":
        return None

    if normalized == "cheap":
        route = _cheap_route_from_config(cfg)
        if route:
            route["routing_reason"] = reason
            route["route_name"] = "cheap"
        return route

    routes_cfg = cfg.get("routes") or {}
    if not isinstance(routes_cfg, dict):
        return None

    selected_name = None
    selected = None
    if raw_name in routes_cfg and isinstance(routes_cfg.get(raw_name), dict):
        selected_name = raw_name
        selected = routes_cfg.get(raw_name)
    else:
        for key, value in routes_cfg.items():
            if isinstance(key, str) and key.strip().lower() == normalized and isinstance(value, dict):
                selected_name = key
                selected = value
                break
    if not isinstance(selected, dict):
        return None

    provider = str(selected.get("provider") or "").strip().lower()
    model = str(selected.get("model") or "").strip()
    if not provider or not model:
        return None

    route = dict(selected)
    route["provider"] = provider
    route["model"] = model
    route["routing_reason"] = reason
    route["route_name"] = str(selected_name or raw_name or normalized)
    return route


def _resolve_named_route(
    primary: Dict[str, Any],
    routing_config: Optional[Dict[str, Any]],
    route_name: str,
    *,
    reason: str,
    label_prefix: str,
) -> Optional[Dict[str, Any]]:
    normalized = str(route_name or "").strip().lower()
    if not normalized or normalized == "primary":
        return _primary_route_result(primary)

    route = _route_from_name(routing_config, route_name, reason=reason)
    if not route:
        return None
    return _resolve_route_result(route, label_prefix=label_prefix)


def _primary_route_result(primary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "model": primary.get("model"),
        "runtime": {
            "api_key": primary.get("api_key"),
            "base_url": primary.get("base_url"),
            "provider": primary.get("provider"),
            "api_mode": primary.get("api_mode"),
            "command": primary.get("command"),
            "args": list(primary.get("args") or []),
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


def _resolve_route_result(route: Dict[str, Any], *, label_prefix: str) -> Optional[Dict[str, Any]]:
    explicit_api_key = None
    api_key_env = str(route.get("api_key_env") or "").strip()
    if api_key_env:
        explicit_api_key = os.getenv(api_key_env) or None

    try:
        runtime = runtime_provider.resolve_runtime_provider(
            requested=route.get("provider"),
            explicit_api_key=explicit_api_key,
            explicit_base_url=route.get("base_url"),
        )
    except Exception:
        return None

    route_name = str(route.get("route_name") or "").strip()
    route_suffix = f" [{route_name}]" if route_name and route_name != "cheap" else ""
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
        "label": f"{label_prefix} → {route.get('model')} ({runtime.get('provider')}){route_suffix}",
        "signature": (
            route.get("model"),
            runtime.get("provider"),
            runtime.get("base_url"),
            runtime.get("api_mode"),
            runtime.get("command"),
            tuple(runtime.get("args") or ()),
        ),
    }


def _tiny_router_is_policy_eligible(
    router_output: Optional["RouterOutput"],
    tiny_router_config: Optional[Dict[str, Any]],
) -> bool:
    if router_output is None:
        return False
    if str(getattr(router_output, "source", "") or "") in ("disabled", "error"):
        return False

    tr_cfg = tiny_router_config or {}
    if not _coerce_bool(tr_cfg.get("enabled"), False):
        return False
    if str(tr_cfg.get("behavior_mode") or "shadow").strip().lower() != "active":
        return False

    th = tr_cfg.get("confidence_thresholds") or {}
    overall_th = _coerce_float(th.get("overall"), 0.45)
    overall = _coerce_float(getattr(router_output, "overall_confidence", 0.0), 0.0)
    return overall >= overall_th


def _tiny_router_prefers_low_stakes_route(
    router_output: "RouterOutput",
    tiny_router_config: Optional[Dict[str, Any]],
) -> bool:
    cfg = tiny_router_config or {}
    th = cfg.get("confidence_thresholds") or {}
    act_th = _coerce_float(th.get("actionability"), 0.5)
    urg_th = _coerce_float(th.get("urgency"), 0.5)
    ret_th = _coerce_float(th.get("retention"), 0.5)

    actionability = getattr(router_output, "actionability", None)
    urgency = getattr(router_output, "urgency", None)
    retention = getattr(router_output, "retention", None)
    relation = getattr(router_output, "relation_to_previous", None)

    action_label = str(getattr(actionability, "label", "none") or "none")
    action_conf = _coerce_float(getattr(actionability, "confidence", 0.0), 0.0)
    urgency_label = str(getattr(urgency, "label", "low") or "low")
    urgency_conf = _coerce_float(getattr(urgency, "confidence", 0.0), 0.0)
    retention_label = str(getattr(retention, "label", "ephemeral") or "ephemeral")
    retention_conf = _coerce_float(getattr(retention, "confidence", 0.0), 0.0)
    relation_label = str(getattr(relation, "label", "new") or "new")
    relation_conf = _coerce_float(getattr(relation, "confidence", 0.0), 0.0)

    if action_label == "act" and action_conf >= act_th:
        return False
    if urgency_label == "high" and urgency_conf >= urg_th:
        return False
    if retention_label == "remember" and retention_conf >= ret_th:
        return False
    if relation_label in ("correction", "cancellation") and relation_conf >= 0.5:
        return False
    return action_label in ("none", "review") and urgency_label == "low"


def resolve_turn_route(
    user_message: str,
    routing_config: Optional[Dict[str, Any]],
    primary: Dict[str, Any],
    *,
    tiny_router_config: Optional[Dict[str, Any]] = None,
    router_output: Optional["RouterOutput"] = None,
    routing_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    route_cfg = routing_config or {}
    if not _coerce_bool(route_cfg.get("enabled"), False):
        return _primary_route_result(primary)

    tr_cfg = tiny_router_config or {}
    ctx = routing_context or {}
    session_id = str(ctx.get("session_id") or "").strip() or None

    tier = "low" if _is_simple_turn(user_message, route_cfg) else "medium"
    tier_source = "simple" if tier == "low" else "default"
    if _tiny_router_is_policy_eligible(router_output, tr_cfg):
        if _tiny_router_prefers_low_stakes_route(router_output, tr_cfg):
            tier = "low"
            tier_source = "tiny-router"
        else:
            tier = "high"
            tier_source = "tiny-router"

    tier = _apply_high_tier_budget(tier, route_cfg, session_id)
    route_name = _route_name_for_tier(route_cfg, tier)
    label_prefix = "tiny-router" if tier_source == "tiny-router" else "smart route"
    reason = f"{tier_source}_{tier}"

    resolved = _resolve_named_route(
        primary,
        route_cfg,
        route_name,
        reason=reason,
        label_prefix=label_prefix,
    )
    if not resolved:
        resolved = _primary_route_result(primary)

    resolved["route_tier"] = tier
    resolved["routing_source"] = tier_source
    if tier == "high":
        _increment_high_tier_usage_count(session_id)
    return resolved
