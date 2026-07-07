"""Observe-mode smart model routing for Hermes turns.

The first implementation is intentionally non-invasive: it evaluates a route
before a turn starts, records the decision, and never mutates the active model.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from agent.smart_router_prompts import build_router_messages, parse_router_json
from agent.task_features import TaskFeatures, extract_task_features

logger = logging.getLogger(__name__)

ROUTES = frozenset({"cheap", "default", "strong", "moa", "no_change"})
MODES = frozenset({"off", "observe", "suggest", "auto"})


@dataclass(frozen=True)
class HistoricalTaskExample:
    summary: str
    route: str = ""
    model: str = ""
    provider: str = ""
    tool_call_count: int | None = None
    api_call_count: int | None = None
    elapsed_seconds: float | None = None
    success: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SmartRouteDecision:
    enabled: bool
    mode: str
    route: str
    confidence: float
    risk: str
    reason: str
    expected_tool_calls: int | None
    should_use_moa: bool
    source: str
    features: TaskFeatures
    historical_examples: list[HistoricalTaskExample] = field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    preset: str | None = None
    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["features"] = self.features.as_dict()
        data["historical_examples"] = [ex.as_dict() for ex in self.historical_examples]
        return data


def evaluate_turn_route(
    agent: Any,
    user_message: Any,
    *,
    messages: list[dict[str, Any]] | None = None,
    effective_task_id: str = "",
    config: dict[str, Any] | None = None,
) -> SmartRouteDecision | None:
    """Return an observe-mode route decision for this turn, if enabled."""

    cfg = _normalize_config(config if config is not None else _read_router_config())
    mode = cfg["mode"]
    if mode == "off":
        return None

    features = extract_task_features(user_message, message_count=len(messages or []))
    historical_examples = _retrieve_historical_examples(user_message, cfg)
    heuristic = _heuristic_decision(features)
    payload = _build_payload(
        agent,
        user_message,
        features,
        historical_examples,
        cfg,
        heuristic,
        effective_task_id=effective_task_id,
    )

    parsed: dict[str, Any] | None = None
    source = "heuristic"
    if _llm_judge_enabled(cfg):
        try:
            parsed = _call_router_llm(agent, cfg, payload)
            source = "llm" if parsed else "heuristic_after_llm_failure"
        except Exception as exc:
            logger.debug("smart_model_router LLM judge failed: %s", exc, exc_info=True)
            source = "heuristic_after_llm_failure"

    selected = _decision_from_parsed(parsed, heuristic)
    route_cfg = _route_config(cfg, selected["route"])
    decision = SmartRouteDecision(
        enabled=True,
        mode=mode,
        route=selected["route"],
        confidence=selected["confidence"],
        risk=selected["risk"],
        reason=selected["reason"],
        expected_tool_calls=selected["expected_tool_calls"],
        should_use_moa=selected["should_use_moa"],
        source=source,
        features=features,
        historical_examples=historical_examples,
        provider=route_cfg.get("provider"),
        model=route_cfg.get("model"),
        preset=route_cfg.get("preset"),
    )
    logger.info(
        "smart_model_router decision mode=%s source=%s route=%s confidence=%.2f "
        "risk=%s provider=%s model=%s task_id=%s reason=%s",
        decision.mode,
        decision.source,
        decision.route,
        decision.confidence,
        decision.risk,
        decision.provider or "-",
        decision.model or decision.preset or "-",
        effective_task_id or "-",
        decision.reason,
    )
    return decision


def _normalize_config(raw: dict[str, Any] | None) -> dict[str, Any]:
    cfg = raw if isinstance(raw, dict) else {}
    mode = str(cfg.get("mode") or "off").strip().lower()
    if mode not in MODES:
        mode = "off"
    routes = cfg.get("routes") if isinstance(cfg.get("routes"), dict) else {}
    router = cfg.get("router") if isinstance(cfg.get("router"), dict) else {}
    return {
        "mode": mode,
        "judge": str(cfg.get("judge") or cfg.get("strategy") or "heuristic").strip().lower(),
        "router": router,
        "routes": routes,
        "top_k": _coerce_int(cfg.get("top_k"), 0),
        "confidence_threshold": _coerce_float(cfg.get("confidence_threshold"), 0.75),
    }


def _read_router_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config() or {}
    except Exception:
        return {}
    cfg = raw.get("smart_model_routing")
    return cfg if isinstance(cfg, dict) else {}


def _retrieve_historical_examples(
    user_message: Any,
    cfg: dict[str, Any],
) -> list[HistoricalTaskExample]:
    """Placeholder for the embedding-backed history lookup.

    Keeping this as a function makes the next step small: replace this body with
    a task embedding index lookup and keep the router prompt contract stable.
    """

    _ = (user_message, cfg)
    return []


def _heuristic_decision(features: TaskFeatures) -> dict[str, Any]:
    score = features.complexity_score
    if features.high_risk_domain or features.destructive_intent:
        route = "strong"
        risk = "high"
        confidence = 0.72
    elif features.architecture_or_analysis and (features.review_or_debug or score >= 6):
        route = "moa"
        risk = "medium"
        confidence = 0.62
    elif score >= 6:
        route = "strong"
        risk = "medium"
        confidence = 0.68
    elif score <= 1:
        route = "cheap"
        risk = "low"
        confidence = 0.65
    else:
        route = "default"
        risk = "medium" if score >= 4 else "low"
        confidence = 0.64

    expected_tool_calls = 0
    if features.likely_needs_file_access:
        expected_tool_calls += 4
    if features.requires_code_edit:
        expected_tool_calls += 4
    if features.likely_needs_tests:
        expected_tool_calls += 2
    if features.likely_needs_web:
        expected_tool_calls += 2

    return {
        "route": route,
        "confidence": confidence,
        "risk": risk,
        "expected_tool_calls": expected_tool_calls,
        "reason": "heuristic score=%s signals=%s" % (score, ",".join(features.signals) or "none"),
        "should_use_moa": route == "moa",
    }


def _build_payload(
    agent: Any,
    user_message: Any,
    features: TaskFeatures,
    historical_examples: list[HistoricalTaskExample],
    cfg: dict[str, Any],
    heuristic: dict[str, Any],
    *,
    effective_task_id: str,
) -> dict[str, Any]:
    return {
        "user_message": _preview(user_message, 3000),
        "platform": getattr(agent, "platform", "") or "",
        "current_provider": getattr(agent, "provider", "") or "",
        "current_model": getattr(agent, "model", "") or "",
        "api_mode": getattr(agent, "api_mode", "") or "",
        "task_id": effective_task_id,
        "features": features.as_dict(),
        "heuristic_decision": heuristic,
        "retrieved_examples": [ex.as_dict() for ex in historical_examples],
        "available_routes": cfg.get("routes") or {},
    }


def _llm_judge_enabled(cfg: dict[str, Any]) -> bool:
    if cfg.get("judge") != "llm":
        return False
    router = cfg.get("router") if isinstance(cfg.get("router"), dict) else {}
    return bool(router.get("provider") and router.get("model"))


def _call_router_llm(
    agent: Any,
    cfg: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any] | None:
    from agent.auxiliary_client import call_llm, extract_content_or_reasoning

    router = cfg.get("router") if isinstance(cfg.get("router"), dict) else {}
    response = call_llm(
        task="smart_routing",
        provider=router.get("provider"),
        model=router.get("model"),
        main_runtime={
            "provider": getattr(agent, "provider", "") or "",
            "model": getattr(agent, "model", "") or "",
            "base_url": getattr(agent, "base_url", "") or "",
            "api_key": getattr(agent, "api_key", "") or "",
            "api_mode": getattr(agent, "api_mode", "") or "",
        },
        messages=build_router_messages(payload),
        temperature=None,
        max_tokens=_coerce_int(router.get("max_tokens"), 400),
        timeout=_coerce_float(router.get("timeout"), 30.0),
    )
    return parse_router_json(extract_content_or_reasoning(response))


def _decision_from_parsed(
    parsed: dict[str, Any] | None,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if not parsed:
        return fallback
    route = str(parsed.get("route") or "").strip().lower()
    if route not in ROUTES:
        route = fallback["route"]
    return {
        "route": route,
        "confidence": max(0.0, min(1.0, _coerce_float(parsed.get("confidence"), fallback["confidence"]))),
        "risk": _risk(parsed.get("risk"), fallback["risk"]),
        "expected_tool_calls": max(
            0,
            _coerce_int(
                parsed.get("expected_tool_calls"),
                fallback["expected_tool_calls"] or 0,
            ),
        ),
        "reason": _preview(parsed.get("reason") or fallback["reason"], 500),
        "should_use_moa": bool(parsed.get("should_use_moa", route == "moa")),
    }


def _route_config(cfg: dict[str, Any], route: str) -> dict[str, Any]:
    routes = cfg.get("routes") if isinstance(cfg.get("routes"), dict) else {}
    entry = routes.get(route)
    return entry if isinstance(entry, dict) else {}


def _risk(value: Any, default: str) -> str:
    risk = str(value or default or "medium").strip().lower()
    return risk if risk in {"low", "medium", "high"} else "medium"


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


def _preview(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."
