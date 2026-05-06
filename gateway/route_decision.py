"""ROI-aware route decision layer for Feishu foreground routing.

P1 keeps this in shadow-compatible pure functions: callers can inspect the same
decision object whether they merely hint, auto-dispatch safe work, or ask for
approval on risky work. This task keeps runtime integration in shadow mode only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from gateway.background_wakeups import BackgroundWakeupPlan, resolve_background_wakeup
from gateway.route_envelope import TaskEnvelope, infer_task_envelope


_HISTORY_WINDOW_DAYS = 30
_ROUTE_HISTORY_GAIN_CAP = 1.0
_WORKER_OUTCOME_GAIN_CAP = 0.75
_TOTAL_HISTORY_GAIN_CAP = 1.5
_FULL_HISTORY_WEIGHT_SAMPLE_COUNT = 3.0


@dataclass(frozen=True)
class RouteScore:
    """Additive ROI score with explainable components."""

    components: Mapping[str, float]
    total: float


@dataclass(frozen=True)
class RouteDecision:
    """Structured routing decision for one inbound task."""

    decision_type: str
    shadow_mode: bool
    envelope: TaskEnvelope
    forced_routes: tuple[str, ...]
    wrapper_commands: tuple[str, ...]
    missing_toolsets: tuple[str, ...]
    confidence: float
    score: RouteScore
    risk_class: str
    auto_dispatch_threshold: float
    reasons: tuple[str, ...]

    @property
    def auto_dispatch(self) -> bool:
        return self.decision_type == "auto_dispatch"


def _dedupe(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _meaningful_routes(plan: BackgroundWakeupPlan) -> tuple[str, ...]:
    return tuple(route for route in plan.route_names if route not in {"default", "work"})


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _route_history_row(summary: Mapping[str, Mapping[str, Any]], route_name: str) -> Mapping[str, Any] | None:
    route_key = str(route_name or "").strip().lower()
    if not route_key:
        return None
    for candidate in (route_key, route_key.replace("_", "-"), route_key.replace("-", "_")):
        row = summary.get(candidate)
        if row is not None:
            return row
    return None


def _coerce_sample_count(value: Any) -> float:
    result = _coerce_float(value)
    if result is None:
        return 0.0
    return max(0.0, result)


def _weighted_average(pairs: Sequence[tuple[float, float]]) -> float:
    total_weight = sum(max(0.0, weight) for _, weight in pairs)
    if total_weight <= 0:
        return 0.0
    return sum(value * max(0.0, weight) for value, weight in pairs) / float(total_weight)


def _history_weight(sample_count: float) -> float:
    if sample_count <= 0:
        return 0.0
    return min(1.0, float(sample_count) / _FULL_HISTORY_WEIGHT_SAMPLE_COUNT)


def _history_gain_components(routes: Sequence[str]) -> dict[str, float]:
    """Return bounded telemetry-fed score components for resolved routes.

    Historical feedback is deliberately capped and sample-weighted. Cold-start
    or telemetry read failures contribute zero, while mature history can nudge
    but not dominate the deterministic route score.
    """

    if not routes:
        return {"route_history_gain": 0.0, "worker_outcome_gain": 0.0}

    try:
        from tools.skill_usage import summarize_route_usage

        summary = summarize_route_usage(window_days=_HISTORY_WINDOW_DAYS)
    except Exception:
        return {"route_history_gain": 0.0, "worker_outcome_gain": 0.0}

    route_signal_pairs: list[tuple[float, float]] = []
    worker_signal_pairs: list[tuple[float, float]] = []
    for route in routes:
        row = _route_history_row(summary, route)
        if row is None:
            continue
        sample_count = _coerce_sample_count(row.get("history_effective_sample_count_30d"))
        if sample_count <= 0:
            sample_count = _coerce_sample_count(row.get("history_sample_count_30d"))
        if sample_count > 0:
            route_signal = _coerce_float(row.get("route_signal_score_30d"))
            if route_signal is not None:
                route_signal_pairs.append((route_signal, sample_count))
        worker_count = _coerce_sample_count(row.get("worker_outcome_effective_sample_count_30d"))
        if worker_count <= 0:
            worker_count = _coerce_sample_count(row.get("worker_outcome_count_30d"))
        if worker_count > 0:
            worker_signal = _coerce_float(row.get("worker_outcome_signal_30d"))
            if worker_signal is not None:
                worker_signal_pairs.append((worker_signal, worker_count))

    route_sample_count = sum(weight for _, weight in route_signal_pairs)
    worker_sample_count = sum(weight for _, weight in worker_signal_pairs)
    route_gain = _clamp(
        _weighted_average(route_signal_pairs) * 0.5 * _history_weight(route_sample_count),
        -_ROUTE_HISTORY_GAIN_CAP,
        _ROUTE_HISTORY_GAIN_CAP,
    )
    worker_gain = _clamp(
        _weighted_average(worker_signal_pairs) * 0.25 * _history_weight(worker_sample_count),
        -_WORKER_OUTCOME_GAIN_CAP,
        _WORKER_OUTCOME_GAIN_CAP,
    )

    total_history_gain = route_gain + worker_gain
    if abs(total_history_gain) > _TOTAL_HISTORY_GAIN_CAP:
        scale = _TOTAL_HISTORY_GAIN_CAP / abs(total_history_gain)
        route_gain *= scale
        worker_gain *= scale

    return {
        "route_history_gain": round(route_gain, 3),
        "worker_outcome_gain": round(worker_gain, 3),
    }


def _route_confidence(plan: BackgroundWakeupPlan, envelope: TaskEnvelope) -> float:
    routes = _meaningful_routes(plan)
    if not routes:
        return 0.0
    confidence = 0.72
    if plan.match_details:
        confidence += 0.1
    if envelope.evidence_required or len(envelope.required_capabilities) > 1:
        confidence += 0.08
    if "orchestration" in envelope.required_capabilities and "multi_agent" in routes:
        confidence += 0.05
    return min(0.95, confidence)


def _score_route_decision(
    *,
    envelope: TaskEnvelope,
    plan: BackgroundWakeupPlan,
    active_toolsets: Sequence[str],
) -> RouteScore:
    routes = _meaningful_routes(plan)
    active_toolset_set = set(active_toolsets or ())
    missing_toolsets = [toolset for toolset in plan.enabled_toolsets if toolset not in active_toolset_set]
    longish = envelope.estimated_duration_class in {"medium", "long"}
    multi_route = len(routes) > 1
    orchestration = "multi_agent" in routes or (plan.execution_plan is not None and plan.execution_plan.dispatch_policy == "parallel")

    quality_gain = 0.0
    if routes:
        quality_gain += 2.0
    if envelope.evidence_required:
        quality_gain += 1.5
    if longish:
        quality_gain += 1.0
    if envelope.required_capabilities:
        quality_gain += min(1.5, len(envelope.required_capabilities) * 0.5)

    components = {
        "quality_gain": quality_gain,
        "speed_gain": 2.0 if longish else (0.75 if routes else 0.0),
        "parallelism_gain": 2.0 if orchestration else 0.0,
        "context_isolation_gain": 1.25 if longish or multi_route else 0.0,
        "missing_tool_gain": min(2.0, float(len(missing_toolsets)) * 0.75),
        "latency_penalty": -1.0 if routes else 0.0,
        "cost_penalty": -0.5 * len(routes),
        "coordination_penalty": -1.0 if orchestration else (-0.25 * max(0, len(routes) - 1)),
        "risk_penalty": {
            "read_only": 0.0,
            "internal_write": -1.0,
            "external_write": -5.0,
            "destructive": -10.0,
        }.get(envelope.risk_class, -2.0),
    }
    components.update(_history_gain_components(routes))
    total = round(sum(components.values()), 3)
    return RouteScore(components=components, total=total)


def resolve_route_decision(
    prompt: str,
    *,
    platform: str = "feishu",
    active_toolsets: Sequence[str] | None = None,
    wake_plan: BackgroundWakeupPlan | None = None,
    auto_dispatch_threshold: float = 3.0,
    confidence_threshold: float = 0.75,
    feishu_auto_dispatch_enabled: bool = True,
    telemetry_source: str | None = None,
) -> RouteDecision:
    """Resolve the P1 ROI-aware routing decision for a prompt."""

    platform_key = str(platform or "feishu").strip().lower() or "feishu"
    envelope = infer_task_envelope(prompt)
    resolved_active_toolsets = tuple(active_toolsets or ())
    plan = wake_plan or resolve_background_wakeup(
        prompt,
        platform=platform_key,
        default_toolsets=resolved_active_toolsets,
    )
    routes = _meaningful_routes(plan)
    score = _score_route_decision(
        envelope=envelope,
        plan=plan,
        active_toolsets=resolved_active_toolsets,
    )
    confidence = _route_confidence(plan, envelope)
    missing_toolsets = tuple(toolset for toolset in plan.enabled_toolsets if toolset not in set(resolved_active_toolsets))
    reasons: list[str] = []

    if envelope.risk_class in {"external_write", "destructive"}:
        decision_type = "approval_required"
        reasons.append(f"risk_gate:{envelope.risk_class}")
    elif envelope.foreground_only or not routes:
        decision_type = "foreground_only"
        reasons.append("foreground_only" if envelope.foreground_only else "no_meaningful_routes")
    elif confidence >= confidence_threshold and score.total >= auto_dispatch_threshold and envelope.risk_class in {"read_only", "internal_write"}:
        if feishu_auto_dispatch_enabled:
            decision_type = "auto_dispatch"
            reasons.append("roi_above_threshold")
        else:
            decision_type = "suggest_wrapper"
            reasons.append("auto_dispatch_disabled")
    else:
        decision_type = "suggest_wrapper"
        if confidence < confidence_threshold:
            reasons.append("confidence_below_threshold")
        if score.total < auto_dispatch_threshold:
            reasons.append("score_below_threshold")

    historical_gain = float(score.components.get("route_history_gain", 0.0)) + float(
        score.components.get("worker_outcome_gain", 0.0)
    )
    if historical_gain > 0:
        reasons.append("route_history_positive")
    elif historical_gain < 0:
        reasons.append("route_history_negative")

    decision = RouteDecision(
        decision_type=decision_type,
        shadow_mode=True,
        envelope=envelope,
        forced_routes=routes,
        wrapper_commands=tuple(plan.wrapper_commands),
        missing_toolsets=missing_toolsets,
        confidence=round(confidence, 3),
        score=score,
        risk_class=envelope.risk_class,
        auto_dispatch_threshold=auto_dispatch_threshold,
        reasons=_dedupe(reasons),
    )
    _record_route_decision_usage(
        decision,
        plan=plan,
        platform=platform_key,
        source=telemetry_source,
    )
    return decision


def _record_route_decision_usage(
    decision: RouteDecision,
    *,
    plan: BackgroundWakeupPlan,
    platform: str,
    source: str | None,
) -> None:
    """Persist route-decision telemetry only when explicitly requested."""

    source_key = str(source or "").strip().lower()
    if not source_key or not decision.forced_routes:
        return

    try:
        from tools.skill_usage import log_route_usage_event
    except Exception:
        return

    details = {
        "platform": str(platform or "").strip().lower(),
        "source": source_key,
        "decision_type": decision.decision_type,
        "route_names": list(decision.forced_routes),
        "selected_routes": list(decision.forced_routes),
        "wrapper_commands": list(decision.wrapper_commands),
        "score": decision.score.total,
        "score_components": dict(decision.score.components),
        "confidence": decision.confidence,
        "risk_class": decision.risk_class,
        "reasons": list(decision.reasons),
        "match_details": list(plan.match_details),
    }
    for route in decision.forced_routes:
        route_details = dict(details)
        route_details["route"] = route
        try:
            log_route_usage_event(
                route_name=route,
                event="route_decision_resolved",
                details=route_details,
            )
        except Exception:
            continue


def should_auto_dispatch_feishu(decision: RouteDecision, *, feishu_auto_dispatch_enabled: bool = True) -> bool:
    """Return whether Feishu may auto-dispatch this decision safely.

    Task 4 exposes this as a pure policy helper only. Runtime auto-dispatch is
    wired in a later phase behind kill switches.
    """

    return (
        feishu_auto_dispatch_enabled
        and decision.decision_type == "auto_dispatch"
        and decision.risk_class in {"read_only", "internal_write"}
        and bool(decision.forced_routes)
    )


def build_feishu_route_decision_shadow_hint(decision: RouteDecision, *, enabled: bool = True) -> str:
    """Build a compact system hint describing the route decision shadow result."""

    if not enabled or decision.decision_type == "foreground_only":
        return ""
    routes = ", ".join(decision.forced_routes) if decision.forced_routes else "none"
    wrappers = ", ".join(decision.wrapper_commands) if decision.wrapper_commands else "none"
    reasons = ", ".join(decision.reasons) if decision.reasons else "none"
    missing = ", ".join(decision.missing_toolsets) if decision.missing_toolsets else "none"
    return (
        "[SYSTEM: RouteDecision shadow: "
        f"decision={decision.decision_type}; risk={decision.risk_class}; "
        f"confidence={decision.confidence}; score={decision.score.total}; "
        f"routes={routes}; wrappers={wrappers}; missing_toolsets={missing}; reasons={reasons}. "
        "This is a routing decision aid, not a worker receipt. Do not claim a worker picked it up unless auto-dispatch or an explicit wrapper produced a real receipt.]"
    )
