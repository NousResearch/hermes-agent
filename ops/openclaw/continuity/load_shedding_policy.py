#!/usr/bin/env python3
"""Deterministic lane-strain evaluator + Slice 11 trigger bridge (Slice 8)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping

SIGNAL_IDS: tuple[str, ...] = (
    "LS_SIG_CONTEXT_UTIL_RATIO",
    "LS_SIG_BOOTSTRAP_TRUNCATIONS_30M",
    "LS_SIG_COMPACTION_TIMEOUTS_30M",
    "LS_SIG_TURN_LATENCY_P95_15M",
    "LS_SIG_LONG_WAIT_EVENTS_15M",
    "LS_SIG_PROVIDER_RESTARTS_30M",
    "LS_SIG_CHAT_ACTION_FAILURES_15M",
    "LS_SIG_CONTINUITY_FRESHNESS_RATIO",
)

SIGNAL_BAND_ENUM: tuple[str, ...] = ("NORMAL", "WARNING", "CRITICAL", "UNKNOWN")
LANE_TIER_ENUM: tuple[str, ...] = ("NORMAL", "WARNING", "CRITICAL", "UNKNOWN")

SLICE11_STATE_FLOOR_TIER: dict[str, str] = {
    "HEALTHY": "NORMAL",
    "WARNING": "WARNING",
    "PRE_FAILOVER": "CRITICAL",
    "FAILOVER_PREP": "CRITICAL",
    "RESET_READY": "CRITICAL",
    "BLOCKED_RESET": "CRITICAL",
    "SUCCESSOR_RESUME_VALIDATION": "CRITICAL",
}

_SIGNAL_THRESHOLDS: dict[str, dict[str, float]] = {
    "LS_SIG_CONTEXT_UTIL_RATIO": {"warning": 0.82, "critical": 0.92},
    "LS_SIG_BOOTSTRAP_TRUNCATIONS_30M": {"warning": 1.0, "critical": 2.0},
    "LS_SIG_COMPACTION_TIMEOUTS_30M": {"warning": 1.0, "critical": 2.0},
    "LS_SIG_TURN_LATENCY_P95_15M": {"warning": 20.0, "critical": 45.0},
    "LS_SIG_LONG_WAIT_EVENTS_15M": {"warning": 2.0, "critical": 4.0},
    "LS_SIG_PROVIDER_RESTARTS_30M": {"warning": 1.0, "critical": 2.0},
    "LS_SIG_CHAT_ACTION_FAILURES_15M": {"warning": 3.0, "critical": 6.0},
    "LS_SIG_CONTINUITY_FRESHNESS_RATIO": {"warning": 0.8, "critical": 1.0},
}


def _now_iso(now: str | None = None) -> str:
    if isinstance(now, str) and now.strip():
        return now.strip()
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    txt = str(value or "").strip()
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        return None


def evaluate_signal_band(signal_id: str, value: Any) -> str:
    key = str(signal_id or "").strip()
    thresholds = _SIGNAL_THRESHOLDS.get(key)
    if not thresholds:
        return "UNKNOWN"

    parsed = _coerce_float(value)
    if parsed is None:
        return "UNKNOWN"

    critical = float(thresholds["critical"])
    warning = float(thresholds["warning"])

    if key == "LS_SIG_CONTINUITY_FRESHNESS_RATIO":
        if parsed > critical:
            return "CRITICAL"
        if parsed > warning:
            return "WARNING"
        return "NORMAL"

    if parsed >= critical:
        return "CRITICAL"
    if parsed >= warning:
        return "WARNING"
    return "NORMAL"


def _tier_rank(tier: str) -> int:
    if tier == "NORMAL":
        return 0
    if tier == "WARNING":
        return 1
    return 2


def effective_lane_tier(*, derived_tier: str, fsm_state: str | None) -> str:
    state = str(fsm_state or "").strip()
    floor = SLICE11_STATE_FLOOR_TIER.get(state, "NORMAL")

    if derived_tier == "UNKNOWN":
        return "CRITICAL"

    if _tier_rank(derived_tier) >= _tier_rank(floor):
        return derived_tier
    return floor


def map_lane_tier_to_failover_trigger(*, lane_tier: str, clear_stable: bool) -> str | None:
    if lane_tier == "UNKNOWN":
        return "TR_UNKNOWN_OR_INCOMPLETE_SIGNAL"
    if lane_tier == "CRITICAL":
        return "TR_CRITICAL_THRESHOLD_REACHED"
    if lane_tier == "WARNING":
        return "TR_WARN_THRESHOLD_REACHED"
    if clear_stable:
        return "TR_WARN_THRESHOLD_CLEARED_STABLE"
    return None


def project_load_shedding_continuity_surface(
    *,
    decision: Mapping[str, Any],
    signal_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    lane_tier = str(decision.get("lane_tier") or "NORMAL").strip().upper()
    derived_tier = str(decision.get("derived_lane_tier") or lane_tier).strip().upper()
    trigger_emitted = str(decision.get("trigger_emitted") or "").strip() or None
    unknown_signals = signal_snapshot.get("unknown_signals") if isinstance(signal_snapshot.get("unknown_signals"), list) else []

    return {
        "lane_health_state": lane_tier,
        "warning_tier": lane_tier == "WARNING",
        "critical_tier": lane_tier == "CRITICAL",
        "escape_triggered": trigger_emitted == "TR_CRITICAL_THRESHOLD_REACHED",
        "clear_stable": trigger_emitted == "TR_WARN_THRESHOLD_CLEARED_STABLE",
        "derived_tier": derived_tier,
        "unknown_signal_count": len(unknown_signals),
        "thin_mode": bool(decision.get("thin_mode") is True),
        "trigger_emitted": trigger_emitted,
    }


def project_load_shedding_operator_surface(
    *,
    decision: Mapping[str, Any],
    signal_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    continuity_projection = project_load_shedding_continuity_surface(
        decision=decision,
        signal_snapshot=signal_snapshot,
    )

    lane_health_state = str(continuity_projection.get("lane_health_state") or "NORMAL")
    next_action = "none"
    if continuity_projection.get("critical_tier") is True:
        next_action = "launch_continuity_worker"
    elif continuity_projection.get("warning_tier") is True:
        next_action = "maintain_thin_mode"
    elif continuity_projection.get("clear_stable") is True:
        next_action = "return_to_normal_mode"

    return {
        "lane_health_state": lane_health_state,
        "warning_tier": bool(continuity_projection.get("warning_tier") is True),
        "critical_tier": bool(continuity_projection.get("critical_tier") is True),
        "escape_triggered": bool(continuity_projection.get("escape_triggered") is True),
        "trigger_emitted": continuity_projection.get("trigger_emitted"),
        "next_action": next_action,
    }


def evaluate_lane_strain_tick(
    *,
    signals: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None = None,
    fsm_state: str | None = None,
    tick_id: int | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    prev = previous_state if isinstance(previous_state, Mapping) else {}

    bands: dict[str, dict[str, Any]] = {}
    unknown_signals: list[str] = []
    warning_count = 0
    critical_found = False

    for signal_id in SIGNAL_IDS:
        value = signals.get(signal_id)
        band = evaluate_signal_band(signal_id, value)
        bands[signal_id] = {"value": value, "band": band}
        if band == "UNKNOWN":
            unknown_signals.append(signal_id)
        elif band == "WARNING":
            warning_count += 1
        elif band == "CRITICAL":
            critical_found = True

    prev_warning_streak = int(prev.get("warning_streak") or 0)
    prev_warning_compound_streak = int(prev.get("warning_compound_streak") or 0)
    prev_normal_streak = int(prev.get("normal_streak") or 0)
    prev_recovery_pending = bool(prev.get("recovery_pending") is True)

    if unknown_signals:
        derived_tier = "UNKNOWN"
        warning_streak = 0
        warning_compound_streak = 0
        normal_streak = 0
    else:
        any_warning = warning_count > 0
        warning_streak = prev_warning_streak + 1 if any_warning else 0
        warning_compound_streak = prev_warning_compound_streak + 1 if warning_count >= 3 else 0

        if critical_found:
            derived_tier = "CRITICAL"
        elif warning_compound_streak >= 2:
            derived_tier = "CRITICAL"
        elif warning_streak >= 2:
            derived_tier = "WARNING"
        else:
            derived_tier = "NORMAL"

        normal_streak = (prev_normal_streak + 1) if derived_tier == "NORMAL" else 0

    previous_tier = str(prev.get("derived_tier") or "NORMAL")
    recovery_pending = prev_recovery_pending
    if derived_tier in {"WARNING", "CRITICAL", "UNKNOWN"}:
        recovery_pending = True
    elif derived_tier == "NORMAL" and previous_tier in {"WARNING", "CRITICAL", "UNKNOWN"}:
        recovery_pending = True

    clear_stable = bool(
        derived_tier == "NORMAL"
        and normal_streak >= 10
        and recovery_pending
    )

    if clear_stable:
        recovery_pending = False

    trigger = map_lane_tier_to_failover_trigger(lane_tier=derived_tier, clear_stable=clear_stable)
    effective_tier = effective_lane_tier(derived_tier=derived_tier, fsm_state=fsm_state)

    decided_at = _now_iso(now)
    decision = {
        "object_type": "clawd.load_shedding.decision.v1",
        "decided_at": decided_at,
        "lane_tier": effective_tier,
        "derived_lane_tier": derived_tier,
        "trigger_emitted": trigger,
        "thin_mode": effective_tier in {"WARNING", "CRITICAL"},
        "suppression_policy": {
            "inline_char_cap": 1200,
            "max_inline_bytes": 4096,
            "max_inline_lines": 120,
        },
        "required_actions": [
            "launch_continuity_worker" if effective_tier == "CRITICAL" else "none"
        ],
    }

    snapshot = {
        "object_type": "clawd.load_shedding.signal_snapshot.v1",
        "evaluated_at": decided_at,
        "tick_id": int(tick_id if tick_id is not None else (int(prev.get("tick_id") or 0) + 1)),
        "signals": bands,
        "unknown_signals": unknown_signals,
        "derived_tier": derived_tier,
    }

    continuity_projection = project_load_shedding_continuity_surface(
        decision=decision,
        signal_snapshot=snapshot,
    )
    operator_projection = project_load_shedding_operator_surface(
        decision=decision,
        signal_snapshot=snapshot,
    )

    decision["projection"] = {
        "continuity_current": continuity_projection,
        "operator": operator_projection,
    }

    new_state = {
        "tick_id": snapshot["tick_id"],
        "warning_streak": warning_streak,
        "warning_compound_streak": warning_compound_streak,
        "normal_streak": normal_streak,
        "derived_tier": derived_tier,
        "recovery_pending": recovery_pending,
    }

    return {
        "signal_snapshot": snapshot,
        "decision": decision,
        "next_state": new_state,
        "projection": {
            "continuity_current": continuity_projection,
            "operator": operator_projection,
        },
    }


def derive_slice11_trigger_from_lane_decision(decision: Mapping[str, Any]) -> str | None:
    tier = str(decision.get("derived_lane_tier") or decision.get("lane_tier") or "").strip()
    clear_stable = str(decision.get("trigger_emitted") or "") == "TR_WARN_THRESHOLD_CLEARED_STABLE"
    return map_lane_tier_to_failover_trigger(lane_tier=tier, clear_stable=clear_stable)
