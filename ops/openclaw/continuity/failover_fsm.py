#!/usr/bin/env python3
"""Deterministic failover FSM core (Slice 11).

Pure helper module:
- canonical enums/contracts for state + trigger + blocker taxonomies
- deterministic trigger precedence
- fail-closed reducer + action authorization gate
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Sequence

FAILOVER_STATE_ENUM: tuple[str, ...] = (
    "HEALTHY",
    "WARNING",
    "PRE_FAILOVER",
    "FAILOVER_PREP",
    "RESET_READY",
    "BLOCKED_RESET",
    "SUCCESSOR_RESUME_VALIDATION",
)

FAILOVER_TRIGGER_ENUM: tuple[str, ...] = (
    "TR_WARN_THRESHOLD_REACHED",
    "TR_WARN_THRESHOLD_CLEARED_STABLE",
    "TR_CRITICAL_THRESHOLD_REACHED",
    "TR_RESET_REQUIRED",
    "TR_FAILOVER_PREP_STARTED",
    "TR_RESET_READINESS_PASS",
    "TR_RESET_READINESS_FAIL",
    "TR_BLOCKERS_CLEARED",
    "TR_RESET_EXECUTED",
    "TR_SUCCESSOR_VALIDATION_PASS",
    "TR_SUCCESSOR_VALIDATION_FAIL",
    "TR_UNKNOWN_OR_INCOMPLETE_SIGNAL",
)

FAILOVER_BLOCKER_ENUM: tuple[str, ...] = (
    "BLK_CONTINUITY_STALE",
    "BLK_PROOF_MISSING",
    "BLK_PROOF_STALE_OR_INVALID",
    "BLK_MUTATION_IN_FLIGHT_UNSAFE",
    "BLK_COHERENCE_INVALID",
    "BLK_QUEUE_AUTHORITY_AMBIGUOUS",
    "BLK_LEASE_OWNERSHIP_AMBIGUOUS",
    "BLK_CONNECTOR_SNAPSHOT_STALE",
)

TRIGGER_PRECEDENCE: tuple[str, ...] = (
    "TR_UNKNOWN_OR_INCOMPLETE_SIGNAL",
    "TR_SUCCESSOR_VALIDATION_FAIL",
    "TR_RESET_READINESS_FAIL",
    "TR_CRITICAL_THRESHOLD_REACHED",
    "TR_RESET_REQUIRED",
    "TR_FAILOVER_PREP_STARTED",
    "TR_RESET_EXECUTED",
    "TR_BLOCKERS_CLEARED",
    "TR_RESET_READINESS_PASS",
    "TR_SUCCESSOR_VALIDATION_PASS",
    "TR_WARN_THRESHOLD_REACHED",
    "TR_WARN_THRESHOLD_CLEARED_STABLE",
)

_ACTION_POLICY: dict[str, frozenset[str]] = {
    "HEALTHY": frozenset({"A1_NORMAL_MUTATION", "A3_CONTINUITY_REFRESH", "A4_TOKEN_GOVERNED_REMEDIATION"}),
    "WARNING": frozenset({"A1_NORMAL_MUTATION", "A2_THIN_MODE_OUTPUT", "A3_CONTINUITY_REFRESH", "A4_TOKEN_GOVERNED_REMEDIATION"}),
    "PRE_FAILOVER": frozenset({"A2_THIN_MODE_OUTPUT", "A3_CONTINUITY_REFRESH", "A4_TOKEN_GOVERNED_REMEDIATION"}),
    "FAILOVER_PREP": frozenset({"A2_THIN_MODE_OUTPUT", "A3_CONTINUITY_REFRESH", "A4_TOKEN_GOVERNED_REMEDIATION"}),
    "RESET_READY": frozenset({"A2_THIN_MODE_OUTPUT", "A3_CONTINUITY_REFRESH", "A5_RESET_EXECUTION", "A6_SUCCESSOR_INTAKE_AND_QUEUE_RECONCILE", "A7_OWNERSHIP_OR_LEASE_TRANSFER_COMMIT"}),
    "BLOCKED_RESET": frozenset({"A2_THIN_MODE_OUTPUT", "A3_CONTINUITY_REFRESH", "A4_TOKEN_GOVERNED_REMEDIATION", "A6_SUCCESSOR_INTAKE_AND_QUEUE_RECONCILE"}),
    "SUCCESSOR_RESUME_VALIDATION": frozenset({"A2_THIN_MODE_OUTPUT", "A3_CONTINUITY_REFRESH", "A6_SUCCESSOR_INTAKE_AND_QUEUE_RECONCILE", "A7_OWNERSHIP_OR_LEASE_TRANSFER_COMMIT"}),
}


def _now_iso(now: str | None = None) -> str:
    if isinstance(now, str) and now.strip():
        return now.strip()
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _state_index(state: str) -> int:
    try:
        return FAILOVER_STATE_ENUM.index(state)
    except ValueError as exc:
        raise ValueError(f"unknown_failover_state:{state}") from exc


def _normalize_trigger(raw: Any) -> str:
    return str(raw or "").strip()


def normalize_trigger_set(triggers: Sequence[Any] | None) -> list[str]:
    known = set(FAILOVER_TRIGGER_ENUM)
    seen: set[str] = set()
    out: list[str] = []
    unknown_seen = False

    for row in triggers or ():
        trig = _normalize_trigger(row)
        if not trig:
            continue
        if trig in known:
            if trig in seen:
                continue
            seen.add(trig)
            out.append(trig)
            continue
        unknown_seen = True

    if unknown_seen and "TR_UNKNOWN_OR_INCOMPLETE_SIGNAL" not in seen:
        out.append("TR_UNKNOWN_OR_INCOMPLETE_SIGNAL")

    return out


def select_highest_precedence_trigger(triggers: Sequence[Any] | None) -> str | None:
    rows = normalize_trigger_set(triggers)
    if not rows:
        return None

    precedence = {name: idx for idx, name in enumerate(TRIGGER_PRECEDENCE)}

    def _key(trigger: str) -> tuple[int, str]:
        return (precedence.get(trigger, len(TRIGGER_PRECEDENCE)), trigger)

    return sorted(rows, key=_key)[0]


def _unknown_failclosed_transition(state: str) -> str:
    if state == "HEALTHY":
        return "WARNING"
    if state == "WARNING":
        return "PRE_FAILOVER"
    return state


def build_state_snapshot(
    *,
    state: str,
    state_version: int = 1,
    entered_at: str | None = None,
    last_transition_trigger: str | None = None,
    reset_required: bool = False,
    reset_allowed: bool = False,
    active_blockers: Sequence[str] | None = None,
    coherence_tuple_ref: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in FAILOVER_STATE_ENUM:
        raise ValueError(f"unknown_failover_state:{state}")

    blockers = [str(item).strip() for item in (active_blockers or ()) if str(item).strip()]
    blockers = sorted(set(blockers))

    return {
        "object_type": "clawd.failover_fsm.state_snapshot.v1",
        "state": state,
        "state_version": max(1, int(state_version)),
        "entered_at": _now_iso(entered_at),
        "last_transition_trigger": str(last_transition_trigger or "").strip() or None,
        "reset_required": bool(reset_required),
        "reset_allowed": bool(reset_allowed),
        "active_blockers": blockers,
        "coherence_tuple_ref": dict(coherence_tuple_ref or {}),
    }


def reduce_failover_state(
    snapshot: Mapping[str, Any],
    *,
    triggers: Sequence[Any] | None,
    now: str | None = None,
    reset_required: bool | None = None,
    blockers: Sequence[str] | None = None,
) -> dict[str, Any]:
    current_state = str(snapshot.get("state") or "").strip()
    if current_state not in FAILOVER_STATE_ENUM:
        raise ValueError(f"unknown_failover_state:{current_state}")

    state_version = int(snapshot.get("state_version") or 1)
    current_reset_required = bool(snapshot.get("reset_required")) if reset_required is None else bool(reset_required)
    current_blockers = [str(item).strip() for item in (snapshot.get("active_blockers") or []) if str(item).strip()]
    if blockers is not None:
        current_blockers = [str(item).strip() for item in blockers if str(item).strip()]
    current_blockers = sorted(set(current_blockers))

    normalized = normalize_trigger_set(triggers)
    selected_trigger = select_highest_precedence_trigger(normalized)
    suppressed = [row for row in normalized if row != selected_trigger]

    next_state = current_state
    next_reset_required = current_reset_required
    if "TR_RESET_REQUIRED" in normalized:
        next_reset_required = True
    next_blockers = list(current_blockers)

    if selected_trigger == "TR_UNKNOWN_OR_INCOMPLETE_SIGNAL":
        next_state = _unknown_failclosed_transition(current_state)
    elif selected_trigger == "TR_CRITICAL_THRESHOLD_REACHED":
        if current_state in {"HEALTHY", "WARNING"}:
            next_state = "PRE_FAILOVER"
    elif selected_trigger == "TR_RESET_REQUIRED":
        next_reset_required = True
        if current_state in {"HEALTHY", "WARNING", "PRE_FAILOVER"}:
            next_state = "FAILOVER_PREP"
    elif selected_trigger == "TR_FAILOVER_PREP_STARTED":
        if current_state == "PRE_FAILOVER":
            next_state = "FAILOVER_PREP"
    elif selected_trigger == "TR_RESET_READINESS_FAIL":
        next_reset_required = True
        if current_state in {"FAILOVER_PREP", "RESET_READY", "BLOCKED_RESET", "SUCCESSOR_RESUME_VALIDATION"}:
            next_state = "BLOCKED_RESET"
    elif selected_trigger == "TR_RESET_READINESS_PASS":
        if current_state in {"FAILOVER_PREP", "BLOCKED_RESET"} and next_reset_required:
            next_state = "RESET_READY"
    elif selected_trigger == "TR_BLOCKERS_CLEARED":
        if current_state == "BLOCKED_RESET":
            next_state = "RESET_READY"
            next_blockers = []
    elif selected_trigger == "TR_RESET_EXECUTED":
        if current_state == "RESET_READY":
            next_state = "SUCCESSOR_RESUME_VALIDATION"
    elif selected_trigger == "TR_SUCCESSOR_VALIDATION_PASS":
        if current_state == "SUCCESSOR_RESUME_VALIDATION":
            next_state = "HEALTHY"
            next_reset_required = False
            next_blockers = []
    elif selected_trigger == "TR_SUCCESSOR_VALIDATION_FAIL":
        if current_state == "SUCCESSOR_RESUME_VALIDATION":
            next_state = "BLOCKED_RESET"
    elif selected_trigger == "TR_WARN_THRESHOLD_REACHED":
        if current_state == "HEALTHY":
            next_state = "WARNING"
    elif selected_trigger == "TR_WARN_THRESHOLD_CLEARED_STABLE":
        if current_state == "WARNING":
            next_state = "HEALTHY"

    reset_allowed = next_state == "RESET_READY"
    entered_at = str(snapshot.get("entered_at") or "").strip()
    if next_state != current_state:
        entered_at = _now_iso(now)
        state_version += 1

    out = build_state_snapshot(
        state=next_state,
        state_version=state_version,
        entered_at=entered_at,
        last_transition_trigger=selected_trigger,
        reset_required=next_reset_required,
        reset_allowed=reset_allowed,
        active_blockers=next_blockers,
        coherence_tuple_ref=snapshot.get("coherence_tuple_ref") if isinstance(snapshot.get("coherence_tuple_ref"), Mapping) else {},
    )
    out["evaluation"] = {
        "selected_trigger": selected_trigger,
        "suppressed_triggers": suppressed,
        "all_triggers": normalized,
        "changed": next_state != current_state,
    }
    return out


def authorize_action_for_state(state: str, action_class: str) -> dict[str, Any]:
    state_key = str(state or "").strip()
    action_key = str(action_class or "").strip()

    if state_key not in FAILOVER_STATE_ENUM:
        return {
            "allowed": False,
            "reason": f"unknown_failover_state:{state_key or 'missing'}",
            "fail_closed": True,
        }

    allowed_set = _ACTION_POLICY.get(state_key, frozenset())
    allowed = action_key in allowed_set
    return {
        "allowed": bool(allowed),
        "reason": "allowed" if allowed else f"forbidden_in_state:{state_key}",
        "fail_closed": not allowed,
        "state": state_key,
        "action_class": action_key,
    }
