#!/usr/bin/env python3
"""
Enhanced Failover FSM with Blocked-Reset Timeout and Recovery Protocols (A3 Extension)

This module extends the deterministic failover FSM with:
- Timeout boundaries for BLOCKED_RESET state
- Force-reset protocol for unresolvable blockers
- Cascading blocker priority resolution
- State validation before resume under stress
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Mapping, Sequence

# Import base FSM to extend
import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FSM_PATH = REPO_ROOT / "ops" / "openclaw" / "continuity" / "failover_fsm.py"

spec = importlib.util.spec_from_file_location("failover_fsm", FSM_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load base FSM from {FSM_PATH}")
base_fsm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_fsm)

# Constants
BLOCKED_RESET_TIMEOUT_SECONDS = 300  # 5 minutes default timeout
BLOCKED_RESET_MAX_DURATION_SECONDS = 900  # 15 minutes max before critical escalation
FORCE_RESET_BLOCKER_THRESHOLD = 3  # Number of unique blockers before force reset considered

# Enhanced state tracking
BLOCKED_RESET_META_ENUM: tuple[str, ...] = (
    "BLOCKED_RESET_META_NONE",
    "BLOCKED_RESET_META_TIMEOUT_PENDING",
    "BLOCKED_RESET_META_FORCE_RESET_PENDING",
    "BLOCKED_RESET_META_ESCALATED_CRITICAL",
)

# Blocker priority for cascading resolution (lower number = higher priority)
BLOCKER_PRIORITY_MAP: dict[str, int] = {
    # Critical blockers - must resolve first
    "BLK_PROOF_MISSING": 1,
    "BLK_PROOF_SCHEMA_INVALID": 1,
    "BLK_PROOF_EXPIRED": 2,
    "BLK_PROOF_INVALIDATED": 2,
    "BLK_PROOF_REFUSED": 2,
    
    # High priority blockers
    "BLK_PROOF_STALE_OR_INVALID": 3,
    "BLK_MUTATION_IN_FLIGHT_UNSAFE": 3,
    "BLK_PROOF_GENERATION_MISMATCH": 3,
    "BLK_PROOF_READ_POINTER_MISMATCH": 3,
    
    # Medium priority blockers
    "BLK_PROOF_MUTATION_UNSAFE": 4,
    "BLK_PROOF_VERIFY_GATE_NOT_PASS": 4,
    "BLK_PROOF_QUEUE_AUTHORITY_AMBIGUOUS": 4,
    "BLK_PROOF_CONNECTOR_STALE": 4,
    
    # Lower priority blockers
    "BLK_CONTINUITY_STALE": 5,
    "BLK_COHERENCE_INVALID": 5,
    "BLK_QUEUE_AUTHORITY_AMBIGUOUS": 5,
    "BLK_LEASE_OWNERSHIP_AMBIGUOUS": 5,
    "BLK_CONNECTOR_SNAPSHOT_STALE": 5,
}

def _now_iso(now: str | None = None) -> str:
    """ISO timestamp helper"""
    return base_fsm._now_iso(now)

def _sort_blockers_by_priority(blockers: Sequence[str]) -> list[str]:
    """Sort blockers by priority (highest priority first)"""
    def priority_key(blocker: str) -> tuple[int, str]:
        priority = BLOCKER_PRIORITY_MAP.get(blocker, 99)
        return (priority, blocker)
    
    return sorted(set(str(b).strip() for b in blockers if str(b).strip()), key=priority_key)

def get_top_blocker(blockers: Sequence[str]) -> str | None:
    """Get the highest priority blocker"""
    sorted_blockers = _sort_blockers_by_priority(blockers)
    return sorted_blockers[0] if sorted_blockers else None

def calculate_blocked_reset_meta(
    *,
    entered_at: str,
    active_blockers: Sequence[str],
    previous_meta_state: str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """
    Calculate BLOCKED_RESET meta-state and timeout status
    
    Returns meta-state information including:
    - meta_state: current meta-state of blocked-reset
    - seconds_in_state: duration in blocked-reset
    - timeout_seconds_remaining: time until timeout
    - should_force_reset: whether force reset should be triggered
    - top_blocker: highest priority blocker
    - blocker_count: total unique blockers
    """
    now_dt = dt.datetime.now(dt.timezone.utc) if now is None else dt.datetime.fromisoformat(now.replace("Z", "+00:00"))
    entered_dt = dt.datetime.fromisoformat(entered_at.replace("Z", "+00:00"))
    
    duration_seconds = int((now_dt - entered_dt).total_seconds())
    timeout_seconds_remaining = max(0, BLOCKED_RESET_TIMEOUT_SECONDS - duration_seconds)
    
    # Determine if we should escalate to force reset
    blocker_count = len(set(active_blockers))
    unique_blockers = set(active_blockers)
    
    # Force reset conditions:
    # 1. Timeout exceeded AND blockers persist
    # 2. Too many unique blockers (system is too broken)
    # 3. Critical blockers that won't auto-resolve
    should_force_reset = False
    force_reset_reason = None
    
    if duration_seconds > BLOCKED_RESET_TIMEOUT_SECONDS and blocker_count > 0:
        should_force_reset = True
        force_reset_reason = f"timeout_exceeded_{duration_seconds}s"
    elif blocker_count >= FORCE_RESET_BLOCKER_THRESHOLD:
        should_force_reset = True
        force_reset_reason = f"blocker_threshold_exceeded_{blocker_count}"
    elif any(b in {"BLK_PROOF_MISSING", "BLK_PROOF_SCHEMA_INVALID"} for b in unique_blockers):
        should_force_reset = True
        force_reset_reason = "critical_blocker_unresolvable"
    
    # Determine meta-state
    if should_force_reset:
        meta_state = "BLOCKED_RESET_META_FORCE_RESET_PENDING"
    elif duration_seconds > BLOCKED_RESET_MAX_DURATION_SECONDS:
        meta_state = "BLOCKED_RESET_META_ESCALATED_CRITICAL"
    elif duration_seconds > BLOCKED_RESET_TIMEOUT_SECONDS:
        meta_state = "BLOCKED_RESET_META_TIMEOUT_PENDING"
    else:
        meta_state = "BLOCKED_RESET_META_NONE"
    
    # Get top priority blocker
    top_blocker = get_top_blocker(active_blockers)
    
    return {
        "meta_state": meta_state,
        "seconds_in_state": duration_seconds,
        "timeout_seconds_remaining": timeout_seconds_remaining,
        "should_force_reset": should_force_reset,
        "force_reset_reason": force_reset_reason,
        "top_blocker": top_blocker,
        "blocker_count": blocker_count,
        "active_blockers": sorted(set(active_blockers)),
        "sorted_blockers": _sort_blockers_by_priority(active_blockers),
    }

def evaluate_force_reset_safety(
    *,
    meta_state: dict[str, Any],
    system_health_signals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Evaluate if force reset is safe given current system state
    
    Returns safety assessment with required actions
    """
    if not meta_state.get("should_force_reset"):
        return {
            "safe": False,
            "reason": "force_reset_not_required",
            "required_actions": ["continue_normal_recovery"],
        }
    
    signals = system_health_signals or {}
    
    # Check system health for force reset safety
    safety_checks = {
        "mutation_quiescent": not signals.get("unsafe_in_flight_mutation", False),
        "queue_authority_clear": signals.get("queue_authority_status") in {"CLEAR", "PASS", "OK"},
        "data_integrity_verified": signals.get("verify_gate_status") in {"PASS", "OK"},
        "connector_fresh": signals.get("connector_freshness_status") in {"FRESH", "PASS", "OK"},
    }
    
    # Force reset is considered safe if:
    # - We meet timeout/blocker thresholds
    # - System is quiescent (no unsafe mutations)
    # - Data integrity is verifiable
    # Note: Some authority ambiguities are acceptable during force reset
    
    safe = (
        safety_checks["mutation_quiescent"] and
        safety_checks["data_integrity_verified"]
    )
    
    required_actions = []
    if safe:
        required_actions.append("execute_force_reset")
        if not safety_checks["queue_authority_clear"]:
            required_actions.append("reconcile_queue_authority_post_reset")
        if not safety_checks["connector_fresh"]:
            required_actions.append("refresh_connectors_post_reset")
    else:
        required_actions.append("escalate_to_manual_intervention")
        if not safety_checks["mutation_quiescent"]:
            required_actions.append("drain_or_abort_unsafe_mutations")
        if not safety_checks["data_integrity_verified"]:
            required_actions.append("restore_from_last_known_good")
    
    return {
        "safe": safe,
        "reason": meta_state["force_reset_reason"],
        "safety_checks": safety_checks,
        "required_actions": required_actions,
    }

def build_enhanced_state_snapshot(
    *,
    state: str,
    state_version: int = 1,
    entered_at: str | None = None,
    last_transition_trigger: str | None = None,
    reset_required: bool = False,
    reset_allowed: bool = False,
    active_blockers: Sequence[str] | None = None,
    coherence_tuple_ref: Mapping[str, Any] | None = None,
    # Enhanced fields
    blocked_reset_meta: Mapping[str, Any] | None = None,
    force_reset_approved: bool = False,
    stress_test_mode: bool = False,
) -> dict[str, Any]:
    """
    Build enhanced state snapshot with blocked-reset metadata
    """
    base_snapshot = base_fsm.build_state_snapshot(
        state=state,
        state_version=state_version,
        entered_at=entered_at,
        last_transition_trigger=last_transition_trigger,
        reset_required=reset_required,
        reset_allowed=reset_allowed,
        active_blockers=active_blockers,
        coherence_tuple_ref=coherence_tuple_ref,
    )
    
    # Add enhanced metadata
    base_snapshot["enhanced"] = {
        "blocked_reset_detection": {
            "meta_state": (blocked_reset_meta or {}).get("meta_state", "BLOCKED_RESET_META_NONE"),
            "seconds_in_state": (blocked_reset_meta or {}).get("seconds_in_state", 0),
            "timeout_seconds_remaining": (blocked_reset_meta or {}).get("timeout_seconds_remaining", BLOCKED_RESET_TIMEOUT_SECONDS),
            "should_force_reset": bool((blocked_reset_meta or {}).get("should_force_reset", False)),
            "top_blocker": (blocked_reset_meta or {}).get("top_blocker"),
            "blocker_count": (blocked_reset_meta or {}).get("blocker_count", 0),
            "sorted_blockers": (blocked_reset_meta or {}).get("sorted_blockers", []),
        },
        "force_reset_approved": force_reset_approved,
        "stress_test_mode": stress_test_mode,
        "enhanced_version": "v1",
    }
    
    return base_snapshot

def reduce_enhanced_failover_state(
    snapshot: Mapping[str, Any],
    *,
    triggers: Sequence[Any] | None,
    now: str | None = None,
    reset_required: bool | None = None,
    blockers: Sequence[str] | None = None,
    system_health_signals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Enhanced state reducer with blocked-reset timeout and force-reset logic
    """
    # Get enhanced metadata if present
    enhanced = snapshot.get("enhanced", {}) if isinstance(snapshot, Mapping) else {}
    
    # Calculate blocked-reset meta if we're in that state
    current_state = str(snapshot.get("state") or "").strip()
    entered_at = str(snapshot.get("entered_at") or "").strip() or _now_iso(now)
    active_blockers = [str(item).strip() for item in (snapshot.get("active_blockers") or []) if str(item).strip()]
    
    if blockers is not None:
        active_blockers = [str(item).strip() for item in blockers if str(item).strip()]
    active_blockers = sorted(set(active_blockers))
    
    blocked_reset_meta = None
    force_reset_trigger = None
    
    if current_state == "BLOCKED_RESET":
        blocked_reset_meta = calculate_blocked_reset_meta(
            entered_at=entered_at,
            active_blockers=active_blockers,
            now=now,
        )
        
        # Evaluate force reset safety
        if blocked_reset_meta["should_force_reset"]:
            force_eval = evaluate_force_reset_safety(
                meta_state=blocked_reset_meta,
                system_health_signals=system_health_signals,
            )
            
            if force_eval["safe"]:
                # Add force reset trigger
                force_reset_trigger = "TR_FORCE_RESET_REQUIRED"
                triggers = list(triggers or [])
                triggers.append(force_reset_trigger)
    
    # Use base FSM reducer
    reduced = base_fsm.reduce_failover_state(
        snapshot,
        triggers=triggers,
        now=now,
        reset_required=reset_required,
        blockers=active_blockers,
    )
    
    # Add enhanced metadata to result
    if blocked_reset_meta:
        reduced["enhanced"] = {
            "blocked_reset_detection": blocked_reset_meta,
            "force_reset_evaluation": force_reset_trigger is not None,
            "force_reset_trigger": force_reset_trigger,
        }
    
    return reduced