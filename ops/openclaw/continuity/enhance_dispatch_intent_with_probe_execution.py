#!/usr/bin/env python3
"""Enhance dispatch intent with probe execution detection.

This script post-processes the execution_supervisor_dispatch_intent to add
probe execution detection for quota-recovered workers.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import sys
sys.path.insert(0, '/home/yeqiuqiu/clawd-architect')

from ops.openclaw.continuity.execution_supervisor_probe_executor import (
    should_trigger_probe_execution,
    build_probe_execution_contract,
    PROBE_EXECUTION_STALE_TICKS,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def enhance_dispatch_intent(
    dispatch_intent: Dict[str, Any],
    probe_plan: Dict[str, Any],
    canary_schedule: Dict[str, Any],
) -> Dict[str, Any]:
    """Enhance dispatch intent with probe execution detection."""
    
    enhanced = dispatch_intent.copy()
    
    # Check if probe execution should be triggered
    should_trigger, target_worker, reason = should_trigger_probe_execution(probe_plan, canary_schedule)
    
    if not should_trigger or not target_worker:
        # Add detection metadata but no action
        enhanced["probe_execution_detection"] = {
            "schema": "clawd.execution_supervisor_probe_execution_detection.v1",
            "detected_at": now_iso(),
            "should_trigger": False,
            "reason": reason,
            "target_worker": None,
            "probe_execution_contract": None,
        }
        return enhanced
    
    # Build probe execution contract
    contract = build_probe_execution_contract(target_worker, probe_plan, canary_schedule)
    
    # Create enhanced dispatch intent with probe execution
    enhanced["probe_execution_detection"] = {
        "schema": "clawd.execution_supervisor_probe_execution_detection.v1",
        "detected_at": now_iso(),
        "should_trigger": True,
        "reason": reason,
        "target_worker": target_worker,
        "probe_execution_contract": contract,
    }
    
    # Add probe execution decision (extending beyond schema for now)
    # In a future slice, we'd update the schema to include EXECUTE_PROBE
    enhanced["_probe_execution_enhancement"] = {
        "original_decision": enhanced.get("decision"),
        "original_decision_reasons": enhanced.get("decision_reasons", []),
        "recommended_decision": "EXECUTE_PROBE",
        "recommended_decision_reasons": [reason],
        "target_worker": target_worker,
        "stale_ticks": probe_plan.get("due_now_cohort_signature_consecutive_ticks", 0),
    }
    
    return enhanced


def main() -> int:
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <dispatch-intent-path> <probe-plan-path> <canary-schedule-path> <output-path>")
        return 1
    
    dispatch_intent_path = Path(sys.argv[1])
    probe_plan_path = Path(sys.argv[2])
    canary_schedule_path = Path(sys.argv[3])
    output_path = Path(sys.argv[4])
    
    try:
        dispatch_intent = load_json(dispatch_intent_path)
        probe_plan = load_json(probe_plan_path)
        canary_schedule = load_json(canary_schedule_path)
        
        enhanced = enhance_dispatch_intent(dispatch_intent, probe_plan, canary_schedule)
        
        save_json(output_path, enhanced)
        
        # Print summary
        detection = enhanced.get("probe_execution_detection", {})
        if detection.get("should_trigger"):
            print(f"Probe execution detected for {detection['target_worker']}: {detection['reason']}")
            print(f"Stale ticks: {enhanced.get('_probe_execution_enhancement', {}).get('stale_ticks', 0)}")
            print(f"Expected artifact: {detection.get('probe_execution_contract', {}).get('expected_artifact')}")
        else:
            print(f"No probe execution needed: {detection.get('reason', 'unknown')}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())