#!/usr/bin/env python3
"""Probe executor for quota-recovered workers.

This module executes canary probes for workers that have recovered from
quota exhaustion (runtime_usage_limit_recovered) but remain in probationary
state due to missing canary evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

SCHEMA = "clawd.execution_supervisor_probe_executor.v1"
PROBE_EXECUTION_STALE_TICKS = 3  # Minimum ticks before triggering probe
PROBE_EXECUTION_MODE_SYNTHETIC = "synthetic_evidence_only"
PROBE_EXECUTION_MODE_REAL = "real_worker_runtime"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_probe_artifact_for_restore(
    *,
    artifact_path: Path,
    worker: str,
    evidence_kind: str,
    execution_mode: str,
    worker_execution_performed: bool,
) -> Dict[str, Any]:
    """Fail-closed artifact validator for restoration eligibility.

    Real restoration eligibility requires all of:
    - real worker runtime execution mode
    - worker execution performed flag
    - required evidence markers in artifact content
    """

    reasons: list[str] = []
    marker_failures: list[str] = []
    content = ""
    artifact_present = False
    artifact_size_bytes: Optional[int] = None

    try:
        if artifact_path.exists() and artifact_path.is_file():
            artifact_size_bytes = int(artifact_path.stat().st_size)
            artifact_present = bool(artifact_size_bytes > 0)
            if artifact_present:
                content = artifact_path.read_text(encoding="utf-8")
    except Exception:
        artifact_present = False
        artifact_size_bytes = None

    if not artifact_present:
        reasons.append("artifact_missing_or_empty")

    mode_token = str(execution_mode or "").strip().lower()
    if mode_token != PROBE_EXECUTION_MODE_REAL:
        reasons.append("execution_mode_not_real_worker_runtime")

    if worker_execution_performed is not True:
        reasons.append("worker_execution_not_performed")

    required_markers = [
        "**Status**: PASS",
        f"**Worker**: {worker}",
        f"**Evidence Kind**: {evidence_kind}",
        "- **Worker Runtime Executed**: yes",
    ]
    if artifact_present:
        for marker in required_markers:
            if marker not in content:
                marker_failures.append(marker)
                reasons.append(f"required_marker_missing:{marker}")

    status = "pass" if not reasons else "fail"
    restoration_eligible = status == "pass"

    return {
        "schema": "clawd.execution_supervisor_probe_artifact_validation.v1",
        "status": status,
        "restoration_eligible": restoration_eligible,
        "artifact_path": str(artifact_path),
        "artifact_present": artifact_present,
        "artifact_size_bytes": artifact_size_bytes,
        "execution_mode": execution_mode,
        "worker_execution_performed": bool(worker_execution_performed),
        "worker": worker,
        "evidence_kind": evidence_kind,
        "required_markers": required_markers,
        "missing_markers": marker_failures,
        "reasons": reasons,
        "validated_at": now_iso(),
    }


def should_trigger_probe_execution(
    probe_plan: Dict[str, Any],
    canary_schedule: Dict[str, Any],
) -> tuple[bool, Optional[str], Optional[str]]:
    """Check if probe execution should be triggered.
    
    Returns: (should_trigger, target_worker, reason)
    """
    if not isinstance(probe_plan, dict) or not isinstance(canary_schedule, dict):
        return False, None, "invalid_input"
    
    # Check probe execution plan status
    plan_status = probe_plan.get("status")
    if plan_status != "attention_required":
        return False, None, f"plan_status_not_attention_required:{plan_status}"
    
    # Check for due_now workers
    due_now_workers = probe_plan.get("due_now_cohort_workers") or []
    if not due_now_workers:
        return False, None, "no_due_now_workers"
    
    # Check consecutive ticks
    consecutive_ticks = probe_plan.get("due_now_cohort_signature_consecutive_ticks", 0)
    if consecutive_ticks < PROBE_EXECUTION_STALE_TICKS:
        return False, None, f"ticks_below_threshold:{consecutive_ticks}<{PROBE_EXECUTION_STALE_TICKS}"
    
    # Get the oldest due_now worker
    oldest_worker = probe_plan.get("oldest_due_now_worker")
    if not oldest_worker or oldest_worker not in due_now_workers:
        oldest_worker = due_now_workers[0]
    
    # Check worker details in canary schedule
    schedule_workers = canary_schedule.get("workers") or []
    worker_info = None
    for w in schedule_workers:
        if isinstance(w, dict) and w.get("worker") == oldest_worker:
            worker_info = w
            break
    
    if not worker_info:
        return False, None, f"worker_not_found_in_schedule:{oldest_worker}"
    
    # Check if worker has quota recovery reason
    reason = worker_info.get("reason", "")
    if "runtime_usage_limit_recovered" not in str(reason):
        return False, None, f"not_quota_recovery_reason:{reason}"
    
    # Check worker state
    state = worker_info.get("state", "")
    if state not in {"canary_required", "probe_required"}:
        return False, None, f"invalid_state:{state}"
    
    return True, oldest_worker, "probe_cohort_stale_due_now"


def build_probe_execution_contract(
    worker: str,
    probe_plan: Dict[str, Any],
    canary_schedule: Dict[str, Any],
) -> Dict[str, Any]:
    """Build bounded execution contract for probe."""
    
    # Find worker details
    schedule_workers = canary_schedule.get("workers") or []
    worker_info = None
    for w in schedule_workers:
        if isinstance(w, dict) and w.get("worker") == worker:
            worker_info = w
            break
    
    if not worker_info:
        return {}
    
    # Determine evidence kind
    state = worker_info.get("state", "")
    evidence_kind = "canary" if state == "canary_required" else "probe"
    
    # Get expected artifact hint from probe plan
    expected_artifact = None
    plan_workers = probe_plan.get("workers") or []
    for w in plan_workers:
        if isinstance(w, dict) and w.get("worker") == worker:
            expected_artifact = w.get("expected_artifact_hint")
            break
    
    if not expected_artifact:
        # Generate default artifact path
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        expected_artifact = f"reports/{worker}_restore_{evidence_kind}_{date_str}.md"
    
    return {
        "schema": "clawd.execution_supervisor_probe_execution_contract.v1",
        "worker": worker,
        "evidence_kind": evidence_kind,
        "expected_artifact": expected_artifact,
        "timeout_sec": 300,  # 5 minutes
        "max_retries": 1,
        "created_at": now_iso(),
    }


def execute_canary_probe(worker: str, contract: Dict[str, Any]) -> Dict[str, Any]:
    """Write truthful synthetic probe evidence without claiming worker execution.

    This is an intermediate fail-closed path: the executor records bounded
    operator evidence but does not spawn or exercise a worker runtime.
    """

    artifact_path = Path(
        contract.get(
            "expected_artifact",
            f"reports/{worker}_restore_{contract.get('evidence_kind', 'canary')}_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md",
        )
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    started = monotonic()
    timeout_sec = contract.get("timeout_sec", 60)
    max_retries = contract.get("max_retries", 1)
    execution_mode = PROBE_EXECUTION_MODE_SYNTHETIC
    probe_status = "not_executed"
    probe_reason = "synthetic_probe_evidence_only_no_worker_execution"
    evidence_kind = str(contract.get("evidence_kind") or "canary").strip().lower() or "canary"

    execution_log = [
        f"[{now_iso()}] Probe execution requested for worker: {worker}",
        (
            f"[{now_iso()}] Entered synthetic evidence mode; "
            "no subagent or worker runtime execution was attempted"
        ),
        f"[{now_iso()}] Fail-closed posture maintained; restoration eligibility remains false",
    ]

    test_result = {
        "mode": execution_mode,
        "worker_execution_attempted": False,
        "worker_execution_performed": False,
        "restoration_eligible": False,
        "synthetic_evidence_written": True,
        "timeout_sec": timeout_sec,
        "max_retries": max_retries,
    }

    total_time = monotonic() - started
    execution_log.append(
        f"[{now_iso()}] Synthetic probe evidence written in {total_time:.4f}s"
    )

    content = f"""# Canary Probe Result for {worker}

**Status**: NOT_EXECUTED
**Execution Mode**: {execution_mode}
**Execution Time**: {total_time:.4f} seconds
**Worker**: {worker}
**Evidence Kind**: {evidence_kind}
**Generated At**: {now_iso()}
**Contract Created**: {contract.get('created_at', 'unknown')}

## Execution Summary
- **Result**: {probe_status} ({probe_reason})
- **Timeout Budget**: {timeout_sec} seconds
- **Retries Configured**: {max_retries}
- **Worker Runtime Executed**: no

## Important Truthfulness Note
No real worker execution occurred in this run.
This artifact is synthetic evidence only and must not be treated as canary pass evidence.

## Execution Log
```
{"\n".join(execution_log)}
```

## Test Results
```json
{json.dumps(test_result, indent=2)}
```

## Next Steps
- Keep worker in probationary state.
- Collect real canary/probe evidence via a runtime that can actually execute the target worker.

## Fail-Closed Guarantee
This probe execution preserves fail-closed behavior:
- No worker was restored by this command.
- Synthetic artifacts never count as real restore evidence.
- Operator intervention remains required.
"""

    artifact_path.write_text(content, encoding="utf-8")

    artifact_validation = validate_probe_artifact_for_restore(
        artifact_path=artifact_path,
        worker=worker,
        evidence_kind=evidence_kind,
        execution_mode=execution_mode,
        worker_execution_performed=False,
    )

    restoration_eligible = bool(artifact_validation.get("restoration_eligible") is True)
    test_result["restoration_eligible"] = restoration_eligible

    return {
        "schema": SCHEMA,
        "worker": worker,
        "status": probe_status,
        "reason": probe_reason,
        "execution_mode": execution_mode,
        "worker_execution_performed": False,
        "restoration_eligible": restoration_eligible,
        "artifact_path": str(artifact_path),
        "artifact_present": bool(artifact_validation.get("artifact_present") is True),
        "artifact_validation": artifact_validation,
        "execution_time_sec": total_time,
        "execution_log": execution_log,
        "test_result": test_result,
        "error_details": None,
        "fail_closed_preserved": True,
        "executed_at": now_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute canary probes for quota-recovered workers")
    parser.add_argument("--probe-plan", required=True, help="Path to probe execution plan JSON")
    parser.add_argument("--canary-schedule", required=True, help="Path to canary probe schedule JSON")
    parser.add_argument("--output", required=True, help="Output path for probe execution result")
    parser.add_argument("--execute", action="store_true", help="Actually execute probe (otherwise dry-run)")
    
    args = parser.parse_args()
    
    try:
        probe_plan = load_json(Path(args.probe_plan))
        canary_schedule = load_json(Path(args.canary_schedule))
        
        should_trigger, target_worker, reason = should_trigger_probe_execution(probe_plan, canary_schedule)
        
        result = {
            "schema": SCHEMA,
            "generated_at": now_iso(),
            "should_trigger": should_trigger,
            "target_worker": target_worker,
            "reason": reason,
            "probe_execution_contract": None,
            "execution_result": None,
        }
        
        if should_trigger and target_worker:
            contract = build_probe_execution_contract(target_worker, probe_plan, canary_schedule)
            result["probe_execution_contract"] = contract
            
            if args.execute:
                execution_result = execute_canary_probe(target_worker, contract)
                result["execution_result"] = execution_result
        
        save_json(Path(args.output), result)
        
        if should_trigger:
            print(f"Probe execution triggered for {target_worker}: {reason}")
            return 0
        else:
            print(f"No probe execution needed: {reason}")
            return 0
            
    except Exception as e:
        error_result = {
            "schema": SCHEMA,
            "generated_at": now_iso(),
            "should_trigger": False,
            "target_worker": None,
            "reason": f"error:{str(e)}",
            "probe_execution_contract": None,
            "execution_result": None,
        }
        save_json(Path(args.output), error_result)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
