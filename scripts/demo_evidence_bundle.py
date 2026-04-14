#!/usr/bin/env python3
"""
Demo script for UI Evidence Bundle Contract functionality.

Shows how to generate and validate evidence bundles for operator warning surfaces.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

def generate_evidence_bundle(warning_reason, severity="warning"):
    """Generate a sample evidence bundle for a given warning reason."""
    
    # Generate bundle ID with timestamp
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    bundle_id = f"bundle_{timestamp}_demo123"
    
    # Create mock screenshot evidence
    screenshot_content = f"Mock screenshot for {warning_reason}"
    screenshot_hash = f"sha256:{hashlib.sha256(screenshot_content.encode()).hexdigest()}"
    
    # Create mock DOM state evidence
    dom_content = {
        "elements": [
            {
                "selector": "#warning-indicator",
                "text_content": f"Warning: {warning_reason}",
                "attributes": {"class": "warning", "data-reason": warning_reason}
            }
        ],
        "capture_info": {
            "url": f"/dashboard/{warning_reason.replace('_', '-')}",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
    dom_hash = f"sha256:{hashlib.sha256(json.dumps(dom_content).encode()).hexdigest()}"
    
    # Create findings based on warning reason
    findings = []
    context = {}
    
    if warning_reason == "execution_supervisor_worker_health_canary_stale":
        findings = [{
            "severity": "warning",
            "category": "worker_health",
            "description": "Worker health canary has not updated recently",
            "rationale": "Health canary should refresh within 15 minutes under normal operation",
            "auto_resolvable": True,
            "metrics": {"age_hours": 19.4, "expected_max_minutes": 15}
        }]
        context = {
            "normal_range": "Health canary should update every 5-15 minutes during normal operation",
            "current_state": "Canary has not updated for 19.4 hours, exceeding normal range",
            "recommended_action": "Refresh worker health canary via: bash ops/openclaw/continuity.sh worker-health-canary --json",
            "when_to_escalate": "If refresh fails or canary remains stale after 30 minutes",
            "auto_resolution_expected": True,
            "estimated_resolution_time": "5-10 minutes"
        }
    elif warning_reason == "execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate":
        findings = [{
            "severity": "info",
            "category": "queue_state",
            "description": "Execution queue is empty with no dispatch candidates",
            "rationale": "System is ready but no work is queued, which is normal during quiet periods",
            "auto_resolvable": True,
            "metrics": {"ready_count": 0, "running_count": 0}
        }]
        context = {
            "normal_range": "Queue can be empty during quiet periods or when all work is complete",
            "current_state": "Queue is empty with 0 ready tasks and no dispatch candidates",
            "recommended_action": "Check queue status via: openclaw queue status --ready",
            "when_to_escalate": "If combined with other execution blockers",
            "auto_resolution_expected": True,
            "estimated_resolution_time": "When new work arrives"
        }
    elif warning_reason == "ground_truth_capture_drift_cooldown_policy_lag":
        findings = [{
            "severity": "info",
            "category": "drift_detection",
            "description": "Ground truth capture is in cooldown period",
            "rationale": "Policy-driven cooldown prevents rapid ground-truth updates to protect against thrashing",
            "auto_resolvable": True,
            "metrics": {"cooldown_remaining_minutes": 29, "total_cooldown_minutes": 30}
        }]
        context = {
            "normal_range": "Cooldown periods of 30-60 minutes are normal after drift detection",
            "current_state": "29 minutes remaining in 30-minute cooldown period",
            "recommended_action": "Monitor for cooldown completion. No action needed unless critical operations are blocked.",
            "when_to_escalate": "If cooldown persists >2 hours or blocks critical operations",
            "auto_resolution_expected": True,
            "estimated_resolution_time": "29 minutes (when cooldown expires)"
        }
    elif warning_reason == "connector_freshness_drift_reconcile_only":
        findings = [{
            "severity": "warning",
            "category": "connector_freshness",
            "description": "Connector freshness drift detected, reconcile in progress",
            "rationale": "Some connectors have exceeded freshness TTL and are being reconciled",
            "auto_resolvable": True,
            "metrics": {"reconcile_progress_percent": 15, "estimated_completion_minutes": 45}
        }]
        context = {
            "normal_range": "Reconcile operations are normal when connectors exceed freshness TTL",
            "current_state": "Connector freshness drift detected, reconcile 15% complete with 45 minutes ETA",
            "recommended_action": "Monitor reconcile progress. No operator action needed unless critical connectors are affected.",
            "when_to_escalate": "If reconcile fails or critical connectors remain stale >1 hour",
            "auto_resolution_expected": True,
            "estimated_resolution_time": "45 minutes (when reconcile completes)"
        }
    
    # Create the complete evidence bundle
    bundle = {
        "schema_version": "ui_evidence_bundle_contract.v1",
        "bundle_id": bundle_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "warning_reason": warning_reason,
        "severity": severity,
        "evidence": {
            "screenshot": {
                "url": f"state/evidence/screenshots/{warning_reason}_{timestamp}.png",
                "capture_time": datetime.utcnow().isoformat() + "Z",
                "dimensions": {"width": 1920, "height": 1080},
                "validation_hash": screenshot_hash,
                "file_size_bytes": len(screenshot_content),
                "capture_method": "browser_screenshot"
            },
            "dom_state": {
                "url": f"state/evidence/dom_snapshots/{warning_reason}_{timestamp}.json",
                "capture_time": datetime.utcnow().isoformat() + "Z",
                "elements_captured": len(dom_content["elements"]),
                "critical_elements": dom_content["elements"],
                "validation_hash": dom_hash
            },
            "findings": findings,
            "context": context
        },
        "audit_trail": {
            "provenance": {
                "source_system": "openclaw_operator_surfaces",
                "capture_method": "demo_evidence_generation",
                "operator_context": "demo_demonstration",
                "capture_trigger": "demo_request"
            },
            "validation_chain": [
                {
                    "validation_step": "evidence_generation",
                    "result": "pass",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "details": "Demo evidence bundle generated successfully"
                }
            ],
            "integrity_checks": {
                "hash_validation": {
                    "status": "valid",
                    "screenshot_hash": screenshot_hash,
                    "dom_hash": dom_hash
                },
                "completeness_check": {
                    "status": "complete",
                    "missing_components": []
                }
            }
        },
        "metadata": {
            "operator_facing": True,
            "retention_policy": "short_term",
            "access_level": "internal"
        }
    }
    
    return bundle

def validate_evidence_bundle(bundle):
    """Validate an evidence bundle against basic requirements."""
    errors = []
    
    # Check required fields
    required_fields = ["schema_version", "bundle_id", "generated_at", "warning_reason", "evidence"]
    for field in required_fields:
        if field not in bundle:
            errors.append(f"Missing required field: {field}")
    
    # Check evidence components
    evidence = bundle.get("evidence", {})
    required_evidence = ["screenshot", "dom_state", "findings", "context"]
    for component in required_evidence:
        if component not in evidence:
            errors.append(f"Missing evidence component: {component}")
    
    # Check audit trail
    audit_trail = bundle.get("audit_trail", {})
    if "integrity_checks" not in audit_trail:
        errors.append("Missing audit trail integrity checks")
    
    # Validate hash integrity
    screenshot = evidence.get("screenshot", {})
    if "validation_hash" in screenshot and not screenshot["validation_hash"].startswith("sha256:"):
        errors.append("Screenshot hash must use SHA256 format")
    
    dom_state = evidence.get("dom_state", {})
    if "validation_hash" in dom_state and not dom_state["validation_hash"].startswith("sha256:"):
        errors.append("DOM state hash must use SHA256 format")
    
    return len(errors) == 0, errors

def main():
    """Demo evidence bundle generation and validation."""
    print("=== UI Evidence Bundle Contract Demo ===\n")
    
    # Generate evidence bundles for current warning reasons
    warning_reasons = [
        "execution_supervisor_worker_health_canary_stale",
        "execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate",
        "ground_truth_capture_drift_cooldown_policy_lag",
        "connector_freshness_drift_reconcile_only"
    ]
    
    for warning_reason in warning_reasons:
        print(f"Generating evidence bundle for: {warning_reason}")
        
        # Generate bundle
        bundle = generate_evidence_bundle(warning_reason)
        
        # Validate bundle
        is_valid, errors = validate_evidence_bundle(bundle)
        
        if is_valid:
            print(f"✅ Evidence bundle generated and validated successfully")
            print(f"   Bundle ID: {bundle['bundle_id']}")
            print(f"   Severity: {bundle['severity']}")
            print(f"   Findings: {len(bundle['evidence']['findings'])}")
            print(f"   Recommended action: {bundle['evidence']['context']['recommended_action'][:80]}...")
        else:
            print(f"❌ Evidence bundle validation failed:")
            for error in errors:
                print(f"   - {error}")
        
        print()
    
    print("=== Demo Complete ===")
    print("Evidence bundles provide operators with:")
    print("• Visual screenshots of relevant dashboards")
    print("• Structured DOM state snapshots")
    print("• Analyzed findings with severity and rationale")
    print("• Clear context and recommended actions")
    print("• Complete audit trail with integrity validation")

if __name__ == "__main__":
    main()