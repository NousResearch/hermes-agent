# Operator Evidence Bundle Mapping v1

**Date**: 2026-04-03  
**Purpose**: Map current operator warning surfaces to comprehensive UI evidence bundles that provide context-rich evidence packages instead of isolated warning signals.

## Overview

This document maps the current warning-level residues in operator-facing surfaces to comprehensive evidence bundles that include screenshots, DOM state snapshots, structured findings, and contextual guidance. The goal is to transform isolated warning signals into auditable evidence packages that help operators understand context and significance under normal READY posture.

## Evidence Bundle Mapping

### 1. execution_supervisor_worker_health_canary_stale

**Current Warning**: `execution_supervisor_worker_health_canary_stale`

**Evidence Bundle Components**:

#### Screenshot Evidence
```json
{
  "screenshot": {
    "url": "state/evidence/screenshots/worker_health_dashboard_20260403T041200Z.png",
    "capture_time": "2026-04-03T04:12:00Z",
    "dimensions": {"width": 1920, "height": 1080},
    "validation_hash": "sha256:abc123def456...",
    "capture_method": "browser_screenshot"
  }
}
```

#### DOM State Evidence
```json
{
  "dom_state": {
    "url": "state/evidence/dom_snapshots/worker_health_dom_20260403T041200Z.json",
    "capture_time": "2026-04-03T04:12:00Z",
    "elements_captured": 47,
    "critical_elements": [
      {
        "selector": "#worker-health-summary .status-indicator",
        "text_content": "Stale (19.4h ago)",
        "attributes": {"class": "status-warning", "data-age-sec": "69864"}
      },
      {
        "selector": "#last-health-update",
        "text_content": "Last updated: 2026-04-02T08:54:37Z",
        "attributes": {"timestamp": "2026-04-02T08:54:37Z"}
      }
    ]
  }
}
```

#### Structured Findings
```json
{
  "findings": [
    {
      "severity": "warning",
      "category": "worker_health",
      "description": "Worker health canary last updated 19.4 hours ago",
      "rationale": "Health canary should refresh within 15 minutes under normal operation. Extended staleness may indicate worker monitoring issues or system overload.",
      "auto_resolvable": true,
      "metrics": {
        "age_hours": 19.4,
        "expected_max_age_minutes": 15,
        "health_score": "unknown"
      }
    }
  ]
}
```

#### Context and Guidance
```json
{
  "context": {
    "normal_range": "Health canary should update every 5-15 minutes during normal operation",
    "current_state": "Canary has not updated for 19.4 hours, exceeding normal range",
    "recommended_action": "Refresh worker health canary via: bash ops/openclaw/continuity.sh worker-health-canary --json",
    "when_to_escalate": "If refresh fails or canary remains stale after 30 minutes",
    "auto_resolution_expected": true,
    "estimated_resolution_time": "5-10 minutes"
  }
}
```

### 2. execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate

**Current Warning**: `execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate`

**Evidence Bundle Components**:

#### Screenshot Evidence
```json
{
  "screenshot": {
    "url": "state/evidence/screenshots/execution_queue_dashboard_20260403T041300Z.png",
    "capture_time": "2026-04-03T04:13:00Z",
    "dimensions": {"width": 1920, "height": 1080},
    "validation_hash": "sha256:def456ghi789...",
    "capture_method": "browser_screenshot"
  }
}
```

#### DOM State Evidence
```json
{
  "dom_state": {
    "url": "state/evidence/dom_snapshots/execution_queue_dom_20260403T041300Z.json",
    "capture_time": "2026-04-03T04:13:00Z",
    "elements_captured": 23,
    "critical_elements": [
      {
        "selector": "#queue-status .ready-count",
        "text_content": "Ready: 0",
        "attributes": {"class": "queue-empty"}
      },
      {
        "selector": "#probe-execution-status",
        "text_content": "Due now: idle (no candidates)",
        "attributes": {"status": "idle", "reason": "no_dispatch_candidate"}
      }
    ]
  }
}
```

#### Structured Findings
```json
{
  "findings": [
    {
      "severity": "info",
      "category": "queue_state",
      "description": "Execution queue is empty with no dispatch candidates",
      "rationale": "System is ready to execute work but no tasks are queued. This is normal during quiet periods or when all work is complete.",
      "auto_resolvable": true,
      "metrics": {
        "ready_count": 0,
        "running_count": 0,
        "idle_duration_hours": 19.4
      }
    }
  ]
}
```

#### Context and Guidance
```json
{
  "context": {
    "normal_range": "Queue can be empty during quiet periods or when all work is complete",
    "current_state": "Queue is empty with 0 ready tasks and no dispatch candidates",
    "recommended_action": "Check queue status via: openclaw queue status --ready",
    "when_to_escalate": "If combined with other execution blockers or if new work is not being processed",
    "auto_resolution_expected": true,
    "estimated_resolution_time": "When new work arrives"
  }
}
```

### 3. ground_truth_capture_drift_cooldown_policy_lag

**Current Warning**: `ground_truth_capture_drift_cooldown_policy_lag`

**Evidence Bundle Components**:

#### Screenshot Evidence
```json
{
  "screenshot": {
    "url": "state/evidence/screenshots/ground_truth_dashboard_20260403T041400Z.png",
    "capture_time": "2026-04-03T04:14:00Z",
    "dimensions": {"width": 1920, "height": 1080},
    "validation_hash": "sha256:ghi789jkl012...",
    "capture_method": "browser_screenshot"
  }
}
```

#### DOM State Evidence
```json
{
  "dom_state": {
    "url": "state/evidence/dom_snapshots/ground_truth_dom_20260403T041400Z.json",
    "capture_time": "2026-04-03T04:14:00Z",
    "elements_captured": 31,
    "critical_elements": [
      {
        "selector": "#drift-cooldown-timer",
        "text_content": "Cooldown active: 29 minutes remaining",
        "attributes": {"status": "active", "remaining_minutes": "29"}
      },
      {
        "selector": "#last-drift-detection",
        "text_content": "Last drift: 2026-04-03T03:45:00Z",
        "attributes": {"timestamp": "2026-04-03T03:45:00Z"}
      }
    ]
  }
}
```

#### Structured Findings
```json
{
  "findings": [
    {
      "severity": "info",
      "category": "drift_detection",
      "description": "Ground truth capture is in cooldown period for 29 more minutes",
      "rationale": "Policy-driven cooldown prevents rapid ground-truth updates to protect against thrashing during rapid changes. This is normal protective behavior.",
      "auto_resolvable": true,
      "metrics": {
        "cooldown_remaining_minutes": 29,
        "total_cooldown_minutes": 30,
        "last_drift_age_minutes": 29
      }
    }
  ]
}
```

#### Context and Guidance
```json
{
  "context": {
    "normal_range": "Cooldown periods of 30-60 minutes are normal after drift detection",
    "current_state": "29 minutes remaining in 30-minute cooldown period",
    "recommended_action": "Monitor for cooldown completion. No action needed unless critical operations are blocked.",
    "when_to_escalate": "If cooldown persists >2 hours or blocks critical operations",
    "auto_resolution_expected": true,
    "estimated_resolution_time": "29 minutes (when cooldown expires)"
  }
}
```

### 4. connector_freshness_drift_reconcile_only

**Current Warning**: `connector_freshness_drift_reconcile_only`

**Evidence Bundle Components**:

#### Screenshot Evidence
```json
{
  "screenshot": {
    "url": "state/evidence/screenshots/connector_status_dashboard_20260403T041500Z.png",
    "capture_time": "2026-04-03T04:15:00Z",
    "dimensions": {"width": 1920, "height": 1080},
    "validation_hash": "sha256:jkl012mno345...",
    "capture_method": "browser_screenshot"
  }
}
```

#### DOM State Evidence
```json
{
  "dom_state": {
    "url": "state/evidence/dom_snapshots/connector_status_dom_20260403T041500Z.json",
    "capture_time": "2026-04-03T04:15:00Z",
    "elements_captured": 38,
    "critical_elements": [
      {
        "selector": "#connector-freshness-summary .drift-status",
        "text_content": "Drift detected (reconcile-only)",
        "attributes": {"status": "drift", "type": "reconcile_only"}
      },
      {
        "selector": "#reconcile-progress",
        "text_content": "Progress: 15% complete",
        "attributes": {"progress": "15", "eta": "45 minutes"}
      }
    ]
  }
}
```

#### Structured Findings
```json
{
  "findings": [
    {
      "severity": "warning",
      "category": "connector_freshness",
      "description": "Connector freshness drift detected, reconcile in progress (15% complete)",
      "rationale": "Some connectors have exceeded their freshness TTL and are being reconciled. This is normal maintenance behavior that prevents stale connector data from affecting system decisions.",
      "auto_resolvable": true,
      "metrics": {
        "reconcile_progress_percent": 15,
        "estimated_completion_minutes": 45,
        "affected_connectors": 2,
        "critical_connectors": 0
      }
    }
  ]
}
```

#### Context and Guidance
```json
{
  "context": {
    "normal_range": "Reconcile operations are normal when connectors exceed freshness TTL",
    "current_state": "Connector freshness drift detected, reconcile 15% complete with 45 minutes ETA",
    "recommended_action": "Monitor reconcile progress. No operator action needed unless critical connectors are affected.",
    "when_to_escalate": "If reconcile fails or critical connectors remain stale >1 hour",
    "auto_resolution_expected": true,
    "estimated_resolution_time": "45 minutes (when reconcile completes)"
  }
}
```

## Evidence Bundle Generation Process

### 1. Trigger Detection
- System detects warning condition in operator surfaces
- Evidence bundle generation triggered automatically
- Bundle ID generated with timestamp and hash

### 2. Evidence Capture
- Screenshot captured of relevant dashboard/UI
- DOM state snapshot captured for critical elements
- Validation hashes computed for integrity

### 3. Finding Analysis
- System analyzes captured evidence for patterns
- Findings generated with severity and rationale
- Metrics extracted from evidence

### 4. Context Assembly
- Normal operation range determined
- Current state assessed
- Recommended actions formulated
- Escalation criteria defined

### 5. Audit Trail Creation
- Provenance information recorded
- Validation chain established
- Integrity checks performed
- Bundle completeness verified

## Integration with Operator Tools

### Command Line Interface
```bash
# Generate evidence bundle for specific warning
openclaw evidence-bundle generate \
  --warning-reason "execution_supervisor_worker_health_canary_stale" \
  --output-format json

# View evidence bundle
openclaw evidence-bundle view \
  --bundle-id "bundle_20260403T041200Z_abc123"

# List recent evidence bundles
openclaw evidence-bundle list \
  --warning-reason "execution_supervisor_worker_health_canary_stale" \
  --limit 10
```

### Web Interface Integration
- Evidence bundles displayed alongside warnings in operator dashboard
- Screenshots shown with hover/click for full view
- DOM state available for inspection
- Findings presented in structured format

### API Integration
```json
{
  "endpoint": "/api/v1/operator/evidence-bundles",
  "method": "GET",
  "parameters": {
    "warning_reason": "execution_supervisor_worker_health_canary_stale",
    "limit": 10,
    "include_screenshots": true
  },
  "response": {
    "bundles": [
      {
        "bundle_id": "bundle_20260403T041200Z_abc123",
        "generated_at": "2026-04-03T04:12:00Z",
        "evidence": { /* evidence bundle data */ }
      }
    ]
  }
}
```

## Quality Assurance

### Evidence Validation
- All screenshots validated with SHA256 hashes
- DOM state snapshots validated for completeness
- Findings cross-referenced with actual system state
- Context information verified against operational norms

### Audit Compliance
- Complete provenance trail maintained
- Integrity checks performed on all evidence
- Validation chain established for each bundle
- Retention policies enforced per evidence type

### Operator Feedback Loop
- Operator engagement tracked for evidence bundles
- Effectiveness measured through resolution times
- Bundle quality assessed through operator surveys
- Continuous improvement based on feedback

## Remaining Limitations

1. **Static Evidence**: Bundles are snapshots, not live streams
2. **Storage Requirements**: Screenshots and DOM states require storage
3. **Capture Timing**: Evidence may not capture transient conditions
4. **Manual Integration**: Requires integration into existing operator tools

## Future Enhancements

1. **Live Evidence Streams**: Real-time evidence updates for critical warnings
2. **Comparative Analysis**: Before/after evidence bundles for changes
3. **Trend Analysis**: Historical evidence bundle analysis
4. **Automated Insights**: ML-generated insights from evidence patterns