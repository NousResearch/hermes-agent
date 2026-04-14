# Enhanced Operator Evidence Surfaces Guide v1

**Date**: 2026-04-03  
**Purpose**: Comprehensive guide for operators on how to interpret and act on enhanced evidence surfaces that provide context-rich evidence packages instead of isolated warning signals.

## Overview

This guide explains how to work with the new enhanced operator evidence surfaces that transform isolated warning signals into comprehensive evidence bundles. These bundles include screenshots, DOM state snapshots, structured findings, and contextual guidance to help operators make informed decisions under normal READY posture.

## What Are Evidence Bundles?

Evidence bundles are comprehensive packages that provide:

- **Visual Evidence**: Screenshots of relevant dashboards and UI states
- **DOM State**: Structured snapshots of critical UI elements
- **Structured Findings**: Analysis of what the evidence shows
- **Contextual Guidance**: Clear explanations of significance and recommended actions
- **Audit Trail**: Complete provenance and integrity information

## Evidence Bundle Structure

### Core Components

```json
{
  "bundle_id": "bundle_20260403T041200Z_abc123",
  "generated_at": "2026-04-03T04:12:00Z",
  "warning_reason": "execution_supervisor_worker_health_canary_stale",
  "severity": "warning",
  "evidence": {
    "screenshot": { /* visual evidence */ },
    "dom_state": { /* structured UI state */ },
    "findings": [ /* analysis results */ ],
    "context": { /* guidance and recommendations */ }
  },
  "audit_trail": { /* provenance and integrity */ }
}
```

### Component Details

#### 1. Screenshot Evidence
- **Purpose**: Visual confirmation of the UI state that triggered the warning
- **Content**: Dashboard screenshots, status indicators, key metrics
- **Validation**: SHA256 hash ensures integrity
- **Timing**: Capture timestamp shows when evidence was collected

#### 2. DOM State Evidence
- **Purpose**: Structured data from critical UI elements
- **Content**: Element selectors, text content, attributes, metrics
- **Scope**: Only elements relevant to the warning condition
- **Format**: JSON snapshot for programmatic analysis

#### 3. Structured Findings
- **Purpose**: Analysis of what the evidence indicates
- **Content**: Severity, category, description, rationale, metrics
- **Classification**: Info, Warning, Critical based on impact
- **Rationale**: Explanation of why this finding matters

#### 4. Context and Guidance
- **Purpose**: Clear guidance on significance and actions
- **Content**: Normal ranges, current state, recommendations
- **Escalation**: When to escalate to higher severity
- **Resolution**: Expected resolution time and method

## Working with Evidence Bundles

### Viewing Evidence Bundles

#### Command Line
```bash
# View specific evidence bundle
openclaw evidence-bundle view --bundle-id bundle_20260403T041200Z_abc123

# List evidence bundles for a warning
openclaw evidence-bundle list --warning-reason "execution_supervisor_worker_health_canary_stale"

# Generate new evidence bundle
openclaw evidence-bundle generate --warning-reason "execution_supervisor_worker_health_canary_stale"
```

#### Web Interface
- Navigate to Operator Dashboard → Evidence Bundles
- Click on bundle ID to view complete evidence package
- Use screenshot viewer to examine visual evidence
- Expand DOM state section to see structured data

#### API Access
```bash
# Get evidence bundle via API
curl -X GET "https://openclaw.local/api/v1/operator/evidence-bundles/bundle_20260403T041200Z_abc123" \
  -H "Authorization: Bearer $TOKEN"
```

### Interpreting Evidence

#### Step 1: Review Severity and Category
- **Info**: Normal operational information, no action required
- **Warning**: Attention needed, may require monitoring or action
- **Critical**: Immediate action required, system functionality affected

#### Step 2: Examine Visual Evidence
- Look at screenshot to understand visual context
- Check for obvious anomalies or error indicators
- Compare with normal operation expectations

#### Step 3: Analyze Structured Findings
- Read description and rationale carefully
- Review metrics for quantitative assessment
- Note auto-resolution capability

#### Step 4: Consider Context and Guidance
- Check if current state is within normal range
- Review recommended actions
- Understand escalation criteria
- Note expected resolution timeline

### Taking Action

#### When No Action is Needed
- Info severity bundles
- Auto-resolvable warnings with short resolution time
- Conditions within acceptable operational range

#### When Monitoring is Appropriate
- Warning severity with auto-resolution expected
- Conditions approaching but not exceeding thresholds
- Normal operational variations

#### When Immediate Action is Required
- Critical severity bundles
- Warnings with no auto-resolution expected
- Conditions exceeding escalation criteria

## Evidence Bundle Categories

### Worker Health Evidence Bundles

**Applies to**: `execution_supervisor_worker_health_canary_stale`

**Typical Evidence**:
- Worker health dashboard screenshot
- Health metric displays
- Worker status indicators
- Historical health trends

**Interpretation Guide**:
- **Normal**: Health updates within 15 minutes
- **Warning**: Health updates 15-60 minutes old
- **Critical**: Health updates >1 hour old or health score critical

**Actions**:
- Refresh health canary: `bash ops/openclaw/continuity.sh worker-health-canary --json`
- Check worker status: `openclaw workers list --health`
- Escalate if refresh fails or health remains poor

### Queue State Evidence Bundles

**Applies to**: `execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate`

**Typical Evidence**:
- Queue status dashboard
- Execution plan display
- Worker availability indicators
- Queue depth metrics

**Interpretation Guide**:
- **Normal**: Empty queue during quiet periods
- **Warning**: Queue empty but work expected
- **Critical**: Queue backed up with no processing

**Actions**:
- Check queue status: `openclaw queue status --ready`
- Verify worker availability: `openclaw workers list --available`
- Escalate if work is not being processed

### Drift Detection Evidence Bundles

**Applies to**: `ground_truth_capture_drift_cooldown_policy_lag`

**Typical Evidence**:
- Ground truth dashboard
- Drift detection indicators
- Cooldown timer displays
- Policy status information

**Interpretation Guide**:
- **Normal**: Cooldown periods 30-60 minutes after drift
- **Warning**: Cooldown approaching 2 hours
- **Critical**: Cooldown >2 hours blocking critical operations

**Actions**:
- Monitor cooldown completion
- Check for blocking operations
- Escalate if critical operations are blocked

### Connector Freshness Evidence Bundles

**Applies to**: `connector_freshness_drift_reconcile_only`

**Typical Evidence**:
- Connector status dashboard
- Freshness timeline displays
- Reconcile progress indicators
- Critical connector classifications

**Interpretation Guide**:
- **Normal**: Reconcile operations in progress
- **Warning**: Reconcile taking >1 hour
- **Critical**: Critical connectors affected >1 hour

**Actions**:
- Monitor reconcile progress
- Check affected connectors
- Escalate if critical connectors remain stale

## Quality Assurance

### Evidence Validation
Every evidence bundle includes:
- **Integrity Checks**: SHA256 validation of screenshots and DOM states
- **Completeness Verification**: All required components present
- **Provenance Trail**: Complete audit trail of evidence collection
- **Validation Chain**: Step-by-step validation of evidence

### Operator Feedback
Provide feedback on evidence bundle effectiveness:
- Use the rating system in the web interface
- Comment on clarity and usefulness
- Suggest improvements via operator feedback channels
- Report any issues with evidence quality

## Best Practices

### Regular Review
- Review evidence bundles during daily operations
- Compare current bundles with historical patterns
- Note changes in evidence patterns over time
- Document lessons learned from evidence interpretation

### Evidence Management
- Don't rely solely on screenshots - read the structured findings
- Use DOM state data for programmatic analysis
- Keep evidence bundles for audit trail compliance
- Follow retention policies for evidence storage

### Escalation Procedures
- Follow escalation criteria in evidence bundle context
- Document actions taken based on evidence
- Communicate evidence-based decisions to stakeholders
- Update procedures based on evidence bundle effectiveness

## Troubleshooting

### Missing Evidence
If evidence bundle components are missing:
- Check if capture failed due to system issues
- Verify evidence generation permissions
- Review system logs for capture errors
- Regenerate bundle if necessary

### Validation Failures
If integrity checks fail:
- Do not trust the evidence bundle
- Regenerate bundle from current system state
- Report validation failures to system administrators
- Check for evidence tampering or corruption

### Interpretation Challenges
If evidence is unclear:
- Compare with historical evidence bundles
- Consult with experienced operators
- Request additional evidence capture
- Use multiple evidence sources for confirmation

## Integration with Operations

### Daily Operations
- Include evidence bundle review in daily checklists
- Use evidence bundles for shift handoffs
- Reference evidence bundles in incident reports
- Track evidence-based decision outcomes

### Incident Response
- Generate evidence bundles during incidents
- Use evidence bundles for post-incident analysis
- Include evidence bundles in incident reports
- Update procedures based on evidence insights

### Continuous Improvement
- Analyze evidence bundle effectiveness
- Identify patterns in evidence interpretation
- Improve evidence capture based on operator needs
- Update training materials with evidence examples

## Conclusion

Enhanced operator evidence surfaces transform raw warning signals into comprehensive evidence packages that support informed decision-making. By providing visual context, structured analysis, and clear guidance, these evidence bundles make operator-facing surfaces clearer, lower-noise, and more stable under normal READY posture.

Operators should embrace evidence-based operations while maintaining awareness of limitations and following best practices for evidence interpretation and management.