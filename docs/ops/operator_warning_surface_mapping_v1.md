# Operator Warning Surface Mapping v1

This document maps current operator warning surfaces to clear, actionable guidance for operators. The goal is to make operator-facing surfaces more boring and predictable under normal posture while preserving fail-closed behavior for actual issues.

## Warning Taxonomy

### Severity Levels
- **Critical**: Requires immediate operator attention, system may be unsafe
- **Warning**: Requires attention but not immediately dangerous
- **Info**: For awareness only, no action typically needed

### Time Sensitivity
- **Immediate**: Action needed within minutes
- **Soon**: Action needed within hours  
- **When Convenient**: Action needed within days
- **Monitor Only**: No action needed unless conditions change

## Current Warning Surface Mapping

### execution_supervisor_worker_health_canary_stale

**Severity**: Warning  
**Category**: worker_health  
**Normal Posture**: Expected during worker rotation or low activity  
**Time Sensitivity**: Soon (within 1 hour)  

**Description**: Worker health monitoring has detected that health evidence is stale. This is designed to detect stuck workers but can occur during normal operation.

**Context**: The system monitors worker health through periodic canary checks. When evidence becomes stale, it may indicate a worker is stuck or the monitoring system needs refresh.

**Operator Action**:
```bash
# Check if any workers are actually stuck
openclaw workers list --stuck

# Review worker health details
openclaw workers health --detailed

# Check recent worker transitions
openclaw workers history --last 24h
```

**Rationale**: Most stale health canary warnings resolve automatically. Manual intervention is only needed if workers are actually stuck.

**Auto-resolution**: Usually resolves within 15-30 minutes as workers cycle or monitoring refreshes.

**Escalation Triggers**:
- Stuck workers persist >1 hour
- Multiple workers show health issues simultaneously
- Combined with other execution blockers

**When NOT to escalate**: During known maintenance windows or low-activity periods.

---

### execution_supervisor_probe_execution_due_now_idle_no_dispatch_candidate

**Severity**: Info  
**Category**: queue_state  
**Normal Posture**: Expected during quiet periods  
**Time Sensitivity**: Monitor Only  

**Description**: The execution system is ready to process work but no tasks are currently queued for execution.

**Context**: This indicates the system is healthy and ready but has no work to do. It's analogous to a factory with workers ready but no raw materials.

**Operator Action**:
```bash
# Check current queue status
openclaw queue status --ready
openclaw queue status --all

# Review recent task completion
openclaw tasks history --last 10

# Check for blocked tasks
openclaw tasks list --status BLOCKED
```

**Rationale**: Idle state is normal and healthy. It means the system has processed all available work and is waiting for more.

**Auto-resolution**: Resolves automatically when new work is queued or existing work becomes unblocked.

**Escalation Triggers**:
- Combined with queue backup or task failures
- During periods when work should be active
- If persists >2 hours during active periods

**When NOT to escalate**: During off-hours, weekends, or known quiet periods.

---

### ground_truth_capture_drift_cooldown_policy_lag

**Severity**: Warning  
**Category**: policy  
**Normal Posture**: Expected after rapid changes  
**Time Sensitivity**: When Convenient (within 2 hours)  

**Description**: Policy-driven cooldown is preventing rapid ground-truth updates to protect against thrashing.

**Context**: The system implements cooldown periods to prevent oscillation during rapid state changes. This is a safety feature, not a bug.

**Operator Action**:
```bash
# Check cooldown status and remaining time
openclaw continuity cooldown --status

# Review recent state changes
openclaw continuity history --last 1h

# Check for rapid change patterns
openclaw continuity changes --rate --window 1h
```

**Rationale**: Cooldown periods are intentional and protect system stability. They should be allowed to complete naturally.

**Auto-resolution**: Resolves automatically when the cooldown period expires.

**Escalation Triggers**:
- Persists >2 hours (may indicate stuck cooldown)
- Blocks critical operations during emergencies
- Combined with other system instability indicators

**When NOT to escalate**: During normal operation after configuration changes or during planned maintenance.

---

### gtc_gateboard_verify_status_lag

**Severity**: Info  
**Category**: verification  
**Normal Posture**: Expected during verify cycles  
**Time Sensitivity**: Monitor Only  

**Description**: Gateboard publication is lagging behind verify status updates, causing temporary inconsistency.

**Context**: The gateboard publishes periodic snapshots while verify status updates continuously. Temporary lag is normal and expected.

**Operator Action**:
```bash
# Check verify evidence freshness
openclaw continuity verify --freshness

# Review gateboard publication timing
openclaw continuity gateboard --timing

# Check for actual verify issues
openclaw continuity verify --status
```

**Rationale**: This is typically a timing issue that resolves as systems synchronize. The underlying verify status is what matters for system health.

**Auto-resolution**: Resolves when gateboard refreshes its publication (usually within 5-10 minutes).

**Escalation Triggers**:
- Verify evidence is actually stale (>30 minutes)
- Combined with real verification failures
- If lag persists >1 hour

**When NOT to escalate**: During normal operation when verify status is actually current.

---

### connector_freshness_drift_reconcile_only

**Severity**: Info  
**Category**: continuity  
**Normal Posture**: Expected during drift reconciliation  
**Time Sensitivity**: Monitor Only  

**Description**: Connector freshness drift has been detected but is being handled through reconcile-only mechanisms, not as a blocking issue.

**Context**: The system detects when connectors have freshness drift but handles it through reconciliation processes rather than blocking operations. This is part of normal drift management.

**Operator Action**:
```bash
# Check current drift status
openclaw continuity drift --status

# Review reconcile-only processes
openclaw continuity reconcile --pending

# Monitor for resolution
openclaw continuity drift --trend
```

**Rationale**: Reconcile-only drift handling is designed to resolve without operator intervention. The system is managing the drift automatically.

**Auto-resolution**: Resolves automatically through system reconciliation processes (usually within 30-60 minutes).

**Escalation Triggers**:
- Drift persists >2 hours without resolution
- Combined with other continuity issues
- If blocking reasons appear in addition to reconcile-only

**When NOT to escalate**: During normal drift reconciliation cycles or after recent system changes.

---

## General Operator Principles

### When to Act
1. **Multiple warnings**: Single warnings often resolve automatically
2. **Persistent warnings**: Warnings lasting >1 hour may need attention
3. **Combined with blockers**: Warnings plus blockers indicate real issues
4. **During active periods**: Warnings during quiet periods are less concerning

### When to Wait
1. **Single warnings**: Most resolve automatically within 30 minutes
2. **After changes**: Give systems time to stabilize after updates
3. **During low activity**: Idle warnings are often normal
4. **Known maintenance**: Expect warnings during planned work

### When to Escalate
1. **Critical severity**: Always escalate critical warnings
2. **Multiple systems**: Issues affecting multiple subsystems
3. **Customer impact**: Any warning affecting user-facing services
4. **Unclear resolution**: When guidance doesn't resolve the issue

## Monitoring Integration

### Key Metrics to Watch
- Warning duration and frequency
- Warning-to-blocker escalation rates
- Time to resolution after operator action
- False positive rates (warnings that resolve without action)

### Alert Thresholds
- Critical warnings: Immediate alert
- Warning severity: Alert after 30 minutes
- Info severity: Alert after 2 hours only if combined with other issues

### Dashboard Integration
- Separate warning tiers by severity
- Show warning age and trend
- Include quick action buttons for common responses
- Link to relevant documentation and runbooks

## Automation Boundaries

### What the System Handles Automatically
- Worker health recovery and rotation
- Queue rebalancing and task redistribution
- Evidence refresh and verification cycles
- Most cooldown and timing issues

### What Requires Operator Attention
- Stuck workers that don't auto-recover
- Queue blockages that persist
- Verification failures with clear causes
- Policy violations requiring manual intervention

### What Should Be Escalated to Engineering
- System-wide failures affecting multiple components
- Repeated automation failures
- Performance degradation trends
- Unclear root causes requiring deep investigation

## Implementation Notes

This mapping is designed to be:
- **Conservative**: When in doubt, recommend checking rather than ignoring
- **Context-aware**: Considers normal operational patterns
- **Actionable**: Provides specific commands and procedures
- **Safe**: Preserves fail-closed behavior for actual issues
- **Evolving**: Should be updated as systems and patterns change

Review and update this mapping quarterly or when significant system changes occur.