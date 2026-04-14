# Enhanced Operator Surfaces Guide v1

This guide provides comprehensive documentation for enhanced operator surfaces, focusing on making warning surfaces clearer and more boring under normal posture while preserving fail-closed behavior for actual issues.

## Overview

The enhanced operator surfaces aim to address the DSG-02 maturity gap by providing:

1. **Evidence-linked critique packets** that connect warnings to actual system evidence
2. **Clear operator guidance** for each warning type with specific actions
3. **Context about normal vs abnormal** operation to reduce false alarms
4. **Automation hints** about when warnings will resolve automatically

## Warning Surface Architecture

### Three-Tier Warning System

#### Tier 1: Surface Warnings (Info Level)
- Expected during normal operation
- No immediate action required
- Monitor for persistence or combination with other warnings
- Examples: idle execution, temporary timing lags

#### Tier 2: Attention Warnings (Warning Level)  
- Require operator attention but not immediate action
- Usually have clear resolution paths
- May resolve automatically within reasonable timeframes
- Examples: stale health evidence, policy cooldowns

#### Tier 3: Action Warnings (Critical Level)
- Require immediate operator attention
- May indicate system safety issues
- Usually require manual intervention
- Examples: stuck workers, verification failures

### Evidence Linkage

Each warning is linked to specific evidence files:

```json
{
  "evidence_linkage": {
    "source": "state/continuity/latest/verify_last.json",
    "field_path": "$.warning_reasons[0]",
    "value": "execution_supervisor_worker_health_canary_stale"
  }
}
```

This allows operators to:
- Verify warning authenticity
- Check evidence freshness
- Understand warning context
- Track warning history

## Operator Guidance Framework

### Action Priorities

#### Immediate (within minutes)
- Critical system safety issues
- Customer-facing service impacts
- Cascade failure risks

#### Soon (within hours)
- Warning-level issues that may escalate
- Performance degradation trends
- Resource exhaustion indicators

#### When Convenient (within days)
- Info-level warnings that persist
- Optimization opportunities
- Documentation updates needed

#### Monitor Only
- Expected during normal operation
- Auto-resolving timing issues
- Single-instance warnings

### Context Provision

Each warning includes context about:

#### Normal Posture Assessment
- **Expected**: Normal part of system operation
- **Warning**: Requires attention but not abnormal
- **Abnormal**: Unusual but not critical
- **Critical**: Indicates serious system issues

#### Time Sensitivity
- **Immediate**: Action needed within minutes
- **Minutes**: Action needed within 1 hour
- **Hours**: Action needed within 4 hours
- **Days**: Action needed within 24 hours
- **Not Time Sensitive**: No urgency

#### Significance Explanation
- Why the warning matters
- What could happen if ignored
- How it relates to other system components
- Historical patterns and trends

## Automation Integration

### Auto-Resolution Hints

Each warning includes automation hints:

```json
{
  "automation_hints": {
    "auto_resolvable": true,
    "resolution_time_estimate": "minutes",
    "escalation_triggers": [
      "persisting_longer_than_1_hour",
      "combined_with_other_warnings",
      "during_high_activity_period"
    ]
  }
}
```

### Automated Response Boundaries

#### What the System Handles Automatically
- Worker rotation and health recovery
- Queue rebalancing and task redistribution
- Evidence refresh and verification cycles
- Cooldown period management
- Temporary timing issue resolution

#### What Requires Operator Judgment
- Whether to wait for auto-resolution
- When to escalate to higher tiers
- How to interpret combined warnings
- Whether customer impact is acceptable

#### What Should Be Escalated Immediately
- System-wide failures
- Repeated automation failures
- Customer-impacting issues
- Unclear root causes

## Implementation Guidelines

### For Operators

#### Daily Operations
1. Review warning dashboard at shift start
2. Acknowledge warnings you understand and are monitoring
3. Take action on warnings requiring attention
4. Document actions taken and outcomes
5. Escalate unclear or persistent issues

#### Warning Response Process
1. **Identify**: Check warning severity and category
2. **Contextualize**: Consider current operational context
3. **Investigate**: Use provided commands to gather more information
4. **Decide**: Choose appropriate action based on guidance
5. **Act**: Execute chosen action and monitor results
6. **Document**: Record what was done and why

#### Escalation Criteria
- Critical warnings of any duration
- Warning-level warnings persisting >1 hour
- Multiple related warnings appearing simultaneously
- Warnings during customer-impacting incidents
- Any warning you don't understand

### For System Administrators

#### Configuration Management
- Warning thresholds should be documented
- Changes to warning logic require review
- New warning types need guidance documentation
- Threshold changes should be gradual and monitored

#### Monitoring Integration
- Warning surfaces should feed into monitoring systems
- Alert fatigue should be monitored and addressed
- False positive rates should be tracked
- Operator feedback should be collected and acted upon

#### Documentation Maintenance
- This guide should be reviewed quarterly
- New warning types require documentation updates
- Operator feedback should inform improvements
- Historical patterns should inform threshold adjustments

## Quality Metrics

### Warning Surface Quality

#### Clarity Metrics
- Percentage of warnings with clear guidance
- Operator confusion reports (qualitative)
- Time to understand warning meaning
- Documentation completeness scores

#### Usefulness Metrics
- Percentage of warnings requiring operator action
- Resolution success rates following guidance
- False positive rates by warning type
- Time to resolution after guidance followed

#### Efficiency Metrics
- Time spent on warning investigation
- Number of escalations due to unclear warnings
- Operator confidence in warning interpretation
- Training time required for new operators

### Continuous Improvement

#### Feedback Collection
- Regular operator surveys about warning clarity
- Post-incident reviews of warning effectiveness
- Analysis of warning-to-blocker escalation patterns
- Documentation improvement suggestions

#### Iteration Process
1. Collect feedback and metrics
2. Identify improvement opportunities
3. Propose changes to warning surfaces
4. Test changes with operator groups
5. Implement approved improvements
6. Monitor effectiveness of changes

## Integration with Existing Systems

### CLI Integration

Enhanced operator commands should include guidance:

```bash
openclaw continuity status --with-guidance
openclaw workers health --explain-warnings
openclaw queue status --suggest-actions
```

### Dashboard Integration

Warning dashboards should display:
- Warning severity and category
- Context about normal vs abnormal
- Quick action buttons for common responses
- Links to relevant documentation

### Alert Integration

Alerting systems should:
- Include warning guidance in alert text
- Provide quick action links
- Escalate based on guidance priorities
- Track alert response effectiveness

## Future Enhancements

### Dynamic Guidance
- Context-aware guidance based on system state
- Historical pattern recognition
- Predictive warning escalation
- Personalized guidance based on operator experience

### Automation Improvements
- Auto-resolution for more warning types
- Intelligent escalation timing
- Automated documentation updates
- Machine learning for false positive reduction

### Multi-Modal Interfaces
- Voice-activated guidance
- Mobile-optimized interfaces
- Integration with collaboration tools
- Augmented reality for complex procedures

## Conclusion

Enhanced operator surfaces aim to make system operation more predictable and less stressful while maintaining safety and reliability. By providing clear guidance and context, operators can make better decisions faster, reducing both alert fatigue and incident response times.

The key principles are:
- **Clarity**: Every warning should have clear meaning and guidance
- **Context**: Operators should understand what's normal vs abnormal
- **Actionability**: Guidance should lead to specific, executable actions
- **Safety**: Fail-closed behavior must be preserved
- **Evolution**: Systems should improve based on operator feedback

Regular review and updates ensure these surfaces remain effective as systems and operational patterns evolve.