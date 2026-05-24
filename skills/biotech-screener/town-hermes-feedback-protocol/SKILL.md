---
name: town-hermes-feedback-protocol
description: Use when reconciling Town findings with Hermes operations via bidirectional routing, weekly sync, and conflict escalation.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [biotech, governance, feedback-loop, town, hermes, operations]
    related_skills: [biotech-screener-ops-ledger, security-alert-response-playbook, memory-steward]
---

# Town-Hermes Feedback Protocol

## Overview
This skill defines a bidirectional feedback protocol between Town and Hermes for the biotech screener operations layer.

Goal:
1) Town findings are routable to Hermes operators/agents.
2) Hermes operational dark matter is routable to Town memory/governance.
3) Dual storage (Town memories and Hermes .learnings/) is reconciled on schedule.

## When to Use
- Town audit findings conflict with Hermes operational policy.
- Hermes detects persistent outages/drift not reflected in Town memories.
- Weekly/monthly governance reconciliation is due.
- You need a decision-rights path for conflict resolution.

## Message Types
1. Operational alerts (Hermes -> Town)
   - Trigger: SLA breach or critical anomaly.
   - Route: email now; GitHub issue for structured tracking.
2. Findings/audit results (Town -> Hermes)
   - Trigger: doc audit, governance finding, memory contradiction.
   - Route: GitHub issue/comment + linked memory reference.
3. Dark matter escalations (Town -> Hermes)
   - Trigger: unresolved systemic pattern.
   - Route: issue + escalation owner assignment.

## Channel Strategy
- Phase 1: email + GitHub issues (current recommended baseline).
- Phase 2: GitHub API automation + conflict linter.
- Phase 3: MCP endpoint-based bidirectional queries.

## Escalation and Decision Rights
If Town finding conflicts with Hermes decision trail:
1) Log conflict with evidence handles.
2) Escalate to ops_supervisor.
3) Decision: amend trigger policy OR reaffirm final decision.
4) Record decision memo in audit trail.
5) Sync both systems (Town + Hermes).

## Sync Cadence
- Weekly: reconcile Town memories vs Hermes .learnings/ deltas.
- Monthly: validate storage allocations and contradiction closure.
- Quarterly: audit protocol effectiveness and update ownership matrix.

## Integration Points
- health-baselines-sla: thresholds create Type-1 alerts.
- failure-patterns-ledger: recurrence >=3 prompts systemic escalation.
- document-lineage-authority-map: fact-authority drift triggers findings.
- decision-audit-trail: conflict adjudication source of truth.

## Operator Governance Defaults (must apply in this environment)
- Diagnose-only first: separate FACTS from INFERENCE in every memo.
- One remediation at a time: draft exact command(s) before execution.
- No scope creep: if the operator marks a path as "over-hot", stop expansion.
- Freeze windows: keep outputs local-only drafts unless explicit clearance is given.
- Dev/prod separation is mandatory: never treat research hypotheses as production-ready changes.

## Practical Minimum Runbook
1) Detect issue and classify message type.
2) Open/append GitHub issue with source links.
3) Assign owner + due date.
4) Update Town memory and Hermes .learnings/ with same canonical handle.
5) Verify closure in both systems.
6) If remediation is approved, execute only one change and re-verify before proposing next action.
7) If not approved, keep recommendations as local-only draft actions.

## Known Limitations
- Email-only routing is slow and semi-manual.
- No native conflict auto-detection without linter.
- Requires explicit ownership discipline for reconciliation.

## Verification Checklist
- [ ] Message routed on correct channel and severity.
- [ ] Conflict owner assigned.
- [ ] Decision memo linked when policy-level contradiction exists.
- [ ] Town memory and Hermes .learnings/ both updated.
- [ ] Recurrence counters/health baselines updated if applicable.
