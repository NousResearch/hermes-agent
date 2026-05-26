# Self-Improvement Proposal

proposal_id: SIP-2026-001
created: 2026-05-26
proposer: Cursor/Hermes preflight

## Observed Problem

observed_problem: Herald Digest outage / missing heartbeat / stale fleet health.

## Evidence

evidence:
  - F-2026-005 matched in `docs/FAILURE_PATTERN_LIBRARY.md`.
  - `herald` heartbeat is missing in the cloud registry-side checkout.
  - `agent_health_summary` status is `attention`.
  - `stale_heartbeats`: 25.
  - `knowledge_graph` artifacts are absent in this checkout.
  - Runtime identity source is unresolved because `/workspace/agents` is registry/index only.

## Related Failure Patterns

related_failure_patterns:
  - F-2026-005

## Affected Surfaces

affected_agents:
  - herald
affected_skills:
  - skills/openclaw/herald
affected_tools:
  - fleet_context_snapshot
  - agent_health_summary
  - agents_get
  - knowledge_query

## Source of Truth Checked

source_of_truth_checked:
  - docs/TOWN_OPENCLAW_AGENT_FLOW_CONTRACT.md
  - docs/FAILURE_PATTERN_LIBRARY.md
  - skills/openclaw/herald/SKILL.md
  - MCP preflight context

## Proposed Change

proposed_change:
  - Create a read-only diagnostic plan to locate the real local `HERMES_AGENTS_DIR`.
  - Verify whether `herald` has SOUL/HEARTBEAT/runtime state in that directory.
  - Verify whether the Herald Digest outage is runtime identity, gateway/mail bridge, cron, or production pipeline related.
  - Do not mutate runtime state during diagnosis.

## Files To Touch

files_to_touch:
  - docs/proposals/SIP-2026-001-herald-recovery-stale-heartbeat.md

## Change Classification

behavior_change: no
runtime_state_change: no
risk_class: GATED
approval_required: yes

## Validation Plan

validation_plan:
  - Proposal passes governance review.
  - No files outside `docs/proposals/` touched.
  - `git diff` confirms proposal-only change.

## Rollback Plan

rollback_plan:
  - Delete proposal document before commit, or revert proposal commit.

## Non-Goals

non_goals:
  - No Herald repair yet.
  - No heartbeat writes.
  - No SOUL/HISTORY creation.
  - No gateway changes.
  - No cron changes.
  - No production signal changes.
  - No identity synthesis.

## Operator Decision

operator_decision: PENDING

## Rules

- This proposal is not implementation.
- No automatic application.
- No identity synthesis.
- No runtime state mutation.
- No skill promotion without explicit approval.
- `recurrence_count >= 3` can justify proposing prevention-rule promotion, not applying it.
