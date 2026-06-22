# HQ Trace Code-State-Space Metrics

## Overview
Code-state-space trajectory metrics for the HQ Agent Trace Index refresh. Design doc only; no live dashboard mutation.

## Why now
- Planner source: `cron_331ecf7312ab_20260621_150248`
- upstream finding: SSA trajectories reveal intent-execution gap as the performance determinant; code state space metrics (edit frequency, test activity, phase transitions) expose per-model problem-solving patterns (arXiv:2606.17454)
- current HQ weakness: Agent Trace Index lacks code-state-space trajectory metrics

## Metrics to Add

### edit_frequency
- per-turn count of file edits in tracked worktree
- bounded to `C:/Users/82109/Desktop/AOE` and active worktree paths
- signal: high edit churn without test activity = possible thrashing

### test_activity
- per-turn count of test invocations + pass/fail delta
- source: terminal tool output regex, or explicit test runner hooks if available
- signal: zero test activity across multiple turns = missing eval step

### phase_transition
- labels per turn:
  - `impl` — new code added
  - `refactor` — existing code restructured
  - `debug` — bug-fix cycle
  - `review` — read-only inspection
  - `plan` — no code change
- heuristic: classify from tool-call sequence + file diff; fall back to `UNKNOWN`

## HQ Integration Points
- script: `scripts/hq_agent_trace_index.py`
- output: `reports/hq_agent_trace_index_latest.json` / `.md`
- dashboard: feeds Task Score v1 `trace_observability` component
- mode: read-only aggregation; no live enforcement

## Implementation Notes (future, approval-gated)
- parse existing `state.db` session messages for tool-call patterns
- join with worktree `git diff --stat` snapshots if available
- produce per-session trace quality score + trend
- tolerates missing fields: if edit/test data unavailable, emit `null` rather than failing

## Safety
- no secret access
- no live cron/Gateway mutation
- no paid-provider calls
- docs-only until worktree is clean
