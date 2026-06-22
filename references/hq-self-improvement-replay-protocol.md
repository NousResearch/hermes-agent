# HQ Self-Improvement Replay Protocol

## Overview
SEAGym-style replay protocol for the HQ Planner-Build-Review cron loop. Read-only policy doc; no live enforcement.

## Why now
- Planner source: `cron_331ecf7312ab_20260621_150248`
- upstream finding: self-evolving agent evaluation requires train/val/test/replay/cost views; frequent harness updates corrupt held-out performance, mid-training snapshots can later collapse (SEAGym, arXiv:2606.17546)
- current HQ: no frozen validation, no replay logging, no cost tracking across loop ticks

## Protocol

### Snapshot Taxonomy
For each Planner/Build/Review tick output:
- `train` — non-silent planner/build/review outputs that change system state
- `val` — harness eval reports (`hq_harness_eval_latest.json`) and dashboard summaries
- `test` — frozen external benchmarks or pinned eval fixtures
- `replay` — historical tick outputs replayed for regression detection
- `cost` — tool-call count + provider/model usage per tick

### Frozen Validation Rule
- Build Packet may only edit skills/references if the frozen val snapshot is `PASS`
- if `REVIEW` or `BLOCKED`, Build emits docs-only output and escalates to Review
- never overwrite frozen val set; append-only log

### Replay Cadence
- weekly: replay last 7 days of Planner/Build/Review ticks
- detect: phase shifts, approval-gate churn, repeated silent ticks with no state change
- output: `reports/hq_self_improvement_replay_latest.md`

### Cost Ledger
- per tick:
  - terminal calls count
  - web search calls count
  - file read/write bytes
  - model invocations (if visible from cron output)
- no live enforcement; read-only report only

## Safety
- no secret access
- no live cron/Gateway mutation
- no paid-provider calls
- docs-only until worktree is clean and approval-gated for script changes
