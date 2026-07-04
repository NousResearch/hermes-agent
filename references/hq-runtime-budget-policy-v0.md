---
name: hq-runtime-budget-policy-v0
description: "Use when reviewing or designing HQ runtime budget and context lifecycle governance. Captures the verified v0 policy contract for token/tool/turn/time budgeting, prefix-cache/tool-trace grounding, read-only dashboard expansion, and approval-gated live enforcement."
version: 0.1.0
author: HQ Build Packet
license: MIT
metadata:
  hermes:
    tags: [hq, budget, context, governance, dashboard, safety]
    related_skills: [cronjob-registry]
---

# HQ Runtime Budget Policy v0

## Overview

This document records the verified HQ policy contract for agent runtime budget and context lifecycle governance. It is a docs-only policy reference. It does not enable live enforcement, cron mutation, provider changes, or secret access.

Verified basis:
- OpenAI / Anthropic official context engineering guidance (provider_doc, high confidence).
- HQ Build findings from `2026-06-17` Planner tick.
- Existing scripts: `hq_long_context_pack_refresh.py`, `hq_context_hygiene.py`.

## Why now

- Context compression and tool-trace efficiency were already validated as high-impact rollout candidates by prior Planner/Build ticks.
- Experiments show that without token + prefix-cache + tool-trace budget alignment, long sessions degrade even if final answers look correct.
- Current HQ scripts measure hygiene and pack efficiency, but enforcement remains prompt-only.

## Governing principles

1. Budget conversations must include **token count**, **prefix-cache integrity**, and **tool-trace grounding** together.
2. Intermediate tool-call states must be measured, not discarded, before compaction decisions.
3. Expand only the **read-only dashboard + validator schema** first; live runtime enforcement remains approval-gated.
4. Keep source trust layered: official provider docs > internal verified reports > arXiv follow-up (pending rate-limit recovery).

## Primary components

### HQBudgetGate
- Split budgets: turns, tool calls, tokens, wall-clock time.
- Label high-level actions as `ASK`, `REFUSE`, or `RECOVER`.

### ContextLifecycleController
- Replace prompt-only load/keep/evict rules with a policy-backed lifecycle.
- Emit audit events on eviction; keep recoverable rollback links to prior context state.

### RuntimeUsageRecorder
- Record per-turn / per-tool usage and prefix-cache effectiveness.
- Feed trace metrics and dashboard summary from reliable on-disk telemetry.

## Safety boundary

- Allowed without extra approval: read-only dashboard expansion, validator schema updates, documentation/reference edits.
- Approval-gated: live enforcement rollout, main/master direct edits, unbounded paid calls, indefinite daemons, secret/ELIOS access.

## Read-only dashboard block

`hq_health_dashboard.py` should expose a compact context budget/fragmentation block summarizing:
- budget estimate
- current token pressure
- prefix-cache hit ratio signal
- tool-trace grounding coverage

No live enforcement logic should be added there.

## Validator / eval integration

- `hq_harness_eval_latest.json` should include a `budget_policy_gate` decision with states: `OK`, `REVIEW`, `BLOCKED`, `RECOVER`.
- `hq_agent_trace_index.py` should add runtime budget indicators: turns, tools, wall-clock, token estimate, prefix-cache effectiveness.

## Script telemetry fields (proposed)

- `hq_context_hygiene.py`: `needs_budget_review`, `budget_estimate`
- `hq_long_context_pack_refresh.py`: per-topic `budget_estimate`

These fields should be additive and non-destructive.

## Deferred

- Learner-based automatic budget adjustment: defer until policy gate evidence is stable.
- arXiv TokenPilot / ContextRL / MemRefine reproduction: defer until rate-limit recovery; on recovery, update citations/applicability notes here.

## Sources / evidence

- OpenAI / Anthropic context engineering docs: high-confidence provider guidance.
- HQ internal roadmap and 2026-06-17 Planner findings.
- Prior HQ harness/adapter artifacts demonstrating existing worktree-only delivery path.
