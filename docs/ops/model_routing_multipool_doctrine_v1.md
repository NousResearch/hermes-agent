# Model Routing / Multipool Doctrine v1

Status: historical archive / non-canonical (retired)
Date: 2026-03-21
Scope: exploratory multipool routing notes from evaluation-era context; superseded by canonical B6 policy

> **Non-canonical status:** Do not treat this file as a live routing baseline or runtime authority. Canonical routing policy lives in:
> - `docs/ops/unified_operating_doctrine_v1.md`
> - `docs/ops/model_routing_no_llm_matrix_v1.md`
> - `docs/ops/model_pool_policy_v1.json`
>
> Kimi-era references in this archive are historical only. Kimi remains retired from default routing/default orchestration.

## Purpose

This document records an exploratory multipool routing approach considered during evaluation.

It is not the final mature qualification system.
It is not the live baseline; it is retained only as reference context for how the evaluation-era routing thinking evolved.
- reduce unnecessary premium Codex burn
- preserve trust on high-risk work
- use cheaper helper lanes for low-risk and intermediate work
- keep routing understandable and stable across sessions

---

## Core principle

Use the cheapest sufficient lane.

Routing priority:
1. no-model / deterministic / shell-first
2. cheap helper lane
3. stronger helper lane
4. premium trust lane

Do not use a stronger or more expensive model unless the task actually needs it.

---

## Lane map

### 1. Deterministic / no-model lane
Default first choice whenever possible.

Use for:
- shell commands
- grep/search
- file reads
- exact transforms
- verification scripts
- schema checks
- status/probe commands
- other mechanical or deterministic tasks

Rule:
- if no model is required, do not call a model

---

### 2. Gemini Flash-Lite — cheap helper lane
Role:
- cheapest broad helper/offload lane currently enabled

Use for:
- first-pass summaries
- long-context helper work
- research triage
- document digestion
- low-risk planning drafts
- broad comparisons
- repetitive helper work
- low-stakes support tasks

Do not use for:
- final trusted coding
- control-plane mutation decisions
- risky edits
- continuity/failover trust-critical logic

---

### 3. Gemini Flash — stronger long-context helper lane
Role:
- stronger helper lane when Flash-Lite is not enough

Use for:
- larger-context synthesis
- better-quality summaries
- harder research/helper tasks
- stronger first-pass planning
- heavier document/context digestion

Do not use for:
- final trusted coding
- risky control-plane changes
- high-trust mutation application

---

### 4. DeepSeek Chat — cheap coding-helper / reasoning lane
Role:
- broad cheap engineering-helper and reasoning lane

Use for:
- repo scans
- code explanation
- first-pass fix plans
- helper function drafting
- low-risk engineering support
- broad analytical helper work
- cheaper coding-helper tasks

Do not use for:
- final trusted implementation on critical paths
- control-plane mutation authority work
- risky final edits without review

---

### 5. DeepSeek Reasoner — deeper analytical lane
Role:
- stronger analytical reasoning lane

Use for:
- harder debugging analysis
- tradeoff reasoning
- root-cause analysis
- reasoning-heavy comparisons
- deeper inference-style tasks
- future brain / inference support patterns

Do not use for:
- final trusted critical coding
- unreviewed risky mutations
- canonical high-risk implementation steps

---

### 6. Codex — premium trust / final coding lane
Role:
- premium high-trust implementation and control lane

Use for:
- dangerous edits
- final implementation passes
- control-plane changes
- continuity / failover / succession logic
- high-risk debugging tied to real mutation
- canonical architecture changes
- trusted final coding work

This lane should be protected.
Do not waste it on routine helper work if a cheaper lane is sufficient.

---

## Practical mental model

- no-model = machinery
- Gemini = long-context / docs / research helper
- DeepSeek = cheap reasoning + coding-helper
- Codex = surgeon / trust lane

Short form:
- Codex = surgeon
- Gemini = reader/research helper
- DeepSeek = analyst/assistant engineer

---

## Escalation doctrine

Escalate only when needed.

Default escalation order:
1. deterministic / shell-first
2. Gemini Flash-Lite
3. Gemini Flash or DeepSeek Chat
4. DeepSeek Reasoner for harder analysis
5. Codex for final trusted work

Notes:
- Prefer Gemini for long-context/doc/research helper tasks
- Prefer DeepSeek for cheap analytical/coding-helper tasks
- Escalate to Codex only when trust, risk, or implementation-critical quality demands it

---

## Cost discipline doctrine

### Preserve Codex
Codex is the premium lane.
Reserve it for work where failure or low quality is expensive.

### Use helper lanes aggressively
Use Gemini and DeepSeek for:
- understanding
- summarizing
- scanning
- planning
- drafting
- comparison
- low-risk helper engineering work

### Avoid fake savings
Do not create a fake cheap lane by sending all cheap-model outputs straight into premium revalidation.
Use premium review selectively for high-stakes outcomes.

---

## Current exclusions / not-yet-mature pieces (historical snapshot)

The following items were open during this evaluation-era snapshot:
- formal multi-provider qualification harness depth across future route classes
- mature provider bakeoff governance
- full canonical doctrine set for every future provider/model
- deeper cockpit UX/action integration over the new rollout dashboard surfaces

Wave 5 closeout note (2026-03-21): long-window route-policy soak/lint, ring-soak automation snapshot, and consolidated rollout dashboard surfaces landed in deterministic runtime form.

---

## Operator policy (current canonical authority)

Do **not** use this archive for live routing decisions.

Canonical policy anchor for runtime enforcement:
- `docs/ops/model_pool_policy_v1.json` (schema: `docs/ops/schemas/model_pool_policy.schema.json`)

When uncertain:
- default to deterministic first
- then follow canonical route-class policy (`NO_LLM` / `SPARK` / `HEAVY`) from the active doctrine set
- escalate only through canonical qualification + rollout governance

---

## Current status label

- doctrine: archived (reference-only)
- routing authority: retired
- multipool phase marker: historical alpha snapshot

This file is retained only for historical context and should not be treated as operational baseline.
