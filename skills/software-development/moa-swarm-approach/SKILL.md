---
name: moa-swarm-approach
description: Plan and verify work with MoA-guided agent swarms.
version: 1.0.0
author: misery-hl, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [swarm, moa, mixture-of-agents, delegation, orchestration]
    category: software-development
    related_skills: [plan, requesting-code-review]
---

# MoA Swarm Approach Skill

Use Mixture of Agents (MoA) to design and review evidence-gated work slices, then use scoped workers to execute only the slices that benefit from parallelism. This skill does not treat model agreement as proof or give workers authority for unapproved side effects.

## When to Use

Use this skill when:

- a broad `/goal` needs decomposition before implementation;
- several independent perspectives could improve a plan or review;
- work has separable slices with explicit inputs and outputs;
- risk, ambiguity, or breadth makes one serial pass unreliable; or
- the user asks for an MoA swarm, mixture of agents, or multi-agent plan.

Do not use it for one deterministic action, a task with no testable success condition, or a request that prioritizes minimum cost and latency. Do not fan out secrets, credentials, raw personal data, or sensitive logs; redact the context or keep the work in the parent session.

## Prerequisites

- Configure at least one MoA preset and its reference/aggregator providers. `/moa <prompt>` runs one prompt through the default preset and then restores the active model.
- Configure `delegation.provider` and `delegation.model` for the native `delegate_task` workers.
- Confirm the parent can inspect the relevant source with Hermes tools such as `read_file`, `search_files`, and `terminal`.
- Identify any action that needs user approval before planning execution.

No optional skill, plugin, or extra toolset is required; these tools ship in the standard Hermes build.

If MoA is unavailable, state `MoA unavailable; using bounded fallback planning.` Use one parent planning pass or a small read-only `delegate_task` panel, and never claim that MoA ran.

## How to Run

1. Define the goal, done condition, non-goals, evidence, and approval boundaries.
2. Give `/moa` a compact, redacted context pack and request an ordered slice plan.
3. Check the plan against source evidence; obtain approval only where cost, risk, or side effects require it.
4. Execute the next ready slice with the smallest useful `delegate_task` fan-out.
5. Require compact worker reports, inspect their evidence, and run objective checks.
6. Use another MoA pass for disputed or high-risk results, then advance, repeat, block, or ask the user.

Run one slice at a time unless slices are independent, write-isolated, and safe to verify separately. The parent remains the overseer and owns every final success claim.

## Quick Reference

| Stage | Mechanism | Required output |
|---|---|---|
| Shape | Parent | Goal, done condition, boundaries |
| Plan | `/moa` | Ordered slices and worker width |
| Execute | `delegate_task` | Scoped evidence or changes |
| Compress | Worker report | Verdict and evidence handles |
| Review | `/moa` when warranted | PASS, BLOCK, or REQUEST_CHANGES |
| Verify | Parent tools | Directly checked result |

Choose worker width by need, not a fixed swarm size:

| Need | Suggested width |
|---|---|
| Deterministic execution | Parent or 1 worker |
| A few independent checks | 2-3 workers |
| Material ambiguity or risk | 3-5 workers |
| Broad search | More only with a stated budget and user approval |

Every worker returns this compact schema:

```text
Verdict: PASS | BLOCK | REQUEST_CHANGES
Finding: one-sentence result
Evidence: paths, source handles, or check output
Changes: files or artifacts, or none
Checks: commands/checks and pass/fail
Risks: unresolved issues
Side effects: none, or exact list
Confidence: low | medium | high
```

## Procedure

### 1. Shape the request

Write a task contract before calling MoA:

```text
Goal:
Done means:
Known evidence:
Non-goals:
Forbidden side effects:
Approval boundaries:
Budget or deadline:
Open questions:
```

Resolve only questions that materially change the plan. Label unknowns instead of letting advisors invent project or runtime facts.

### 2. Ask MoA to design the slices

Use `/moa` with the contract and this instruction:

```text
Design the smallest evidence-gated execution plan for this goal.
Compare advisor proposals by evidence, risk, and simplicity; do not vote.
Do not perform actions or invent facts.

For each slice return:
- id, objective, required input, and required output
- dependencies and non-goals
- worker angles and suggested worker count
- read-only or edit-capable status
- allowed tools and forbidden side effects
- verification evidence and completion gate

Also return overall risks, user approval points, and the next ready slice.
```

Reject a plan whose slices overlap ownership, lack a completion gate, or require evidence no worker can obtain.

### 3. Gate the plan

The parent checks cited files, APIs, constraints, and commands before execution. MoA is advisory: consensus cannot override source-of-truth evidence.

Ask the user before expensive fan-out, live external actions, persistent jobs, production changes, credential or permission changes, commits, pushes, deployments, or migrations unless the request already authorized that exact action. Do not pause for routine read-only checks or ordinary local edits already in scope.

### 4. Dispatch the next ready slice

Use `delegate_task` with one `tasks` entry per independent angle. Keep workers as `leaf` roles unless a slice explicitly needs nested orchestration and the configured depth allows it.

Give every worker a self-contained prompt:

```text
Slice and assigned angle:
Objective:
Required input:
Required output:
Scope and owned paths:
Read-only or edit-capable:
Allowed tools:
Forbidden side effects:
Evidence required:
Completion gate:
Stop and report when:

Return only the compact worker schema.
```

Read-only workers may inspect the same source. Editing workers must have non-overlapping ownership or isolated branches/worktrees; never allow concurrent edits to the same live files.

### 5. Compress and inspect reports

Do not paste raw worker transcripts into the overseer context. Preserve verdicts, disagreements, evidence handles, changed paths, checks, and side effects.

For a wide slice, assign one read-only synthesis worker to deduplicate reports without deciding success. The parent opens decisive artifacts and reruns checks when the claim matters.

### 6. Review and advance

Use a second `/moa` pass when workers disagree, risk is high, the plan changed, or evidence is incomplete:

```text
Review the slice contract, compact reports, and parent verification.
Judge evidence rather than vote count. Do not invent missing facts.
Return PASS, BLOCK, or REQUEST_CHANGES; list missing evidence and the safest next step.
```

Advance only when the slice output exists and its completion gate passes:

- **PASS with direct evidence:** mark the slice complete and start the next ready slice.
- **Missing or contradictory evidence:** repeat or narrow the current slice.
- **Changed scope, cost, or side effects:** ask the user before continuing.
- **Blocked dependency:** report the exact blocker and the evidence needed to unblock it.

After the final slice, run integration-level verification rather than assuming individually passing slices compose correctly.

## Pitfalls

1. **Blind fan-out:** More workers add cost and synthesis load. Start with the smallest width that produces distinct evidence.
2. **Duplicate angles:** Rephrased copies create fake consensus. Give workers different failure modes or responsibilities.
3. **Context flooding:** Large transcripts defeat the design. Return compact reports with drill-down handles.
4. **Sensitive fan-out:** Advisor or worker traces may persist. Redact first or keep the task local.
5. **Self-reported success:** Worker and MoA verdicts are inputs, not verification. The parent checks evidence directly.
6. **Overlapping edits:** Shared write ownership causes conflicts and invalidates checks. Serialize or isolate edits.
7. **Unbounded nesting:** Nested orchestrators multiply cost and obscure ownership. Prefer leaf workers and explicit slices.
8. **Approval theater:** Ask only at meaningful boundaries, while respecting authority already granted by the user.
9. **Plan drift:** If execution changes inputs, outputs, or risk, return to the affected slice gate before proceeding.

## Verification

Before reporting completion, confirm:

- [ ] Goal, done condition, non-goals, and approval boundaries were explicit.
- [ ] Context sent to MoA and workers was minimal and redacted.
- [ ] Each slice had defined inputs, outputs, ownership, and a completion gate.
- [ ] Worker width matched the task rather than a fixed preset.
- [ ] Worker reports named evidence, checks, changes, risks, and side effects.
- [ ] The parent inspected decisive evidence and reran relevant checks.
- [ ] Disagreement or high-risk work received an evidence-based review.
- [ ] No worker exceeded its scope or performed an unapproved side effect.
- [ ] Integration verification passed after the final slice.
- [ ] The final response distinguishes completed work, blockers, and remaining risk.
