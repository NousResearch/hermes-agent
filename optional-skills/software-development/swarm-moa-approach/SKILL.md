---
name: swarm-moa-approach
description: "Use when planning or reviewing complex agentic work with a MoA-backed swarm checkpoint: run a bounded Mixture-of-Agents planning/review pass, then execute with normal Hermes delegation, Kanban, worktrees, and parent-owned verification."
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [swarm, moa, mixture-of-agents, delegation, planning, review, verification]
    category: software-development
    related_skills: [plan, subagent-driven-development, requesting-code-review, systematic-debugging]
---

# Swarm MoA Approach

## Overview

Swarm MoA Approach is a conservative protocol for using Hermes' Mixture-of-Agents (MoA) provider as a **planning and review checkpoint** before launching more expensive agentic work.

The pattern is:

```text
bounded MoA planning/review checkpoint
→ user approval when needed
→ normal Hermes execution substrate
→ parent verifies objective evidence
→ optional final MoA review checkpoint
→ parent final PASS/BLOCK/REQUEST_CHANGES
```

MoA is useful when the hard part is thinking clearly: decomposition, risk spotting, architecture trade-offs, or skeptical final review. It is not a replacement for tool-grounded execution. Advisors may suggest; the parent/aggregator still owns side effects, verification, and the final decision.

## When to Use

Use this skill when the user asks for:

- MoA-backed planning or review
- a "bigger model" style reasoning checkpoint
- swarm planning before subagents, Kanban, or worktrees
- multi-agent architecture/design/debugging review
- a skeptical final review of compact worker evidence
- an experimental alternative to a normal subagent planning panel

Best fit:

- architecture choices with several plausible paths
- implementation plans where a bad decomposition would waste worker calls
- code review or security/isolation review before risky changes
- hard debugging strategy before launching tool-using workers
- Kanban plan review before creating durable work
- final review of worker findings before parent approval

Do **not** use this skill for:

- one obvious tool call or deterministic task
- arithmetic, hashes, time/date, or current system state checks
- tasks where the user asked to be cheap, fast, or single-pass
- source-of-truth operational answers where speculation is unsafe
- raw PII, secrets, credentials, or sensitive logs unless redacted and explicitly approved for fan-out
- live sends, deploys, commits, profile changes, or cron creation without normal approval gates

## Current Hermes MoA Contract

Hermes MoA is a virtual model provider, not a normal tool call:

- Provider: `moa`
- Model: a configured preset such as `default`, `review`, `ll-plan`, or `ll-review`
- One-shot CLI/TUI entry: `/moa <prompt>`
- Reference advisors run privately and do not call tools
- The aggregator is the acting model and may use the normal tool schema
- Optional full-turn traces can persist advisor and aggregator inputs/outputs

If `/moa` or provider `moa` is unavailable, do not pretend a MoA checkpoint happened. Say that the MoA checkpoint is unavailable in the current runtime and fall back to the `plan` skill, `subagent-driven-development`, or a small `delegate_task` planning panel.

## Core Principle

Use MoA for reasoning checkpoints. Use Hermes primitives for execution.

| Need | Prefer |
|---|---|
| Several model perspectives before acting | MoA checkpoint |
| Tool-grounded inspection by separate agents | `delegate_task` |
| Durable, auditable, multi-day coordination | Kanban |
| Parallel code edits | isolated git worktrees/branches |
| Final risky-surface verification | requesting-code-review / hard PASS-BLOCK review |
| One straightforward task | direct execution, no swarm |

Do not majority-vote facts into existence. Evidence beats consensus.

## Default Protocol

### 1. Check MoA-worthiness

Before invoking MoA, decide whether 2-3 independent advisor perspectives could plausibly change the plan, catch a major risk, or prevent wasted worker calls.

Skip MoA when:

- the task is simple
- the needed facts are missing and must be retrieved first
- context is too sensitive to fan out
- the user prioritized speed/cost
- advisor output would not change the next action

### 2. Build a grounded context pack

Gather facts with tools before asking MoA to reason over them.

Use this schema:

```text
Goal:
Done means:
Verified facts:
Files/docs/diffs inspected:
Known constraints:
Non-goals:
Risky surfaces:
Forbidden side effects:
Candidate verification commands/checks:
Open questions:
Budget: lean | standard | deep
```

Rules:

- Redact secrets, tokens, PII, customer contact info, and unnecessary logs.
- Include evidence handles and concise excerpts, not huge raw dumps.
- Mark unknowns explicitly instead of letting advisors infer them.
- If the context is too sensitive to share with advisor models, skip MoA.

### 3. Run a one-shot MoA planning checkpoint

Prefer one-shot `/moa` so the main session returns to its normal model/provider afterward.

Prompt template:

```text
You are a MoA planning checkpoint for a guarded Hermes swarm workflow.
Reference advisors should critique from different angles, but the final answer must be one actionable plan.
Do not majority-vote. Judge by evidence and risk.
Do not invent repo/runtime facts beyond the context pack.
Do not perform side effects.

Context pack:
<insert compact context pack>

Return only:
Goal:
Done means:
Non-goals / forbidden side effects:
MoA-worthiness: yes | no + why
Recommended substrate: single-pass | delegate_task | Kanban | worktree | cron | mixed
Why not simpler:
Swarm nodes:
- name:
  purpose:
  suggested width: 1 | 3 | 5 | 10
  worker angles:
  needs tools: yes | no
  needs worktree: yes | no
  durable state needed: yes | no
Single-pass/tool nodes:
Approval gates:
Risky surfaces:
Hard-pass needed: yes | no + why
Verification plan:
Budget risk: low | medium | high
Recommended next step:
```

### 4. Pause for approval when needed

Do not launch expensive, durable, or risky work just because the MoA plan recommends it. Ask for approval before:

- wide fan-out or K=10 worker swarms
- Kanban board/task creation
- worktree implementation branches
- commits, pushes, PRs, deploys, migrations
- cron/background jobs
- live customer/vendor/employee messages
- profile, skill, plugin, memory, or production config mutations

Suggested wording:

```text
I have not deployed the full swarm yet.
This is the MoA-backed plan. Approve, narrow, or tell me to fall back to a normal planning panel.
```

### 5. Execute with normal Hermes rules

After approval, choose the smallest safe substrate:

- `delegate_task` for bounded in-session worker analysis
- Kanban for durable, auditable, or interruptible work
- worktrees for parallel code edits
- cron only after manual/dry-run proof and explicit approval
- parent applies changes and performs live side effects

Every worker prompt should include:

```text
Assigned angle:
Goal:
Scope/root/profile/worktree:
Allowed tools:
Forbidden side effects:
Read-only or edit-capable:
Evidence required:
Output schema:
Stop condition:
```

### 6. Verify directly

Before claiming success, the parent must verify objective evidence directly:

- files, diffs, artifacts, or task records
- targeted tests, lint, typecheck, build, or smoke checks
- browser/API/service behavior where relevant
- CI/deployment status if applicable
- Kanban/handoff state for durable work
- independent hard-pass review for risky surfaces

MoA and worker self-reports are evidence candidates, not proof.

### 7. Optional final MoA review

Use a final MoA checkpoint when risk or disagreement warrants it:

- worker reports conflict
- the plan changed during execution
- code touches auth, permissions, data, deployment, or automation surfaces
- results will become a reusable process, cron, skill, or tool
- the user asked for extra confidence

Prompt template:

```text
You are a skeptical MoA final review checkpoint for a Hermes swarm run.
Do not majority-vote. Judge evidence quality.
Do not invent facts. If evidence is missing, BLOCK or REQUEST_CHANGES.
MoA may recommend, but the parent owns final PASS/BLOCK.

Goal:
<goal>

Plan used:
<brief>

Worker findings:
<compact summaries>

Parent verification evidence:
<commands, outputs, changed paths, unresolved risks>

Return only:
Verdict: PASS | BLOCK | REQUEST_CHANGES
Confidence: low | medium | high
Blocking issues:
Non-blocking issues:
Evidence sufficient:
Evidence weak or missing:
Contradictions resolved:
Hard-pass required: yes | no + why
Safe next step:
Do not do yet:
```

## Recommended Defaults

Keep MoA bounded:

- advisors: 2-3
- fanout: `user_turn` for planning/review checkpoints
- reference output: 400-800 tokens, default around 600
- traces: off by default
- one-shot `/moa` checkpoints, not long MoA-active execution sessions

If you maintain MoA presets, a practical naming scheme is:

```yaml
moa:
  default_preset: ll-plan
  save_traces: false
  presets:
    ll-plan:
      enabled: true
      fanout: user_turn
      reference_max_tokens: 600
      # reference_models and aggregator depend on the user's configured providers
    ll-review:
      enabled: true
      fanout: user_turn
      reference_max_tokens: 600
```

Do not include provider-specific model names in the skill unless the user asks for help configuring MoA. Use the user's existing MoA presets when available.

## Budget and Fallback Criteria

Set a small budget before the checkpoint:

```text
Max MoA checkpoints:
Max advisor width:
Max reference tokens:
Max wall time:
Fallback trigger:
```

Fallback to a normal planning panel or direct execution when:

- MoA is unavailable
- advisor outputs are generic, unsupported, or contradictory without evidence
- the checkpoint adds latency without changing the plan
- context is too sensitive to fan out
- the task becomes tool-heavy and iterative
- the user says cheap/fast/stop/fallback
- cost or latency exceeds the agreed budget

## Output Note

At the end of a run, include a short note:

```text
Swarm MoA note:
- checkpoints used:
- execution substrate:
- workers used:
- what MoA changed:
- verification evidence:
- cost/latency concern: low | medium | high
- fallback recommended next time: yes | no
```

Keep the user-facing answer concise. Do not dump advisor transcripts.

## Common Pitfalls

1. **MoA everywhere.** This skill is a checkpoint layer, not the whole execution engine.
2. **Token burn disguised as confidence.** More model calls can add noise. Use budget gates.
3. **Tool-blind advisors.** Advisors need a grounded context pack and cannot inspect files themselves.
4. **Consensus illusion.** Agreement among advisors is not evidence.
5. **Leaving MoA active.** Long MoA-provider sessions can multiply every tool iteration.
6. **Trace leakage.** Full traces may persist sensitive context.
7. **Skipping approval.** Expensive or risky deployment still needs user approval.
8. **Replacing verification.** MoA review does not replace tests, source inspection, or hard-pass review.

## Verification Checklist

- [ ] MoA-worthiness was checked.
- [ ] Sensitive context was redacted or MoA was skipped.
- [ ] The checkpoint was a real MoA run, or fallback was clearly labeled.
- [ ] The plan named substrate, worker roles, approval gates, and verification.
- [ ] User approval was obtained before expensive, durable, or risky actions.
- [ ] Execution used normal Hermes isolation and side-effect rules.
- [ ] Parent verified objective evidence directly.
- [ ] Risky surfaces received independent review where required.
- [ ] Final answer states what MoA changed and whether fallback is recommended next time.
