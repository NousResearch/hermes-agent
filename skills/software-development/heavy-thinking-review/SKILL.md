---
name: heavy-thinking-review
description: "Use when a task needs high-confidence synthesis from multiple independent reasoning paths: hard debugging, architecture decisions, code review, research-to-implementation, Kanban plan review, security/isolation review, or complex strategy. Runs parallel thinkers followed by a skeptical deliberator; avoid for simple facts or source-of-truth operational answers where speculation is unsafe."
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [parallel-reasoning, deliberation, review, kanban, subagents, synthesis]
    related_skills: [subagent-driven-development, systematic-debugging, requesting-code-review, writing-plans, kanban-orchestrator]
---

# Heavy Thinking Review

## Overview

Heavy Thinking Review is a fan-out / fan-in reasoning protocol inspired by the paper "HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness" (`arXiv:2605.02396`). It turns a hard question into several independent analyses, then asks a skeptical deliberator to compare evidence, resolve contradictions, and re-derive the final recommendation.

The point is **not** majority vote. Three weak answers that agree are still weak. The value comes from viewpoint diversity, compact synthesis, contradiction handling, and tool-grounded verification.

## Background

This skill adapts the fan-out / fan-in workflow described in ["HeavySkill: Heavy Thinking as the Inner Skill in Agentic Harness"](https://arxiv.org/abs/2605.02396). The paper frames heavy thinking as an inner harness skill: run several independent reasoning attempts, serialize their outputs, then have a separate deliberation pass synthesize the result. This skill turns that research pattern into a practical Hermes workflow using `delegate_task` for quick in-session fan-out or Kanban for durable, auditable fan-out/fan-in work.

Core pattern:

1. Spawn multiple independent thinkers.
2. Give each thinker a distinct angle, not a duplicate prompt.
3. Compress each result into a structured memory cache.
4. Deliberate skeptically: critique evidence, reconcile contradictions, re-derive the answer, and verify decisive claims with tools when possible.

## When to Use

Use this skill when a single reasoning path is likely to miss something important:

- hard debugging or root-cause analysis
- architecture or implementation strategy
- code review before merge
- security, isolation, privacy, or boundary review
- Kanban suite planning or final review
- research paper synthesis into practical implementation changes
- high-stakes harness, workflow, or operations changes
- complex strategy where multiple viewpoints improve the final answer

Do **not** use this skill for:

- simple factual lookup
- arithmetic or deterministic tool checks
- tasks with one obvious tool call or one obvious edit
- short user answers where extra analysis would add latency but not quality
- source-of-truth operational questions where unsupported speculation is unsafe
- subjective preference polishing where extra agents mostly add noise

If the task depends on an authoritative policy, approved memory, legal/compliance source, or production runbook, heavy thinking cannot invent missing authority. Use it to identify gaps and escalation paths, not to bypass missing evidence.

## Default Protocol

### 1. Decide whether fan-out is justified

Before spawning workers, decide whether independent viewpoints are worth the cost. Use the smallest useful width.

Recommended widths:

| Task class | Width |
|---|---:|
| Light hard task | 3 thinkers |
| Serious design/debug/review | 4-5 thinkers |
| Durable or audit-heavy suite | Kanban fan-out/fan-in |
| Broad research program | staged Kanban or multiple batches |

Avoid 8+ thinkers unless outputs are short, checkable, and cheap. Large serialized context can become noisy and can make the deliberator worse.

### 2. Assign different roles, not duplicate prompts

Bad:

```text
Three agents, all solve this.
```

Better:

```text
A — implementation feasibility
B — security / isolation / failure modes
C — operator or end-user workflow
D — testability / observability
```

Example role sets:

Architecture review:

```text
A — implementation feasibility and migration path
B — security, privacy, isolation, and failure modes
C — user/operator workflow and support burden
D — testability, observability, and rollback
```

Debugging:

```text
A — reproduce and inspect logs/errors
B — inspect recent diffs, dependency changes, and config changes
C — reason from architecture, invariants, and expected data flow
D — propose minimal fixes and regression tests
```

Research-to-implementation:

```text
A — summarize source claims and evidence
B — map claims to current system capabilities
C — identify risks, overclaims, and mismatches
D — propose concrete implementation or process changes
```

Product or strategy review:

```text
A — customer/user value and workflow fit
B — technical feasibility and data requirements
C — risk, compliance, and failure modes
D — positioning, narrative, and adoption friction
E — skeptical reviewer: what would make this fail?
```

### 3. Use `delegate_task` for quick parallel work

When the work can complete inside the current session, use `delegate_task` batch mode. Each child must be self-contained and should return a compact structured result.

Worker prompt template:

```text
You are one independent thinker in a Heavy Thinking Review. Do not read or infer other thinkers' outputs. Analyze only from this assigned angle: <angle>.

Return only:
- approach
- key findings
- evidence / files / commands / sources used
- risks or uncertainty
- recommendation
- confidence: low | medium | high
```

Use the narrowest relevant toolsets per thinker:

- `['terminal', 'file']` for repo inspection, tests, logs, and diffs
- `['web']` for current external research
- `['browser']` only when interactive pages are required
- `['terminal', 'file', 'web']` only when the same thinker truly needs all of them

### 4. Use Kanban for durable or human-interruptible work

Use Kanban instead of `delegate_task` when:

- the work should survive restart
- multiple profiles or isolated workspaces matter
- human review or approval may interrupt the workflow
- artifacts and audit trail matter
- long validation or final review is expected

Kanban graph shape:

```text
T1  specialist-A   independent angle A
T2  specialist-B   independent angle B
T3  specialist-C   independent angle C
T4  reviewer       synthesize / re-derive from T1-T3     parents: T1,T2,T3
```

The fan-in task must explicitly say:

```text
Do not majority-vote or concatenate parent outputs.
Identify contradictions.
Judge evidence quality.
Re-derive the recommendation from the underlying facts.
If all parent outputs are weak, say so and produce a better answer or block for missing evidence.
```

### 5. Build a serialized memory cache

Before final deliberation, normalize worker outputs into a compact cache. This prevents the longest, most confident, or first result from dominating.

Use this schema:

```text
Thinker ID:
Assigned angle:
Approach:
Key findings:
Evidence:
Contradictions / disagreements:
Failure modes:
Recommendation:
Confidence:
```

Keep each thinker summary roughly the same length. If order bias matters, sort by assigned angle rather than perceived quality.

### 6. Deliberate skeptically

The final deliberator must not be a stenographer. It should:

1. classify the task type and success criteria
2. compare evidence quality, not vote count
3. identify contradictions between thinkers
4. verify decisive claims with tools when possible
5. re-derive the recommendation from underlying facts
6. state uncertainty explicitly
7. produce the final user-facing answer in the requested format

Deliberator prompt template:

```text
You are the Heavy Thinking deliberator. You have independent thinker outputs below.

Do not majority-vote.
Do not concatenate summaries.
Critique each output for evidence quality and missing assumptions.
Resolve contradictions using provided evidence and, if tools are available, verify decisive facts.
If all thinkers are wrong or shallow, re-derive the answer yourself.
Return a concise final answer plus only the reasoning details that materially affect the decision.
```

## Tool Verification Rules

Heavy thinking improves reasoning; it does not replace grounding.

Always use tools for:

- file contents, diffs, repo state, tests, logs, and generated artifacts
- current facts, papers, versions, package docs, and external claims
- arithmetic, timing, benchmark numbers, and comparisons
- claims about services, ports, processes, system state, or deployed behavior

For side effects, verify the handle yourself before claiming success:

- read the file back
- inspect the diff
- run the test, lint, render, or validation command
- check task status
- inspect the output artifact
- capture exact paths, IDs, commit hashes, URLs, or status codes

## Output Format

Final answers usually include:

```text
Verdict:
Key evidence:
Contradictions resolved:
Recommendation:
Risks / follow-ups:
```

For concise user-facing tasks, compress this to the minimum useful form. Do not dump hidden chain-of-thought or long internal debate.

## Practical Patterns

### Code review

Thinkers:

- correctness and bug risk
- security, privacy, and data boundaries
- tests, coverage, and regression risk
- maintainability, design, and migration cost

Deliberator output:

- blocking findings first
- non-blocking issues second
- explicit pass/fail recommendation
- exact files/lines when possible

### Architecture decision

Thinkers:

- simplest implementation path
- long-term maintainability
- failure modes and operational risk
- observability, migration, and rollback

Deliberator output:

- recommended option
- rejected alternatives and why
- minimum viable implementation
- test and rollback plan

### Kanban suite design

Thinkers:

- task graph and dependencies
- workspace/profile/tool boundary
- validation artifacts and PASS/FAIL criteria
- failure modes and cleanup

Deliberator output:

- task IDs or dependency graph
- durable boundary contract
- artifact paths
- final reviewer checklist

### Research paper to implementation

Thinkers:

- source summary and claims
- empirical evidence strength
- applicability to the current system
- risks, missing pieces, and overclaims

Deliberator output:

- what to adopt now
- what to avoid
- concrete skill/config/process changes
- experiments to validate

## Common Pitfalls

1. **Majority voting.** If three weak outputs agree, they are still weak. Evidence beats count.

2. **Duplicate agents.** Parallelism helps only when thinkers are independent and meaningfully diverse.

3. **Context flooding.** Long raw outputs degrade synthesis. Normalize into the memory-cache schema.

4. **Letting the first thinker anchor the decision.** Keep summaries similarly sized and order them by assigned angle.

5. **Using heavy thinking to bypass authority.** If the task needs an approved source, policy, or runbook, fan-out can identify the missing authority but cannot safely invent it.

6. **No verification step.** Tool-grounded tasks still need actual file/test/status checks.

7. **Over-iterating.** Repeated deliberation can introduce cumulative noise. Prefer one strong fan-in pass unless the first pass identifies a specific gap.

8. **Stopping at a plan when execution was requested.** If tools can complete the task safely, execute and verify.

## Verification Checklist

Before finalizing a Heavy Thinking Review:

- [ ] The task justified fan-out.
- [ ] Thinkers had different roles or angles.
- [ ] Outputs were compressed into a structured memory cache.
- [ ] The deliberator resolved contradictions instead of majority-voting.
- [ ] Decisive claims were verified with tools where possible.
- [ ] The final answer is concise and user-facing.
- [ ] Risks and uncertainty are explicit.
- [ ] Any side effects were verified by reading/checking the resulting artifact.
