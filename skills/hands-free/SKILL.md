---
name: hands-free
description: >-
  Turn a plain-language desired end result into a meticulously crafted `/goal [...]`
  command that launches a Lead-orchestrated, parallelized multi-agent workflow. Use
  when the user states an outcome they want and wants it driven to completion
  hands-free — e.g. "just make X happen", "I want to end up with Y", "hands free:
  ship the landing page", "set it and forget it". The skill captures the end
  result, runs a short bounded round of context-refining exploration, then emits a
  single self-contained `/goal` block for a Lead orchestration agent to execute
  step-by-step while fanning out subtasks to skilled sub-agents.
---

# Hands Free

Convert a desired end result, expressed in natural language, into one precise
`/goal [...]` invocation that a **Lead orchestration agent** can execute
autonomously — decomposing the goal into a linear sequence of milestones and, at
each milestone, deploying the right **skilled sub-agents** in parallel to reach
the outcome efficiently.

The deliverable of this skill is **not** the finished work — it is the crafted
`/goal` command. You are the architect who hands a flawless blueprint to the
orchestrator.

## When to use

Trigger when the user describes an **outcome** rather than a single mechanical
step, and signals they want it driven to done without micromanaging:

- "I want to end up with a working onboarding email sequence."
- "Hands free — get the Q3 launch page researched, built, and reviewed."
- "Just make our docs site searchable."
- "Set up the thing so customers can book a call." 

Do **not** use for a single atomic action the user could run directly
(e.g. "rename this file", "what does line 40 do") — there is no orchestration to
design there.

## Operating principles

1. **Outcome first, mechanism second.** Anchor everything to the end state the
   user wants. Every milestone and subtask must trace back to it.
2. **Exploration is bounded.** Hands-free means the user does *not* want a long
   interview. Ask only what materially changes the plan. Prefer one batched
   round of 2–4 sharp questions over a drip of follow-ups. If you can infer a
   sane default, state the assumption instead of asking.
3. **Linear spine, parallel ribs.** The goal executes as an ordered sequence of
   milestones (the spine). Within a milestone, independent subtasks fan out to
   sub-agents in parallel (the ribs). Never parallelize across a dependency.
4. **Every subtask names an owner skill and a definition of done.** No vague
   "improve the thing." Each is assignable, verifiable, and bounded.
5. **Approval and review gates are explicit.** Anything outward-facing
   (publishing, sending, deploying) gets a named gate. Mirror the repo's hard
   rule: never post/ship without approval.
6. **The `/goal` is self-contained.** A fresh orchestrator with no prior context
   must be able to execute it from the block alone.

## Workflow

### Phase 1 — Capture the end result
Restate the user's desired outcome in one crisp sentence and reflect it back
inside your reasoning. Identify: the artifact(s) that must exist at the end, who
consumes them, and what "done" observably looks like.

### Phase 2 — Bounded exploration
Probe only the gaps that would change the orchestration. Use the
`AskUserQuestion` tool to batch 2–4 high-leverage questions. Draw from the
checklist in `references/exploration.md`, but include a question only if its
answer would change milestones, owners, gates, or success criteria. For
everything else, choose a sensible default and record it as a stated assumption.

If the user said "hands free" / "don't ask me", skip straight to assumptions —
ask at most one blocking question, and only if proceeding would otherwise risk
the wrong outcome.

### Phase 3 — Decompose & design
Build the orchestration plan:
- Break the outcome into an **ordered list of milestones** (the linear spine).
- For each milestone, list the **parallelizable subtasks** and assign each to a
  **skilled sub-agent** from `references/agent-roster.md`.
- Mark **dependencies** between milestones and the **gates** (review/approval)
  between them.
- Define **success criteria** for the goal and a **definition of done** per
  milestone.

### Phase 4 — Emit the `/goal`
Render one fenced `/goal [...]` block following the exact structure in
`references/goal-spec.md`. After the block, add a 2–3 line plain-language
summary and list any assumptions you made so the user can correct course before
launch. Do **not** start executing the work yourself — emitting the command is
the finish line for this skill.

## Reference material

- `references/goal-spec.md` — the canonical `/goal [...]` structure and field
  definitions. Read before emitting.
- `references/agent-roster.md` — the catalog of skilled sub-agents the Lead can
  deploy, their specialties, and parallelization rules.
- `references/exploration.md` — the context-refinement question checklist.
- `examples/example-run.md` — a full worked example from request to `/goal`.

## Hard rules

- Never emit a `/goal` that ships, sends, publishes, or deploys anything
  outward-facing without an explicit approval gate before that step.
- Never invent capabilities or agents not in the roster; if the outcome needs a
  skill that does not exist, say so and propose the closest fit.
- Keep the `/goal` block self-contained and copy-paste runnable.
- Surface assumptions; never bury a guess inside the goal as if it were a fact.
