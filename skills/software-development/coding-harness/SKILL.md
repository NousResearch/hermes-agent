---
name: coding-harness
description: "Use when a coding task spans many steps, files, or a long session: imposes a phased execution loop, evidence-driven verification, and tracked falsifiable progress so long-horizon work doesn't drift, stall, or silently regress."
version: 1.0.0
author: Teddy Tennant
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [coding, long-horizon, harness, execution-loop, verification, orchestration, reliability]
    related_skills: [plan, test-driven-development, systematic-debugging, requesting-code-review, subagent-driven-development]
---

# Coding Harness

## Overview

Long-horizon coding fails from **drift and unverified assumptions**, not from inability to
write code. Over 50+ tool calls the agent forgets the original goal, acts on stale beliefs
about the codebase, re-solves problems it already solved, and — worst — declares "done" on
its own say-so when the build is actually broken.

This skill is a **conductor**: it wraps a multi-step coding task in a structured loop, forces
every increment to be proven against external reality, and keeps falsifiable state on disk so
progress survives context compaction and interruptions. It trades a little raw speed for
structure that holds up over a long session.

The three pillars (adapted from agentic-harness-engineering / NexAU):

1. **Structured execution loop** — discrete phases with explicit entry/exit criteria. You
   always know which phase you are in and what proves you can leave it.
2. **Evidence-driven verification** — the agent's belief that something works is never
   sufficient. Only the *external* signal (build, test, lint, running the app) closes a step.
3. **Falsifiable progress** — each increment records a prediction; the next verification keeps
   it, reverts it, or marks it partial. No change is "trusted" until reality confirms it.

This skill does NOT replace the per-phase skills — it orchestrates them. Defer to `plan` for
planning, `test-driven-development` for the verify-first discipline, `systematic-debugging`
when a verification fails, `requesting-code-review` for an independent verifier, and
`subagent-driven-development` for delegation.

## When to Use

- The task spans **5+ distinct steps** or **multiple files / subsystems**.
- It will run across a long session where context will fill and compact.
- Correctness matters and "looks done" is not good enough (refactors, migrations, features
  with tests, bug hunts across modules).
- The user says things like "build X end-to-end", "migrate Y", "make Z reliable", or hands
  you a multi-part spec.

**Don't use for:** single-file edits, a one-line fix, a quick question, or anything finishable
in under ~5 tool calls. The harness overhead is not worth it there.

## The Execution Loop

Work **one increment at a time** through these phases. Never skip VERIFY.

| Phase | Action | Exit criterion |
|-------|--------|----------------|
| **SCOPE** | Restate the goal in your own words; list constraints, unknowns, success signals. Run the helper `init`. | Goal + done-definition written to state. |
| **PLAN** | Break the goal into ordered, independently-verifiable increments. Use the `plan` skill. | An ordered increment list exists in state. |
| **IMPLEMENT** | Take the next increment. Make the smallest change that could satisfy it. Record a prediction. | Code written; prediction recorded via `add-increment`. |
| **VERIFY** | Run the *external* check for this increment (see verification-protocol). | Pass/fail/partial recorded via `record-verification`. |
| **ATTRIBUTE** | Compare the result to the prediction. Verdict: keep / revert / partial. On fail → ROOT-CAUSE. | Verdict recorded; tree is in a known-good state. |
| **ITERATE** | If increments remain, go to IMPLEMENT. If all done, do a final full-suite VERIFY and stop. | All increments kept; final verify green. |

**ROOT-CAUSE (on VERIFY fail):** do not patch blindly. Switch to the `systematic-debugging`
skill, find the actual cause, then either fix forward or revert the increment. A failed
increment must return the tree to a known-good state before you move on.

Full phase spec, transitions, and termination rules: `references/execution-loop.md`.

## Harness State

The loop's memory lives in `.hermes/coding-harness/state.json` in the workspace, managed by
the helper script. It is the single source of truth that survives context compaction — re-read
it (`status`) whenever you lose the thread or resume after an interruption.

```bash
HS="python3 skills/software-development/coding-harness/scripts/harness_state.py"
# (use the absolute path to the script; it writes ./.hermes/coding-harness/state.json by default)

$HS init "Migrate auth from sessions to JWT, all tests green"
$HS add-increment "Add JWT issue/verify helpers" --predict "new unit tests pass" --risk "breaks existing session test"
$HS record-verification ch_001 pass --note "pytest tests/auth -q: 14 passed"
$HS status
```

State holds: the goal + done-definition, the ordered increment list, each increment's
prediction / risk / verification result / verdict, and a running log. See
`references/change-manifest.md` for the schema and a worked example.

## Verification Protocol (the core discipline)

**Never mark an increment done on self-report.** The single most common long-horizon failure
is the agent believing it succeeded when the external test says otherwise. Always close the
loop with a signal the agent does not author:

- **Build / compile** — the command exits 0.
- **Tests** — the relevant suite passes; for new behavior, write the test first (`test-driven-development`).
- **Lint / typecheck** — clean (or no new findings vs. baseline).
- **Run it** — for behavior the test suite can't cover, actually run the app/CLI and observe.

Cross-check belief vs. reality: if you *expected* pass and got fail (or vice-versa), that gap
is the most valuable signal in the loop — investigate it, don't paper over it.

For high-stakes increments, spawn an **independent verifier subagent** (no agent should verify
its own work) — defer to `requesting-code-review`. Full protocol and per-task-type proof
standards: `references/verification-protocol.md`.

## Falsifiable Progress

Every increment is a hypothesis. When you `add-increment`, state `predicted_impact` (what
should get better) and the at-risk regression (what might break). After VERIFY, the verdict is:

- **keep** — prediction held; advance.
- **revert** — change was ineffective or caused a regression; undo it, log the lesson.
- **partial** — partially worked; refine and re-verify.

This makes regressions visible immediately and stops the agent from accreting changes it can't
account for. Verdicts are recorded in state and form the audit trail of the whole task.

## Tool Orchestration

- **Parallel vs. sequential:** independent read-only ops (reading several files, separate
  greps) → fire concurrently in one turn. Edits/commands touching the **same path** → run
  sequentially to avoid races (matches Hermes's own `tool_executor.py` heuristic). Never
  parallelize two writes to the same file.
- **Delegate** large, independent sub-investigations or fan-out work to subagents
  (`subagent-driven-development`) — keep the main context focused on the loop, not on raw
  search output.
- **Context discipline:** before context fills, checkpoint to state (`add-increment` /
  `status` capture progress) so a compaction can't lose the plan. Prefer running a command and
  reading its summary over pasting huge outputs into context. This mirrors the compaction /
  long-output middleware that keeps the AHE loop in a valid state.

## Common Pitfalls

1. **Declaring done without running anything.** "The code looks correct" is not verification.
   Run the external check every time.
2. **Skipping ATTRIBUTE.** If you don't compare result to prediction, silent regressions
   accumulate. Record a verdict for every increment.
3. **Big-bang increments.** A change that touches ten things can't be verified or reverted
   cleanly. Keep increments small and independently checkable.
4. **Patching a failing verify without root-causing.** Switch to `systematic-debugging`; a
   symptom fix usually breaks something else.
5. **Letting state go stale.** If `status` doesn't match the working tree, the harness is
   lying to you. Update state as you go, not at the end.
6. **Re-restating instead of resuming.** After a compaction, run `status` and continue — don't
   re-plan from scratch and lose verified progress.
7. **Using the harness for trivial tasks.** Overhead with no payoff. See "When to Use".

## Verification Checklist

- [ ] `init` run with a concrete, testable done-definition
- [ ] Work proceeded one small increment at a time
- [ ] Every increment has a recorded prediction AND an external verification result
- [ ] Every VERIFY used a signal the agent did not author (build/test/lint/run)
- [ ] Every failed verify was root-caused (not blind-patched) and the tree returned to green
- [ ] Final full-suite verification passed
- [ ] `status` matches the actual working tree at the end
- [ ] High-stakes changes got an independent reviewer (`requesting-code-review`)
