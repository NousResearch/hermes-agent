---
name: leaf-brief-standards
version: 1.0.0
description: "Use when dispatching implementer or fixer leaves."
author: KelpME (field-verified in HermesForge foreman sessions) + Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [orchestration, delegation, briefs, reliability]
    category: software-development
    related_skills: [test-driven-development, requesting-code-review]
---

# Leaf Brief Standards Skill

How to write dispatch briefs for implementation subagents (leaves) so they
produce work instead of dying on their timeout ceiling. Verified empirically:
subagents given prose briefs + a worktree spent 35–60 min on open-ended recon,
then hit their timeout ceiling with ZERO lines banked — three times on one task.
Root cause: the brief described the OUTCOME; the leaf had to discover the REPO.
When the model gateway degrades (measured p90 of 361s per call, vs ~5s when
healthy), ~25 recon reads consume the entire budget. The same tasks briefed with
pre-digested recon were implemented in minutes.

## When to Use

- Dispatching any implementation or fix leaf via `delegate_task`.
- Re-dispatching after a leaf timeout or failure (continuation briefs).
- Reviewing a brief someone else wrote before it burns a leaf's budget.

Not for: review-only leaves (they need the diff, not pace rules) or architect
dispatches when the orchestrator already holds the context (author the plan
yourself — dispatching an architect to re-discover it wastes a leaf).

## Prerequisites

No special setup. The orchestrator must have already read the relevant code
itself (the recon it will paste into the brief comes from that read).

## How to Run

Write the brief, then verify it against the checklist below BEFORE calling
`delegate_task`. The checklist is the deliverable: a brief that fails it will
cost a leaf.

## Quick Reference

The iron rule: **a leaf's first productive action (writing the failing test or
the fix) must be reachable within its first 1–3 tool calls.** If a leaf needs to
"read the code to understand the task," the orchestrator failed to brief, not
the leaf.

## Procedure

### Brief anatomy (in order)

1. **PACE RULE** (first line, always): "implement within your first 1–3 tool
   calls; ≤N tool calls total" — N sized to the task (small fix ≤20, module
   ≤25, integration ≤30).
2. **DO-NOT-READ list** (explicit): planning docs, status files, session logs,
   anything not listed. Curiosity + a slow gateway = a dead leaf.
3. **Pre-digested recon** — paste the relevant surface: exact function
   signatures, the data shapes (including wrapper-vs-array traps), file:line
   anchors for the 3–6 functions that matter, the test conventions to mirror,
   and any helper the leaf must reuse instead of re-implement.
4. **The contract, pinned**: exact API signatures, exact file paths (NEW vs
   MODIFY), exact commands + expected output, the commit command (message
   convention + the exact file list to stage).
5. **FROZEN artifacts called out**: any test file / contract the leaf must not
   edit — with the line-audit note that the pinned values were verified
   satisfiable BEFORE dispatch.
6. **ESCALATION rule**: "report NEEDS_CONTEXT with the exact pinned line if a
   requirement is unsatisfiable — do NOT edit the contract to fit."
7. **REPORT format**: status/SHA/counts/summary, <200 words, long detail written
   incrementally to a log file (leaves die composing long final messages — the
   file survives; the chat message may never land).

### Verification before dispatch (the orchestrator's job)

- **Line-audit any test contract a leaf authored** before treating it as truth:
  hand-compute every numeric assertion. A leaf-authored test can be
  mathematically unsatisfiable — no implementation can pass it, and the leaf
  will burn its whole budget discovering that (or worse, "fix" it by weakening
  the assertion).
- **Probe data-shape assumptions yourself**: a common silent killer is a
  wrapper-vs-array misread (`{v:[…]}` vs `[…]`) — the code reads a field off the
  wrong object, gets `undefined`, and every downstream computation NaNs. One
  probe script beats a timeout death.
- **Confirm the seam exists**: if the brief says "call X from Y", verify Y can
  actually call X (one grep) before the leaf burns its budget discovering it
  can't.

### Task-type disposal

- **Pure-module tasks** (new file, no integration): ideal for leaves — full API
  in the brief, zero recon needed.
- **Integration tasks**: split — the leaf implements against a frozen contract;
  the orchestrator (or a dedicated fixer leaf) does the wiring if the seam is
  ambiguous.
- **After a leaf death**: mine the transcript + `git status` FIRST. Banked RED
  tests are the contract for the continuation (never re-author). Zero banked
  work after 2 continuations = the orchestrator implements directly (sanctioned
  escalation), then runs the normal review gates on its own code.
- **Timeout with banked work ≠ failure**: it's inventory. The continuation brief
  says "DONE (do not touch): X. YOUR REMAINING WORK: Y" — never re-author from
  scratch.

## Pitfalls

- "Read the repo first to understand the codebase" — no. The orchestrator read it.
- Long prose context with the API buried in paragraphs — paste signatures, not
  prose.
- Dispatching an architect leaf when the orchestrator already holds all the
  context — author the decomposition yourself.
- Letting a leaf commit without spelling out the exact file list to stage —
  scope discipline erodes otherwise.
- Trusting a worker-authored test suite without hand-verifying the numbers.
- A leaf-authored test suite can be *doubly unsatisfiable* (an assertion
  measured from spawn position; a threshold requiring a coefficient ≥0.29 when
  the documented default was 0.1). The leaf burns its budget proving no
  implementation can pass.

## Verification

- The leaf's first transcript entries show it writing tests or code, not
  reading exploration targets.
- No leaf dies with zero banked work on a well-briefed task.
- Continuations resume from banked artifacts ("DONE (do not touch): X") instead
  of re-authoring.
