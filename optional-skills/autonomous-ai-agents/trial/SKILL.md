---
name: trial
description: "Gated judging that stops false-done: an agent may not ship until executable evidence survives an independent verdict. Use for high-stakes work where a green test is not enough proof."
version: 0.3.0
author: Da7_Tech
license: MIT
homepage: https://github.com/Da7-Tech/trial-skill
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [verification, judging, false-done, quality-assurance, evidence, high-stakes]
    related_skills: [subagent-driven-development, test-driven-development, systematic-debugging]
---

# Trial

## Overview

Trial runs a task through **gates**: nothing advances without a written artifact,
and nothing ships until the work is verified with evidence. One rule governs it:

> **No execution before the decision is sound. No delivery before an independent
> verdict.** "Verified" means nothing without a written artifact showing the
> method, the evidence, and the result.

But rigor must be **fast**. The amount of ceremony — and whether to spawn
subagents at all — scales to the task. Most tasks run the gates *inline* in
minutes; only genuinely large or high-stakes work spawns the full parallel
tribunal.

Deep specs: `references/gates.md` (gates + lens matrix), `references/verdicts.md`
(verdict rules), `references/console.md` (the live console). Artifact scaffolds
in `templates/`.

## How this differs from `subagent-driven-development`

They are complementary, not competing:

- **`subagent-driven-development`** is about *execution*: dispatch a fresh
  subagent per task and review between tasks (2-stage review of the build).
- **`trial`** is about *judging the finished claim*: it attacks the "done" itself
  with executable evidence and an independent verdict. Its target is **false-done**
  — work that passes the builder's own tests but is actually broken.

Measured on 5 tasks with hidden SWE-bench-style oracles, a baseline that trusts
the builder's "strong evidence" ships 4/5 false-done tasks; Trial ships 0/5 and
blocks 0/5 correct ones (see the [benchmark](https://github.com/Da7-Tech/trial-skill/blob/main/benchmark/results/2026-06-29-judge-accuracy.md)).


## Step 0 — Triage FIRST (this decides speed)

Before doing anything, size the task. This is the most important decision.

| Tier | What it is | Path | Subagents | Target time |
|---|---|---|---|---|
| **fast** (default) | a function, a script, a small fix, one component, a focused analysis/doc | gates run **inline**, verify by running | 0 (maybe 1) | minutes |
| **standard** | a multi-file feature, a real module | light fan-out where it helps | ≤ 3 | tens of minutes |
| **strict** | shipping a product, an important audit | full gated fan-out, 3 judges | ~6–9 | longer |
| **maximum** | irreversible / high-risk / long-lived | full path, max reasoning | 9+ | longest |

Default is **fast** unless the task is clearly bigger. Read the mode from the
invocation (`/trial strict <task>`) if given; otherwise infer it. When unsure,
pick the lighter tier — you can escalate mid-run if it proves harder.

> **Speed is a feature.** Every delegated subagent costs real wall-clock —
> minutes each on slower models. A simple task that takes half an hour, or
> stalls at "gate 3/8", is a **bug, not rigor**. Never run the full fan-out on
> small work.

## Autonomous run

`/trial <prompt>` is one standing instruction (like goal mode): once started,
run to completion without stopping to check in. Don't pause to summarize between
gates or ask "shall I continue". The only stops: a 1–3 question clarification at
framing if the task is genuinely underspecified, or a hard escalation (rework cap
hit with a blocker needing a human) — set `verdict: escalated` and stop.

## The Fast Path (default — most tasks)

You (the orchestrator) run all eight gates **inline and quickly, with no
delegation**. The gates still happen — they're just fast:

1. **Frame** — short `mission-brief.md`: the real goal, testable acceptance
   criteria, non-goals. (If underspecified, ask 1–3 short questions now.)
2. **Research** — a quick inline scan: read the relevant files / recall the
   standard. Not a delegated batch. One short note in the decision brief is enough.
3. **Decide** — a 3-line `decision-brief.md`: approach + the acceptance bar +
   anything the build must not do.
4. **Council** — self-check the decision against the acceptance criteria inline.
5. **Build** — do the work directly with `terminal`/`file`/`patch`. Delegate ONE
   worker only if it genuinely needs an isolated context (rare for small tasks).
6. **Judge** — **verify by RUNNING it**: execute the code, run the tests, hit the
   edge cases yourself. Objective execution IS the independent check here — it's
   evidence, not self-opinion.
7. **Rework** — fix what the run exposed; re-run until green.
8. **Deliver** — short `final-delivery.md` mapping each acceptance criterion to
   the run evidence (command → result). Set `verdict: delivered`.

Keep `status.json` + the console updated so the user watches it advance live —
the gates just move fast. **Honesty:** the fast path verifies by running the
work directly instead of spawning fresh subagent judges; that's correct for
low-stakes work. For high-stakes work, use the full path (fresh subagents with
isolated context judge the claim independently).

## The Full Path (strict / maximum, or a genuinely complex task)

Same eight gates, but with **parallel delegated** research, council, and judging —
because the work is big enough that isolation + parallelism + independent judges
earn their cost. The gate-by-gate detail and the triple-lens matrix are in
`references/gates.md`; verdict rules in `references/verdicts.md`. In short:

2. **Triple research** — one parallel `delegate_task` batch of independent lenses.
4. **Council** — parallel batch of 3 judges on the decision brief.
5. **Execution team** — builders sized to the task (architecture/backend/frontend/…).
6. **Triple judging** — fresh judges, three different methods (requirement-match /
   real execution / adversarial), each a written `judge-verdict.md` with evidence.
7. **Targeted rework** — a blocking defect goes back to the responsible role only,
   bounded by the mode's rework rounds; on deadlock, escalate.
8. **Final review** — *you*, independently: re-run the tests, read the diff, check
   every criterion. Never trust subagent summaries.

`standard` is between the two: fan out only the part that genuinely benefits
(e.g. one research lens or one builder), keep judges to 1–2.

## Speed rules (apply on every path)

- **Don't delegate what you can do inline faster.** Delegation buys parallelism
  and context isolation on heavy work — it is pure latency cost on small work.
- **Bound subagents.** Tight goals; cap the count (~3 unless the task truly needs
  more). On slow models, prefer inline.
- **Never let a gate stall.** If a subagent is slow or looping, cut it and do that
  gate inline. Finishing is the job.

## Setup (all paths)

- `mkdir -p .hermes/trial/<timestamp>-<slug>` (relative to the workspace).
- Start `status.json` from `templates/status.example.json` (set `task`, `mode`,
  `lang`, `creator`, gates `pending`, `verdict: in-progress`) and `decision-log.md`.
- Render + open the console once (below), then update it at every gate.

## Language

Trial is written in English but **operates in the user's language** (detect from
how the user writes / `display.language`; default English). Every question,
artifact, the ledger, and the final report follow the user's language. Set `lang`
in `status.json` so the console matches (RTL for Arabic). Templates are English
scaffolds — fill them in the user's language.

## Live console (the visual the user watches)

See `references/console.md`. At every gate transition: (1) update `status.json`
(`current_gate`, each `gates[].state`, `judges[]`, `builders[]`, append a
`ledger[]` line, `verdict`); (2) re-render:
`python3 <skill_dir>/scripts/render_console.py --status <run>/status.json --out <run>/trial-console.html`;
(3) on the FIRST render only, open it (`open <run>/trial-console.html` on macOS,
`xdg-open` on Linux). It auto-reloads while `verdict` is `in-progress`.
`<skill_dir>` is the absolute path injected with this skill.

On the fast path the gates advance quickly — still update status.json so the
console (and any in-app trace) shows live progress, but don't let console
bookkeeping slow you down.

## Delegation recipes (full path)

`delegate_task` spawns a real subagent with an isolated context + terminal; the
parent blocks until children return; concurrency caps at
`delegation.max_concurrent_children` (default 3). Parallel batch:

```
delegate_task(tasks=[
  {goal: "<lens 1 ...>", context: "<run dir + brief>", toolsets: ["search","web","file"]},
  {goal: "<lens 2 ...>", context: "...", toolsets: ["search","web","file"]},
])
```

Single worker: `delegate_task(goal="<slice>", context="<brief + run dir>", toolsets=["terminal","file"])`.
Leaf limits: subagents cannot `delegate_task`, `clarify`, `memory`,
`send_message`, or `execute_code` — put everything in `goal` + `context`, and
have each write its artifact + return a short summary. Evidence in, evidence out:
a judge's goal must demand method + exact evidence + verdict; reject "looks good".

## Artifacts

Under `.hermes/trial/<timestamp>-<slug>/`: `mission-brief.md`, `decision-brief.md`,
`final-delivery.md`, `decision-log.md`, `status.json`, `trial-console.html`. The
full path adds `research/`, `council/`, `build/`, `judging/`, `rework/`. The fast
path writes the core set only. If `decision-log.md` has no line for a gate, that
gate didn't happen.

## Common Pitfalls

1. **Over-ceremony on a simple task.** The #1 failure. A function does NOT need a
   3-judge tribunal. Triage first; default to the fast path; finish in minutes.
2. **Stalling.** A run stuck at gate N for many minutes is broken — cut the slow
   subagent and do the gate inline.
3. **Trusting summaries.** Verify by running; on the full path the orchestrator
   re-runs tests at Gate 8.
4. **Verdict with no evidence.** "Looks correct" is not a verdict.
5. **Letting the builder judge itself** (full path) — judges are fresh subagents.
6. **Unbounded rework** — cap rounds; on deadlock, escalate.
7. **Wrong language / stale console** — render artifacts in the user's language;
   set `lang`; keep `status.json` current.

## Verification Checklist

- [ ] Triaged first; effort + subagent count matched the task size (no fan-out on small work).
- [ ] `mission-brief.md` has explicit, testable acceptance criteria.
- [ ] The result was **verified by running it** (fast path) or by fresh judges with evidence (full path).
- [ ] Every blocking defect fixed and re-verified; rework bounded.
- [ ] `status.json` + console updated through to a final `verdict`.
- [ ] `final-delivery.md` maps each acceptance criterion to its evidence.
- [ ] All artifacts, ledger, and the report are in the user's language.
- [ ] It finished in a time that fits the task — fast for small work.
