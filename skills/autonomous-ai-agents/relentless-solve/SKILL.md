---
name: relentless-solve
description: >
  Use when a prompt should be driven to a solution without letting go. The `solve` entry
  gates the intent to one of three engines: a trivial passthrough for one-shot answers,
  method-explorer's (fka resilient-planner) own AND/OR backtracking search when there's exactly one clear method,
  or — for failure-prone, uncertain, multi-step intents — its own full loop: clarifies the
  prompt first (next-best-questions ranks candidate questions by EVSI; the investigator
  researches them), asks the task-decomposer for a fresh plan.json (intent + everything learned
  so far → ordered oneshot-sized tasks with success criteria), executes one hermes oneshot
  per task (with its own bounded local retry and mid-cycle staleness-triggered replan), and
  harvests the per-task verdicts — failures as dead-ends, successes as facts — into the
  evidence folded into the next clarify/replan round, looping until success or the search is
  provably information-dry. Deterministic outer loop (no LLM decides control flow), durable
  and resumable (resumable-script flow). The `scope` entry runs the CLARIFY→PLAN prefix
  ONLY (read-only grounded research in a disposable sibling worktree, never executes) and
  emits a scope package — use it to scope work you'll author yourself. Triggers:
  "relentlessly solve this", "keep going until it works", "clarify then plan, execute, and
  learn from failures", "scope this out for me / what would it take".
version: 0.1.0
author: agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [orchestrator, clarify-execute-loop, value-of-information, backtracking, durable, autonomous]
    related_skills: [investigator, next-best-questions, task-decomposer, method-explorer, resumable-script]
---

# Relentless Solve — clarify → plan → execute per task → harvest → repeat

## Overview

One deterministic loop over three existing layers. Each cycle:

1. **CLARIFY** — `investigator/scripts/iterate.py` (in-process): information-gain ranks the
   next-best questions given *everything known so far*; the top-K are researched by a full
   Hermes agent; answers and gaps come back as tombstones.
2. **PLAN** — `task-decomposer` (one `hermes -z` oneshot): the prompt, re-rendered with the
   evidence ledger, becomes one validated `plan.json` at `relentless/<slug>/c<N>/` —
   ordered oneshot-sized tasks with success criteria (the final task always verifies the
   intent), or an honest `needs_decision` / `exhausted` verdict. Malformed plans retry
   with the violations echoed back (3 strikes → visible, resumable failure).
3. **EXECUTE** — one `hermes -z` oneshot per task; the executor writes
   `result-<id>.json` `{verdict, evidence}` judged against the task's success criterion.
   CODE reads the artifact (missing/malformed ⇒ failed); tasks whose dependency failed
   are skipped. EXECUTE is itself two nested levels (`run_intent_path`):
   - **LEVEL 2 — local retry** (`run_task_with_local_retry`): on failure, run a clarify
     SCOPED to why THIS task failed (not the whole intent), fold the fact straight into
     the shared ledger, and reattempt the SAME task — up to `--local-retry-budget` extra
     tries (default 2). A task that succeeds on its first attempt is byte-identical to
     today (same step key, same `result-<id>.json`). On exhaustion, escalation is
     unchanged: the failure becomes a dead-end exactly as a single-attempt failure would —
     UNLESS a cheap gate (`rp_delegation_gate`, one oneshot) judges genuine alternative
     methods plausible, in which case a scoped `method-explorer` sub-run
     (`run_task_delegation`, capped at `MAX_RP_DELEGATIONS_PER_CYCLE`=1/cycle) gets one
     shot at the task first — the same diagnose→retry→backtrack→cap policy
     method-explorer already owns, invoked here at task grain instead of
     re-implemented. Never a hard dependency: sibling absent or any failure falls back to
     the unchanged escalation path.
   - **LEVEL 1 — staleness gate** (`stale_tail`): after EVERY task (success or failure),
     a pure-code, no-LLM check asks whether the REMAINING to-do list still holds —
     dead-method reuse, vocabulary overlap with the fresh evidence (evidence AND any
     `learnings`), or local-retry exhaustion. If it trips, LEVEL 1 first runs its own
     next-best-questions clarify pass scoped to the fresh evidence and the remaining
     tasks (`run_replan_clarify` — same investigator primitive as LEVEL 0/2, folding any
     answers into the ledger before replanning), then requests a bounded
     (`MAX_REPLANS_PER_CYCLE`, default 3) mid-cycle **partial replan** (one task-decomposer
     oneshot) that reforges the untouched tail into its own `replan-<seq>.json` —
     `plan.json` itself is never overwritten. Every "map intent to task" moment in the
     loop — the whole-cycle plan (LEVEL 0), a local retry (LEVEL 2), and a partial replan
     (LEVEL 1) — is preceded by a clarify pass sized to its scope; none of them skip
     leveraging what's already been learned. A `needs_decision` from a partial replan
     always assume-and-notes; it never suspends, even under `--gate` — the human gate
     lives only at the top of the cycle loop.
4. **HARVEST** — `scripts/harvest.py` (pure fold): failed tasks become dead-ends, worked
   tasks become facts, seeded into the next clarify round, where the EVSI gate naturally
   retires resolved questions and surfaces the next-best unknowns. An executor attempt can
   ALSO report `learnings` — optional, free-form incidental facts beyond the pass/fail
   evidence itself (capped at 5 per attempt, 500 chars each) — regardless of whether that
   attempt worked or failed. Each one folds as its own `fact` record (fp'd on its own
   text), so it's visible to the next clarify/plan round exactly like any other evidence,
   and it also feeds LEVEL 1's staleness-gate vocabulary check — a learning from a
   *successful* task can still flag a later task as stale before it even runs.

The **prompt is immutable** (intent); the **ledger only grows** (facts / gaps / dead-ends).
Stop conditions, in order of honor: **SUCCESS** (every plan task verified worked — the
final task is intent verification, so this equivalence stays in pure code) ·
**information-dry** (a full cycle yields zero fresh facts — clarify converged AND every
harvested record was already known; the anti-flap guard that separates relentless from
flailing) · `max_cycles` · outer wallclock.

`needs_decision` forks default to **assume-and-note**: the open fork is recorded as a gap
and the next clarify round ranks it. `--gate` opts into suspending instead (engine exit 10;
answer with `resume --answer`).

## How to run (inside the hermes container)

```bash
docker exec -u 10000 hermes python3 \
  /opt/data/hermes-agent/skills/autonomous-ai-agents/relentless-solve/scripts/relentless.py run \
  --slug my-task --answer-cwd /path/to/target/project \
  --prompt 'the intent, stated once' \
  [--max-cycles 5] [--wallclock 14400] [--k 6] [--inv-rounds 3] [--capability act] [--gate] \
  [--plan-timeout 300] [--task-timeout 600] [--local-retry-budget 2] [--dod path/to/dod.md]
# after a --gate suspension:
... relentless.py resume --slug my-task --answer 'prefer source D'
```

`--answer-cwd` pins where the clarify answerer researches — always set it to the target
project (the known failure mode is researching the install dir). Host-side runs are
tests-only (`tests/`, fakes; iterate.py needs the container's model_utils live).

`--dod path/to/dod.md` (also on `solve`; full route only) fronts the intent with a
define-done requirements spec: the rendered prompt carries the unmet R-ids, every plan
task must name the requirement ids it `serves` (coverage is BINDING —
`planfile.coverage_violations` rides the same retry-echo channel as schema violations,
alongside the now-binding dead-method check), and each executed cycle lands
`c<N>/report.json` — task-decomposer's completion contract (`{status, tasks,
requirements, delta}`; the delta returns only knowledge NOT already in the ledger when
the cycle was planned). The final `report.md` opens with the requirements rollup. The
dod TEXT rides inside the immutable engine input, so resumes replay the exact spec the
run started with (define-done must be on disk wherever the run plays); a spec failing
`spec.lint` is refused at the CLI.

## `solve` — the one-argument entry (gate → route → run)

```bash
# host-side; the only required input is the intent:
python3 ${HOME}/.hermes/skills/relentless-solve/scripts/relentless.py solve \
  --prompt 'the intent, stated once' \
  [--budget 1800] [--risk act|experiment|read] [--gate] \
  [--slug S] [--route trivial|single_method|full] [--gate-only] [--dod path/to/dod.md]
```

Interface principle: **everything else is derived, conventional, or adaptive** — a knob is
a confession the system can't read the evidence. The honest residue is four parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `--prompt` | the intent (the ONE required input) | — |
| `--budget` | total wallclock seconds, one pool; routes subdivide it | 1800 |
| `--risk` | may the agent touch the world? → clarify capability + planner constraints | act |
| `--gate` | suspend on an unresolvable GUARD-HALT fork (else assume-and-note) | off |

Derived (never asked): **slug** — deterministic kebab-case of the intent's key nouns (same
prompt → same `relentless/<slug>/`, so re-invocations resume); **route** — one cheap
`hermes -z` classify: `trivial` (one pass-through answer) · `single_method` (one
method-explorer run, all-but-60s of budget) · `full` (the clarify→execute→harvest loop).
**Any gate failure — error, garbage, unknown verdict — routes to `full`**: misrouting a
trivial prompt wastes a few calls; misrouting a hard task silently under-serves it.

Adaptive: on the full route the cycle budget is a **cascading share** — each cycle gets
`remaining / cycles-left`, recomputed from the memoized clock at every boundary, split
20% to the planning oneshot (clamp 60–300s), 70% across the task oneshots — divided by
`n_tasks * (1 + local_retry_budget)` so LEVEL 2's local retries stay inside the same
per-cycle envelope (clamp 120–900s each) — and 10% across up to `MAX_REPLANS_PER_CYCLE`
mid-cycle partial-replan oneshots (clamp 60–240s each), so a cheap cycle's unspent time
flows back to later cycles. A mid-cycle wallclock deadline (this cycle's share, checked
only right before a retry or a replan attempt — never before a task's own first attempt)
backstops aggregate overshoot from chained retries/replans; evidence-based stops
(SUCCESS, information-dry) remain primary, the budget is only the ceiling.

Receipts (an invisible default is a bug): the verdict + derivations persist to
`relentless/<slug>/gate.json` (**reused on re-invocation — never re-classified**; force
with `--route`, inspect with `--gate-only`); the gate verdict is the full route's first
ledger row; `report.md` opens with a header recording slug/route/budget/risk/stop.

## `scope` — the planning entry (CLARIFY → PLAN, never EXECUTE)

```bash
python3 .../relentless.py scope --prompt 'the intent' --answer-cwd /path/to/project \
  [--rounds 2] [--budget 1800] [--dod path/to/dod.md] \
  [--fact 'a human answer' ...] [--evidence-file facts.txt] [--knowledge on|off]
```

`solve` = "do it"; `scope` = "tell me what you'd do." The write-contract line (see
`skills/ARCHITECTURE.md`): investigation writes KNOWLEDGE, planning writes a PROPOSAL,
only EXECUTE writes the world — scope stops at the proposal. Per round: one grounded
clarify (capability FORCED to `read`; no override) then one task-decomposer oneshot.
`tasks` → **scoped** (done) · `needs_decision` → the question is folded so the NEXT
round's clarify targets it; if it survives, it's emitted as an open decision (scope
output, not failure) · `exhausted` → **infeasible-as-stated** · zero fresh facts +
converged → **dry**.

The deliverable is the **scope package** — `relentless/<slug>/scope/scope.md` (+
machine-readable `scope.json`): intent · dod rollup (when `--dod`; its `OPEN:` line
seeds the first clarify) · facts learned · the task breakdown (your authoring
work-list) · open decisions · **Unanswerable** (attempted, NOT_FOUND, with reasons) ·
**Next questions** (EVSI-ranked, above-floor, never attempted — answer these, in this
order, then re-run with `--fact`/`--evidence-file` to fold your answers back in).

**Isolation** (mechanical, not just polite): research runs in a disposable **sibling
worktree** created FROM `--answer-cwd`'s own (wor)tree at its HEAD, with uncommitted
tracked changes carried across (untracked files: documented limitation). The answerer's
`cwd` is pinned there at the subprocess level. Layers, stated honestly: read directive
(instructional) · toolsets minus terminal (mechanical) · subprocess cwd → disposable
worktree (mechanical) · per-round `git status` **dirty receipts** vs the post-setup
baseline (detection; violations land in the package AND keep the worktree as evidence)
· the container boundary (outer wall). Clean teardown removes the worktree; a non-git
`--answer-cwd` degrades to directive-level with a warning.

**Containment on a violation**: a receipt mismatch archives evidence (`violation.patch` +
a bounded copy of untracked files + `manifest.json`) under `scope/violations/round-<n>/`
BEFORE the worktree is reset back to the baseline commit; either step failing (archival or
reset) surfaces as the `containment-failed` verdict. The violating round's clarify facts
are tainted — excluded from later clarify seeding, plan rendering, and knowledge
promotion — and render in scope.md with a `⚠` marker and a "(violating round —
re-verify)" suffix. scope.json's isolation record distinguishes `had_violation` (a
violation happened at ANY point in the run) from `currently_diverged` (the worktree was
STILL dirty vs baseline at teardown). Documented limitation, unchanged by containment:
git-ignored writes are invisible to the receipt (`git status --porcelain` doesn't see
them) and are silently wiped by the reset's `git clean -ffdx` — a containment blind
spot, not a detected violation. Scope flow is now **version 2**: an in-flight v1
`<slug>/scope-flow` journal will NOT resume against it — delete `<slug>/scope-flow` and
re-run scope from scratch (do **not** reach for `--accept-flow-change` here, unlike the
solve-side precedent in Exit codes below).

## Building-block resolution (env overrides)

| Dependency | Default | Override |
|---|---|---|
| next-best-questions ranker (fka information-gain; env prefix kept) | sibling `../next-best-questions/scripts` → `../information-gain/scripts` (pinned before importing iterate) | `INFOGAIN_SCRIPTS_DIR` |
| ask model dispatch | `${HERMES_HOME}/skills/productivity/ask/scripts` | `ASK_SCRIPTS_DIR` (read inside the ask/investigator skills, not by relentless.py) |
| task-decomposer (plan-as-data contract: planfile.py + envelope.py + report.py, loaded from ONE dir) | sibling `../task-decomposer/scripts` → `${HERMES_HOME}/skills/task-decomposer/scripts` | `TASK_DECOMPOSER_DIR` |
| define-done spec grammar (`--dod` runs only, every play incl. replays) | sibling `../define-done/scripts` → `${HERMES_HOME}/skills/define-done/scripts` | `DEFINE_DONE_DIR` |
| method-explorer driver (solve `single_method` route only) | `${HERMES_HOME}/skills/method-explorer/scripts/drive.py` (falls back to the old `resilient-planner` dir name) | `METHOD_EXPLORER_DRIVE` (`RESILIENT_DRIVE` still accepted) |
| method-explorer envelope (`single_method` invocation contract) | sibling `../method-explorer/scripts` → `${HERMES_HOME}/skills/method-explorer/scripts` (old dir name accepted at each rung) | `METHOD_EXPLORER_DIR` (`RESILIENT_ENVELOPE_DIR` still accepted) |
| resumable-script engine | `${HERMES_HOME}/skills/resumable-script/scripts` | `RESUMABLE_ENGINE_DIR` (locator read by relentless.py; engine.py itself never reads it) |

The engine must be deployed at `${HERMES_HOME}/skills/resumable-script/` (sync it there if
only present in a staging tree).

## Subroutine posture (topology B) — the dev loop above, relentless below

The family rule (2026-07-03, `skills/ARCHITECTURE.md`): the DIRECT layer — your dev
loop, outside this family — is the top-level orchestrator; relentless is a
**per-problem subroutine** (`scope` for planning, `solve` for hard steps). Three
mechanical consequences live here:

- **Recursion guard (exit 4).** Every entry sets `RELENTLESS_ACTIVE=<slug>` in its
  process env; task executors and clarify answerers inherit it, and a nested
  invocation refuses with exit 4 — nothing below relentless may call upward. An
  oversized task should return verdict `needs_split` (with a `split` list) instead:
  no local retry, no delegation — the loop folds the SPLIT HINT and forces an
  immediate partial replan through the task-decomposer. `--allow-nested` is the
  explicit escape hatch.
- **Result contract.** Every `solve` route writes `relentless/<slug>/solve.json`
  (`{slug, route, outcome, detail, report_path, spent_s, artifacts}`; artifact keys
  pinned per route — trivial: `answer` · single_method: `plan_tree` · full: `ledger`,
  `last_plan` · all routes: `journey`, the path to journey.json IF this invocation
  produced it, else null); `scope` writes `scope/scope.json`. Parse those, not report.md.
- **Global knowledge tier** (`scripts/knowledge.py` → `${HERMES_HOME}/knowledge/
  global.jsonl`): at run end, the run's facts and dead-ends are promoted (flock-guarded,
  fp-deduped, tagged with the repo identity of `--answer-cwd` — worktrees of one repo
  share a key); a new run seeds its first clarify with the most recent same-project
  records as provenance-prefixed EVIDENCE ONLY (never the ledger — a prior run's
  dead-end is never binding here; null-project records never seed; cross-project
  seeding isn't offered). `--knowledge off` = hermetic: no seeding, no promotion.

## State layout

```
${HERMES_HOME}/relentless/<slug>/
  flow/           # resumable-script engine state (journal.jsonl, state.json, blobs/, lock)
  prompt-c<N>.md  # the rendered body (intent + ledger) each cycle actually planned from
  ledger.jsonl    # human-readable evidence ledger snapshot (flow journal is the durable truth)
  journey.json    # the CONSOLIDATED decision record (see "The journey record" below) —
                  #   written by every route and every outcome; hindsight spliced on success
  retro.json      # the hindsight judge's raw emission (success + full route only;
                  #   .rej sibling on a validation-rejected attempt)
  report.md       # the journey's FULL render + the ledger-by-kind appendix (a pure view
                  #   of journey.json — regenerable offline)
  c<N>/           # per-cycle scratch receipts (read by humans; folded once from memory):
    plan.json           # the cycle's ORIGINAL plan (never overwritten by a partial replan)
    plan.json.rej<K>    # validation-rejected planner attempts (audit)
    report.json         # --dod runs: the cycle's completion contract (status/requirements/delta)
    result-<id>.json    # each task's self-reported {verdict, evidence} — first attempt
    result-<id>-retry<K>.json  # LEVEL 2: each local reattempt's own result artifact
    replan-<seq>.json   # LEVEL 1: a mid-cycle partial replan's tail (1..MAX_REPLANS_PER_CYCLE)
    replan-<seq>.json.rej<K>   # validation-rejected partial-replan attempts (audit)
    clarify/            # the investigator's per-cycle journal (crash mid-clarify resumes):
      tombstones.jsonl  #   answered facts + gaps as they land (header pins the problem fp)
      answer-<fp>.json  #   per-question answer artifacts (omitted under risk=read)
    clarify-scoped/<task-id>-retry<K>/   # LEVEL 2: one scoped-clarify journal per local retry
      tombstones.jsonl  #   same durability mechanism as clarify/, scoped to one task's failure
    clarify-scoped/replan<seq>-after-<task-id>/   # LEVEL 1: one pre-replan clarify journal
      tombstones.jsonl  #   scoped to the fresh evidence/learnings that tripped the gate
    rp-<task-id>/       # LEVEL 2's method-explorer delegation for ONE exhausted task
      prompt.md         #   the scoped intent handed to method-explorer (real_prompt)
  solve.json      # B2 subroutine contract: {slug, route, outcome, detail, report_path,
                  #   spent_s, artifacts}; artifacts.journey is the path to journey.json
                  #   IF this invocation produced it, else null — present for all routes
  scope-flow/     # scope mode's OWN engine state (flow id relentless-scope)
  scope/          # scope mode artifacts (never collide with a solve run on the slug):
    scope.md          # the scope package (human)
    scope.json        # the package, machine-readable (+ isolation record)
    worktree/         # the disposable sibling research worktree (removed when clean;
                      #   kept + .violated-<n> siblings as evidence on a write-contract
                      #   violation); worktree-baseline.json = post-setup porcelain
    c<N>/plan.json    # per-round decomposer artifacts (same shapes as the solve loop)
    s<N>/clarify/     # per-round investigator journals
${HERMES_HOME}/knowledge/global.jsonl   # the global knowledge tier (see Subroutine posture)
${HERMES_HOME}/plans/<slug>-single/   # method-explorer tree (solve single_method route only)
${HERMES_HOME}/plans/<slug>-c<N>-<task-id>/   # LEVEL 2 delegation's OWN method-explorer
                                              # tree (plan-tree.md, journal.jsonl) — never
                                              # collides with -single or across cycles
```

## Exit codes / result

Exit codes are the resumable-script engine's, plus the recursion guard: `0` completed —
read `result.outcome` from the final stdout JSON (`success` | `information-dry` |
`max-cycles` | `wallclock`; exit 0 covers all four; scope: `scoped` | `open-decisions` |
`infeasible` | `dry` | `budget` | `containment-failed`) · `4` **nested invocation refused** (RELENTLESS_ACTIVE
was set; see Subroutine posture — engine codes 3/10/11/12/13 are distinct) · `10`
suspended on a `--gate` fork · `1/2/3` failed/usage/skew. Machine consumers should read
`solve.json`/`scope.json` rather than parsing stdout. A crash or kill is
resumable: re-run the same `run` command and completed steps replay from the journal — the
plan step is memoized, so execution resumes at the first unfinished task/local-retry/replan
(after editing relentless.py mid-run, add `--accept-flow-change`).

Flow version is **5** (v3: LEVEL 1/2 nested retry + partial-replan key topology —
`c<N>/t/<id>/retry<K>[/clarify]`, `c<N>/replan/after-<id>`; v4: LEVEL 2's method-explorer
delegation adds `c<N>/t/<id>/rp-delegate`; v5: the journey record adds `retro/journey`
always and `retro/clock` + `retro/judge` on a successful cascade route). For the v2→v3
jump, the engine's `flow_hash()` forces `--accept-flow-change` for ANY in-flight run
automatically. **v3→v4 was different**: the delegation change lived entirely in
`run_task_with_local_retry`/`run_intent_path`, not in `relentless_flow`'s own source —
`flow_hash()` only hashes `relentless_flow`'s source, so it did not trip on its own; the
real safety net is the engine's separate strict-replay key-sequence check — an in-flight
v3 run where every task so far had **worked** resumes cleanly under
`--accept-flow-change` (identical key sequence either way); a run interrupted with an
**already-failed, still-exhausting** task hits the engine's own `NonDeterminism` guard
(exit 3) if replay now wants an `rp-delegate` step the old journal doesn't have — a
clean, understood failure, not corruption; re-run `solve` instead (`gate.json` is
reused). **v4→v5 trips `flow_hash()` again** (relentless_flow's source changed), so an
in-flight v4 journal needs `--accept-flow-change` — and since every v5 step key sits
AFTER the cycle loop, the accepted journal replays its completed prefix cleanly and only
the retro/report tail runs fresh. Runs started before v2 (drive-based cycles) cannot
resume at all.

## The journey record (journey.json) + hindsight

Two journals, two jobs: the engine's `journal.jsonl` is the RAW activity log (replay/
durability); `journey.json` is the CONSOLIDATED record — written by **every route and
every outcome** — and `report.md` is its pure FULL render (`journey.py:render_journey`;
the view cannot drift from the record). The model is one node shape,
**Node = (evidence, options, taken)**, on a chain (`S0, S1, ...`) with one node wherever
the plan of record changed: cycle plans, mid-cycle partial replans, LEVEL 2 retries, and
delegation gates. Everything else dissolves into the three primitives:

- a **failed branch** is dead-end *evidence* at a later node (`from`-linked to the task
  that produced it — harvest already works this way);
- a **superseded replan tail** is a `not_taken` *option* ("continue-old-tail", why_not =
  the staleness reason) at the replan node;
- **evidence lives only as per-node deltas** — "known at S_k" is positional (the union
  of evidence at nodes ≤ k), the flat ledger is `derive_ledger()`'s concatenation, and a
  rendered journey therefore reads in exactly the order the knowledge grew (LLM-context
  first: the primary consumer is a model ingesting it as prompt material);
- **options are captured prospectively**: the task-decomposer prompts ask for an optional
  `alternatives` list (≤3, `{method, why_not_now}`) — never binding, dropped silently
  when malformed — so `not_taken` options are what was recorded *at the time*, rendered
  as "options on the table (as recorded)", never "all possible options".

Renders degrade `FULL` (+ Mermaid tail for humans) → `COMPACT` (retry/delegate nodes
collapse to one line; what the hindsight judge consumes) → `SPINE`; the citation
skeleton (chain, taken/not_taken, fps) survives every level, and all free text is
hard-capped so a journey can never blow out a consumer's prompt budget. When feeding an
LLM, use `render_journey(journey, level)` / journey.json directly — report.md is the
human file (its legacy ledger appendix intentionally repeats the render's evidence).

**Hindsight** (`retro_envelope.py` + `run_hindsight`): on `success` + full route +
leftover budget only, one oneshot judges "was there a more optimal path?" against the
COMPACT render. Its claims must cite node keys and evidence fps (code-validated, one
violation-echo retry, then dropped); pure positional logic stamps each avoidable-branch
claim **genuinely-avoidable** (a recorded `not_taken` option whose enabling evidence was
already known) / **blind-spot** (evidence known, option never recorded) /
**honest-exploration** (the evidence didn't exist yet — the dead end had to be run to be
learned; the system working as designed). INVARIANT: hindsight is advisory ink — it can
never un-succeed a run (every failure mode is a `{"skipped": reason}` receipt) and never
triggers execution. Valid `promoted_learnings` ride into the knowledge-plane promotion
(source=retro), so later runs on the same project seed with hindsight baked in.

## Design notes

- Dead-end fingerprints key on the **method label**, not the reason — a method dying twice
  with fresh wording is the flap the information-dry guard exists to catch. An early dry stop
  therefore means "no new methods and no new facts", not "no new error text".
- Task verdicts are self-reported artifacts read by code — no LLM judge. The plan contract's
  mandatory final verification task bounds optimistic self-reports; what slips through is
  caught by the next cycle's ledger.
- **LEVEL 2's local retry and LEVEL 1's staleness gate own exactly one decision each** —
  LEVEL 2 asks "does THIS task get another local shot"; LEVEL 1 asks "does the REMAINING
  to-do list still hold." Neither changes LEVEL 0's cycle-boundary behavior: escalation on
  local-retry exhaustion folds the identical dead-end record a single-attempt failure
  always did, and a mid-cycle `needs_decision`/`exhausted` from a partial replan reuses the
  same assume-and-note/fold vocabulary as the top-level fork.
- The staleness gate is deliberately **pure code, no LLM** — dead-method-fp reuse, a cheap
  stopword-filtered keyword overlap between fresh evidence and a remaining task's
  `method`/`description`/`success_criterion`/`intent_link`, or unconditional local-retry
  exhaustion. False positives cost one extra oneshot (capped); false negatives fall through
  to the next full cycle's clarify+replan — the pre-existing safety net.
- **`learnings` are asked for as mini post-mortems, not one-liners** — the executor prompt
  asks for the systems/materials involved, the hypothesis going in (what/why we expected
  success), what actually happened, and why it succeeded or failed, so a learning read out
  of context by a later clarify/plan call is still self-contained and actionable. This is
  distinct from `evidence` (strictly "what you observed vs the criterion") — `learnings`
  is for everything else worth remembering.
- **Intent stays mechanics-free by construction**: `render()`/`render_partial()` only ever
  append ledger-derived sections below the verbatim intent — no fold function anywhere
  (including LEVEL 2's scoped-clarify fold) mutates `prompt` itself. The task-decomposer
  schema's `intent_link` field is the one intentional, bounded bridge from task-level
  mechanics back to intent-level meaning; it never touches the intent text.
- **LEVEL 1/2's retry→backtrack→cap policy deliberately parallels `method-explorer`'s
  AND/OR search**, at a lighter weight and a different grain: LEVEL 2's local retry ≈
  method-explorer's cheap-retry rungs, LEVEL 1's staleness-gate-triggered partial replan
  ≈ its backtrack step, `dead_fps`/`stale_tail` trigger A ≈ its dead-set tombstoning,
  `MAX_REPLANS_PER_CYCLE`/the mid-cycle `deadline` ≈ its GUARD-HALT cap. This is a known,
  accepted parallel, not an oversight — method-explorer's own docs explicitly exclude
  "tasks with exactly one method," which is precisely what LEVEL 2 handles, so the two
  aren't solving the same problem even though the *shape* recurs. When a single task's
  local-retry budget is exhausted, LEVEL 2 can hand that ONE task to a scoped
  `method-explorer` sub-run (`run_task_delegation`) instead of relying on LEVEL 1's
  task-decomposer-oneshot guess at a new method — see DESIGN.md for the delegation contract,
  budget, and fallback.
- Full design + locked decisions: `src/hermes/skills/relentless-solve/DESIGN.md` (staging).
