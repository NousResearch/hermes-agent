# relentless-solve — DESIGN (v4, 2026-07-02)

*Family-wide roles, state contracts, call graph, and isolation rules: `skills/ARCHITECTURE.md`.*

## Scope mode + subroutine posture (2026-07-03, topology B)

`scope` runs the CLARIFY→PLAN prefix only and stops at the proposal — the pipeline
prefix as a product for a human who will author the work themselves. It is its OWN
engine flow (`flow(id="relentless-scope", version=2)(scope_flow)`, state at
`<slug>/scope-flow`, artifacts under `<slug>/scope/`) so `relentless_flow`'s hashed
source stays byte-identical — no `--accept-flow-change` fallout for in-flight solve
runs; the same constraint shaped every topology-B addition: global-tier seeding lives
in `run_clarify`, promotion in `write_report`/`write_scope_package`, the recursion
guard and `solve.json` in the cmd handlers — all helper source, none of it hashed.
Rationale for the posture (dev loop above, relentless as per-problem subroutine, the
`RELENTLESS_ACTIVE` exit-4 guard, `needs_split` instead of recursion, the global
knowledge tier with its two poisoning guards) lives in ARCHITECTURE.md; mechanics in
SKILL.md's "scope" and "Subroutine posture" sections.

Containment revision (Codex cross-model review, 2026-07-03): the original content-blind
porcelain-only baseline was replaced by full receipts (porcelain + HEAD + a
baseline-commit diff hash), so a same-shape edit or a hidden commit no longer slips
past detection; evidence is archived before any destructive reset (evidence-before-
destruction); and `clean_ledger` gives one ledger projection reused by clarify seeding,
plan rendering, AND knowledge promotion, so a tainted round can't leak through any of
the three paths individually. This is why the flow bumped to version 2 above.

One script that takes a prompt and does not let go of it: clarify → plan → execute per
task → harvest verdicts as new evidence → re-clarify → re-plan, until the intent is
satisfied or the search space is *provably* dry. It is a thin orchestrator — every hard
problem is solved by an existing skill; this script only owns the loop, the evidence
ledger, and the stop-conditions.

## Building blocks (all exist today)

| Role in the loop | Skill / script | Contract used |
|---|---|---|
| "next-questions" | `investigator` — `skills/autonomous-ai-agents/investigator/scripts/iterate.py` (wraps `information-gain` EVSI ranker) | in-process: `apply_capability(cfg, capability)` + `iterate(problem, cfg, seed_evidence=[...])` → `{tombstones: [{status, question, evidence}], stop_reason, n_answered, n_gaps}`; cfg keys `k`, `max_rounds`, `floor`, `answer_cwd`, capability `act\|experiment\|read` (the `--problem <text> --json` CLI mirrors it). Pinned by InvestigatorContract |
| "task decomposer" | `task-decomposer` — `scripts/planfile.py` (schema+validate) + `scripts/envelope.py` (invocation contract) | one `hermes -z` oneshot: (intent + rendered ledger) → validated `plan.json` at `relentless/<slug>/c<N>/` — ordered oneshot-sized tasks with success criteria, or `needs_decision` / `exhausted`. Bounded violation-echo retries. Pinned by PlanContract |
| per-task executor | `relentless.py run_intent_path` (LEVEL 1) → `run_task_with_local_retry` (LEVEL 2) → `run_task` — one `hermes -z` oneshot per attempt | executor writes `result-<id>[-retry<K>].json` `{verdict: worked\|failed, evidence}` judged against the task's success_criterion; CODE reads the artifact (missing/malformed ⇒ failed — disk beats stdout). LEVEL 2 bounds local reattempts (own scoped clarify each time); LEVEL 1 evaluates a pure-code staleness gate after every task and may request a mid-cycle `replan-<seq>.json` for the untouched tail |
| durable outer loop | `resumable-script` engine (`skills/resumable-script/scripts/engine.py`) | flow = ordinary function; each phase a memoized `ctx.step(key, fn)`; human gates via `ctx.ask` (exit 10, resume with `--answer`) |

(`method-explorer` + `drive.py` remain the engine of the solve `single_method` route —
one AND/OR backtracking run when the gate says there is exactly one clear method.)

## The loop

```
input: { prompt, slug, k=6, inv_rounds=3, max_cycles=5, wall_budget }

evidence = []          # append-only ledger: facts, gaps, dead-ends (all tombstone-shaped)
seen     = set()       # fingerprints of records already folded (anti-flap)

for cycle in 0..max_cycles-1:

  # A — CLARIFY (next-questions)
  inv = ctx.step(f"c{cycle}/clarify", iterate(render(prompt, evidence), ...))
  evidence += inv.tombstones                      # answered facts + known gaps

  # B — PLAN (plan-as-data)
  plan = ctx.step(f"c{cycle}/plan",
                  task-decomposer oneshot(render(prompt, evidence)) -> c<N>/plan.json)

  # E — the planner's human fork / honest exhaustion
  if plan.disposition == needs_decision:
      gap/ask(f"c{cycle}/fork", plan.question)    # assume-and-note, or suspend under --gate
  elif plan.disposition == exhausted:
      evidence += fact("planner declared exhaustion")

  # C — EXECUTE per task + HARVEST verdicts as state
  else:
    for task in plan.tasks:                       # ids from the MEMOIZED plan step ⇒
        skip if a dependency failed               #   replay-deterministic step keys
        r = ctx.step(f"c{cycle}/t/{task.id}", run_task(task))   # one hermes -z each
    evidence += harvest_tasks(plan, results)      # failed → dead-end, worked → fact
    if all results worked: return report(success) # final task = intent verification

  # D — STOP HONESTLY (relentless ≠ flailing)
  if zero fresh evidence anywhere and clarify converged:
      return report(information-dry)              # no new information → dry

return report(max-cycles / wallclock, resumable)
```

## Locked decisions

1. **Intent is immutable; evidence grows.** The original prompt is never rewritten
   (mirrors method-explorer's intent/method split). "Refining the prompt" =
   re-rendering it with the evidence ledger appended. This keeps EVSI re-ranking honest:
   answered questions become derivable and self-retire (the U-gate), dead ends suppress
   re-planning down the same branch.

2. **Prompt rendering** (`render(prompt, evidence)`):
   ```
   <original prompt verbatim>

   ## Established facts (do not re-derive)
   - <found tombstones>
   ## Known gaps (proceed on the stated assumption)
   - <gap tombstones + assumption>
   ## Dead ends — do NOT re-attempt these methods
   - Tried <branch>: failed — <reason>     (from prior cycles' ✝ dead-sets)
   ```

3. **Fresh plan.json per cycle** (`relentless/<slug>/c<N>/`), dead-ends injected via
   the rendered ledger — NEVER resume a stale plan. Rationale: the refined prompt
   materially changes the method space; a fresh plan constrained by the prior dead-set
   is cleaner than patching a plan authored under weaker knowledge. The engine's
   memoized steps handle *intra*-cycle interruption (a crash resumes at the first
   un-journaled task).

4. **Failure conditions are tombstone-shaped evidence.** A failed task's verdict becomes
   exactly the kind of fact `information-gain --evidence` was built to fold in
   (verified live: answering a question retires it and surfaces the next-best — same
   mechanism, the "answer" here is "that method is dead and why"). Dead-end
   fingerprints key on the task's METHOD label, not the failure evidence.

5. **Stop conditions, in order of honor:**
   - `SUCCESS` → done.
   - **Information-dry:** a full cycle yields zero fresh facts (investigator converged
     with nothing new AND harvest is a subset of `seen`) → report genuine exhaustion.
     This is the anti-flap guard: relentless means never stopping *while there is new
     information*, not looping on the same failure.
   - Hard caps: `max_cycles` (default 5), wall-clock budget → stop `active`, resumable.

6. **The outer script is itself a resumable-script flow.** Each phase is a `ctx.step`
   (a crash mid-cycle replays completed phases from the journal, exactly-once); the
   only `ctx.ask` is the GUARD-HALT fork that grounded research cannot resolve —
   user-only constraints are the one thing the loop may block on, and it blocks
   *suspended*, not spinning. Everything else self-answers (investigator default
   `--capability act`).

7. **Runs in-container** (`docker exec hermes`, uid 10000, `HERMES_HOME=/opt/data`).
   State under `${HERMES_HOME}/relentless/<slug>/` (journal, rendered prompts, report) —
   NOT `/tmp` (write_file guard). Both sub-scripts already run there.

8. **Plans are data; control flow stays code** (v2). Plan *content* comes from the
   task-decomposer oneshot; ordering, dependency skips, verdict reading, and stop rules
   stay in relentless.py. The plan contract makes the FINAL task an intent-verification
   task, so "SUCCESS = all tasks verified worked" is a pure-code equivalence. Verdicts
   are self-reported per task via a `result-<id>.json` artifact (missing/malformed ⇒
   failed) — no LLM judge; the mandatory verification task bounds optimistic
   self-reports, and the next cycle's ledger catches what slips through.

## File layout (implementation target)

```
skills/autonomous-ai-agents/relentless-solve/     # hermes-agent repo, next to investigator
  SKILL.md
  scripts/relentless.py      # the flow (engine imported from resumable-script skill)
  scripts/harvest.py         # task results → ledger records fold (pure, unit-testable)
  tests/test_harvest.py      # pin the worked/failed/skipped folds + fp-on-method rule
  tests/test_loop.py         # scripted plan/task fakes, no container
  tests/test_plan_request.py # the request_plan/run_task seams (scripted invoke_hermes)
skills/autonomous-ai-agents/task-decomposer/         # the plan-as-data sibling (v2)
  SKILL.md
  scripts/planfile.py        # schema constants + validate + load/dump (pure)
  scripts/envelope.py        # plan_prompt/retry_suffix — the oneshot invocation contract
  tests/test_planfile.py     # validation + envelope wording; fixtures/plan-golden.json
```

## Isolation posture (hardened 2026-07-01, commit a20bb3045)

Every skill in the stack is importable + unit-testable standalone: pipeline.py's model_utils
(ask) import and iterate.py's infogain import are graceful (raise/degrade at call time, not
import time — pipeline's raw_chat keeps its error-as-data contract); relentless.py imports the
investigator lazily (`_investigator()`, cfg-building folded into `run_clarify(problem, seeds,
inp)`). Proof: A/B/C suites pass with ASK_SCRIPTS_DIR and INFOGAIN_SCRIPTS_DIR masked.
Drift surfaces pinned by CONTRACT TESTS (runtime stays decoupled), in relentless-solve tests/:
PlanContract (relentless._planner() resolves the SAME planfile.py/envelope.py the task-decomposer
skill ships; the golden plan validates; the planning prompt names the exact plan.json path
request_plan reads back); EnvelopeContract (the solve `single_method` seam —
_envelope().real_prompt resolution + the extra= kwarg solve_single pins risk through);
EngineContract (relentless_flow under the REAL engine: result parity with FakeCtx,
replay-executes-nothing, gate suspend→resume — already caught that engine resume replays the
prefix so scripted fakes feed only the live tail). All contract tests SKIP when the
counterpart skill is absent. (v1's GrammarContract died with tree harvesting — no plan-tree
parsing remains in this skill.) Known flag: method-explorer staging (src/hermes,
un-versioned) vs deployed (~/.hermes/skills) have diverged (SKILL.md, scenario assets,
tests/) — unresolved by choice.

## Layering (agreed 2026-07-01)

Four layers — intelligence at the leaves, determinism at the top:

```
L3  relentless.py — the loop            code only, no LLM calls (the `solve` entry's
    owns: cycle control, stop rules,    routing gate makes exactly ONE, receipted —
          evidence ledger, the one      see "the `solve` entry" below; the loop
          human gate                    itself stays LLM-free)
L2  clarify: investigator iterate.py    plan: task-decomposer envelope + planfile
                                        execute: one hermes -z oneshot per task
L1  information-gain ranker             task-decomposer SKILL.md (plan-as-data contract)
L0  hermes -z runtime + ollama models
```

Boundary rules: (1) code decides control flow, models decide content (the drive.py lesson);
(2) one blackboard — the ledger is the only cross-layer state; clarify = ledger→ledger',
plan = ledger→plan.json, execute = plan→results, harvest = results→ledger' (pure); (3)
every layer terminates on its own; (4) the human gate lives only at L3. Rejected: agent-run
loop (turn-cap trap), merging execution into investigator (blast radius + termination
semantics + reusability). State-store roles stay strict: engine journal = replay mechanism
(never read by domain code), ledger = domain truth (the only thing rendered into prompts),
per-cycle `c<N>/{plan.json, result-*.json}` = scratch receipts (folded once from in-memory
results, then inert). Prerequisite this demanded: `seed_evidence`/`--evidence-file` on
iterate.py so cycle N+1's clarify conditions on the ledger through the ranker's proper
evidence path.

## Resolved (was Open) — decided by jim 2026-07-01, implemented in v1

- GUARD-HALT fork default = **assume-and-note** (never block): the open fork is recorded as a
  gap and the next clarify round ranks it via EVSI (no separate fork-investigate pass — one
  mechanism). `--gate` opts INTO `ctx.ask` suspension instead.
- Budget shape: outer `--wallclock` checked at cycle boundaries + drive.py's own per-run
  backstops within a cycle (`--drive-wallclock ≲ wallclock/max_cycles`).
- Cycle 0 **always clarifies** (no bucket=0 skip — one code path; convergence is cheap on
  tight prompts).
- Implemented at `hermes-agent/skills/autonomous-ai-agents/relentless-solve/` (branch
  feature/hybrid-port): scripts/relentless.py (flow+CLI), scripts/harvest.py (pure parser),
  tests (harvest fixtures from real plan-trees + FakeCtx loop tests incl. replay determinism),
  SKILL.md. Prereq landed in investigator/scripts/iterate.py. resumable-script synced to
  ~/.hermes/skills/ for in-container engine import.

## Resolved — the `solve` entry (decided by jim + implemented 2026-07-01)

The one-argument front door: `solve --prompt TEXT [--budget SECS] [--risk act|experiment|read]
[--gate]`. Interface principle: **a knob is a confession the system can't read the evidence** —
every parameter is derived (slug, route, constraints), conventional (paths), or adaptive
(budget shares); the four flags above are the honest residue (preferences evidence can't decide).

- **Gate ownership**: solve owns triage. One cheap `hermes -z` classify → `trivial` |
  `single_method` | `full`; **any gate failure (error/garbage/unknown verdict) → `full`** —
  misrouting trivial wastes a few calls, misrouting hard silently under-serves (asymmetric risk).
- **gate.json idempotence**: the verdict persists to `relentless/<slug>/gate.json` and is
  REUSED on re-invocation (never re-classified; `--route` forces, `--gate-only` inspects).
  Deterministic slug (kebab intent-nouns) is what makes re-invocations land on the same state.
- **Budget = cascading shares of ONE pool**: full route recomputes execute's share each cycle
  boundary (~80% of remaining/cycles-left, floor 300s, ceiling the static drive default) from
  the memoized clock — replay-safe, and unspent time flows back. Evidence stops stay primary.
- **Receipted defaults**: an invisible default is a bug — gate verdict + derivations in
  gate.json, the verdict as the full route's first ledger row, and a report.md header
  (slug/route/budget/risk/stop) on every route.
- **Envelope de-dup**: the runtime copy of the planner invocation was replaced by a direct
  import of method-explorer/scripts/envelope.py (same-repo sibling = same layout in-container);
  tests/test_contracts.py remains the drift tripwire (now executing, not skipping).
- **Deferred (separately gauntlet-gated inner-loop levers, from the 2026-07-01 design
  discussion)**: (a) reason-diversity weighting for upstream jumps — K deaths sharing ONE root
  cause is the real jump signal, not the count; (b) probe-aware candidate scoring — add
  discriminative value (EVSI) to next-move selection so the loop can buy information when
  failure-level attribution is ambiguous.

## v1 → v2 — plan-as-data (decided by jim + implemented 2026-07-01)

The plan-generation phase was extracted from the loop into a standalone **task-decomposer**
skill: (immutable intent + rendered ledger) → one validated `plan.json` per cycle; the
loop became a per-task driver (one `hermes -z` oneshot per task, verdict via a
`result-<id>.json` artifact read by code). Decision set:

- **Refactor, not a sibling**: relentless-solve itself is the driver around the swappable
  planner (the swap seam is the planfile/envelope module contract + `TASK_DECOMPOSER_DIR`).
  method-explorer + drive.py left the full route entirely; they remain the
  `single_method` route's engine.
- **Plan format**: new tasks-JSON schema (task-decomposer `planfile.py`, schema 1) —
  id/method/description/success_criterion/depends_on/status, plus plan-level
  `disposition` (`tasks` | `needs_decision` | `exhausted`). `needs_decision` replaced the
  GUARD-HALT fork as the loop's one human gate; `exhausted` replaced EXHAUSTION-STOP.
- **Execution**: one oneshot per task, fine-grained worked/failed feedback per cycle;
  dependency-failed tasks are skipped (no ledger record — the blocker recorded the
  dead-end). `method` stays the anti-flap fp identity; worked facts fingerprint as
  `"ok <method>"` so fail-then-work records both transitions.
- **Flow version bumped to 2** (key topology changed to `c<N>/plan` + `c<N>/t/<id>`):
  v1 journals refuse to resume instead of skewing. Runs started before v2 cannot resume;
  re-run `solve` — gate.json is reused.
- **Deferred**: a cheap per-cycle judge oneshot if optimistic self-reports bite in
  practice (the mandatory final verification task + next cycle's ledger bound them for
  now); prompt iteration on the envelope's "oneshot-sized tasks" rule is expected.

## v2 → v3 — nested per-task retry + mid-cycle staleness replan (decided by jim, implemented 2026-07-02)

The same "clarify → evaluate sufficiency → proceed or escalate" primitive that governs the
outer cycle turns out to apply recursively one and two levels down: after every task, does
the REMAINING to-do list still hold (LEVEL 1); inside one task's execution, does IT deserve
another local shot before its failure escalates (LEVEL 2). Neither level existed before v3
— a task got exactly one attempt, and any correction only happened at the next full cycle
boundary. Four locked decisions:

- **LEVEL 2 exists**: `run_task_with_local_retry` — on failure, a clarify SCOPED to why
  THIS task failed (built from task fields only, never `inp["prompt"]`), fold the fact
  into the shared ledger, reattempt the SAME task id. Bounded by `local_retry_budget`
  (default 2, `--local-retry-budget` on `run`, DEFAULTS-only elsewhere — not a `solve`
  flag, matching the "a knob is a confession" posture). On exhaustion, escalation is
  UNCHANGED: the failure folds into `harvest.harvest_tasks()` exactly as a single-attempt
  failure always did — byte-identical dead-end record, so decision #2 below holds by
  construction, not by promise.
- **The global assume-and-note default stays UNCHANGED** — no new default human-prompting
  behavior anywhere, including LEVEL 1's own fork: `needs_decision` from a mid-cycle
  partial replan ALWAYS assume-and-notes, even under `--gate`. The human-suspend gate
  stays confined to L3 (layering rule #4, above) — a second suspend point nested inside
  task execution would need continuation/resume semantics that don't exist and aren't
  worth inventing for a mid-cycle optimization. If the fork persists, it resurfaces at
  the next cycle boundary, where `--gate` already works as tested.
- **The staleness check is a cheap gate first, replan only on trigger**:
  `stale_tail` (LEVEL 1) is pure code, no LLM — three triggers, any one sufficient:
  dead-method-fp reuse against the ledger's dead-ends (exact), stopword-filtered keyword
  overlap between fresh evidence and a remaining task's
  method/description/success_criterion/intent_link (blunt but cheap), or unconditional
  local-retry exhaustion. Only a positive trigger costs a real oneshot
  (`request_partial_replan`, hard-capped at `MAX_REPLANS_PER_CYCLE`=3/cycle); false
  negatives fall through to the pre-existing next-cycle clarify+replan safety net.
  `request_partial_replan` writes its OWN `c<N>/replan-<seq>.json` — it never overwrites
  `c<N>/plan.json` — and asks the model only for the new tail (not a verbatim
  reproduction of completed tasks); the driver splices `tasks[:i+1] + result["tasks"]` in
  code. A replan that fails validation repeatedly, or throws, becomes a non-fatal
  `{"disposition": "replan-failed"}` sentinel (`_attempt_partial_replan`) — a failed
  OPTIMIZATION must never sink a cycle that could otherwise finish its original tail.
- **Definition-of-done + intent/mechanics separation, enforced in the SCHEMA**:
  `planfile.SCHEMA_VERSION` bumped 1→2 for a new required per-task field, `intent_link`
  — deliberately NOT folded into the existing plan-level `rationale` (which explains the
  whole decomposition; `intent_link` explains why THIS task, at task granularity — same
  name at two levels would read ambiguously in flattened JSON/logs) and deliberately NOT
  merged into `description` (which stays the executor's literal, verbatim instruction).
  `intent_link` is what LEVEL 1's staleness gate keys its vocabulary-bleed trigger on.
  `envelope.py`'s `plan_prompt`/`partial_replan_prompt` share one `_TASK_RULES` block
  (can't drift apart) that also tightens `success_criterion` to a STRICT,
  objectively-checkable definition of done (with an explicit good/bad example) and states
  the intent/mechanics boundary explicitly for the model reading the rendered body. This
  is a documentation/enforcement addition, not a behavior change — `render()` already
  never mutated `prompt`, confirmed while auditing every fold function (`fold_clarify`,
  `fold_one`, `fold_records`, and the new LEVEL 2 scoped-clarify fold all only ever
  `ledger.append(...)`).

**Executor `learnings` (added same session, no flow-version impact — purely additive data,
same step keys)**: a task attempt may optionally report `learnings` — free-form incidental
facts beyond the pass/fail `evidence` line, regardless of verdict. The prompt asks for a
self-contained mini post-mortem (systems/materials involved, the hypothesis going in, what
happened, why it succeeded or failed) so a learning read out of context later is still
actionable. `run_task_with_local_retry` accumulates learnings across EVERY attempt (a
failed attempt's learning isn't lost once a later attempt recovers); `run_intent_path`
merges them onto the harvested result so each one folds as its own `fact` record
(`harvest.harvest_tasks`, fp'd on its own text) — visible to the next clarify/plan call
exactly like any other ledger fact. They also feed `stale_tail`'s vocabulary-bleed trigger,
so a learning from a *successful* task can flag a downstream task as stale before it runs.
Capped (`LEARNINGS_MAX_COUNT`=5, `LEARNINGS_MAX_CHARS`=500) as a runaway-ledger backstop.

**LEVEL 1's pre-replan clarify (added same session)**: mapping intent to task should
always leverage whatever's already known (the ledger — empty the first time) and be
refined by a next-best-questions pass before the mapping happens. LEVEL 0 already did
this (`run_clarify` before `request_plan`) and LEVEL 2 already did this
(`run_scoped_clarify` before a local retry) — LEVEL 1's partial replan was the one
mapping moment that skipped it, replanning straight off whatever was already in the
ledger. Closed by adding `run_replan_clarify` (delegates to `run_clarify` — the SAME
investigator primitive, not a new skill; `investigator` already IS the "wrapper that
takes a problem, ranks next-best-questions, gathers answers, refines" — no reason to
duplicate it), scoped to the fresh evidence/learnings that tripped the staleness gate
plus the remaining tasks about to be replanned, using LEVEL 2's tighter local_k/
local_inv_rounds. Its own `clarify-scoped/replan<seq>-after-<id>/` run_dir gives it the
same tombstones.jsonl resume durability as every other clarify call; its facts fold with
`source="replan-clarify"` before `render_partial` builds the replan prompt, so the
partial replan itself sees them. No flow-version impact — new step key
(`c<N>/replan/after-<id>/clarify`, inserted before the existing `c<N>/replan/after-<id>`)
but no change to the retry/replan control-flow shape already covered by the v3 bump.

Also decided during implementation (not user-facing product decisions, but load-bearing):
mid-cycle wallclock backstop (`deadline`, cascade-only) is checked ONLY right before an
actual retry or replan attempt — never before a task's own first attempt (already bounded
by `task_to`) — so a cycle where nothing fails or replans issues zero new clock reads;
budget cascade becomes a three-way split (plan 20% / task attempts 70%, divided by
`n_tasks * (1 + local_retry_budget)` / mid-cycle replans 10%, `REPLAN_TO_FLOOR/CAP`=60/240,
`MAX_REPLANS_PER_CYCLE`=3); flow version bumped to 3 (the engine's own `flow_hash()` +
strict-replay guard already force `--accept-flow-change` / `NonDeterminism` correctly on
this upgrade — the version bump is the project's documented convention, matching the
v1→v2 precedent, not the actual enforcement mechanism); `harvest.py` needed ZERO changes
(it already only reads `plan.get("tasks")` + a results list — LEVEL 1 calls it per-task
with the current, possibly-replanned task list, same signature, same fp/anti-flap
namespace).

## v3 → v4 — LEVEL 2 delegates to method-explorer on exhaustion (decided by jim, implemented 2026-07-02)

A skill-family naming/overlap audit (rename `task-planner`→`task-decomposer`; dedup the
`run_scoped_clarify`/`run_replan_clarify` boilerplate; extract the `invoke_hermes`/
`drive.py`-`invoke` bare-oneshot pattern into `resumable-script/scripts/oneshot.py`)
surfaced a real finding: LEVEL 1/2's local-retry→staleness-gate→partial-replan→cap
machinery independently converges on the SAME diagnose→retry→backtrack→cap *policy*
`method-explorer` already owns, for a genuinely different *domain* — method-explorer
explicitly excludes "tasks with exactly one method" (LEVEL 2's whole job), while LEVEL
1's task-decomposer-oneshot partial replan has no disciplined search over alternatives, no
scored frontier, no anti-flap tombstoning beyond a flat dead-end-fp set (method-explorer
has all three). A full design pass produced a buildable per-task delegation mechanism;
the investigation's own honest recommendation was to DEFER it (this exact coupling was
deliberately removed from the full route in the v1→v2 redesign; cost/value at single-task
grain was unconfirmed) — **jim overrode that recommendation and asked to build it now.**

- **Trigger**: only after LEVEL 2's `local_retry_budget` is exhausted (not
  deadline-starved — no point starting an expensive sub-run with no wallclock left), AND
  a cheap one-call classifier (`rp_delegation_gate`, same try/except-degrades-safely
  shape as `classify()`) judges the failure has plausible alternative methods rather than
  looking environmental/unfixable by any method. ANY gate failure (parse error,
  exception, timeout) → no delegation — never escalate into an expensive sub-run on
  ambiguity. Deliberately NOT a new `plan.json` schema field: the signal is often only
  knowable after seeing the failure evidence, not at plan-authoring time, and a field
  would mean coordinating a schema bump across `task-decomposer` too.
- **Scope**: `task_delegation_intent` builds method-explorer's `intent` from TASK
  FIELDS ONLY (method/description/success_criterion/intent_link) plus the failed method
  named as a known-dead approach — never touches `inp["prompt"]`, same intent/mechanics
  separation as `scoped_clarify_problem`.
- **Invocation**: `run_task_delegation` reuses the EXACT contract `solve_single` already
  uses for the whole-intent case (`_envelope().real_prompt` + `run_drive`) — no new
  invocation contract invented. Its own scoped slug
  (`<slug>-c<cycle>-<task-id>`, derived from `cycle_dir`'s basename) can't collide with
  the whole-intent `<slug>-single` route or across cycles.
- **Budget**: `DEFAULTS["task_drive"]` (max_ticks=4, per_tick_timeout=300, wallclock=900)
  — static, deliberately far smaller than the whole-intent `DEFAULTS["drive"]`, NOT a new
  cascade fraction (avoids a 4th cascade dimension the investigation warned about).
  `MAX_RP_DELEGATIONS_PER_CYCLE=1` mirrors `MAX_REPLANS_PER_CYCLE`'s cap pattern but is a
  fully independent counter, owned by `run_intent_path` (`delegations_used`, threaded
  into `run_task_with_local_retry` as `allow_delegation`) — a delegation conceptually
  substitutes for what would otherwise become a LEVEL 1 partial-replan attempt for that
  task, so no separate cascade math was added for it.
- **Outcome handling** (inside `run_task_with_local_retry`, right where the retry loop's
  exhaustion is detected): on drive `SUCCESS`, the task's `result` is overridden to
  `worked` and a `learnings` entry names the alternate method found (reusing the
  already-shipped learnings machinery — no new fold path) — this un-sets `exhausted`, so
  LEVEL 1 never even sees this task as needing a replan. On any OTHER terminal
  (EXHAUSTION-STOP/GUARD-HALT/error), `result`/`exhausted` are left UNCHANGED and a
  richer dead-end note folds into the ledger — LEVEL 1's existing partial-replan path
  fires exactly as it did before this feature existed, just with better dead-set context.
- **Fallback, never a hard dependency**: `_attempt_task_delegation` checks `_envelope()`
  availability FIRST and explicitly catches the `SystemExit` it raises when
  method-explorer isn't on disk (`SystemExit` is a `BaseException`, not caught by a
  bare `except Exception` — a real gap the naive first draft had, since this call can be
  the FIRST thing in a `full` route run to ever touch method-explorer, unlike
  `request_partial_replan`'s equivalent risk with `task-decomposer`, which is always
  already-resolved by the time a replan fires) → `{"disposition":
  "delegation-unavailable"}`. Any other runtime failure (subprocess, bad JSON) →
  `{"disposition": "delegation-failed"}`. Both degrade identically: the original failed
  `result` is returned unchanged.
- **Flow version bumped to 4 — for a DIFFERENT reason than v2→v3.** This change lives
  entirely in `run_task_with_local_retry`/`run_intent_path`, not in `relentless_flow`'s
  own body, so the engine's `flow_hash()` (which only hashes `relentless_flow`'s own
  source via `inspect.getsource`) does NOT change and will NOT force
  `--accept-flow-change` on its own this time — unlike v2→v3, where the changed function
  WAS `relentless_flow` itself. The version bump is therefore the only signal; the actual
  safety net is the engine's separate strict-replay key-sequence check, which raises
  `NonDeterminism` (exit 3) if a v3 journal's replay ever reaches a point that now wants
  an `rp-delegate` step the old journal doesn't have.
- **Test-infrastructure lesson** (worth recording since it cost real debugging time): a
  pre-existing `self.setUp()`-called-twice-mid-test pattern in `test_gate_suspends_then_
  answer_continues` silently corrupted `self._orig` (capturing the ALREADY-patched value
  instead of the true original), so `tearDown()` left a stale mock in
  `relentless._maybe_delegate_task` for whatever test ran next in the same process. This
  was latent and harmless for every OTHER patched name (every test that cares about them
  calls `wire()` to set its own fresh fake regardless of what was left behind) — LEVEL 2
  delegation was the first mechanism where a DIFFERENT test's silent leftover value
  actually mattered. Fixed at the root (reset the instance-level capture lists directly
  instead of re-calling `setUp()`), not worked around per-test.

## v4 + --dod — the skill-family dedup verdict lands here (jim's decisions, 2026-07-02)

A duplication audit across the six staging skills found ONE genuine duplicate role:
`intent-to-tasks` (built 2026-07-01 as the WHAT/HOW pair's mapper) and `task-decomposer`
were two renderings of the same "(intent + failure knowledge) → candidate tasks"
transformer — same three evidence sections in, byte-identical anti-flap `fp()`, same
exhaustion verdict, same reinvoked-per-cycle contract; they differed only in artifact
grammar (taskmap.md vs plan.json) and in whether a dod.md fronted the intent. Verified
NOT duplicative in the same audit: investigator vs next-best-questions (ranker vs
resolver wrapping it) and this skill's LEVEL 1/2 vs method-explorer (already resolved
by the v4 delegation; method-explorer uniquely owns scored-frontier alternative-method
search). Jim's calls: fold intent-to-tasks INTO task-decomposer and retire it; the
completion contract lives in task-decomposer; wire define-done consumption now.

What landed in THIS skill:

- **`--dod path/to/dod.md`** (run + solve, full route only). The dod TEXT rides inside
  the immutable engine input (read+linted at CLI time, `_load_dod`), so every play —
  live or replay — parses the exact spec the run started with; parse_dod is pure, so
  this adds NO step key. `dodctx` ({parsed, unmet, known, section}) is derived at the
  top of `relentless_flow` from `inp["dod"]` via the lazy `_spec()` loader
  (`DEFINE_DONE_DIR` → sibling → deployed — the same resolution shape as `_planner()`).
- **Binding validation** through the EXISTING `_plan_attempt_loop` retry-echo channel
  (`extra_checks`): `planfile.coverage_violations` (every unmet R-id served, no
  dangling serves) and `planfile.dead_violations` (never re-propose a dead method —
  the old prompt-convention made contractual; `dead_fps` is a pure fold of the ledger).
  The partial-replan path passes the TAIL's unmet ids (head-served ids subtracted).
- **Per-cycle completion contract**: `_reporter()` (task-decomposer's report.py, loaded
  from the SAME dir `_planner()` resolved so schema/report can't drift) computes
  `completion_report(plan, results, dod_parsed, knowledge_in_fps=k_in, cycle)` — `k_in`
  is a frozenset snapshot of `seen` taken right before the plan step, so the report's
  delta is exactly what the cycle taught. Written inside the existing `c<N>/plan-out`
  step as `c<N>/report.json`; the final `report.md` opens with the requirements rollup.
- **`harvest.py` deliberately UNCHANGED**: it stays this skill's own copy of the fold
  (standalone testability), now pinned behaviorally from the task-decomposer side
  (`task-decomposer/tests/test_contracts.py::HarvestContract` — fp parity + record
  parity modulo `source`).
- **NO flow-version bump**: the key topology is identical with and without --dod (no
  new ctx.step anywhere — dodctx/k_in/dead_fps are derived values; report.json is
  written inside plan-out). relentless_flow's source DID change, so the engine's
  flow_hash forces `--accept-flow-change` on in-flight v4 journals — a code-edit
  consequence, not a topology change, hence no v5 (the v2→v3/v3→v4 bumps were
  topology).

## v4 → v5 — the journey record + post-success hindsight (decided by jim, implemented 2026-07-03)

The requirement, in jim's words: keep the raw journal of activities, but also produce a
CONSOLIDATED journal — the facts we knew, the environment/context and the location at
which we stood, the options that existed at each state point (including the ones not
taken), the successful path front and center — and, on success, ask "was there a more
optimal way?" grounded in that record. Five design rounds converged on ONE primitive.

- **Node = (evidence, options, taken)** — journey.json is a CHAIN of decision nodes
  (kinds: plan / replan / retry / delegate / terminal), one wherever the plan of record
  changed. jim's consolidation insight closed the design: **failed paths ARE evidence**
  (harvest already folds a failed task into a dead-end record), so a failed branch is
  never its own category — it is dead-end evidence at a later node, `from`-linked. The
  decision TREE is a derived view of the chain (exploration is sequential; failures fold
  back as evidence, not branches): a spine with untaken-option stubs.
- **All the old special cases dissolve**: superseded replan tail → a `not_taken`
  "continue-old-tail" option (why_not = the staleness reason) at the replan node; a
  mid-cycle fork/failed replan → "continue-old-tail" is the TAKEN option (the original
  tail really did continue); worked steps → `from`-linked facts; retries/delegations →
  nodes with their real small option sets ({retry-with-insight, accept-dead-end},
  {delegate, fold-dead-end} with the gate verdict as why).
- **No ledger array, no watermarks**: evidence is stored ONLY as per-node deltas;
  "known at S_k" is positional (union of evidence at nodes ≤ k); `derive_ledger()`
  reconstructs the flat ledger by concatenation. Chosen over the earlier
  ledger+`knew_until`-watermark draft explicitly for the PRIMARY CONSUMER — an LLM
  reading the render as prompt context: document order = evidence order, so a model
  reading top-to-bottom has, at any node, read exactly what the system knew there. No
  indexes to dereference.
- **Prospective alternatives capture** (task-decomposer `_ALTERNATIVES_RULE`, shared by
  plan_prompt AND partial_replan_prompt): options are recorded AT DECISION TIME, never
  reconstructed — that is what lets hindsight distinguish "saw it and passed" from
  "nobody saw it". Advisory only: planfile.validate ignores the field entirely; the
  journey fold drops malformed entries silently. Honesty rule: rendered as "options on
  the table (as recorded at the time)", never "all possible options".
- **Hindsight = one oneshot, judged by code**: success + full route + leftover budget
  only (clamp 60–180s of remaining; skipped-with-receipt otherwise; never on failure —
  on non-success the same journey skeleton already serves the next attempt, and a
  "how could you have failed better" critique has no consumer). Citation contract:
  every claim names a node key + (avoidable) an evidence fp exactly as rendered;
  code validates (one violation-echo retry, then drop) and stamps tiers PURELY
  POSITIONALLY: (a) genuinely-avoidable — a recorded not_taken option at S_k with
  enabling evidence born at ≤ k; (b) blind-spot — evidence known, option never
  recorded; (c) honest-exploration — evidence born after S_k → reclassified
  unavoidable (the prompt says it outright: dead ends whose disproof required running
  them are the system working as designed).
- **INVARIANTS**: hindsight can never un-succeed a run (every failure mode → a
  `{"skipped": reason}` sentinel in the journey's advisory slot; SUCCESS stands);
  advisory-only — it triggers no execution, ever; `exploration.ratio` is a NEUTRAL
  number (dead ends / decisions) — "waste" is a judgment only the tiered hindsight may
  make, never the arithmetic's vocabulary.
- **report.md = render_journey(FULL)** — a pure view of journey.json (+ the legacy
  ledger-by-kind appendix); renders degrade FULL → COMPACT (hindsight's input;
  retry/delegate nodes collapse) → SPINE, with the citation skeleton (chain,
  taken/not_taken, fps) surviving every level and hard text caps at render time
  (a journey must never blow out a consumer's prompt budget). Mermaid is a FULL-only
  human garnish — for an LLM the node blocks ARE the graph serialization.
- **Uniformity**: journey.json on every route and every outcome — trivial/single_method
  write a degenerate one-node chain (`_write_degenerate_journey`); failure runs keep the
  same skeleton with "Where it stopped" instead of "The path that worked"; delegation
  sub-runs keep their own journey under their own slug, linked by `sub_run` reference,
  never inlined.
- **Learnings close the loop through the knowledge plane**: valid `promoted_learnings`
  are promoted as source=retro fact records (fp on own text, dedup by append()'s fp
  guard) — the in-memory ledger is never mutated in the report step (a live-vs-replay
  divergence there would trip the engine's determinism check).
- **Flow v5**: `retro/journey` (always; memoized so replay is byte-identical),
  `retro/clock` + `retro/judge` (success ∧ cascade ∧ budget). All new keys sit AFTER the
  cycle loop, so an in-flight v4 journal under `--accept-flow-change` replays its
  completed prefix and only the retro/report tail runs fresh.
- **Deferred levers** (recorded, not built): cross-slug learning reuse beyond the
  project-scoped knowledge plane; alternatives-quality scoring (are the recorded
  not_taken options genuinely viable, or filler?); hindsight over a delegation sub-run's
  own journey (today the parent only links it).

## v5 polish pass — fixes from the build's own learnings (2026-07-03, same day)

Smoke-rendering the fixture surfaced four gaps; fixed per [[fix-after-learnings]]:
(1) tier stamping survives judge paraphrase — verbatim-label rule added to the retro
prompt AND a normalized word-overlap fallback in `_option_matches` (fp equality first);
(2) learnings no longer render as completed steps — evidence keeps `via: "learning"`,
and "The path that worked" derives from the new key instead of from-linked facts;
(3) evidence `from` pointers resolve to DECISION COORDINATES (`"S0:env-override"`,
bare id kept as `from_task`) — tree edges reconstructible without scanning options;
(4) `success_path` derived key (worked tasks in execution order; fp-global outcome
annotation means a fail-then-work method marks both occurrences — dedupe keeps the
LAST). Degenerate routes derive it naturally via a synthetic t0 + the "ok <method>"
fp namespace. No relentless.py change, no flow bump.

Considered and REJECTED in the same review: per-retry-attempt evidence in the ledger
(breaks dead-end fp anti-flap; the retry node's chose.outcome already records it);
a journey for scope mode (every scope decision is "plan again"; scope.json serves the
machine consumer); de-duplicating report.md's ledger appendix against the render
(the appendix is the grep-familiar human view — LLM consumers feed on
render_journey/journey.json, now said outright in SKILL.md).

## Round 3 (post-Codex-review)

- `solve.json` now surfaces `artifacts.journey` on every route. Full runs and resumes
  provenance-gate it: the path appears only when this invocation has a terminal engine
  result and journey.json exists. A successful resume refreshes solve.json from the
  original gate.json verdict plus current engine state; its elapsed time is best-effort 0.
- Under `risk=read`, the executor prompt carries an explicit read-only HARD CONSTRAINT:
  observe and verify only, without modifying files, configuration, or external state.
- A judged non-empty hindsight path always yields a reusable route learning. Code
  synthesizes a missing restatement so it reaches both journey.json and promotion;
  `journey.validate_hindsight` deliberately stays permissive because an otherwise-valid
  judgment must not be discarded merely for omitting redundant learning text.
- Before judging, an existing retro.json is moved to retro.json.prior. Artifact-first
  reading therefore cannot mistake a prior run's judgment for the current run's output.

Deliberate non-changes this round: `--hindsight` remains deferred for bare `run` because
it would bump flow_hash and pay judge cost without a gate receipt; tier stamping's
"seen at or before the cited node" horizon remains cumulative; task-id/S-key collisions
are impossible because task ids are lowercase while S-keys are `S<digit>`; and
fp-normalization's word-overlap fallback retains known non-ASCII limits, mitigated by
identity-based outcome annotation elsewhere in the system.
