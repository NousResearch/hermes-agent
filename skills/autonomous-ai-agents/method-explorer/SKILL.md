---
name: method-explorer
description: >
  Use for any non-trivial, failure-prone task whose success path is uncertain.
  Charts a happy-path sequence of stages, pre-mortems each stage (assume it
  fails, ask why), and executes with autonomous diagnose -> next-best-action ->
  backtrack recovery that preserves the original intent and never dead-ends.
  Separates intent (the outcome that must hold) from method (how), runs an
  AND/OR backtracking search over alternative methods, and maintains a durable
  markdown plan-tree artifact (intent, stages, branch log, frontier, dead-set,
  decision-log). Built for headless/autonomous (hermes -z) runs: self-answers
  the diagnostic questions and logs genuine human-decision forks to the artifact
  for async review rather than blocking. Use when a task has real failure risk
  and plausible alternative approaches; skip trivial single-step tasks or tasks
  with only one possible method.
metadata:
  hermes:
    tags: [planning, backtracking, pre-mortem, recovery, autonomous]
    related_skills: [task-decomposer, relentless-solve, resumable-script]
---

# Method Explorer

*(fka `resilient-planner` — renamed 2026-07-03: within the skill family the PLAN
role belongs to `task-decomposer`; this skill's role is EXPLORE — resilient
search over alternative methods. See `skills/ARCHITECTURE.md`.)*

## Overview

A linear plan dies at its first failed stage, or quietly drops the goal when one
approach stops working. This skill prevents both. It **separates intent from
method**: the *intent* (the outcome that must hold) is fixed and protected; the
*method* (the particular stages used to get there) is disposable. When a stage
fails it does not abandon the task — it asks *why*, generates the next-best
action, and if the local branch is exhausted it **unwinds back upstream** to a
different method that still satisfies the same intent. The search only stops at
success, or when every method under the stated constraints is genuinely
exhausted — never at a silent dead end.

Mental model: an **AND/OR backtracking search**. Stages are AND-steps (all must
pass). Decision points are OR-nodes (any one viable child reaches the goal). A
durable plan-tree on disk records the frontier (open, untried branches) and the
dead-set (exhausted ones) so the search resumes across turns and never re-walks
a dead branch.

**Not to be confused with `task-decomposer`**: `task-decomposer` is a stateless,
schema-owning oneshot — given an intent plus another driver's accumulated
evidence, it emits ONE `plan.json` (ordered task DATA) and returns; it never
executes anything or runs its own search. `method-explorer` is the opposite
shape: a self-contained, standalone driver that owns its own durable plan-tree,
executes stages itself, and backtracks on its own. `relentless-solve` uses BOTH,
for different situations: `task-decomposer` for its own multi-task, multi-cycle
loop (via the `full` route), and `method-explorer` (this skill, via the
`single_method` route) when the gate decides there's exactly one clear method
and no branching search is needed. `relentless-solve`'s own bounded local
retry/replan (inside its `full` route) independently converges on a lighter
version of this same diagnose→retry→backtrack→cap policy at a finer grain (one
task, not a whole intent) — see `relentless-solve`'s design notes for how the
two are bridged.

## When to Use / NOT to Use

**Use** when a task is multi-stage, has real risk of a stage failing, and has
plausible alternative approaches — and especially in headless/autonomous runs
where there is no human to course-correct mid-flight.

**Skip** for trivial single-step tasks (just do it), for tasks with exactly one
possible method (no branching to search), or for pure information lookups.

## Startup & Per-Cycle Contract

The operational core. Every section below elaborates these steps — when in doubt,
follow the contract.

**STEP 0 — Route before you act.**
1. If `$HERMES_SIM_SCENARIO` (or a `scenario:` line in the prompt) is set, verify the
   file is readable **now**. Unreadable → write the plan-tree with `STATE: GUARD-HALT`
   and the note `INFRA — sim scenario unreadable; do not bump`, and stop. **Never fall
   through to real execution** (see Simulation Mode).
2. Check for an existing plan-tree at `${HERMES_HOME}/plans/<task-slug>/plan-tree.md`
   (slug rule under *Artifact location*) and route on its `STATE:` header:
   - `SUCCESS` / `EXHAUSTION-STOP` → already done: report that result, don't redo it.
   - `active` → **RESUME**: reload INTENT, the **✝ dead-set**, and **FRONTIER**;
     continue from the frontier. Never re-choose a ✝ method — the on-disk dead-set
     overrides any preference order in the prompt (see *Resuming an interrupted run*).
   - no file → fresh start (STEP 1).

**STEP 1 — Create the artifacts (fresh start).** Frame the intent (P0), chart and
pre-mortem the path (P1–P3), then `write_file` the plan-tree **exactly in this shape**:

```markdown
# Plan-Tree: <task-slug>   STATE: active | SUCCESS | EXHAUSTION-STOP | GUARD-HALT

INTENT: <one sentence — the outcome that must hold>
SUCCESS: <the checklist that decides "done", one line>
HARD (inviolable): <list>
SOFT (relaxable, ranked): 1) ...  2) ...        [append "(relaxed)" when given up]

NODES   (markers: ○ open/untried · ▶ active · ✝ dead · ✓ done)
- S1   <method/tag>              ✝ <symptom · root cause · LOCAL/STRUCTURAL · why dead>
- S1b  <method/tag>  (parent S1) ✝ <one-line reason>
- S2   <method/tag>              ✓ <receipt: what was verified, and how>
- S3   <method/tag>              ▶
FRONTIER: <untried node/method>, ...            (empty ⇒ candidate for EXHAUSTION-STOP)
```

The journal is **one compact single-line JSON object per cycle**, appended to
`${HERMES_HOME}/plans/<task-slug>/journal.jsonl`:

```json
{"node":"S1","q":"fetch valid JSON from the primary source?","chosen":"alfa","expected":"HTTP 200 + JSON with key ok","verdict":"fail","evidence":"curl exit 6 (DNS); re-ran once, same — no success receipt","next":"backtrack->cache"}
```

Fields: `node` · `q` · `chosen` (the method) · `expected` (predicted before acting)
· `verdict` (`success`|`progress`|`fail`) · `evidence` (a re-checked receipt, or
`UNVERIFIED`) · `next` (exactly ONE move). No other fields.

**STEP 2 — The cycle (repeat until STEP 3).**
1. **PREDICT** — pick the next frontier node; record `q` and `expected` (commit the
   prediction *before* acting).
2. **ACT** — one tool call (in Simulation Mode: read the node's declared outcome
   instead).
3. **RECONCILE** — `verdict` + re-checked `evidence` + one-move `next`; append ONE
   journal line (quoted-heredoc — see The Journal); update the node's marker +
   one-line receipt in the plan-tree. On failure, run **The Key Questions** and apply
   the lowest viable rung of the **Next-Best-Action Ladder**.
4. **Persistence (canonical).** After reconciling, **your next action is the next
   cycle's tool call.** Do not emit a final/summary response while the intent is
   unmet, a viable branch remains, and budget remains. A record whose `next` is
   `backtrack->X` must be immediately followed by actually doing X — writing "I will
   now try X" is not trying X.

**STEP 3 — Terminate.** Set the plan-tree `STATE:` and stop:
- `SUCCESS` — all success criteria met; cite the receipt.
- `EXHAUSTION-STOP` — frontier empty **and** all soft constraints relaxed; report the
  blocking hard constraint + the external change that would unblock it. Never fabricate.
- `GUARD-HALT` — a budget guard fired with branches still open (say which); never
  mislabel it EXHAUSTION-STOP. Record the halt as one `GUARD-HALT: <which guard
  fired, open branches, smallest bump to continue>` body line under the header —
  downstream parsers read that line, not prose.

## Tooling (Hermes)

This skill runs inside the Hermes agent. Use Hermes tool names, not other
frameworks':

| Need | Hermes tool |
|------|-------------|
| Parallel branch explorers | **`delegate_task`** — `tasks: [...]` for a batch; capped by `delegation.max_concurrent_children` (default 3) |
| Write / update the artifact | **`write_file`** (`path`, `content`) to create; **`patch`** (`mode:"replace"`, `path`, `old_string`, `new_string`) to update a section |
| Read / search | **`read_file`**, **`search_files`** |
| Long-lived cross-task memory | **`memory`** (optional; the artifact is the per-task record) |
| MCP servers | the **`mcp`** tool |

**Headless semantics (`hermes -z` / oneshot):** all tool and shell approvals are
auto-bypassed, and `clarify` is auto-answered with an arbitrary default. So
**do not** use `clarify` to resolve a real fork — you will get a coin-flip.
Instead, decide autonomously using **The Key Questions** and **record the fork
in the journal** (`next`/evidence) and as the chosen node's one-line note, for
asynchronous human review.

**Operational note — drive with retry; a single turn can't self-heal a dead backend.**
The model backend occasionally produces *no output at all* (slow → timeout, or an empty
response), leaving an empty journal — a "no-op". A lone `hermes -z` turn cannot retry
itself, so drive this skill with a **resume-until-terminal wrapper**. One ships with the
skill: **`scripts/drive.py`** — it re-invokes `hermes -z`, reads the plan-tree `STATE:`
header, and resumes (`active`) tick after tick until a terminal STATE
(`SUCCESS`/`EXHAUSTION-STOP`/`GUARD-HALT`), with no-op retry, livelock/stuck detection,
and a generous per-tick timeout. Run it in-container as
`python3 ${HERMES_HOME}/skills/method-explorer/scripts/drive.py --slug <slug>
--prompt-file <f> --in-container`. Retries/resumes are cheap because a re-run **resumes**
from the plan-tree (see *Resuming an interrupted run*) instead of repeating completed
work. **The wrapper is for runs that ran out of room** (a dead/slow backend, or an
exhausted oneshot iteration budget) — never an excuse to stop while you still have
budget (contract **STEP 2.4**); a genuinely long task that exhausts its budget is
finished by the retry/resume — intended design, not a failure.

**Artifact location:** write the plan-tree to
`${HERMES_HOME}/plans/<task-slug>/plan-tree.md`. In the container deployment
`HERMES_HOME=/opt/data`, so that is `/opt/data/plans/<task-slug>/plan-tree.md`.
If unsure of `HERMES_HOME`, resolve it once with a terminal `echo $HERMES_HOME`.

**Slug derivation — must be deterministic.** Resume-by-artifact only works if a
re-invocation computes the **same** slug. If the prompt names a slug (or a plan
path), use it **verbatim**. Otherwise derive it from the intent's key nouns —
lowercase kebab-case, 2-4 words, no dates/counters/randomness (e.g. intent
"migrate the billing DB to Postgres" → `migrate-billing-db`). The same task
re-invoked must land on the same `plans/<task-slug>/` directory, or the resume
machinery silently never engages.

**`delegate_task` notes:** a single delegation runs in the background and its
result re-enters the conversation as a new message; a batch (`tasks` array) runs
its children in parallel. Children **cannot** call `delegate_task`, `clarify`,
`memory`, or `execute_code`, but they do get `terminal`/`file`/`web` — enough to
scout viability. A child returns its verdict **as its final summary**; the
parent (this skill) owns the artifact and integrates the verdicts.

## The Loop

```
FRAME INTENT → CHART HAPPY PATH → PRE-MORTEM → EXECUTE ─┐
     ▲                ▲                            │ (a stage fails)
     │                └──── BACKTRACK ◄── DIAGNOSE ◄┘
     │                                       │ (frontier near-empty)
     └──── RELAX SOFT CONSTRAINT ◄───────────┘
                                     ↓
                           SUCCESS  /  EXHAUSTION-STOP
```

## The Recursive Loop

The phases below are one pass of a deeper engine: a **self-similar question
loop**. The unit of work is a *question-set*, not a stage — and the same
procedure that answers a question is re-applied to figure out *what to ask next*
whenever an attempt stalls.

```
solve(node):
  1. ASK      derive this node's key questions
              · progress node → "what must be true to satisfy the postcondition?
                                 what is the cheapest probe / attempt?"
              · failure node  → The Key Questions (symptom · root cause · locality
                                · reachability · cheapest recovery · intent check)
  2. ATTEMPT  act / answer  (real execution, a cheap probe, or — in Simulation
              Mode — read this node's declared outcome from the scenario)
  3. ASSESS   progress?  →  postcondition met
                            | >=1 new viable downstream branch opened
                            | a hypothesis eliminated that narrows the search
        progress → journal it; advance to the next node
        NO       → TOMBSTONE this node; journal the no-progress + why
                   REGENERATE: re-run solve() on the meta-question "what next?"
                     → backtrack to an ancestor decision-point, substitute a
                       method, or reframe (relax a ranked soft constraint)
                   solve(next_node)                  # same procedure, recursively
  4. STOP     SUCCESS (intent met) · EXHAUSTION (frontier empty + soft
              constraints relaxed) · GUARD (depth / branch / iteration budget)
```

- **Progress is the pivot.** If an attempt makes progress, advance. If it does
  not, the branch becomes a **tombstone** — recorded so it is never re-asked — and
  the *same loop* generates the next questions. This is what "answer a couple of
  questions, then figure out the next questions if you didn't make progress" means
  operationally.
- A **tombstone** generalizes the **Dead-set**: not only failed *methods* but any
  *question-branch* that yielded no progress.
- The recursion is genuine: when you are stuck on *what to even ask*, run `solve`
  on that meta-question too.

## Quick Reference

| Phase | Do | Plan-tree updated | Journal |
|-------|----|-------------------|---------|
| P0 Frame Intent | Separate intent vs method; list hard + ranked soft constraints | INTENT + constraints header | — |
| P1 Chart Happy Path | Ordered stages; per stage record its load-bearing assumption + risk; mark decision-points | NODES (○) | — |
| P2 Pre-mortem | For each stage assume failure; record the detection signal + likely next-best branch | NODES (pre-staged) | — |
| P3 Next-Best Actions | Enumerate ranked recovery branches; fan out `delegate_task` for uncertain forks | FRONTIER | — |
| P4 Execute + Recover | Walk the path; on real failure run The Key Questions; apply lowest viable ladder rung | NODES (✝/✓ + receipt) | one lean record/cycle |
| P5 Backtrack / Terminate | Maintain frontier + dead-set; backtrack or stop per the rules | FRONTIER, STATE | terminal record |

## Phases

### P0 — Frame Intent
Restate the task as **intent** (the outcome that must be true at the end), held
separate from **method** (how you intend to get there). Record:
- **Goal** — one sentence; the outcome backtracking must preserve.
- **Success criteria** — a checklist that decides "done".
- **Hard constraints** — inviolable. A candidate action that breaks one is *not*
  a candidate.
- **Soft constraints** — relaxable, **ranked** by how reluctantly you'd give
  each up. These are the release valves when the frontier runs low.

Write the artifact with `write_file` now (template below).

### P1 — Chart the Happy Path
Decompose the intent into the ordered sequence of stages most likely to succeed.
For each stage record: action, precondition, expected postcondition, the
**single load-bearing assumption** that makes it work, a risk rating, and
whether it is a **decision-point** (a stage with plausible alternative methods —
these are the OR-nodes you may backtrack to).

### P2 — Pre-mortem
For each stage, assume it fails. Answer **The Key Questions** hypothetically and
record: the runtime **detection signal** (how you will know it failed) and the
pre-staged **likely next-best branch**. High-risk stages deserve more than one.

### P3 — Generate Next-Best Actions
For each anticipated failure, enumerate ranked recovery branches (the
**Next-Best-Action Ladder**). When a fork is genuinely uncertain — competing
methods whose viability you cannot judge from reasoning alone — **fan out with
`delegate_task`** (see Parallel Branch Exploration) to scout them in parallel.
Add every candidate branch to the **Frontier**.

### P4 — Execute + Recover
Walk the chosen path. On a **real** stage failure:
1. Run **The Key Questions** against the actual evidence.
2. If a pre-staged branch matches, take it; otherwise generate fresh branches.
3. Apply the **lowest viable rung** of the ladder.
4. Update the NODES markers + one-line receipts; record any judgement-call fork in
   the journal (and the node's note).

### P5 — Backtrack / Terminate
Maintain the **Frontier** and **Dead-set**. When the current branch has no
viable child, mark it dead and **backtrack** to the nearest ancestor
decision-point that still has frontier children. Apply the Stop Conditions.

## The Key Questions

Run these at every failure (real or hypothetical). They are the "ask why" core —
answer them in the journal's `evidence`/`next` and the failed node's one-line
receipt.

1. **Symptom** — what actually happened, versus the stage's expected
   postcondition?
2. **Root cause** — which load-bearing assumption broke, and *why*? (Not the
   symptom — the assumption underneath it.)
3. **Locality** — classify on two axes: is the fix *here* or *upstream*, and is the
   block *transient* or *standing*?
   - **LOCAL-transient** — the method + capability are sound; the failure is
     time-dependent and may clear on retry or on a sibling of the same class: **host /
     network down, DNS blip, connection refused, timeout, HTTP 429/500/503, a
     flaky/non-deterministic tool.** → retry once (**rung 0**) or substitute a sibling
     (**rung 2**). A whole *class* being unreachable *right now* (every network source
     down) is **still LOCAL-transient** — the discriminator is time-dependence, not how
     many siblings it knocks out.
   - **LOCAL-method** — this specific method is wrong but everything upstream is valid →
     fix in place (**rung 1**) or substitute (**rung 2**).
   - **STRUCTURAL** — a *standing* decision or missing capability makes the stage
     unreachable by this whole class of method and will **not** change on retry:
     **permission/auth denial (401/403/EACCES), missing binary/capability,
     protected/read-only path, quota/policy denial, an endpoint that does not exist.** →
     **stop trying variants**, escalate to **backtrack (rung 3)** or **relax (rung 4)**;
     do not brute-force.
   - **Discriminating test:** *would waiting-and-retrying, or a sibling of the same class,
     plausibly succeed?* Yes → **LOCAL-transient**. *Would every method of this class hit
     the same wall no matter when you try?* → **STRUCTURAL**. "The host is down" answers
     **yes to the first → LOCAL-transient**, not structural.
   - **In Simulation Mode, answer this test from the declared reason's *semantics*, not
     from retry behavior** — a sim retry always re-declares the same outcome by scenario
     rule, so "the retry also failed" is *not* evidence of standing-ness there. "Host/
     mirror is down" stays **LOCAL-transient** in sim even though no sim retry can succeed.
   - **Running out of alternatives does not change the label.** Locality classifies the
     *blocker's* time-dependence, not whether siblings remain. A transient outage with
     zero siblings left is still **LOCAL-transient** — the terminal state
     (EXHAUSTION-STOP) records "nothing left to try"; the ✝ label records *why this node
     died*. Keeping them separate is what makes the exhaustion report actionable: a
     LOCAL-transient blocker's unblock condition is *time* (retry later), a STRUCTURAL
     one's is an external change (permission, capability).
4. **Reachability** — does any *untried* method still reach this stage's
   postcondition? List them; they become new frontier branches.
5. **Cheapest recovery** — the lowest viable rung of the ladder.
6. **Intent check** — does the chosen candidate still satisfy the original
   intent and **every** hard constraint? If it violates a hard constraint it is
   not a candidate, no matter how convenient.

## Next-Best-Action Ladder

Cheapest first; climb only when a rung is non-viable.

0. **Retry as-is** — only when **Locality = LOCAL-transient** (flaky network, rate limit,
   timeout, 5xx, host down *right now*). A **STRUCTURAL** verdict skips this rung — but a
   transient network-down *mislabeled* STRUCTURAL wrongly forfeits this one legit retry,
   so apply Key Question #3's discriminating test before you skip it. Retry once or twice,
   not many times (see **Budget discipline**).
1. **In-place fix** — repair the broken assumption, keep the same method.
2. **Substitute method** — same stage postcondition, a different approach.
3. **Backtrack** — unwind side effects to the nearest ancestor decision-point
   and take an untried branch. Use when Locality = STRUCTURAL, or when rungs 0–2
   are exhausted.
4. **Relax a soft constraint** — give up the lowest-ranked soft constraint to
   open methods that were previously off-limits, then re-chart from the affected
   stage. Log which constraint and why.
5. **EXHAUSTION-STOP** — only after the Frontier is empty **and** all soft
   constraints are relaxed. Report the blocking hard constraint and the external
   change that would unblock it. **Never fabricate a path** and never silently
   declare failure.

**Budget discipline — don't grind a blocked method.** Climbing the ladder means
trying a *genuinely different* rung, not many variants of the same blocked action.
When a method fails for a **STRUCTURAL** reason (permission/auth denial, missing
capability, protected/unwritable path, an endpoint that doesn't exist — **NOT** a
transient host-down, which is LOCAL-transient → retry/substitute), **do not enumerate
5–20 variations of it** (other write tools, other syscalls, other shells) — tombstone the
method and jump straight to **rung 3 (backtrack)** or **rung 4 (relax a soft
constraint)**. Many-variant brute-forcing is itself a failure mode: a oneshot turn has
a finite iteration budget, and grinding a structurally-blocked method **exhausts it
before the loop reaches the move that actually works** (e.g. writing the deliverable
to a writable path when the path is only a *soft* constraint).

## Parallel Branch Exploration

When P3 surfaces a fork between competing methods whose viability you cannot
settle by reasoning, scout them concurrently. Dispatch one `delegate_task` batch
with a `tasks` array — one task per candidate method. Each child investigates
(reads files, runs commands, probes the web) and **returns a structured verdict
as its summary**:

```
delegate_task(tasks=[
  {
    "goal": "Assess whether <method A> can reach <stage postcondition>, given <intent + hard constraints>. Do NOT implement it — only assess. Return JSON: {viability: high|medium|low|none, cost: low|med|high, risk: <one line>, evidence: <what you checked>, blocking: <hard-constraint conflict or none>}.",
    "context": "<paths, constraints, what 'done' means for this stage>",
    "toolsets": ["terminal", "file", "web"]
  },
  { "goal": "...assess <method B>...", "context": "...", "toolsets": ["terminal","file","web"] }
])
```

Keep the batch within `max_concurrent_children` (default 3). When the verdicts
return: commit to the highest-viability, hard-constraint-clean branch; move the
others to the FRONTIER (untried) or mark them **✝** (if a child found a
hard-constraint conflict). Record the choice and the runners-up in the journal
(and the node notes). Do not over-fan-out: scout only forks you genuinely cannot judge
inline.

## Dead-End Avoidance & Stop Conditions

This is what keeps the search from dead-ending or looping.

- **Frontier** = open, untried, intent-satisfying branches. **Dead-set** =
  branches that are exhausted or proven to violate a hard constraint. **Never
  expand a node in the Dead-set** — that is the cycle/dead-end guard.
- **SUCCESS** — all success criteria met. Stop; summarize the path taken.
- **BACKTRACK** — current branch has no viable child → mark it dead → pop to the
  nearest ancestor decision-point with frontier children.
- **EXHAUSTION-STOP** — Frontier empty **and** all soft constraints relaxed →
  emit the structured "no viable path" report (blocking hard constraint +
  unblock condition). Never declare failure while any open branch or any
  unrelaxed soft constraint remains.
- **Guards** (loop/cost backstops, distinct from true exhaustion): max backtrack
  depth, max branches explored, max iterations. When a guard halts the run,
  **say so explicitly: set the plan-tree STATE to `GUARD-HALT`** — a guard-halt
  ("hit branch budget, N branches still open") must never be mistaken for genuine
  exhaustion. Suggest
  the smallest budget bump that would continue.
- **Persistence** — the canonical rule is contract **STEP 2.4**: after reconciling,
  your next action is the next cycle's tool call; never voluntarily return with the
  frontier open and budget left. The distinct nuance here: **a turn can legitimately
  run out of room.** A oneshot turn has a finite iteration/tool-call budget; if you
  exhaust it (or are externally killed) mid-task, that is a **resumable interruption,
  not a failure** — the plan-tree carries the open frontier forward and a
  re-invocation resumes (see *Resuming an interrupted run*). So: never *choose* to
  stop with budget left, **and** spend the budget well — on a STRUCTURAL blocker,
  relax/backtrack **fast** rather than brute-forcing many variants of a blocked
  method (that is what burns the whole budget).

## Choosing the Next Question

When REGENERATE offers several viable next-questions, don't just grab the
cheapest — choose deliberately:

- **Marker-grounded anti-repeat.** Before choosing, read the **✝ nodes in the
  plan-tree** — they are the operative dead-set. Never re-derive a tombstoned
  question-branch. This is the durable anti-loop guard. (The journal corroborates,
  but under the driver wrapper it is archived per tick to `journal.tick*.jsonl`,
  so the live `journal.jsonl` may be empty on resume — the tree's markers, not the
  journal, are what you can rely on being present.)
- **Score the candidates.** Rank viable next-questions by `{expected progress,
  cost, reversibility, intent-fit, novelty-vs-journal}` and take the top. Fall
  back to the Next-Best-Action Ladder's cheapest-first only when scores tie.
- **Two different climbs — don't conflate them.**
  - **Ran out of siblings (ordinary back-up).** When the current parent has **no untried
    method left** (its frontier is spent — e.g. only two network sources existed and both
    tombstoned), simply **backtrack to the next parent/branch**. This is the normal P5
    move; it is **NOT** the K-heuristic — do not narrate it as a "K-jump."
  - **Upstream-jump heuristic (K=5).** Only when **K consecutive siblings tombstone while
    *more* untried siblings remain** (default **K=5**) do you *pre-emptively* stop trying
    the rest and suspect the **parent decision** — jump further upstream or relax a ranked
    soft constraint. K exists to abandon a *long* sibling run early; it is **meaningless
    when fewer than K siblings exist**, because you exhaust them before reaching K. This is
    how the loop "unwinds back upstream and creates new ideas" instead of grinding a dead
    sub-tree.
  - Record which happened in `next`: `backtrack->P` for a ran-out; `jump->P` / `relax->C`
    for a genuine K-jump.

## Decision Records — predict → act → reconcile (show your work, lean)

Every cycle leaves **one lean decision record** so the path is visible and
*evidence-based*, not a story told after the fact — and **recorded once**: the
journal holds the per-cycle facts; the plan-tree holds only the current marker
state. Never re-narrate a cycle in both — that duplication is wasted tokens and a
drift risk. Reason through predict → act → reconcile, but **persist only the
essentials**.

**BEFORE acting (commit — the outcome isn't known yet, so this can't be rationalized):**
- **q** — the question this step answers; name the **desire** it serves. Weigh the
  candidate methods in your reasoning and pick the cheapest viable one — but the
  *deferred* alternatives live **once** in the plan-tree FRONTIER, not copied into
  every record.
- **expected** — the postcondition you predict, concrete enough to check later.
  Committing the prediction *before* acting is what makes the record a real test of
  your model, not a post-hoc story.

**ACT** — make the tool call.

**AFTER acting (reconcile — the actual evidence):**
- **verdict** — `success` / `progress` / `fail`, naming the method **chosen**.
- **evidence (receipt)** — an *independently re-checked* observable: a returned
  artifact, a row count, an exit code, a file read back. **Verify; do not trust the
  tool's own "success."** If you cannot cite a receipt, mark the verdict
  **UNVERIFIED**. **Verify *correctness*, not just presence:** a receipt must confirm
  the result is the *right* answer for the intent, not merely that a tool returned
  *something* — "the API returned a file" is not success if the intent was "the *most
  recent* file" and you haven't confirmed it. A plausible-but-wrong result is a
  **failure**, and tombstoning it is the win.
- **next** — the move that follows. If the prediction missed, add a one-word
  *surprise* note: a mismatch is the most valuable signal in the loop (your model was
  wrong → real learning) and usually reframes the next question.

Keep it honest and lean:
- **Evidence over eloquence.** A record earns trust by citing a *re-checked* receipt,
  not by length — one compact line per cycle (next section).
- **Record once.** The journal is the per-cycle fact log; the plan-tree is the
  current-state map (markers + one-line receipts). Do **not** keep a second prose
  "decision-log" that re-states the journal — that duplication is exactly the token
  waste this format removes.
- **In context, keep only the pinned summary** (intent · hard constraints ·
  dead-set = ✝ nodes · frontier) so the trail survives compaction; full records live
  on disk.

## Simulation Mode

Simulation Mode plays the recursive loop out **deterministically and with no real
side effects**, so you can watch it converge or correctly dead-end. It is driven
by a scenario file; activate it when `$HERMES_SIM_SCENARIO` is set or a
`scenario: <path>` is given in the prompt.

**Validate the scenario file FIRST.** If `$HERMES_SIM_SCENARIO` (or `scenario:`) is
set but the file is missing or unreadable, **halt immediately**: write the plan-tree
with `STATE: GUARD-HALT` and the note `INFRA — sim scenario unreadable; do not bump`,
and stop. **Never fall through to real execution** — a run that was asked to simulate
must not silently perform real side effects because its scenario failed to load.

**No delegation in sim.** Do **not** use `delegate_task` in Simulation Mode — children
execute for real (they get `terminal`/`file`/`web`), which breaks the no-side-effects
guarantee. Answer forks from the scenario's declared semantics instead.

In Simulation Mode the **ATTEMPT** step does not execute — it reads this node's
declared outcome from the scenario:

- `tombstone` / `no-progress` → treat as no-progress: journal it, REGENERATE
  (backtrack). **Do not run the real action.**
- `progress` / `success` → treat as progress: journal it, advance.
- silent (no matching rule) → fall back to the scenario's `default`:
  `passthrough` runs the real action (mixed sim/real), or `progress` /
  `tombstone`.

Rules are **situational**: they match on stage id / action substring / occurrence
count, and may depend on prior journal state (e.g. `after_tombstones: 2`).
Everything else about the loop is unchanged — the planner does not behave
differently because it is simulated; only the *source* of each node's outcome
changes. That is the point: the same diagnose→tombstone→regenerate machinery
runs, journaled, against declared outcomes. Scenario format + worked examples:
`references/simulation.md`. Two ready scenarios ship in `assets/scenarios/`.

**Sim-mode `evidence` is a declared-outcome echo, not a receipt.** In Simulation Mode the
journal's `evidence` is copied from the scenario rule's declared `reason` — nothing really
ran, so the Decision-Records receipt discipline (an *independently re-checked* receipt ·
"don't trust the tool's own success" · UNVERIFIED) is **inert**. A sim run demonstrates the
loop's *control flow* (tombstone → regenerate → backtrack), **not** evidence-gathering. To
exercise the receipt discipline, use a **real-mode** run where a file read-back / exit code /
row count is the actual receipt. Likewise, classify **Locality from the declared reason's
semantics** ("source is down" → LOCAL-transient), never from the fact that a sim retry
re-fails — sim rules re-declare outcomes by design (see Key Question #3).

## The Journal — `journal.jsonl` (lean, append-only)

Append-only, **one compact single-line JSON object per loop cycle** at
`${HERMES_HOME}/plans/<task-slug>/journal.jsonl` (`/opt/data/plans/...` in the
container). It is the replayable, evidence-based record of the decision path —
recorded **once** (the plan-tree does not repeat it). **The literal record shape +
field list live in the Startup Contract, STEP 1** — no `candidates` / `rationale` /
`confidence` / `surprise` arrays. The reasoning still happens; only the essentials
are persisted, and deferred options live **once** in the plan-tree FRONTIER.

**`next` names exactly ONE immediate move** — the single tool call you make next
(contract STEP 2.4 requires executing it immediately). Do **not** stack actions
(`relax->X; backtrack->Y`) in `next`; pick the one you'll do now and let the deferred
alternative live once in the FRONTIER. A two-action `next` breaks the "do X now"
contract.

Journal **every** cycle (success and failure). Write each record as ONE compact
single-line JSON object on its own line, then a newline (valid JSONL — never
pretty-print across lines, never concatenate two objects without a newline between
them). **Append with a terminal quoted-heredoc** — one cheap call, no re-reading the
file, no escaping problems (the quoted `'EOF'` means quotes inside the JSON pass
through untouched):

```sh
cat >> ${HERMES_HOME}/plans/<task-slug>/journal.jsonl <<'EOF'
{"node":"S1","q":"...","chosen":"...","expected":"...","verdict":"fail","evidence":"...","next":"..."}
EOF
```

Do **not** read-and-rewrite the whole journal to append — that costs tokens
per cycle and is exactly what produces truncated/concatenated (`}{`) journals.
(`write_file`/`patch` remain a fallback if the terminal is unavailable.) **Do not**
roll the journal up into a second prose decision-log — the plan-tree's marker map
already reflects the final state (rolling it up was pure duplication).

## Resuming an interrupted run

> **Resuming is for runs that ran out of room** — an external kill (timeout/crash)
> *or* an exhausted oneshot iteration budget — **never a substitute for finishing
> when you still have budget** (contract **STEP 2.4**). Only a run whose
> `STATE: active` + open FRONTIER was left by a genuine interruption is a resume
> point, not one you chose to end.

A headless turn can be killed mid-task (a slow/empty model backend → timeout → no
output; see the operational note in Tooling). Because the plan-tree + journal are
written **incrementally**, such an interrupted run is **resumable** — don't restart
from scratch. **The STATE routing is contract STEP 0** (SUCCESS/EXHAUSTION-STOP →
report, don't redo · active → resume · none → fresh). What resume means in detail:
re-load INTENT, the **✝ nodes** (dead-set), and the **FRONTIER**, and continue from
the open frontier. **Never re-expand a ✝ node**, and don't re-run nodes already
marked **✓**. **A `✝` method in the plan-tree OVERRIDES any "preference order" in
the prompt** — even if a method is listed first or "preferred", if it is `✝` on disk
it is DEAD evidence, not a fresh suggestion: do **not** re-choose it. The on-disk
dead-set wins over the prompt.

On resume, **append to `journal.jsonl`** (never truncate or overwrite it); the
plan-tree is overwritten in place. Note: the driver wrapper **archives the journal
every tick** (`journal.jsonl` → `journal.tick<n>.jsonl`), so finding an empty or
short `journal.jsonl` next to a populated plan-tree is **normal on resume — not
evidence of lost work**. Just keep appending to `journal.jsonl`; the full decision
path is the union of the tick archives, and the **plan-tree's ✝/✓ markers are the
operative resume state** (that's why each fact must reach the tree as a marker,
not live only in the journal).

This turns the dominant runtime flake (a backend-induced timeout) from "lost work" into
"pick up where it stopped" — the on-disk plan-tree *is* the resume point.

### Artifact self-repair (degraded resume)

Interruptions can tear the artifacts themselves. Rules, strictly gated:

- **Malformed plan-tree** — repair **only** when the file exists but has **no parseable
  `STATE:` header** (nothing matching `STATE: <word>`). If a STATE header parses, the
  tree is NOT malformed — do not "repair" (i.e. wipe) a valid tree you merely found
  confusing; a populated FRONTIER is unrecoverable once overwritten. When it truly is
  headerless/torn: **harvest the dead-set** = every `✝` line still readable in the
  garbled text **plus** every `fail` verdict in `journal.jsonl` ∪ `journal.tick*.jsonl`;
  **re-derive the frontier by re-running P3** against the intent (deferred options were
  never in the journal — record-once); write a **fresh tree** with the STEP-1 template
  (harvested ✝ nodes included, `STATE: active`); journal the cycle with
  `next: repair->tree`. Then resume normally.
- **Invalid journal lines** (truncated/concatenated JSON) — **skip them** when reading;
  **never truncate or rewrite** the journal to "clean it up". Keep appending new valid
  lines; the tick archives preserve what the live file lost.
- **Tree/journal contradiction** (e.g. the journal's last record is a verified `success`
  receipt for the intent but the tree says `active`, or vice versa) — **receipts win**:
  re-derive STATE from the journal's evidence, rewrite the tree to match, and journal
  the correction (`next: repair->state`). Never report a terminal result the receipts
  don't support.
- **Artifact write fails** (EACCES/read-only on the plans dir) — fall back to the
  terminal: `mkdir -p` the plan dir, heredoc-append the journal, `cat > file` the tree.
  If the terminal also cannot write, this is **INFRA**: say so explicitly in the final
  response (what failed, the exact path) — do **not** continue unjournaled as if
  nothing happened.

## Plan-Tree Artifact — compact marker map

One markdown file per task at `${HERMES_HOME}/plans/<task-slug>/plan-tree.md`,
created with `write_file` and **overwritten in place** with `patch`
(`mode:"replace"`) as nodes change state. It is the durable **current-state map**
— not a history (that's the journal) — that makes backtracking and cross-turn
resumption possible, and the thing a human reviews after a headless run.

Keep it compact: a pinned header + a node list with **status markers** and a
**one-line receipt/reason** per node. The markers carry the state that used to need
separate Branch-log / Dead-set / Decision-log sections, so there is **no
duplication** of the journal. **The literal template lives in the Startup Contract,
STEP 1** — use it exactly; this section defines the semantics.

- The **✝ nodes are the dead-set** — never re-expand one (the anti-loop guard).
- The **FRONTIER line is the open set** — EXHAUSTION-STOP only when it is empty
  *and* all soft constraints are relaxed.
- **STATE** in the header is the terminal classification: a GUARD-HALT (budget hit,
  branches still open) must read `GUARD-HALT`, never `EXHAUSTION-STOP`.
- Overwrite the file as state changes; do **not** append per-cycle history to it
  (that is the journal's job — recording it in both is the duplication to avoid).

## Common Mistakes

- **Confusing intent with method.** Backtracking must preserve the *intent*; if
  you anchor on the first method you abandon the goal the moment it fails.
- **Declaring failure with branches still open.** Only EXHAUSTION-STOP (empty
  frontier + all soft constraints relaxed) is real failure.
- **Re-expanding a dead node.** Always check the Dead-set first; that is the
  anti-loop guard.
- **Relaxing a *hard* constraint to make progress.** Hard constraints are part
  of the intent; relax only ranked *soft* ones, lowest first, and log it.
- **One-and-done diagnosis.** Skipping the Locality question leads to patching
  locally when the real fix is upstream (or vice-versa).
- **Using `clarify` for a fork in headless mode.** It auto-answers with a
  default — decide autonomously and log the fork instead.
- **Over-fanning-out.** Only `delegate_task`-scout forks you cannot judge by
  reasoning; needless batches burn tokens.
- **Advancing without a progress signal.** If an attempt didn't meet the
  postcondition, open a new viable branch, or eliminate a hypothesis, it did
  **not** make progress — tombstone and regenerate; don't pretend forward motion.
- **Grinding a dead sub-tree.** After **K** sibling tombstones *with more siblings still
  untried* (K=5), jump upstream — don't keep trying near-identical siblings of a failed
  parent decision. But if you simply **ran out** of siblings (fewer than K existed and all
  tombstoned), that is an ordinary back-up, **not** a K-jump — don't label it one.
- **Duplicating the journal into the plan-tree.** The journal is the per-cycle fact
  log; the plan-tree is the current-state marker map. Re-narrating each cycle in both
  (a prose "decision-log" that restates the journal) wastes tokens and risks drift.
  Record each fact once.

## Cross-References

- `delegate_task` — batch/parallel branch exploration (`tasks` array).
- `write_file` / `patch` — create and update the plan-tree artifact + journal.
- `skills_list` / `skill_view` — discovery; invoke this skill as
  `/method-explorer`.
- `references/simulation.md` — scenario spec, journal schema, worked examples.
- `assets/scenarios/*.json` — ready-to-run demo scenarios; point
  `$HERMES_SIM_SCENARIO` at one to drive Simulation Mode.
- `journal.jsonl` — the per-cycle journal written under
  `${HERMES_HOME}/plans/<task-slug>/`.
- Deployed at `<HERMES_HOME>/skills/method-explorer/SKILL.md`
  (`/opt/data/skills/...` in the container).
