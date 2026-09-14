# Phase 5 — NOVA Supervisor

Business objectives, routed under the tenant's own delegation policy, submitted to the
runtime's existing board. **Core patches: still 1** (the `AGENTS.md` routing row). No Hermes
file was modified.

---

## The audit that shaped this phase

Phase 5 was planned as "decompose → route → task board → collect". Inspecting the runtime
first changed what was worth building, because **three of those four already exist**:

| Capability | Where | What it does |
|---|---|---|
| Decomposition | `hermes_cli/kanban_decompose.py` | Asks an LLM to break a triage card into a task graph with dependencies |
| Routing | same, `_build_roster` / `_normalize_assignee_choice` | Picks an assignee per child by matching the task to a profile's description |
| Task board | `hermes_cli/kanban_db.py`, `kanban.db` | Task graph, parents, priorities, retries, idempotency keys |
| Fan-out and collect | `kanban_db_dispatch.py`, `kanban_swarm.py` | Runs independent children in parallel; the root wakes when the graph completes |

Building a NOVA decomposer on top of that would have been duplication, and would have meant
taking a model-provider dependency in a layer whose entire dependency surface is the standard
library and PyYAML.

**What the runtime does not do is govern the routing.** Its decomposer selects from every
profile on the host and validates only that the name exists:

```python
# hermes_cli/kanban_decompose.py
def _build_roster() -> tuple[list[dict], set[str]]:
    all_profiles = profiles_mod.list_profiles()      # every profile on the host
    ...
    return roster, {p.name for p in all_profiles}    # the only validation
```

NOVA agents have declared `delegation.may_assign_to` since Phase 1. It was validated at
bundle load, compiled into the profile under the `nova:` key the runtime ignores, and
enforced **nowhere** — verified by grep and by reading a materialized `config.yaml`. A tenant
could declare that support delegates to operations, and nothing prevented support's work
landing on finance.

That gap is this phase.

---

## What was built

```
nova/spec/objective.py         the declaration — steps, owners, dependencies
nova/supervisor/route.py       the decision, pure: may this step go to this agent?
nova/supervisor/submit.py      plan -> work items, idempotent and audited
nova/supervisor/report.py      the objective's state, derived from the runtime
nova/runtime/hermes/submit.py  the write, through the runtime's own API
```

The objective *spec* lives with the other specs rather than in `supervisor/`, both because it
is a declaration like an agent or an identity and because keeping it there stops the
dependency arrow looping back through the runtime contract.

---

## The rule

> A step may be assigned to the objective's owner, or to an agent the owner declared in
> `delegation.may_assign_to`. Nothing else.

Classified `hard_preemptive` in the vocabulary of `nova/policy/limits.py`, and it earns that
for one specific reason: **NOVA is the writer**. A refusal means no task is created, rather
than a task created and reported on afterwards. Verified: with the delegation revoked, the
board held 4 tasks before the attempt and 4 after.

A refused objective creates **nothing** — not "the valid steps, with the rest reported". Half
a month-end close running while the other half is refused is worse than none of it.

Refusals carry a reason as data (`not_delegable`, `unknown_assignee`, `disabled_assignee`,
`unknown_owner`) and a detail sentence that names the fix, because an operator who is told
only "refused" has to go and read three YAML files to find out why.

---

## The boundary of that claim

Stated here because it belongs next to the claim, not in a footnote:

**NOVA governs the work NOVA submits.** Three things are outside it:

1. A human running `hermes kanban create --assignee finance` directly.
2. The runtime's own decomposer, which routes the children of a triage card.
3. An agent delegating in-session through the runtime's own delegation tools.

For (2), a step may opt in with `decompose: true`. NOVA records that the step's children are
routed by the runtime, `RoutingDecision.ungoverned_steps` names them, and the routing
decision carries a warning saying so. Claiming those children were governed would be exactly
the control that looks present in a review and does nothing — which is the thing this phase
was built to remove, not to add somewhere else.

---

## Writing to the runtime's board

This is the first time NOVA writes to a runtime-owned database, so the reasoning is explicit.

**Which database.** `kanban.db` is the shared work board at the home root, written by the
runtime's own CLI, dashboard, dispatcher and in-agent tools. It is **not** `state.db`,
`hermes_state.db`, `response_store.db`, `memories` or `sessions` — those hold conversation
and credential state, they are on `materialize.NEVER_WRITE`, and nothing here touches them.

**Through the runtime's API, never raw SQL.** `kanban_db.create_task` mints ids, writes the
task-event rows the dashboard and notifier read, honours the idempotency key, and holds
whatever locking the schema requires. Hand-rolling an INSERT would reimplement all of it
against a schema upstream is free to change. Calling it is a supported seam; copying it is a
patch.

**Imported lazily, inside an adapter.** The same seam Phase 4 established for document
extraction, and policed by the same three boundary tests.

**Idempotent by construction.** The key is `<objective id>:<step id>`. Submission touches an
external system one item at a time, so a crash halfway leaves half a plan on the board;
re-running finishes it rather than duplicating it. Verified: a second submit created 0 and
resolved all 4 to the same task ids.

---

## Verified against the real runtime

Empirically, not by reading:

| Claim | Evidence |
|---|---|
| Steps with no dependency run in parallel | `pull-ledger` and `handbook-thresholds` land `ready`; `compare` and `write-findings` land `todo` |
| Dependencies are real | `compare` waits on both of its parents |
| Re-submission is safe | second run: 0 created, 4 existing, identical task ids |
| A refusal writes nothing | 4 tasks before, 4 after |
| The assignee reaches the worker | `kanban_db_dispatch.py:2197` maps assignee → profile → `HERMES_HOME` |

---

## Surfaces

```
nova objective list   <bundle>                     declared objectives and how each routes
nova objective plan   <bundle> <id>                route and show the plan — writes nothing
nova objective submit <bundle> <id> [--dry-run]    place the steps on the board
nova objective status <bundle> [<id>] [--json]     read the objective back from the runtime
```

`plan` is the rehearsal worth running before a rollout: it prints steps in execution order
with their routing verdict, and exits non-zero if any step is refused.

`GET /platform/v1/objectives` returns routing and progress together. The dashboard's
Objectives panel shows `refused` as its own state rather than folding it into "not started" —
nothing is waiting to happen, and nothing will until someone changes the policy, which is the
difference between a person investigating and a person waiting.

---

## Two bugs this phase exposed

**The dashboard banner was last-writer-wins.** A governance refusal was silently replaced by
"no knowledge index yet", purely because of panel render order — the least important message
winning by arriving last. Notices now accumulate with problems sorted first.

**A single missing column rendered the board empty.** `work.py` selected a fixed column list
and both readers convert a `sqlite3.Error` into an empty result, so one column upstream had
not added yet would show "no tasks at all" rather than an error. The reader now asks the
store which columns it has and selects the intersection; the defensive accessors in
`_row_to_task` were always written for exactly that.

---

## Known limitations

- **Declared plans, not generated ones.** NOVA does not decompose. A step needing it delegates
  to the runtime's decomposer and is marked ungoverned.
- **No scheduling.** An objective is submitted when someone submits it. Recurrence is the
  runtime's `cron/` or an external scheduler.
- **No cross-objective concurrency control.** Two objectives can both queue work for one
  agent; `limits.max_concurrent_tasks` bounds what that agent runs at once, and nothing
  bounds what is waiting.
- **Progress is derived, never stored.** Correct by construction — the board is a shared
  surface and a NOVA-side copy would drift — but it means an objective whose tasks were
  archived out of the runtime reads as not started.
- **`may_assign_to` is not transitive.** If A may assign to B and B to C, an objective owned
  by A still may not route a step to C. That is deliberate: transitive delegation is how a
  narrow grant silently becomes a broad one.

---

## Tests

`tests/platform/test_supervisor.py` (28) — declaration, cycles, and the routing rule tested
as a rule: every refusal reason, self-assignment, disabled agents, and a test proving the
example bundle is governed by its declaration rather than passing by accident.

`tests/platform/test_supervisor_submit.py` (15) — against a real `kanban.db`: parallelism,
dependency ordering, idempotency, a refusal writing nothing, and the write-ahead audit pair.
Skipped when the runtime is not importable, which is NOVA's own normal test state.
