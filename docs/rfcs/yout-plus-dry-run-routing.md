# Yout Plus Routing — Dry-Run Mode (Design Spec)

> Status: design only, not yet implemented.
> Source files affected: `hermes_cli/kanban_decompose.py`, `hermes_cli/kanban_db.py`,
> `hermes_cli/kanban.py` (CLI surface), `tools/kanban_tools.py` (MCP surface).
> Related: `docs/profile-routing.md` (message routing — a different system; this doc
> covers *task* routing, i.e. the Kanban auto-decomposer that predicts an owning
> profile and fans a task out into a child dependency graph).

## Problem

Yout Plus routes a task through `kanban_decompose.decompose_task()`: it resolves the
owning profile, builds the context a worker will see, asks the LLM for a child
dependency graph, then persists that graph (`kb.decompose_triage_task` /
`kb.specify_triage_task`) and lets the dispatcher pick up the new rows and spawn
workers. There is currently no way to see the predicted routing decision without
committing it — every call mutates the board and can trigger real worker dispatch.

Dry-run mode makes the *prediction* observable without the *mutation*.

## Input contract

Identical to live routing: a task already sitting in the DB (any status; live routing
requires `triage`, dry-run does not enforce that, since it never transitions status).
Callers pass either:

- `task_id` of an existing task (read-only lookup), or
- ad-hoc `title` + `body` with no backing row at all, for previewing routing on text
  that hasn't been created as a card yet.

```python
def dry_run_route(
    *,
    task_id: str | None = None,
    title: str | None = None,
    body: str | None = None,
) -> DryRunResult: ...
```

Exactly one of `task_id` or `title` must be given. When `task_id` is given, the
function reads the row (`kb.get_task`, a plain `SELECT`) but does not require any
particular status and does not lock or update it.

## Output contract — `DryRunResult`

```python
@dataclass
class DryRunResult:
    ok: bool
    reason: str = ""                       # populated when ok=False
    predicted_owner: str | None = None     # (a)
    context_envelope: dict | None = None   # (b)
    dependency_graph: list[dict] | None = None  # (c)
    rationale: str = ""                    # decomposer's one-line "why"
    fanout: bool = False                   # False => single-task routing, no children
```

- **(a) predicted_owner** — the assignee the *root* task would land on: for
  `fanout=false` this is the single resolved assignee; for `fanout=true` this is the
  orchestrator profile that stays parent of the graph (mirrors `root_assignee` in
  `kb.decompose_triage_task`).
- **(b) context_envelope** — the exact `worker_context` shape a dispatched worker
  would receive: title, body, parent handoffs (empty for a fresh task), recent related
  work by the predicted assignee, and the roster/prompt inputs used to produce the
  prediction (roster snapshot, orchestrator, default_assignee). This is assembled with
  the same helper the live path uses (`kanban_show`'s `worker_context` formatter) so
  dry-run output cannot drift from what a live worker actually sees.
- **(c) dependency_graph** — for `fanout=true`, the validated children list exactly as
  `decompose_triage_task` would insert it, but as plain dicts, never written:
  `[{"index": 0, "title": ..., "body": ..., "assignee": ..., "parents": [...]}]`.
  Parent indices are 0-based into this same list, pre-cycle-checked. For
  `fanout=false`, this is `None` and the single-task title/body/assignee is reported
  via `predicted_owner` plus `context_envelope`.

## Toggle

- **Library**: `dry_run_route(...)` is a new, separate entrypoint — not a boolean flag
  threaded through `decompose_task()`. Keeping it a distinct function (rather than an
  `if dry_run:` branch inside the existing one) is the smallest way to guarantee the
  mutating path is physically unreachable from the dry-run call, instead of relying on
  a conditional that could regress under future edits.
- **CLI**: `hermes kanban decompose <task_id> --dry-run` — prints the `DryRunResult` as
  JSON to stdout, exit 0 on `ok=True`. No DB write occurs; this is verifiable with
  `git diff` on the DB file's mtime / a `PRAGMA data_version` check in tests.
- **MCP tool surface**: `kanban_tools.py` gets a `dry_run: bool = False` parameter on
  the decompose-triggering tool. `True` routes to `dry_run_route()` instead of
  `decompose_task()` before any tool-level side effect fires.

## Where the code path diverges from live routing

`decompose_task()` today does, in order:

1. `kb.get_task()` — read.
2. Build roster (`_build_roster`), resolve orchestrator/default assignee — pure.
3. `call_llm(...)` — external call, no DB effect.
4. Parse + validate the LLM's JSON (title/parents/cycle check) — pure.
5. **Mutate**: either `kb.specify_triage_task(...)` (single-task path) or
   `kb.decompose_triage_task(...)` (fan-out path) — the *only* two calls in the entire
   function that touch the database. `decompose_triage_task` wraps everything (row
   inserts, `link_tasks`, root status flip) in one `write_txn`; per its own docstring
   nothing outside that transaction opens a competing one.
6. Return an outcome. (No worker spawn happens here — that is a separate dispatcher
   loop that later polls for tasks in `ready` status and spawns a process per row.)

Steps 1–4 are prediction; step 5 is the sole mutation boundary; step 6 is reporting.
`dry_run_route()` shares steps 1–4 via an extracted pure helper (`_predict_routing()`,
new) and **stops before step 5**, returning the validated in-memory result instead of
calling either mutating function.

Because the dispatcher only ever spawns workers for rows that exist and are `ready`,
and dry-run never calls `create_task` / `link_tasks` / `decompose_triage_task` /
`specify_triage_task`, no row is ever created for the dispatcher to see — worker
invocation is prevented as a structural consequence of never crossing step 5, not by a
separate guard that has to be kept in sync.

## Functions wrapped / short-circuited

| Function | Live routing | Dry-run |
|---|---|---|
| `kb.get_task` | called | called (read-only, no status requirement) |
| `_build_roster`, `_resolve_orchestrator_profile`, `_resolve_default_assignee` | called | called unchanged (extracted into shared `_predict_routing`) |
| `call_llm` | called | called unchanged — prediction must reflect real decision logic |
| JSON parse + validation (title/parents/cycle check) | called | called unchanged |
| `kb.specify_triage_task` | called (single-task path) | **never called** |
| `kb.decompose_triage_task` (and everything inside its `write_txn`: task INSERTs, `link_tasks`, root status flip, audit comment) | called (fan-out path) | **never called** |
| Dispatcher worker spawn (separate process, polls `ready` tasks) | triggered indirectly once rows exist | never triggered — no rows are ever created |

## Tests to add (per parent task `t_2930a1af`)

1. Single-owner request → `fanout=false`, `predicted_owner` set, `dependency_graph is
   None`, zero DB writes (assert `PRAGMA data_version` unchanged, or row count
   unchanged for `tasks`/`task_deps`/`comments`).
2. Cross-functional request → `fanout=true`, `dependency_graph` has ≥2 entries with a
   real parent edge, zero DB writes.
3. Engineering-owned request → predicted_owner resolves to `engineering` via roster
   description match, zero DB writes.
4. Failure-path test: run `dry_run_route()` against a fixture DB, snapshot
   `sqlite3.connect(db).iterdump()` before and after, assert byte-identical. This is
   the strongest proof dry-run cannot mutate — asserting on the specific two function
   calls only proves the code doesn't call them today, not that nothing else did.

## Non-goals

- Not changing live routing behavior at all — `decompose_task()` keeps its current
  signature and effect.
- Not adding a dry-run mode to the dispatcher itself — irrelevant, since dry-run never
  produces rows for the dispatcher to act on.
