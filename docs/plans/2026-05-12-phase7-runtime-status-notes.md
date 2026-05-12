# Phase 7 Runtime Orchestration Context Notes

## Summary

Phase 7 adds the smallest live "place" where CLI/gateway/session code can hold
and reach the Phase 3 `TaskRegistry` + Phase 4 `WorkerLaneRegistry`, so the
Phase 6 read-only status formatter becomes an *observable runtime* rather than a
formatter library only.

New module: `agent/orchestration_runtime.py`

- `OrchestrationRuntime` — a tiny dataclass holding one `TaskRegistry` and one
  `WorkerLaneRegistry`, with thin pass-throughs to Phase 6:
  - `OrchestrationRuntime.create()` → fresh, empty, in-memory `TaskRegistry`
    (`path=None` — no file persistence) + empty `WorkerLaneRegistry` (no lanes).
  - `snapshot(*, session_key=None)` → `OrchestrationSnapshot` (JSON-safe `to_dict`).
  - `format_tasks(*, session_key=None, compact=True)` → delegates to
    `orchestration_status.format_tasks` against the held task registry.
  - `format_agents(*, compact=True)` → delegates to `orchestration_status.format_agents`.
  - `format_overview(*, session_key=None, compact=True)` → delegates to
    `orchestration_status.format_overview`.
- Per-owner helpers (no global singleton; storage attribute is `RUNTIME_ATTR =
  "_orchestration_runtime"`):
  - `get_orchestration_runtime(owner) -> OrchestrationRuntime | None`
  - `set_orchestration_runtime(owner, runtime) -> OrchestrationRuntime`
    (rejects a non-`OrchestrationRuntime`)
  - `get_or_create_orchestration_runtime(owner) -> OrchestrationRuntime`
    (attaches a fresh empty runtime if absent)
  - `format_runtime_tasks(owner, ...)` / `format_runtime_agents(owner, ...)` /
    `format_runtime_overview(owner, ...)` — get-or-create on `owner`, then format.

New tests: `tests/agent/test_orchestration_runtime.py` (12 tests).

Slash command wiring (`/tasks`, `/agents`) is **deferred** — see "Risks /
Follow-up" for why; the runtime helpers above are exactly what a later, focused
command-wiring phase builds on.

## PURPOSE-FIT DESIGN RATIONALE

Phase 6 deliberately did not wire `/tasks` / `/agents` because "there is not yet a
long-lived runtime task/worker registry for those commands to read". Phase 7's
*only* job is to remove that blocker with the minimum surface:

- A **container, not a runtime engine.** `OrchestrationRuntime` owns no threads,
  no locks, no background work, no worker dispatch. It holds the two registries
  the substrate already provides and forwards reads to the Phase 6 formatter.
  Construction is "explicit injection" (`OrchestrationRuntime(task_registry=...,
  worker_registry=...)`, used by tests / a future phase that already owns them)
  with `create()` as the only sugar — there is no implicit shared default.
- **Per-owner attachment, not a module global.** The packet was explicit: no
  global singleton. So a runtime lives on whatever object owns it (`HermesCLI`,
  a gateway runner, a session object, a test dummy) under one private
  `_orchestration_runtime` attribute, reached only via `getattr`/`setattr` —
  fully duck-typed about the owner, zero module-level mutable state, two owners
  get two independent runtimes.
- **"Create if absent" attaches an *empty* runtime, never a fabricated one.**
  `get_or_create_orchestration_runtime` / the `format_runtime_*` helpers attach a
  fresh empty `OrchestrationRuntime.create()` when the owner has none, so a
  status surface always has *something truthful* to read — an empty board
  ("No active tasks are currently registered.") — and later work that populates
  the registries is visible through the same handle. No fake state, no
  misleading dashboard.
- **Read-only on the same path Phase 6 already vetted.** `snapshot()` /
  `format_*` go through `build_snapshot` / `format_tasks` / `format_agents` /
  `format_overview`, which count `pending_followups` via `len(...)` and never
  iterate their payloads — so `PendingTurnItem.raw` is never inspected,
  serialised, copied, or deep-copied anywhere on this path. Everything
  `snapshot().to_dict()` returns is plain JSON-safe data.
- **No Phase 6 changes.** `agent/orchestration_status.py` already exposes exactly
  the functions a runtime needs; `OrchestrationRuntime` just calls them. No
  "small adapter hook" turned out to be necessary. `runtime.format_tasks()`
  therefore inherits Phase 6's documented registry behaviour — active-only,
  session-scoped, worker *linkage* shown (`worker: <id> (<kind>)`); the live
  worker *status* annotation (`… [running]`) appears in `format_agents()` and is
  cross-linked into the task block by `format_overview()` (which builds a
  snapshot of both registries internally).

## WHAT YOU INTENTIONALLY DID NOT BUILD

This phase does not implement:

- `/tasks` / `/agents` slash-command rewiring (deferred — see below).
- gateway/Telegram natural-language status auto-routing.
- append/correction attachment from live gateway messages.
- worker dispatch or new-task creation (the runtime starts/polls/kills no
  workers; a future phase that wants a lane calls
  `runtime.worker_registry.register(lane)` itself).
- cancel / stop / force-kill task behaviour.
- a durable routing DB / SQLite schema (the task registry stays `path=None`
  in-memory; no persistence is wired).
- the public `delegate_task(background=True)` API.
- the Ralph / focused-agent runtime.
- an LLM / model-based follow-up classifier.
- a global singleton runtime.
- a broad CLI / gateway / TUI refactor.
- a worker result synthesis / delivery pipeline.

## RALPH/FUTURE FOCUSED-AGENT NOTES

The layering is now:

```text
(future) Ralph / focused-agent runtime, worker dispatch
  → OrchestrationRuntime            # Phase 7: holds the live registries on an owner
    → TaskRegistry + WorkerLaneRegistry   # Phase 3/4: state
      → orchestration_status.*      # Phase 6: read-only presentation
        → /tasks /agents, natural-language status   # later: thin handlers
```

Concretely, a later phase:

- **Command wiring** — once the integration surface is chosen, a thin handler is
  `format_runtime_overview(self)` (for a combined view) or
  `format_runtime_tasks(self)` / `format_runtime_agents(self)` against the
  CLI/gateway object — no new plumbing, the helper get-or-creates an empty
  runtime so even a not-yet-populated owner answers gracefully. (It must pick
  fresh command names, or consciously repurpose the existing `/tasks` / `/agents`
  — see below.)
- **Natural-language status (gateway)** — gate on
  `orchestration_status.looks_like_orchestration_status_query(text)`, then call
  `format_runtime_overview(session_or_runner, session_key=...)`. Read-only;
  mutates nothing.
- **Follow-up routing / worker dispatch** — `FollowupRouter.route(...)` already
  takes a `TaskRegistry` (+ optional `WorkerLaneRegistry`); pass
  `runtime.task_registry` / `runtime.worker_registry`. A worker lane (Claude
  Code, terminal, …) is registered with `runtime.worker_registry.register(lane)`;
  task↔worker linkage uses `worker_lanes.link_worker_to_task(runtime.task_registry,
  task_id, handle)`.
- **A "Ralph" unit** is then naturally "one `FocusedTask` + one `WorkerLane`
  handle, observable through this runtime" — no new abstraction needed yet.

This keeps the separation clean: runtime owns *where the state lives*, the
registries own *state*, the router owns *follow-up policy*, the status module
owns *presentation*, and Hermes main owns *synthesis / accountability*.

## Validation

```text
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/agent/test_orchestration_runtime.py \
  tests/agent/test_orchestration_status.py \
  tests/agent/test_followup_router.py \
  tests/agent/test_worker_lanes.py \
  tests/agent/test_task_registry.py \
  tests/agent/test_pending_turn_queue.py -q

129 passed, 8 warnings
```

```text
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m pytest \
  tests/cli/test_busy_queue_coalescing.py \
  tests/cli/test_busy_input_mode_command.py \
  tests/gateway/test_restart_drain.py \
  tests/gateway/test_session_race_guard.py -q

65 passed, 8 warnings
```

```text
/Users/wookim/.hermes/hermes-agent/venv/bin/python -m compileall -q \
  agent/orchestration_runtime.py agent/orchestration_status.py agent/followup_router.py \
  agent/worker_lanes.py agent/task_registry.py agent/pending_turn_queue.py cli.py gateway/run.py

git diff --check

both clean / passed
```

New-test coverage (`tests/agent/test_orchestration_runtime.py`):

- `create()` → fresh empty in-memory registries; two creates independent.
- explicit registry injection used as-is.
- `get` / `set` / `get_or_create` helpers on a dummy owner; idempotent;
  replacement; `set` rejects non-runtime; `get` ignores a foreign value in the slot.
- no global-singleton leakage between owners; a brand-new owner reads empty.
- empty runtime → graceful empty `format_tasks` / `format_agents` / `format_overview`
  and an empty, JSON-safe snapshot; `format_runtime_*` helpers create an empty
  runtime on a missing owner.
- injected task + running worker → correct `format_runtime_tasks` /
  `format_runtime_agents` / `format_runtime_overview` (active-only tasks, live
  worker line, cross-linked worker status in the overview), correct snapshot
  counts, no consistency warnings; JSON-safe.
- `runtime.snapshot()` JSON-safe and not deep-copy-traversed even when a task has
  a follow-up carrying a non-JSON, deepcopy-hostile `raw` passthrough.
- `runtime.format_tasks()` session-scoped and active-only.
- a worker carrying non-JSON-safe spec metadata degrades (no goal) instead of
  raising, through the runtime.

## Risks / Follow-up

- **Command wiring deferred — why.** `/agents` (with `/tasks` as an alias)
  *already exists* in `hermes_cli/commands.py::COMMAND_REGISTRY` and resolves to
  `_handle_agents_command` in both `cli.py` and `gateway/run.py`, where it lists
  *background processes / subagent state* — an unrelated, established feature.
  Pointing those names at the orchestration runtime would change existing
  behaviour (the packet requires "no normal chat/gateway behavior changes"), and
  *adding* runtime output to those handlers means threading an `OrchestrationRuntime`
  through CLI construction / gateway session dispatch — touching the 600k-char
  `cli.py` and `gateway/run.py` — which is exactly the broad CLI/gateway refactor
  this phase stops short of. So Phase 7 ships the runtime + helpers and leaves the
  command surface to a later, focused phase that decides: new names
  (`/orchestration`, `/board`, …) vs. consciously merging into the existing
  `/agents`. The `format_runtime_*` one-liners make either path small.
- Natural-language gateway status routing remains a later thin integration
  (`looks_like_orchestration_status_query` + `format_runtime_overview`).
- Durable persistence stays deferred: `OrchestrationRuntime.create()` builds an
  in-memory `TaskRegistry` (`path=None`); restart recovery is a later phase and
  must still avoid serialising `PendingTurnItem.raw`.
- The runtime helpers `getattr`/`setattr` a private attribute on the owner; an
  owner with restrictive `__slots__` lacking that slot would raise on `set` —
  acceptable, since the intended owners (`HermesCLI`, gateway runner, session,
  test dummy) all allow attribute assignment.
- `runtime.format_tasks()` inherits Phase 6's registry behaviour, so the inline
  worker line is linkage-only (`worker: <id> (<kind>)`); the live worker status
  shows in `format_agents()` / `format_overview()`. If a future `/tasks` wants the
  live status inline too, that is a small Phase 6 adapter (a `worker_registry=`
  kwarg on `format_tasks`) — intentionally not done now.
