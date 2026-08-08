# ManagedRun / ProfileRun Manifest Contract Plan

> **Status:** proposal / contract draft. This document is intentionally docs-only and
> does not implement a new runtime API. It captures the smallest shared manifest
> shape needed to align follow-up work around Kanban task runs, SessionDB, ACP
> session persistence, possible future control-plane work, and external
> workbenches.

**Goal:** Define an initial ManagedRun / ProfileRun manifest contract that lets an
orchestrator verify what actually ran, where it ran, which runtime identity was
used, what artifacts were produced, and whether the intended live agent process
advanced.

**Architecture:** Phase 1 is grounded in current durable Kanban `task_runs`
attempts, joined to parent `tasks` only to enrich an already-existing attempt
manifest with requested-work data; SessionDB for
session/runtime metadata and transcripts; and ACP's scoped restoration of
persisted ACP sessions. The manifest is a contract layer over those records.
Fields without a current source or explicit correlation remain `null` or are
reported as unsupported. A future control plane could fill additional fields
after its runtime implementation lands.

**Tech Stack:** Python, Kanban's SQLite `tasks` + `task_runs`, SQLite SessionDB, ACP
session persistence/resume, and JSON/YAML manifest serialization. Terminal
sandbox data and a profile-control plane are later, conditional integrations,
not Phase 1 data sources.

---

## Background

Hermes already has several primitives that are close to a durable managed-agent
runtime:

- Profiles provide named agent configurations.
- `delegate_task` provides lightweight synchronous subagents.
- Kanban provides durable multi-agent task dispatch.
- SessionDB stores conversation history and related session metadata.
- ACP persists and resumes ACP-sourced sessions using stored cwd/model/provider
  data; it does not provide profile leases or claim records.
- The closed, unmerged `agent_control` PR describes a possible future
  profile-control direction, but its handles, leases, and run records do not
  exist on current main.
- Terminal backends provide local, Docker, SSH, and other execution surfaces.

The missing piece is a shared run contract that sits above those primitives. A
caller should not have to trust a free-form summary like `done`. It should be
able to inspect a durable manifest with machine-readable status, runtime
identity, artifacts, validation results, logs, and failure state.

Related discussion:

- [#26675](https://github.com/NousResearch/hermes-agent/issues/26675) — Managed
  Agent Runtime contracts on top of `agent_control` / Kanban / SessionDB.
- [#18420](https://github.com/NousResearch/hermes-agent/issues/18420) —
  multi-agent orchestration pipeline direction.
- [#18493](https://github.com/NousResearch/hermes-agent/pull/18493) — closed,
  unmerged historical proposal for an `agent_control`-based direction; any use
  here is conditional on a future runtime implementation.
- [#8943](https://github.com/NousResearch/hermes-agent/issues/8943) — Docker
  sandbox discussion relevant to worker lifecycle.

---

## Non-goals

- Do not replace `delegate_task`. Keep it as the fast synchronous subtask path.
- Do not require Redis, S3, Postgres, or a visual workflow editor for the first
  implementation.
- Do not require every sandbox backend to provide identical isolation guarantees.
- Do not define model-routing policy, budget policy, or loop policy here. Those
  can layer on top after the manifest exists.
- Do not make external workbenches trust HTTP success alone. A control-plane
  update is not the same as target-runtime advancement.

---

## Proposed manifest shape

The manifest should be serializable as JSON and YAML. Field names below are a
contract draft; implementations may start with a subset, but the subset should be
explicit in `schema_version` and `capabilities`.

```yaml
schema_version: managed-run-manifest-0
run_id: string  # non-null stringified task_runs.id in Phase 1
parent_run_id: string | null
profile: string | null
status: claimed | starting | running | blocked | succeeded | failed | cancelled | timed_out | scheduled | stale | rate_limited | unknown
phase: accepted | claimed | sandbox_started | runtime_started | runtime_advanced | validating | completed | null
capabilities:
  live_target_receipt: boolean
  artifact_validation: boolean
  session_resume: boolean
source:
  kind: string | null
  task_id: string | null
  raw_status: string | null
  raw_outcome: string | null
  session_correlation_provenance: string | null
runtime:
  provider: string | null
  model: string | null
  api_mode: string | null
  toolsets: [string] | null
  mcp_servers: [string] | null
  profile_config_source: string | null
workspace:
  kind: scratch | dir | worktree | null
  path: string | null
  git_ref: string | null
  dirty: boolean | null
sandbox:
  backend: local | docker | ssh | modal | daytona | other | null
  sandbox_id: string | null
  process_id: string | null
  process_generation: string | null
  resource_limits: object | null
  ttl_seconds: number | null
session:
  session_id: string | null
  parent_session_id: string | null
  forked_from: string | null
  resume_mode: new | resume | fork | clone | null
inputs:
  task: string | null
  artifacts: [path] | null
  metadata: object | null
output_contract:
  required_artifacts: [path] | null
  schema: object | null
  validation_command: string | null
outputs:
  summary: string | null
  artifacts: [path] | null
  metadata: object | null
validation:
  status: not_run | passed | failed | skipped | null
  command: string | null
  exit_code: number | null
  output_excerpt: string | null
live_target_receipt:
  status: not_required | pending | confirmed | stale | rejected | wrong_runtime | cursor_not_advanced | null
  session_id: string | null
  workspace_path: string | null
  runtime_instance_id: string | null
  process_generation: string | null
  input_cursor_before: string | number | null
  input_cursor_after: string | number | null
  event_cursor_before: string | number | null
  event_cursor_after: string | number | null
observability:
  logs: [path] | null
  recent_events: [object] | null
  token_usage: object | null
  cost: object | null
  started_at: timestamp | null
  ended_at: timestamp | null
failure:
  kind: none | spawn_failed | timeout | validation_failed | crashed | cancelled | blocked | gave_up | released | reclaimed | stale | rate_limited | rejected | wrong_runtime | cursor_not_advanced | unknown | null
  error: string | null
  retry_count: number | null
  block_reason: string | null
```

---

## Lifecycle states

A managed profile run should distinguish these transitions:

1. **accepted** — the orchestration layer accepted a requested run.
2. **claimed** — a runner claimed responsibility for the run.
3. **sandbox_started** — a concrete workspace / sandbox / session was allocated.
4. **runtime_started** — the target runtime process exists and has a runtime
   identity.
5. **runtime_advanced** — the intended live runtime consumed the input and
   advanced its cursor.
6. **validating** — output contract validation is running.
7. **completed** — the run reached a terminal state and wrote its final manifest.

These are contract states, not all states currently observable in Phase 1.
A queued/pre-claim Kanban `tasks` row is a pre-run request projection, not a
Phase 1 ManagedRun manifest. Phase 1 emits a manifest only after a `task_runs`
row exists, including any persisted task-run row and synthesized terminal rows
for transitions such as startup failure or scheduling when no open run exists.
Its non-null `run_id` is the stringified `task_runs.id`; the parent task only
enriches that attempt manifest. `phase` is `null` whenever the current records
do not establish one of these contract transitions. In particular, a Kanban
heartbeat is attempt liveness only and never maps to `runtime_started` or
`runtime_advanced`.
Sandbox allocation, target runtime identity/advancement, and validation require
later producers; until then their phases and dependent fields are unsupported.
This distinction prevents a Kanban claim or heartbeat from being misreported as
proof that a particular live runtime advanced.

---

## Live target receipt

External workbenches and control planes often drift at the boundary between
`request accepted` and `the intended live process actually moved`. A durable run
manifest should make that boundary explicit.

A `live_target_receipt` is a later capability that confirms that the correct
target runtime advanced. Current Kanban, SessionDB, and ACP persistence expose
no live control-plane signal sufficient to produce that confirmation. Phase 1
therefore sets `capabilities.live_target_receipt` to `false` and every receipt
field, including `status`, to `null`/unsupported (using `not_required` only when
the run contract truly does not require a receipt). When a future producer exists, it is
not enough for an API endpoint to return 200 or for a database row to update. The
receipt should record:

- session id observed by the target runtime;
- workspace or worktree path observed by the target runtime;
- runtime instance id or lease id, if a future control plane provides one;
- process generation, so restarted processes do not masquerade as old ones;
- input cursor before and after the action;
- event cursor before and after the action;
- typed failure state when the runtime is stale, rejects the input, or is the
  wrong target.

Recommended receipt statuses:

| Status | Meaning |
| --- | --- |
| `not_required` | This run type does not require live target confirmation. |
| `pending` | A receipt is required but has not arrived yet. |
| `confirmed` | The intended runtime advanced and cursors moved as expected. |
| `stale` | The claimed runtime/session is too old to accept the action. |
| `rejected` | The intended runtime explicitly rejected the action. |
| `wrong_runtime` | The action landed on a different runtime/session/workspace. |
| `cursor_not_advanced` | The runtime was contacted, but no input/event cursor moved. |

---

## Artifact contract and validation

A managed run should declare its required outputs before work begins. The caller
should receive a validation result, not only a natural-language claim that files
were written.

Minimum artifact fields:

- `inputs.artifacts` — input files or artifact ids made available to the run.
- `output_contract.required_artifacts` — files or artifact ids that must exist at
  completion.
- `output_contract.schema` — optional JSON Schema or equivalent structured
  contract.
- `output_contract.validation_command` — optional command run in the workspace or
  sandbox to prove the output is usable.
- `validation.status` — `passed`, `failed`, `skipped`, or `not_run`.

Validation failures should be terminal unless the orchestrator has an explicit
retry policy.

---

## Session and workspace semantics

The manifest should separate three concepts that are easy to conflate:

- **Context window** — the model-visible prompt state for the current turn.
- **Session history** — durable event/transcript history stored by SessionDB or a
  future SessionStore abstraction.
- **Workspace** — filesystem state for the run, such as a scratch directory or
  git worktree.

A managed run may compact its context window without losing durable session
history. It may also fork or resume a session while using a new workspace. The
manifest should make these choices visible through `session.resume_mode`,
`session.forked_from`, and `workspace.kind`.

---

## Mapping to existing primitives

### `delegate_task`

`delegate_task` can remain summary-oriented and synchronous. If it emits a
manifest later, it can use a reduced form:

- no durable async claim lease required;
- no sandbox lifecycle required unless the child uses one;
- no live target receipt required by default;
- still useful to record profile/model/toolsets/session/artifacts when
  available.

### Kanban task and task-run attempt/lifecycle

`hermes_cli/kanban_db.py` currently defines durable parent `tasks` and child
`task_runs`, linked by `task_runs.task_id = tasks.id`. The task supplies title,
optional body, assignee, task status, workspace kind/path, branch name, optional
model override, and originating `session_id`. Joined `task_attachments` rows
supply input attachment paths and metadata when present. For `inputs.task`, the
exact Phase 1 mapping is the task title followed by the body when the body is
present; it describes requested task-level work, not worker transcript input.
`inputs.artifacts` contains the joined attachment `stored_path` values, or is
`null` when none are recorded; no empty list is invented. Other task/input
metadata is populated only when its documented meaning and provenance are
preserved, otherwise it is `null`.

A task-run record grounds manifest attempt identity and lifecycle using `id`,
`task_id`, optional `profile`, `step_key`, `status`, claim lock/expiry, worker
pid, maximum runtime, heartbeat, start/end timestamps, outcome, summary,
metadata, and error. `outputs.summary` and `outputs.metadata` map respectively
from task-run summary and metadata only when present. `failure.retry_count`
remains `null`: current task/run records do not store a canonical per-attempt
retry ordinal, and task-level failure counters or retry policy are not the same
thing.

A pre-claim `tasks` row is only a pre-run request projection and does not emit a
Phase 1 run manifest. Phase 1 includes any persisted `task_runs` row, including
synthesized terminal attempts for transitions such as startup failure or
scheduling when no open run exists. Phase 1 sets the
manifest's non-null `run_id` to the stringified `task_runs.id`. The joined task
can additionally populate task-level workspace and model-override fields, but
these describe requested/resolved task configuration, not independently
observed run-specific runtime state. A heartbeat proves liveness in the Kanban
worker path, not that a specific model runtime consumed input.

The proposed manifest's `source` record preserves provenance from the current
rows; it is not a claim that a separate source record exists today. For a
Kanban attempt, `source.kind` is `kanban_task_run`, `source.task_id` preserves
the parent id, and `source.raw_status` and `source.raw_outcome` preserve the
unmodified `task_runs` values. `source.session_correlation_provenance` records
how an optional SessionDB correlation was established (for example, a named
task-run metadata key); it remains `null` without explicit correlation.

Kanban normalization is an adapter/view contract. It does not mutate `tasks` or
`task_runs`, and it preserves the raw values above even when they are unknown.

| Current `task_runs` source values | Manifest `status` | Manifest `phase` | `failure.kind` |
| --- | --- | --- | --- |
| open row: `status=running`, `outcome=null` | `running` | `claimed` | `null` |
| `status=done`, `outcome=completed` | `succeeded` | `completed` | `none` |
| `status=blocked`, `outcome=blocked` | `blocked` | `completed` | `blocked` |
| `status=crashed`, `outcome=crashed` | `failed` | `completed` | `crashed` |
| `status=timed_out`, `outcome=timed_out` | `timed_out` | `completed` | `timeout` |
| `status=failed`, `outcome=spawn_failed` | `failed` | `completed` | `spawn_failed` |
| status/outcome `gave_up` | `failed` | `completed` | `gave_up` |
| status/outcome `released` | `cancelled` | `completed` | `released` |
| status/outcome `reclaimed` | `cancelled` | `completed` | `reclaimed` |
| status/outcome `scheduled` | `scheduled` | `completed` | `null` |
| status/outcome `stale` | `stale` | `completed` | `stale` |
| status/outcome `rate_limited` | `rate_limited` | `completed` | `rate_limited` |
| Any future/unrecognized status/outcome | `unknown` | `null` unless established conservatively | `null` unless established conservatively |

Heartbeat data does not change the open-row mapping: it is liveness only. An
adapter must not guess lifecycle or failure details for an unrecognized pair.

`tasks.session_id` is the originating chat/agent session association propagated
through `HERMES_SESSION_ID` when the task was created. It is not a run-specific
worker SessionDB id and is not proof that an attempt executed in that session.
It may be exposed as explicitly labelled task provenance, but must not populate
`session.session_id` for the target run without a separate explicit
provenance/correlation contract.

### SessionDB runtime and session metadata

`hermes_state.py` currently stores session `id`, `source`,
`parent_session_id`, model/model configuration, start/end/end reason, cwd/git
fields, billing provider/base URL/mode, usage, cost, and transcript history.
When a task run is explicitly correlated to a SessionDB record, these values can
populate the corresponding session, runtime, workspace, observability, and
transcript lookup fields.

There is no canonical Kanban `task_runs` to SessionDB session foreign key today.
Phase 1 must record a correlation explicitly (for example, a session id in
task-run metadata) and identify its provenance; it must not infer one from
timestamps or profile names. Without explicit correlation, SessionDB-derived
manifest fields remain `null`/unsupported. SessionDB also does not provide a
Kanban claim/lease or proof of live-runtime advancement.

### ACP session persistence and resume

`acp_adapter/session.py` restores a persisted session only when its SessionDB
`source == "acp"`, recreating an ACP agent from persisted cwd/model/provider
data. This can support `capabilities.session_resume` and resume-related fields
only for explicitly correlated ACP-sourced sessions.

ACP persistence is not a profile-control plane. It supplies neither Kanban
attempt claims/leases nor a live-runtime receipt, and it must not be used to
infer them.

### Future `agent_control` integration

The `agent_control` proposal in PR #18493 is closed and unmerged. Current main
has none of its proposed run handles, leases, or profile-control records, so it
is not a Phase 1 source. If a compatible runtime implementation lands later, an
adapter could map its handles and control events into manifest identity and
lifecycle fields while keeping control-plane acknowledgement distinct from
`live_target_receipt.confirmed`. Until then, every field that depends on that
proposal remains `null`/unsupported.

### Phase 1 data-source summary

| Manifest information | Current source | Phase 1 limit |
| --- | --- | --- |
| Requested task, assignee, workspace kind/path/branch, model override, originating task/session association | Kanban `tasks` joined to an existing `task_runs` attempt | Enrichment only; a pre-claim task is a pre-run request projection and emits no Phase 1 manifest; originating `tasks.session_id` is not the worker run session |
| Input attachment paths/metadata | `task_attachments` joined on task id | Task inputs only; absence maps to `null`, not an invented empty list |
| Attempt id/profile/status, claim, heartbeat, timestamps, outcome/summary/metadata/error | Kanban `task_runs` joined to `tasks` by `task_runs.task_id = tasks.id` | Manifest begins only when this row exists; `run_id` is its stringified id; raw status/outcome are preserved and normalized by the table above; no run-specific session, logs, output artifacts, or runtime receipt |
| Session/parent, model configuration, cwd/git, provider/mode, usage/cost, transcript | Explicitly correlated SessionDB session | No canonical foreign key from Kanban; otherwise `null`/unsupported |
| Session resume | Explicitly correlated ACP session with `source == "acp"` | Persistence/resume only; no profile lease or control-plane record |
| Sandbox/process identity, output artifacts/log paths, live target receipt | No verified general current source | `null`/unsupported; later producer required |
| Proposed control handles/leases | Future `agent_control`-like implementation | Not available on current main |

---

## Incremental implementation path

### Step 1: Document and validate the schema

- Add a small schema representation for the manifest without adding a runtime
  or storage API.
- Add JSON fixture examples for a Kanban-only attempt and an explicitly
  SessionDB-correlated attempt, with unavailable fields represented as `null`
  and unsupported capabilities set to `false`.
- Add tests that fixtures contain required top-level sections and terminal
  states, and do not claim unsupported capabilities.

### Step 2: Fill manifest fields from existing records

- Join `task_runs.task_id` to `tasks.id` and populate current task-level request,
  workspace/model-override, originating-session provenance, and attachment data
  separately from attempt/lifecycle fields on `task_runs`.
- Emit no Phase 1 manifest before `task_runs` exists. Set `run_id` to the
  stringified attempt id, preserve raw status/outcome in `source`, and apply the
  documented normalization without mutating Kanban rows.
- Optionally populate session/runtime metadata from an explicitly correlated
  SessionDB record, including ACP resume information only when the source is
  `acp`.
- `tasks.session_id` is only the originating task/session association. There is
  no canonical Kanban-task-run to worker SessionDB-session foreign key today.
  Represent correlation explicitly (for example in task-run metadata) and
  record its provenance; otherwise keep all dependent fields `null`.
- Keep every other unknown or unsupported field explicit as `null` instead of
  omitting it or inferring it.

### Step 3: Add artifact contract validation

- Support required output artifacts first.
- Add optional schema validation and validation command later.
- Fail with `failure.kind = validation_failed` when required artifacts are
  missing or validation fails.

### Step 4: Add live target receipts where a live runtime exists

- First add a runtime/control-plane producer that can identify the target and
  report advancement. A future `agent_control`-like adapter is one candidate,
  but can be integrated only after its runtime implementation lands.
- Record before/after input and event cursors around a workbench/control-plane
  action.
- Confirm receipt only when the intended runtime identity matches and the cursor
  advances.
- Fail with typed states such as `stale`, `wrong_runtime`, or
  `cursor_not_advanced` instead of a generic failure.

### Step 5: Expose manifests to observability surfaces

- CLI: show current phase, status, profile, workspace, validation status, and
  receipt status.
- Gateway/API: return a safe manifest summary for dashboards and external
  workbenches.
- Logs: link run manifest to recent events and session id.

---

## Acceptance criteria for the first implementation PR

- A manifest schema or fixture exists in the repository.
- A Phase 1 Kanban fixture joins a `task_runs` attempt to its parent `tasks` row,
  reports task-level request/workspace/model override/input attachments only
  when present, and reports attempt-level lifecycle, timing, outcome/summary,
  metadata, and error only when present.
- Every Phase 1 fixture has a non-null `run_id` equal to the stringified
  `task_runs.id`; no manifest fixture exists for a pre-claim task projection.
- Kanban fixtures preserve source kind, task id, raw status, raw outcome, and
  explicit session-correlation provenance when applicable, and cover every
  documented normalization, including `scheduled`, `stale`, `rate_limited`,
  and conservative `unknown` handling.
- An explicitly correlated SessionDB fixture may additionally report its
  session, model/provider, cwd/git, usage/cost, and transcript linkage.
- Missing values (including optional profile, task inputs, output summary and
  metadata, retry count, phase, and validation state), missing run-specific
  SessionDB correlation, and unavailable sandbox/log/output-artifact/live-receipt
  fields are explicit `null`/unsupported rather than inferred or promised.
- A pre-claim task is represented only as a pre-run request projection, not a
  Phase 1 run manifest. `tasks.session_id` is labelled as originating
  provenance and never used as the run's `session.session_id`.
- A failed task run maps its existing error/outcome into the documented failure
  representation without inventing a new runtime signal.
- Existing Kanban, SessionDB, ACP, and `delegate_task` behavior remains
  backward-compatible; the first implementation requires no `agent_control`
  behavior.

## Acceptance criteria for live target receipts

These criteria apply to a later implementation after a live control-plane
signal exists; they are not Phase 1 acceptance criteria.

- A workbench/control-plane action can distinguish API acceptance from live
  runtime advancement.
- The receipt records target session, workspace/worktree, runtime instance or
  process generation, and before/after cursors.
- Wrong target, stale runtime, rejection, and no-cursor-advance cases are typed
  and testable.
- Dashboards can display receipt state without exposing private transcript
  content.

---

## Open questions

1. Should `schema_version` be numeric or named by contract family, e.g.
   `managed-run-manifest-0`?
2. Would a separate future TaskRequest manifest be useful for pre-claim request
   projections? Phase 1 ManagedRun identity is already decided: `run_id` is the
   stringified Kanban `task_runs.id`.
3. Should `live_target_receipt` be required for all managed profile runs, or only
   for long-lived live runtimes / workbench-driven actions?
4. Should validation commands run inside the target sandbox, in the caller's
   workspace, or in a separate verifier sandbox?
5. Which manifest fields are safe to expose through public dashboard/API
   surfaces by default?
6. If a future profile-control runtime lands, how should its identity and
   lifecycle records correlate with existing Kanban and SessionDB records?
