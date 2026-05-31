# ManagedRun / ProfileRun Manifest Contract Plan

> **Status:** proposal / contract draft. This document is intentionally docs-only and
> does not implement a new runtime API. It captures the smallest shared manifest
> shape needed to align follow-up work around `agent_control`, Kanban, SessionDB,
> profile invocation, and external workbenches.

**Goal:** Define an initial ManagedRun / ProfileRun manifest contract that lets an
orchestrator verify what actually ran, where it ran, which runtime identity was
used, what artifacts were produced, and whether the intended live agent process
advanced.

**Architecture:** Treat `agent_control` as the likely profile-control plane,
Kanban as a durable dispatcher / DAG backend, SessionDB as the durable transcript
source, and terminal backends as sandbox lifecycle providers. The manifest is a
contract layer over those primitives: implementations can fill the fields from
existing run records first, then add stricter artifact validation and live target
receipts incrementally.

**Tech Stack:** Python, SQLite SessionDB, Kanban task metadata, `agent_control`
run records, ACP profile control, terminal sandbox backends, JSON/YAML manifest
serialization.

---

## Background

Hermes already has several primitives that are close to a durable managed-agent
runtime:

- Profiles provide named agent configurations.
- `delegate_task` provides lightweight synchronous subagents.
- Kanban provides durable multi-agent task dispatch.
- SessionDB stores conversation history and related session metadata.
- ACP and `agent_control` work provide a profile-control direction.
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
- [#18493](https://github.com/NousResearch/hermes-agent/pull/18493) —
  `agent_control` durable ACP-based profile-agent orchestration direction.
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
run_id: string
parent_run_id: string | null
profile: string
status: queued | claimed | starting | running | blocked | succeeded | failed | cancelled | timed_out
phase: accepted | claimed | sandbox_started | runtime_started | runtime_advanced | validating | completed
capabilities:
  live_target_receipt: boolean
  artifact_validation: boolean
  session_resume: boolean
runtime:
  provider: string
  model: string
  api_mode: string | null
  toolsets: [string]
  mcp_servers: [string]
  profile_config_source: string | null
workspace:
  kind: scratch | dir | worktree
  path: string
  git_ref: string | null
  dirty: boolean | null
sandbox:
  backend: local | docker | ssh | modal | daytona | other
  sandbox_id: string | null
  process_id: string | null
  process_generation: string | null
  resource_limits: object | null
  ttl_seconds: number | null
session:
  session_id: string
  parent_session_id: string | null
  forked_from: string | null
  resume_mode: new | resume | fork | clone
inputs:
  task: string
  artifacts: [path]
  metadata: object
output_contract:
  required_artifacts: [path]
  schema: object | null
  validation_command: string | null
outputs:
  summary: string
  artifacts: [path]
  metadata: object
validation:
  status: not_run | passed | failed | skipped
  command: string | null
  exit_code: number | null
  output_excerpt: string | null
live_target_receipt:
  status: not_required | pending | confirmed | stale | rejected | wrong_runtime | cursor_not_advanced
  session_id: string | null
  workspace_path: string | null
  runtime_instance_id: string | null
  process_generation: string | null
  input_cursor_before: string | number | null
  input_cursor_after: string | number | null
  event_cursor_before: string | number | null
  event_cursor_after: string | number | null
observability:
  logs: [path]
  recent_events: [object]
  token_usage: object | null
  cost: object | null
  started_at: timestamp | null
  ended_at: timestamp | null
failure:
  kind: none | spawn_failed | timeout | validation_failed | crashed | cancelled | blocked | stale | rejected | wrong_runtime | cursor_not_advanced | unknown
  error: string | null
  retry_count: number
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

This prevents ambiguous states such as `running` from hiding where the system is
stuck. For example, a run that is accepted by the control plane but never
claimed by a worker should fail differently from a run that is claimed but
rejected by the live runtime.

---

## Live target receipt

External workbenches and control planes often drift at the boundary between
`request accepted` and `the intended live process actually moved`. A durable run
manifest should make that boundary explicit.

A `live_target_receipt` confirms that the correct target runtime advanced. It is
not enough for an API endpoint to return 200 or for a database row to update. The
receipt should record:

- session id observed by the target runtime;
- workspace or worktree path observed by the target runtime;
- runtime instance id or lease id;
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

### Kanban

Kanban can act as the durable graph / dispatcher backend:

- board task id maps to `run_id` or `parent_run_id`;
- task status maps to manifest `status` and `phase`;
- worker claim / heartbeat maps to `claimed`, `runtime_started`, and
  observability fields;
- task attachments or generated files map to manifest artifacts;
- blockers and retries map to `failure.block_reason` and `failure.retry_count`.

### `agent_control`

`agent_control` can provide the profile invocation and runtime-control plane:

- run handle maps to `run_id`;
- accepted / claimed / lease records map to lifecycle phases;
- ACP profile execution maps to runtime identity fields;
- control-plane acks should be distinct from `live_target_receipt.confirmed`.

### SessionDB / SessionStore

SessionDB can be the initial source for `session_id`, parent session links, and
transcript lookup. A future SessionStore abstraction can expose fork, clone,
resume, and event-slice operations without changing the manifest shape.

---

## Incremental implementation path

### Step 1: Document and validate the schema

- Add a small dataclass / TypedDict / Pydantic-style schema for the manifest.
- Add JSON fixture examples for success, validation failure, and wrong-runtime
  receipt.
- Add tests that fixtures contain required top-level sections and terminal
  states.

### Step 2: Fill manifest fields from existing records

- Populate `run_id`, `profile`, `status`, timestamps, runtime provider/model,
  workspace path, session id, logs, and basic failure metadata from the current
  `agent_control` and/or Kanban records.
- Keep unknown fields explicit as `null` instead of omitting them silently.

### Step 3: Add artifact contract validation

- Support required output artifacts first.
- Add optional schema validation and validation command later.
- Fail with `failure.kind = validation_failed` when required artifacts are
  missing or validation fails.

### Step 4: Add live target receipts where a live runtime exists

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
- A successful managed run can report profile, provider/model, workspace,
  session id, status, logs, artifacts, and timestamps.
- A failed managed run reports a typed `failure.kind`.
- Output artifact validation can fail a run with a structured validation result.
- Existing `delegate_task`, Kanban, and `agent_control` behavior remains
  backward-compatible.

## Acceptance criteria for live target receipts

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
2. Which existing record should own the canonical `run_id` when both Kanban and
   `agent_control` are involved?
3. Should `live_target_receipt` be required for all managed profile runs, or only
   for long-lived live runtimes / workbench-driven actions?
4. Should validation commands run inside the target sandbox, in the caller's
   workspace, or in a separate verifier sandbox?
5. Which manifest fields are safe to expose through public dashboard/API
   surfaces by default?
