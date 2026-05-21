# Jeeves Swarm Operator Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a production-safe Jeeves operator mode where every Garrett chat is received by Jeeves, routed into direct work / parallel subagents / durable jobs / scripts / dumb-pipe automations, then synthesized back into one verified response.

**Architecture:** Do not build a second agent runtime. Reuse Hermes gateway ingress, `AIAgent`, `delegate_task`, cron, kanban, Honcho/autopilot ingest, existing approval flows, and gateway status/progress callbacks. Add a thin swarm/job state layer plus operator routing policy and shadow rollout controls.

**Tech Stack:** Python, Hermes Agent gateway/runtime, SQLite/JSONL state, existing `delegate_task`, Honcho via existing autopilot ingest plugin, cron/kanban for durable work, pytest.

---

## Validated Existing Primitives

- Gateway ingress/response: `gateway/platforms/base.py`, `gateway/run.py`.
- Main runtime: `run_agent.py`, `agent/conversation_loop.py`.
- Parallel in-turn agents: `tools/delegate_tool.py` via `delegate_task(tasks=[...])`.
- Session/transcript state: `hermes_state.py`, `gateway/session.py`, `~/.hermes/state.db`.
- Durable scheduled work: `cron/jobs.py`, `cron/scheduler.py`.
- Durable worker/task substrate: `hermes_cli/kanban_db.py`, `tools/kanban_tools.py`.
- Approval flow: existing gateway `/approve` and `/deny`, `tools/approval.py`.
- Honcho signal path: `~/.hermes/plugins/autopilot-ingest/tools.py::ingest_signal`.
- Live canonical approval ledger: `~/.hermes/state/open_decisions.json`.
- Live machine audit pattern: `~/.hermes/state/*_state.json`, `~/.hermes/state/*_metrics.jsonl`.

## Non-Negotiables

1. Jeeves owns all user-facing synthesis; child agents do not DM Garrett.
2. Side effects are default-denied unless already permitted by Hermes policy or explicitly approved.
3. Subagents remain bounded by existing `delegation.max_concurrent_children`.
4. Honcho writes come from the operator/parent, not noisy subagents.
5. `delegate_task` is for synchronous fan-out; kanban/cron are for durable work.
6. n8n/Docker are dumb execution pipes, not reasoning engines.
7. Feature starts in shadow/dry-run mode before intercepting normal Slack behavior.

---

## Phase 1: Swarm State Model + Store

### Task 1: Add swarm model tests

**Objective:** Define durable, inspectable job/task/audit objects before implementation.

**Files:**
- Create: `tests/swarm/test_swarm_state.py`
- Create: `agent/swarm_state.py`

**Test cases:**
- `SwarmJob.create(...)` sets `status='received'`, stable ID, timestamps, original request, and empty task/audit lists.
- `SwarmJob.add_task(...)` appends a task and audit event.
- `SwarmJob.transition(...)` records status changes and audit events.
- JSON round-trip preserves job/task/permission/eval data.

**Run:**

```bash
python -m pytest tests/swarm/test_swarm_state.py -q -o 'addopts='
```

**Expected:** fail before implementation, pass after.

### Task 2: Implement `agent/swarm_state.py`

**Objective:** Add small dataclass/Pydantic-free model to avoid introducing runtime dependencies.

**Files:**
- Modify/Create: `agent/swarm_state.py`

**Implementation shape:**
- Dataclasses/enums for:
  - `SwarmJob`
  - `SwarmTask`
  - `RoutingPlan`
  - `PermissionGrant`
  - `EvalResult`
  - `AuditEvent`
- Helpers:
  - `new_job(...)`
  - `add_task(...)`
  - `transition(...)`
  - `to_dict()` / `from_dict()`
- Statuses:
  - job: `received`, `triaging`, `planning`, `awaiting_permission`, `running`, `verifying`, `summarizing`, `completed`, `failed`, `cancelled`, `partially_completed`
  - task: `queued`, `blocked`, `awaiting_permission`, `running`, `needs_review`, `verifying`, `completed`, `failed`, `cancelled`, `superseded`

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_state.py -q -o 'addopts='
```

---

## Phase 2: Swarm Store + Shadow Audit

### Task 3: Add JSONL store tests

**Objective:** Persist swarm audit without touching production `state.db` first.

**Files:**
- Create: `tests/swarm/test_swarm_store.py`
- Create: `agent/swarm_store.py`

**Test cases:**
- Store writes a job snapshot to `${HERMES_HOME}/state/swarm_operator_state.json`.
- Store appends events to `${HERMES_HOME}/state/swarm_operator_metrics.jsonl`.
- Store loads existing jobs on restart.
- Store handles corrupt/missing files gracefully with explicit errors and no silent data loss.

### Task 4: Implement `agent/swarm_store.py`

**Objective:** Add profile-safe state using `get_hermes_home()`.

**Files:**
- Modify/Create: `agent/swarm_store.py`

**Implementation shape:**
- `SwarmStore(base_dir=None)`
- `save_job(job)`
- `load_jobs()`
- `append_event(event)`
- Atomic write for JSON snapshot.
- Append-only JSONL for metrics/audit.

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_store.py tests/swarm/test_swarm_state.py -q -o 'addopts='
```

---

## Phase 3: Operator Routing Policy

### Task 5: Add routing policy tests

**Objective:** Classify requests deterministically before involving LLM routing.

**Files:**
- Create: `tests/swarm/test_swarm_routing.py`
- Create: `agent/swarm_router.py`

**Test cases:**
- Simple explanatory prompt -> `direct`.
- Multiple independent lookups or code/research/review request -> `swarm`.
- More than 3 procedural/validation/source-of-truth steps -> `script` candidate.
- External send/deploy/destructive wording -> permission required.
- n8n/docker wording -> `pipe` candidate with permission gate.

### Task 6: Implement `agent/swarm_router.py`

**Objective:** Add first-pass deterministic router used by Jeeves before tool fan-out.

**Files:**
- Modify/Create: `agent/swarm_router.py`

**Implementation shape:**
- `route_request(text, platform_context=None, config=None) -> RoutingPlan`
- Conservative heuristics only; no LLM required in v1.
- Include `reason`, `mode`, `suggested_tasks`, `permission_requests`, `verification_required`.

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_routing.py -q -o 'addopts='
```

---

## Phase 4: Gateway Shadow Hook

### Task 7: Add gateway shadow-mode hook tests

**Objective:** Every inbound Garrett message can create a shadow job record without changing response behavior.

**Files:**
- Create: `tests/swarm/test_gateway_swarm_shadow.py`
- Modify/Create: `gateway/builtin_hooks/swarm_operator.py`

**Test cases:**
- Hook receives inbound event metadata and message text.
- If disabled, it does nothing.
- If enabled in dry-run, it writes a `SwarmJob` and metrics event.
- Hook never blocks normal message handling on store failure; it logs a clear warning.

### Task 8: Implement shadow hook

**Objective:** Deploy the safest possible first slice: observe only, no child agents.

**Files:**
- Modify/Create: `gateway/builtin_hooks/swarm_operator.py`
- Modify: hook registration location if needed, likely `gateway/builtin_hooks/__init__.py` or `gateway/run.py` hook setup.

**Implementation shape:**
- Config keys, default disabled:
  - `swarm_operator.enabled: false`
  - `swarm_operator.dry_run: true`
  - `swarm_operator.max_children: 3`
  - `swarm_operator.persist_to_honcho: false`
- On inbound message:
  - create job
  - route
  - write state/metrics
  - do not alter response

**Verification:**

```bash
python -m pytest tests/swarm/test_gateway_swarm_shadow.py tests/swarm -q -o 'addopts='
```

---

## Phase 5: Operator Prompt + Delegation Contract

### Task 9: Add operator prompt builder tests

**Objective:** Ensure Jeeves receives explicit swarm operating rules without bloating every session when disabled.

**Files:**
- Create: `tests/swarm/test_swarm_prompt.py`
- Modify: `agent/prompt_builder.py` or a new `agent/swarm_prompt.py`

**Test cases:**
- Disabled config does not alter prompt.
- Enabled config includes operator routing rules, delegation limits, permission model, and subsystem policy.
- Prompt includes “scripts for >3 procedural steps/source-of-truth/validation/eval.”
- Prompt includes “n8n/docker are dumb pipes.”

### Task 10: Implement swarm operator prompt section

**Objective:** Make Jeeves behave as operator using existing model/tool loop.

**Files:**
- Modify/Create: `agent/swarm_prompt.py`
- Modify: `agent/prompt_builder.py`

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_prompt.py tests/agent/test_prompt_builder.py -q -o 'addopts='
```

---

## Phase 6: Bounded Parallel Swarm Execution

### Task 11: Add executor tests with mocked `delegate_task`

**Objective:** Prove operator can translate routing tasks into bounded delegate calls.

**Files:**
- Create: `tests/swarm/test_swarm_executor.py`
- Create: `agent/swarm_executor.py`

**Test cases:**
- Executes at most `max_children` independent tasks.
- Uses declared toolsets per task.
- Aggregates results into job summary data.
- Partial child failure marks job `partially_completed` unless critical.
- Permission-required tasks remain blocked and are not dispatched.

### Task 12: Implement `agent/swarm_executor.py`

**Objective:** Add thin wrapper over existing `delegate_task`; do not build a new scheduler.

**Files:**
- Modify/Create: `agent/swarm_executor.py`

**Implementation shape:**
- `build_delegate_tasks(routing_plan, job)`
- `execute_swarm(job, routing_plan, delegate_fn)`
- Use injected `delegate_fn` for tests; production can call existing delegate path/tool later.
- Return structured task results and update job state.

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_executor.py tests/swarm -q -o 'addopts='
```

---

## Phase 7: Status + Approval Integration

### Task 13: Add status formatting tests

**Objective:** Garrett can ask “what are you doing?” and get swarm/job status.

**Files:**
- Create: `tests/swarm/test_swarm_status.py`
- Create: `agent/swarm_status.py`

**Test cases:**
- Formats active jobs with status, tasks, blockers, and last event.
- Redacts secrets from audit metadata.
- Shows permission requests clearly.

### Task 14: Implement status formatter and wire to `/agents` or `/status`

**Objective:** Surface swarm jobs alongside existing running agents.

**Files:**
- Modify/Create: `agent/swarm_status.py`
- Modify: `gateway/run.py` status command area, if minimal and safe.

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_status.py -q -o 'addopts='
```

---

## Phase 8: Honcho Persistence

### Task 15: Add Honcho persistence adapter tests

**Objective:** Persist final swarm summaries through existing autopilot ingest path without exposing every child-agent scratchpad.

**Files:**
- Create: `tests/swarm/test_swarm_honcho.py`
- Create: `agent/swarm_honcho.py`

**Test cases:**
- Builds a compact signal from job summary, task results, and verification state.
- Uses stable `external_id` to dedupe.
- Does not persist raw secrets/tool dumps.
- No-op when disabled or ingest plugin unavailable.

### Task 16: Implement Honcho summary adapter

**Objective:** Parent/operator writes useful memory; children stay quiet.

**Files:**
- Modify/Create: `agent/swarm_honcho.py`

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_honcho.py -q -o 'addopts='
```

---

## Phase 9: Verification/Eval Layer

### Task 17: Add verification planner tests

**Objective:** Require tests/evidence for code/config/automation/source-of-truth work.

**Files:**
- Create: `tests/swarm/test_swarm_verification.py`
- Create: `agent/swarm_verify.py` (`agent/swarm_verifier.py` compatibility alias)

**Test cases:**
- Code/config jobs require test/lint/typecheck or explicit skipped reason.
- n8n/docker/script tasks require dry-run or payload contract evidence.
- External/client-facing output requires reviewer/human approval.

### Task 18: Implement verification planner

**Objective:** Add structured eval requirements; execution can remain manual/tool-driven in v1.

**Files:**
- Modify/Create: `agent/swarm_verify.py`
- Modify/Create: `agent/swarm_verifier.py` alias if external callers use verifier naming

**Verification:**

```bash
python -m pytest tests/swarm/test_swarm_verifier.py -q -o 'addopts='
```

---

## Phase 10: Rollout

### Task 19: Config defaults and docs

**Objective:** Ship disabled-by-default with clear rollout instructions.

**Files:**
- Modify: `hermes_cli/config.py`
- Create: `website/docs/user-guide/features/swarm-operator.md` or suitable docs path.

**Config defaults:**

```yaml
swarm_operator:
  enabled: false
  dry_run: true
  max_children: 3
  persist_to_honcho: false
  honcho_summary_enabled: false
```

### Task 20: Focused and adjacent tests

**Objective:** Prove the feature does not break normal Hermes behavior.

**Commands:**

```bash
python -m pytest tests/swarm -q -o 'addopts='
python -m pytest tests/agent/test_prompt_builder.py tests/tools/test_delegate.py tests/cron/test_scheduler.py -q -o 'addopts='
```

### Task 21: Shadow rollout on Garrett’s local stack

**Objective:** Enable dry-run shadow mode only after tests pass.

**Manual steps, requiring Garrett approval before changing live config:**

```bash
hermes config set swarm_operator.enabled true
hermes config set swarm_operator.dry_run true
hermes config set swarm_operator.intercept_gateway false
hermes gateway restart
```

**Verification:**
- Send a Slack DM.
- Confirm normal response still works.
- Confirm `${HERMES_HOME}/state/swarm_operator_state.json` updates.
- Confirm `${HERMES_HOME}/state/swarm_operator_metrics.jsonl` receives one event.
- Confirm no subagents run in shadow mode.

---

## First Implementation Sprint Recommendation

Start with Tasks 1-8 only:

1. State model.
2. Store.
3. Router.
4. Gateway shadow hook.

This gives us the spine without risking the “oops I accidentally made a self-driving clown car with root access” problem. Once shadow data looks sane, wire Jeeves prompt/executor and then Honcho/status/evals.

## 2026-05-21 Safe Slice Notes

Implemented the next disabled-by-default slice:

- Repo `DEFAULT_CONFIG` now includes `swarm_operator` defaults; no user config changes.
- Gateway `/status` safely appends persisted swarm status when available.
- Honcho summary adapter is inert unless enabled and accepts injected writers for tests.
- Verification planner records parent-owned checklist/eval metadata without executing tools.
- User docs added at `website/docs/user-guide/features/swarm-operator.md`.
