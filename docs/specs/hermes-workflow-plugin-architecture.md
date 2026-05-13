# Hermes Workflow Plugin Architecture Spec

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task after PRD/spec approval.

**Status:** Draft v0.1

**Related PRD:** `docs/prd/hermes-workflow-system.md`

**Goal:** Define the backend/source-of-truth architecture for the Hermes workflow system: state, schemas, validation, Kanban materialization, CLI/API surfaces, and auditability.

**Architecture:** Implement the workflow system as a plugin-shaped Hermes Core capability. The workflow layer owns deterministic process state and exposes normalized data to CLI, gateway/API, no-agent reports, and the WebUI. Kanban remains the execution substrate for worker dispatch in MVP.

**Tech Stack:** Python, SQLite, YAML, JSON-schema-style validation, existing Hermes plugin/CLI patterns, existing Kanban DB/dispatcher primitives.

---

## 1. Scope

This spec covers the Hermes-side architecture only.

Included:

- workflow plugin package shape
- persistent workflow storage model
- `.hermes/workflow.yaml` policy schema
- workflow artifact schemas
- DAG normalization and validation
- state machine and audit events
- Kanban materialization model
- worktree allocation metadata
- CLI/API contract for WebUI consumption
- implementation plan slices
- test strategy

Not included:

- WebUI component implementation details
- graph editing
- autonomous replanning
- replacement of Kanban dispatcher
- project-specific workflow profiles

## 2. Repository Fit

Relevant existing Hermes areas:

- `hermes_cli/plugins.py` — bundled/user/project/pip plugin discovery and lifecycle.
- `plugins/` — bundled plugin packages with `plugin.yaml` and optional CLI/runtime modules.
- `hermes_cli/kanban_db.py` — SQLite-backed board, tasks, dependency links, comments, events, workspace metadata.
- `hermes_cli/kanban.py` — Kanban CLI surface.
- `tools/kanban_tools.py` — worker-facing Kanban tools.
- `gateway/platforms/api_server.py` — OpenAI-compatible/API-server surface used by Workspace/WebUI.
- `hermes_cli/web_server.py` and dashboard/WebUI-related handlers — local status surfaces.

MVP should add workflow as a new coherent package rather than burying graph logic inside existing Kanban code.

Recommended package layout:

```text
plugins/workflow/
  plugin.yaml
  __init__.py
  cli.py
  api.py
  audit.py
  artifacts.py
  dag.py
  errors.py
  materialize.py
  policy.py
  schemas.py
  state.py
  store.py
  worktrees.py
  README.md
```

If plugin CLI/API registration cannot cleanly support all required surfaces yet, implement reusable domain code under `hermes_cli/workflow/` and keep `plugins/workflow/` as the registration boundary:

```text
hermes_cli/workflow/
  __init__.py
  audit.py
  artifacts.py
  dag.py
  materialize.py
  policy.py
  schemas.py
  state.py
  store.py
  worktrees.py

plugins/workflow/
  plugin.yaml
  __init__.py
  cli.py
  api.py
```

The code should not depend on the WebUI. WebUI consumes workflow APIs.

## 3. Source-of-Truth Rule

Hermes workflow state is authoritative.

The workflow plugin owns:

- workflow identity and status
- DAG topology
- node status
- approval/gate state
- artifact references
- audit events
- Kanban task mappings
- worktree/branch allocation metadata
- normalized workflow API responses

Clients may request transitions, but cannot directly mutate state without going through workflow validation.

## 4. Storage Model

Use SQLite for authoritative state, aligned with existing Kanban durability patterns.

Default location:

```text
<hermes-root>/workflow/workflow.db
```

Associated artifact directory:

```text
<hermes-root>/workflow/artifacts/<workflow_id>/
  workflow.yaml
  policy.snapshot.yaml
  prd.yaml
  spec.yaml
  dag.yaml
  dag.normalized.json
  events.jsonl              # optional export mirror; DB remains source of truth
  nodes/<node_id>/
    handoff.yaml
    review.yaml
    test-results.json
```

Board-aware workflow records should include the Kanban board slug. This lets workflows coexist across projects without mixing task queues.

### 4.1 SQLite Tables

Initial schema:

```sql
CREATE TABLE workflows (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  description TEXT DEFAULT '',
  workspace_path TEXT,
  board TEXT NOT NULL DEFAULT 'default',
  scale TEXT NOT NULL CHECK (scale IN ('small','medium','large','xl')),
  status TEXT NOT NULL,
  current_gate TEXT,
  policy_path TEXT,
  policy_snapshot_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  created_by TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE workflow_artifacts (
  id TEXT PRIMARY KEY,
  workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  path TEXT,
  sha256 TEXT,
  mime_type TEXT,
  schema_version INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'active',
  created_at REAL NOT NULL,
  created_by TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE workflow_nodes (
  workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  node_id TEXT NOT NULL,
  title TEXT NOT NULL,
  role TEXT NOT NULL,
  profile TEXT,
  status TEXT NOT NULL,
  gate_level INTEGER NOT NULL DEFAULT 1,
  gate_type TEXT,
  kanban_task_id TEXT,
  branch TEXT,
  worktree_path TEXT,
  base_ref TEXT,
  definition_of_done_json TEXT NOT NULL DEFAULT '[]',
  scope_json TEXT NOT NULL DEFAULT '{}',
  evidence_json TEXT NOT NULL DEFAULT '{}',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (workflow_id, node_id)
);

CREATE TABLE workflow_edges (
  workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  parent_node_id TEXT NOT NULL,
  child_node_id TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'depends_on',
  PRIMARY KEY (workflow_id, parent_node_id, child_node_id)
);

CREATE TABLE workflow_gates (
  id TEXT PRIMARY KEY,
  workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  node_id TEXT,
  gate_type TEXT NOT NULL,
  level INTEGER NOT NULL,
  status TEXT NOT NULL,
  verdict TEXT,
  required_actor TEXT NOT NULL,
  resolved_by TEXT,
  resolved_at REAL,
  artifact_id TEXT,
  reason TEXT,
  metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE workflow_events (
  id TEXT PRIMARY KEY,
  workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  node_id TEXT,
  event_type TEXT NOT NULL,
  actor_type TEXT NOT NULL,
  actor_id TEXT,
  message TEXT NOT NULL DEFAULT '',
  data_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL
);

CREATE TABLE workflow_kanban_mappings (
  workflow_id TEXT NOT NULL REFERENCES workflows(id) ON DELETE CASCADE,
  node_id TEXT NOT NULL,
  board TEXT NOT NULL,
  task_id TEXT NOT NULL,
  materialized_at REAL NOT NULL,
  PRIMARY KEY (workflow_id, node_id),
  UNIQUE (board, task_id)
);
```

Implementation note: the schema can be trimmed if existing Kanban tables already carry some fields, but workflow-specific graph/gate/audit state should not be inferred from Kanban prose.

## 5. Workflow Policy Schema

Policy is loaded from the selected workspace path:

```text
<workspace>/.hermes/workflow.yaml
```

If absent, use a safe default policy and emit a warning in validation output.

Minimal schema:

```yaml
version: 1
project:
  name: string
  board: default

roles:
  planner: planner
  architect: architect
  reviewer: reviewer
  publisher: publisher
  decomposer: decomposer
  engineer: engineer
  integrator: integrator
  retro: retro
  historian: historian

review:
  default: llm_auditable
  gates:
    small:
      prd: optional
      spec: optional
      dag: none
      review: llm_auditable
    medium:
      prd: light
      spec: required
      dag: optional
      review: llm_auditable
    large:
      prd: full
      spec: required
      dag: human
      review: human_at_major_breakpoints
    xl:
      prd: full
      spec: required
      dag: human
      review: human_at_major_breakpoints

dag:
  format: yaml
  require_integrator_for_large: true
  require_definition_of_done: true
  max_parallel_engineers: 4

worktrees:
  enabled: true
  root: .worktrees
  branch_prefix: workflow
```

Validation rules:

- `version` must be supported.
- every canonical role may map to a profile name or `null` if intentionally unavailable.
- mapped profiles should be discoverable via Hermes profile listing before materialization.
- review gate settings must be one of known values.
- `max_parallel_engineers` must be a positive integer.
- `worktrees.root` must stay inside the workspace unless explicitly allowed by future policy.

## 6. Canonical State Machine

Workflow statuses:

```text
inbox
brief_draft
prd_draft
prd_review
prd_approved
spec_draft
spec_review
spec_approved
dag_proposed
dag_validated
dag_approved
materialized
running
integration
implementation_review
publish_ready
published
retro
done
blocked
cancelled
```

Node statuses:

```text
waiting
ready
running
blocked
review
publish
done
failed
cancelled
```

Gate statuses:

```text
pending
approved
rejected
blocked
waived
```

Transitions are allowed only through workflow service functions. Every accepted transition writes a `workflow_events` row.

### 6.1 Transition Examples

- `dag_proposed -> dag_validated` only after deterministic DAG validation passes.
- `dag_validated -> dag_approved` requires a gate verdict when policy says DAG approval is human or LLM-auditable.
- `dag_approved -> materialized` only after Kanban task creation and dependency linking succeeds transactionally.
- `materialized -> running` when one or more materialized nodes move to ready/running.
- `running -> integration` when all implementation nodes are done and an integrator node exists.
- `publish_ready -> published` only after publisher evidence artifact is attached.

## 7. Artifact Model

Artifacts are YAML on disk for human readability and normalized to JSON for validation/API use.

Artifact kinds:

```text
brief
prd
spec
dag
handoff
review
publish_report
retro
status_report
```

Every artifact should have:

```yaml
schema_version: 1
kind: dag
workflow_id: wf_...
created_at: "2026-05-12T20:00:00Z"
created_by:
  actor_type: llm_profile
  actor_id: decomposer
body: {}
```

The plugin computes SHA-256 for persisted artifacts and stores it with the artifact row for auditability.

## 8. DAG Schema

Human-authored/proposed YAML:

```yaml
schema_version: 1
workflow_id: wf_example
name: Example workflow DAG
scale: large
nodes:
  - id: spec-review
    title: Review architecture spec
    role: reviewer
    profile: reviewer
    status: waiting
    parents: []
    gate:
      level: 1
      type: spec_review
    scope:
      summary: Review spec for correctness and implementation readiness.
      non_goals: []
    definition_of_done:
      - Review artifact exists.
      - Verdict is approved or blocked with reasons.

  - id: backend-api
    title: Implement backend workflow API
    role: engineer
    profile: engineer
    parents: [spec-review]
    workspace:
      kind: worktree
      base_ref: origin/main
    definition_of_done:
      - Tests cover list/detail/DAG endpoints.
      - API returns normalized nodes and edges.

  - id: integration
    title: Integrate implementation nodes
    role: integrator
    profile: integrator
    parents: [backend-api]
    definition_of_done:
      - Integration branch includes all approved node outputs.
      - Tests pass.
```

Normalized internal JSON should denormalize parent edges and fill defaults:

```json
{
  "schema_version": 1,
  "workflow_id": "wf_example",
  "nodes": [
    {
      "id": "backend-api",
      "role": "engineer",
      "profile": "engineer",
      "status": "waiting",
      "parents": ["spec-review"],
      "children": ["integration"],
      "gate_level": 1,
      "workspace": {"kind": "worktree", "base_ref": "origin/main"}
    }
  ],
  "edges": [
    {"source": "spec-review", "target": "backend-api", "kind": "depends_on"}
  ]
}
```

## 9. DAG Validation Rules

`workflow dag validate` must reject invalid DAGs with structured errors.

Rules:

1. YAML parses.
2. Schema version is supported.
3. Node IDs are unique, slug-safe, and stable.
4. Parent references exist.
5. Graph is acyclic.
6. Roles are canonical.
7. Profiles mapped by policy exist before materialization.
8. Every implementation node has `definition_of_done`.
9. Every node has a bounded scope summary.
10. Large/XL DAG with multiple engineer nodes has exactly one reachable `integrator` node unless policy overrides.
11. Gate nodes have valid gate level/type.
12. No node is materialized if an upstream required approval is missing.
13. Worktree roots/branches are valid and do not escape configured workspace root.
14. Node count and max parallelism respect policy caps.

Return shape:

```json
{
  "ok": false,
  "errors": [
    {
      "code": "unknown_profile",
      "path": "nodes[2].profile",
      "message": "Profile 'frontend-engineer' is not installed",
      "severity": "error"
    }
  ],
  "warnings": []
}
```

## 10. Kanban Materialization

Materialization converts an approved workflow DAG into Kanban tasks.

Input:

- workflow id
- normalized DAG
- policy snapshot
- target board

Output:

- Kanban tasks
- Kanban dependency links
- workflow node ↔ task mappings
- audit events

Task body should include a compact generated header plus node scope:

```markdown
# Workflow Node: backend-api

Workflow: wf_123
Node: backend-api
Role: engineer
Profile: engineer
Branch: workflow/wf_123/backend-api
Worktree: .worktrees/wf_123-backend-api

## Scope
...

## Definition of Done
- ...

## Required Artifacts
- handoff.yaml
- test-results.json where applicable
```

Task metadata should carry machine-readable fields if Kanban supports metadata JSON. If not, add a reserved fenced YAML block in body as a transitional bridge; do not rely on prose parsing in workflow APIs.

Dependency links:

- for every DAG edge `parent -> child`, create Kanban task link parent task -> child task.
- gate/review nodes materialize like any other task unless represented internally as workflow gates only.

Transaction rule:

- materialization must be all-or-nothing from the workflow perspective.
- if Kanban task creation partially succeeds and later work fails, the workflow should record a failed materialization event with created task IDs and expose a repair command.

## 11. Worktree Allocation

For each node requiring a worktree:

```text
branch: <branch_prefix>/<workflow_id>/<node_id>
worktree: <workspace>/<worktrees.root>/<workflow_id>-<node_id>
base_ref: policy or node override, default origin/main
```

Example:

```text
workflow/wf_123/backend-api
/mnt/c/Users/colebienek/pepchat/.worktrees/wf_123-backend-api
```

Validation:

- branch/node IDs must be shell-safe slugs.
- worktree path must remain under allowed root.
- no two active nodes may allocate same branch/worktree.
- existing non-empty worktree path blocks unless explicitly repairing/resuming.

The plugin should initially allocate metadata. Actual `git worktree add` can remain owned by dispatcher/worker if that is how Kanban currently works, but the selected paths/branch names should be deterministic and visible before dispatch.

## 12. API Contract

Expose read-only status APIs first. Mutation APIs can follow after visibility MVP.

Recommended endpoints under the Hermes API server/dashboard namespace:

```text
GET /api/workflows
GET /api/workflows/{workflow_id}
GET /api/workflows/{workflow_id}/dag
GET /api/workflows/{workflow_id}/nodes/{node_id}
GET /api/workflows/{workflow_id}/events
GET /api/workflows/{workflow_id}/artifacts
```

Future mutation endpoints:

```text
POST /api/workflows/{workflow_id}/validate-dag
POST /api/workflows/{workflow_id}/approve
POST /api/workflows/{workflow_id}/materialize
POST /api/workflows/{workflow_id}/nodes/{node_id}/retry
POST /api/workflows/{workflow_id}/nodes/{node_id}/block
POST /api/workflows/{workflow_id}/nodes/{node_id}/unblock
```

API responses must separate deterministic `facts` from optional LLM `insights`.

## 13. CLI Contract

Initial CLI:

```bash
hermes workflow init [--workspace PATH] [--board BOARD]
hermes workflow policy validate [--workspace PATH]
hermes workflow create --title TITLE --scale small|medium|large|xl [--workspace PATH]
hermes workflow status [WORKFLOW_ID]
hermes workflow dag validate WORKFLOW_ID DAG.yaml
hermes workflow dag import WORKFLOW_ID DAG.yaml
hermes workflow materialize WORKFLOW_ID
hermes workflow events WORKFLOW_ID
hermes workflow node show WORKFLOW_ID NODE_ID
```

Later:

```bash
hermes workflow approve WORKFLOW_ID --gate dag --artifact review.yaml
hermes workflow export WORKFLOW_ID --format yaml|json
hermes workflow repair materialization WORKFLOW_ID
```

## 14. Auditability

Every meaningful event writes an audit row.

Actor types:

```text
human
llm_profile
system
external
```

Important event types:

```text
workflow_created
policy_loaded
artifact_attached
artifact_validated
dag_validation_passed
dag_validation_failed
gate_created
gate_approved
gate_rejected
dag_imported
materialization_started
kanban_task_created
kanban_link_created
materialization_completed
node_status_changed
workflow_status_changed
```

Audit events should contain enough metadata to reconstruct who/what acted, what changed, and which artifact hash was involved.

## 15. Error Types

Define stable error codes. Examples:

```text
policy_not_found
policy_invalid_yaml
unknown_role
unknown_profile
invalid_state_transition
invalid_dag_yaml
duplicate_node_id
missing_parent
cycle_detected
missing_definition_of_done
integrator_required
worktree_path_escape
materialization_partial_failure
kanban_board_unavailable
```

Stable errors matter because WebUI can render actionable messages.

## 16. Implementation Plan

### Task 1: Add workflow package skeleton

**Objective:** Create plugin/domain package without behavior.

**Files:**

- Create: `plugins/workflow/plugin.yaml`
- Create: `plugins/workflow/__init__.py`
- Create: `hermes_cli/workflow/__init__.py`
- Create: `hermes_cli/workflow/errors.py`

**Verification:** import modules in Python.

### Task 2: Add policy loader and validator

**Objective:** Load `.hermes/workflow.yaml`, apply defaults, return structured validation errors.

**Files:**

- Create: `hermes_cli/workflow/policy.py`
- Test: `tests/hermes_cli/test_workflow_policy.py`

**Verification:** pytest covers valid policy, missing policy default, invalid YAML, unknown gate setting.

### Task 3: Add workflow store schema

**Objective:** Initialize SQLite workflow DB and CRUD base records.

**Files:**

- Create: `hermes_cli/workflow/store.py`
- Test: `tests/hermes_cli/test_workflow_store.py`

**Verification:** temp `HERMES_HOME` creates DB, inserts workflow, inserts event, fetches list/detail.

### Task 4: Add DAG parser/normalizer/validator

**Objective:** Convert YAML DAG to normalized JSON and reject invalid graphs.

**Files:**

- Create: `hermes_cli/workflow/dag.py`
- Test: `tests/hermes_cli/test_workflow_dag.py`

**Verification:** tests cover duplicate IDs, missing parents, cycles, unknown role, missing DoD, integrator required.

### Task 5: Add artifact persistence

**Objective:** Persist YAML artifacts, compute SHA-256, attach to workflow.

**Files:**

- Create: `hermes_cli/workflow/artifacts.py`
- Test: `tests/hermes_cli/test_workflow_artifacts.py`

**Verification:** artifact path stays under workflow artifact root and hash is stable.

### Task 6: Add read-only CLI commands

**Objective:** Expose `hermes workflow` commands for init/status/policy/dag validate.

**Files:**

- Modify: `hermes_cli/main.py` or relevant argparse command registration
- Create: `plugins/workflow/cli.py` if plugin CLI registration supports this path
- Test: `tests/hermes_cli/test_workflow_cli.py`

**Verification:** CLI command tests pass with temp Hermes home.

### Task 7: Add Kanban materializer

**Objective:** Create Kanban tasks and links from an approved normalized DAG.

**Files:**

- Create: `hermes_cli/workflow/materialize.py`
- Test: `tests/hermes_cli/test_workflow_materialize.py`

**Verification:** materialized tasks have correct dependency links and workflow mapping rows.

### Task 8: Add read-only API serialization

**Objective:** Return normalized workflow list/detail/DAG/node/events payloads for WebUI.

**Files:**

- Create: `hermes_cli/workflow/api.py` or `plugins/workflow/api.py`
- Modify API-server/dashboard routing in the smallest appropriate location.
- Test: `tests/gateway/test_workflow_api.py` or equivalent.

**Verification:** endpoints return JSON with correct facts/insights separation.

### Task 9: Add no-agent/report export

**Objective:** Provide deterministic status report export for cron/no-agent use.

**Files:**

- Create: `hermes_cli/workflow/reports.py`
- Test: `tests/hermes_cli/test_workflow_reports.py`

**Verification:** report renders without LLM calls.

## 17. Test Strategy

Run targeted tests first:

```bash
python -m pytest tests/hermes_cli/test_workflow_policy.py -q -o 'addopts='
python -m pytest tests/hermes_cli/test_workflow_dag.py -q -o 'addopts='
python -m pytest tests/hermes_cli/test_workflow_store.py -q -o 'addopts='
python -m pytest tests/hermes_cli/test_workflow_materialize.py -q -o 'addopts='
```

Then run broader affected suites:

```bash
python -m pytest tests/hermes_cli tests/gateway/test_api_server.py tests/tools/test_kanban_tools.py -q -o 'addopts='
```

Full suite before merge:

```bash
python -m pytest tests/ -q -o 'addopts='
```

## 18. Acceptance Criteria

MVP backend is acceptable when:

- `.hermes/workflow.yaml` can be loaded and validated.
- workflow records persist durably in SQLite.
- DAG YAML can be normalized to JSON.
- invalid DAGs produce structured errors.
- approved DAGs can materialize into Kanban tasks and links.
- worktree/branch allocation metadata is deterministic.
- workflow list/detail/DAG/node/events APIs return normalized JSON.
- every state transition and materialization action emits audit events.
- WebUI can render a Level 2 DAG view without parsing Kanban prose.

## 19. Open Questions

1. Should workflow DB live beside Kanban DB per board, or globally under `<root>/workflow/workflow.db` with `board` columns?
2. Which existing API surface is best for workflow endpoints: API server, dashboard server, or both behind shared serializers?
3. Can plugin CLI registration cleanly add `hermes workflow`, or should the first version add a core CLI command and keep the plugin boundary internal?
4. Should gate-only nodes materialize as Kanban tasks, or should workflow gates remain internal until action endpoints exist?
5. How much Kanban task metadata support exists today, and is it enough for workflow mapping without body fences?
