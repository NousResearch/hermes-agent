# Hermes Workflow System PRD

## Status

Draft v0.1 for discussion and architecture handoff.

## Summary

Hermes should provide a general workflow system that coordinates a reusable team of project-agnostic agents through product planning, architecture, decomposition, implementation, integration, review, publish, and retro stages. The system should combine LLM judgment with deterministic middleware: LLMs reason, design, decompose, implement, and review; code enforces process integrity, validates artifacts, materializes DAGs, tracks state, allocates worktrees, and exposes an auditable control surface.

The feature is delivered as a Hermes Core workflow capability/plugin plus a WebUI DAG control surface. Hermes Core remains the source of truth for workflow state and execution. The WebUI is a client that visualizes, inspects, and eventually operates on workflow state through stable APIs.

## Problem

Hermes already has powerful primitives: profiles, skills, Kanban, gateway dispatch, cron, worktree mode, persistent sessions, and tool access. However, large multi-agent software work still relies too much on LLMs remembering process rules from prompts and manually creating correct task graphs.

That creates failure modes:

- Agents can start implementation before a spec is approved.
- Decomposition can produce invalid or poorly linked task graphs.
- Workers can be assigned to nonexistent profiles.
- Parallel workers can collide without predictable worktree allocation.
- Human operators lack a clear DAG-level view of runtime state.
- LLM reviews may be useful but not auditable enough for trust.
- WebUI surfaces can accidentally become orchestration logic instead of a client over Hermes state.

The goal is to make multi-agent workflows stable without removing LLM agility.

## Product thesis

LLMs should be used where judgment is required. Deterministic middleware should own anything that can be specified, validated, or enforced.

Related architecture bridge: `docs/specs/oh-my-hermes-agent-architecture-bridge.md` records which `oh-my-hermes-agent` operating-model lessons are being adopted as first-party workflow primitives.

In short:

> LLMs are allowed to be smart. They are not allowed to be the database, scheduler, validator, lock manager, CI parser, state machine, or source of truth.

## Goals

1. Define a general workflow layer for Hermes that works across projects.
2. Use canonical, project-agnostic role names.
3. Load project/workspace-specific workflow policy from `.hermes/workflow.yaml`.
4. Support adaptive planning depth: light briefs for small work, full PRDs for large work.
5. Represent large execution plans as validated DAGs.
6. Require an integration node for large DAGs with multiple implementation nodes.
7. Materialize approved DAGs into Kanban tasks with correct dependencies.
8. Allocate and record isolated worktrees for parallel implementation work.
9. Provide auditable review gates, with human review at major breakpoints for large work and LLM-auditable review for smaller work.
10. Expose a WebUI DAG view with Level 2 interactivity: graph visualization plus node inspection.
11. Keep Hermes Core/plugin as source of truth and WebUI as a client/control surface.

## Non-goals

- Replace Hermes Kanban in the MVP.
- Build a full graph-editing UI in the MVP.
- Build autonomous replanning in the MVP.
- Hard-code any project-specific workflow into core profiles.
- Require maximum ceremony for tiny chores and bugfixes.
- Make deterministic middleware choose product or architecture tradeoffs.

## Canonical team roles

The workflow system should use these canonical role names:

- `planner` — shapes rough user intent into a PRD or lightweight brief.
- `architect` — turns approved product intent into an engineering requirements document/spec.
- `reviewer` — reviews PRDs, specs, DAGs, implementations, and publishing artifacts. Adversarial review is a mode/skill, not a separate canonical profile name.
- `publisher` — owns git/GitHub/release mechanics, PRs, CI observation, and publish/merge gates.
- `decomposer` — reads approved specs and emits validated execution DAGs.
- `engineer` — implements one bounded DAG node in an assigned workspace/worktree.
- `integrator` — combines multiple implementation branches/worktrees into a coherent feature result.
- `retro` — analyzes completed workflow for process, skill, and policy improvements.
- `historian` — maintains durable project/workflow state summaries and decision trails.

Project policy maps canonical roles to installed profiles.

## Workflow lifecycle

A large feature should normally flow through:

```text
Chat / Inbox request
→ planner PRD
→ PRD approval
→ architect ERD/spec
→ reviewer spec review
→ publisher lands approved spec if required
→ decomposer DAG proposal
→ DAG validation
→ DAG approval
→ Kanban materialization
→ engineer nodes in isolated worktrees
→ integrator
→ reviewer implementation review
→ publisher PR/merge/publish
→ retro / historian update
```

Small work should compress this lifecycle. For example:

- Tiny chore: direct engineer or assistant action, optional LLM-auditable review.
- Small bugfix: lightweight brief → engineer → LLM-auditable reviewer → publisher if needed.
- Medium feature: architect/spec → engineer → reviewer → publisher.
- Large/XL feature: full PRD/spec/DAG/human gates/integration.

## Review gate levels

### Level 0 — No review

Allowed only for trivial mechanical actions and status-only artifacts.

### Level 1 — LLM-auditable review

Appropriate for small chores, small bugfixes, low-risk implementation tasks, docs, and test-only work.

Requirements:

- reviewer profile/model recorded
- verdict artifact persisted
- evidence links included
- reasons recorded
- diff/test facts gathered deterministically where possible

### Level 2 — Human breakpoint review

Required for large projects and major breakpoints:

- PRD approval
- architecture/spec approval when high impact
- DAG launch approval
- risky publish/merge approval
- security/privacy-sensitive changes

### Level 3 — External mandatory gate

Used when Hermes cannot decide:

- secrets/credentials
- billing/account changes
- destructive data migrations
- legal/product decisions
- branch protection or external owner review

## Deterministic middleware responsibilities

Hermes workflow middleware should own:

- workflow state machine and legal transitions
- `.hermes/workflow.yaml` loading and validation
- profile discovery and role/profile mapping validation
- YAML artifact schema validation and JSON normalization
- DAG validation: unique node IDs, valid parents, acyclic topology, known roles, known profiles, required integration node for large DAGs
- DAG materialization into Kanban tasks
- dependency links between materialized tasks
- worktree/branch allocation metadata
- artifact and handoff schema validation
- required command/test result collection where configured
- PR/CI/git factual status collection where available
- audit event persistence
- status APIs for WebUI/CLI/no-agent reporting

Middleware may reject invalid LLM proposals with structured errors.

## LLM responsibilities

LLM agents should own:

- product discussion and PRD drafting
- architecture tradeoff reasoning
- decomposition proposal generation
- implementation of bounded tasks
- semantic review of specs/code
- synthesis of status insights from deterministic facts
- retro analysis

LLMs may propose state transitions and artifacts, but deterministic middleware decides whether those transitions/artifacts are valid.

## Project policy: `.hermes/workflow.yaml`

Workflow policy should be project/workspace-specific and machine-readable.

Example shape:

```yaml
version: 1
project:
  name: example-project

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
  normalize_to: json
  require_integrator_for_large: true
  layout: elk

execution:
  default_workspace_mode: worktree
  base_branch: origin/main

artifacts:
  root: .hermes/workflows
```

## Artifact model

Human-authored or LLM-authored artifacts should be YAML where practical, normalized to JSON internally for validators and APIs.

Initial artifact types:

- PRD / feature brief
- architecture/spec artifact
- DAG proposal
- node definition
- worker handoff
- review verdict
- publish report
- retro report

## DAG proposal requirements

A DAG proposal should include:

```yaml
workflow_id: optional-before-materialization
feature: Human-readable feature name
size: small|medium|large|xl
spec:
  path: optional/path/to/spec.md
  required: true
nodes:
  - id: room-state
    title: Implement room lifecycle state
    role: engineer
    parents: []
    workspace: worktree
    definition_of_done:
      - Max capacity enforced
      - Lifecycle tests pass
    non_goals:
      - Do not implement UI
```

Validation rules:

- node IDs are unique
- every parent exists
- graph is acyclic
- every role is canonical
- every canonical role maps to an installed profile
- every implementation node has a definition of done
- large/XL graphs with multiple implementation nodes include an `integrator` node
- review/publish gates exist according to workflow policy
- graph is not materialized until required approvals are satisfied

## Kanban integration

Kanban remains the MVP execution substrate.

The workflow system should map:

```text
workflow DAG node ↔ Kanban task
```

Materialization should create Kanban tasks with:

- title
- body
- assignee profile
- parent dependency links
- workflow ID
- node ID
- artifact links
- definition of done
- workspace/worktree metadata if applicable

The LLM should not manually create a large graph of Kanban cards when a validated DAG artifact exists. Middleware should materialize the graph to avoid missing links or invalid assignees.

## Worktree allocation

For implementation nodes using worktree mode, middleware should allocate and record:

- task ID
- workflow ID
- node ID
- branch name
- worktree path
- base branch
- creation timestamp
- cleanup status

Example:

```yaml
workflow_id: wf_123
node_id: room-state
task_id: t_abc123
branch: workflow/wf_123/room-state
worktree: .worktrees/wf_123-room-state
base: origin/main
```

Workers receive this as assigned context rather than improvising branch/worktree setup.

## WebUI DAG control surface

The WebUI should provide a Level 2 DAG experience in the MVP:

- read-only graph visualization
- React Flow graph rendering
- Dagre or ELK auto-layout
- status color/icon legend
- clickable nodes
- node detail drawer
- live or periodically refreshed status
- artifact/test/handoff links where available
- approval gate visibility

The WebUI must treat Hermes workflow APIs as source of truth. It must not own orchestration state or independently enforce workflow rules except for client-side affordances and display validation.

### Node detail drawer

Clicking a node should show:

- title
- role
- assigned profile
- status
- parents and children
- critical path marker if available
- task body/scope
- definition of done
- non-goals
- linked spec section/artifacts
- Kanban task ID
- worker profile/run ID
- worktree path
- branch
- start/finish timestamps
- changed files when available
- test commands/results when available
- handoff artifact
- review verdict
- PR/CI links
- audit events
- blocker/retry/reclaim history

### Stretch UI actions

Not MVP, but future graph actions should include:

- approve gate
- retry node
- reclaim worker
- reassign node
- block/unblock node
- split node
- add dependency
- open PR
- open logs
- open worktree

All actions must call Hermes workflow APIs and pass deterministic validation before mutating state.

## API expectations

The workflow system should expose stable APIs for CLI and WebUI clients.

Potential endpoints or equivalent CLI/API functions:

```text
GET /api/workflows
GET /api/workflows/:workflow_id
GET /api/workflows/:workflow_id/dag
GET /api/workflows/:workflow_id/nodes/:node_id
GET /api/workflows/:workflow_id/events
GET /api/workflows/:workflow_id/artifacts
POST /api/workflows/:workflow_id/approve
POST /api/workflows/:workflow_id/materialize
```

Exact endpoint naming is an architecture decision, but WebUI should consume normalized workflow/DAG data rather than scraping Kanban prose.

## CLI expectations

Potential commands:

```bash
hermes workflow init
hermes workflow validate
hermes workflow status
hermes workflow dag validate dag.yaml
hermes workflow materialize dag.yaml
hermes workflow approve <workflow_id> --gate dag
hermes workflow node show <workflow_id> <node_id>
```

## Auditability

Every meaningful decision should leave an audit event:

- workflow created
- artifact produced
- artifact validated
- gate requested
- gate approved/rejected
- reviewer verdict
- DAG materialized
- Kanban task created
- worker claimed/completed/blocked
- worktree allocated
- publish attempted/completed/blocked

Audit events should include:

- timestamp
- actor type: human, LLM profile, system middleware
- actor identifier where available
- model/profile where applicable
- artifact references and hashes where practical
- structured reason/verdict

## Status reporting

The system should support hybrid status reporting:

1. Deterministic facts, suitable for no-agent reports and WebUI status.
2. Optional LLM insights, clearly labeled as synthesis over facts.

Example deterministic status:

```text
Workflow: wf_123 voice rooms
State: implementation_running
Nodes: 8 total, 3 done, 2 running, 1 blocked, 2 waiting
Critical path: room-state → signaling → integration → review → publish
Human action required: none
```

## MVP scope

MVP should include:

1. PRD/spec artifact definitions.
2. `.hermes/workflow.yaml` draft schema.
3. DAG YAML schema and JSON normalization.
4. Deterministic DAG validator.
5. DAG-to-Kanban materialization.
6. Required integration node rule for large DAGs.
7. Basic workflow state and audit events.
8. Worktree metadata allocation/recording for implementation nodes.
9. CLI/status surfaces sufficient for verification.
10. WebUI read-only DAG view with node inspection.

## Stretch goals

- Graph editing and runtime actions from WebUI.
- Auto-replanning from blocked/failed nodes.
- Critical path and cost/runtime estimates.
- Multi-repo workflows.
- Rich CI/provider integrations.
- Artifact diffing and review comparison.
- Workflow templates by project/domain.
- Marketplace/distribution packaging for workflow plugins and canonical team profiles.

## Acceptance criteria

1. A user can configure canonical roles through `.hermes/workflow.yaml`.
2. A decomposer can produce a YAML DAG artifact for a large feature.
3. The validator rejects invalid DAGs with structured errors.
4. A valid approved DAG can be materialized into Kanban tasks with correct dependencies.
5. Large DAGs with multiple implementation nodes require an integrator node.
6. Implementation nodes can carry deterministic worktree metadata.
7. Review gates record whether approval came from a human, LLM-auditable reviewer, or external blocker.
8. The WebUI can render the workflow DAG and inspect node details using Hermes-provided normalized data.
9. The WebUI does not become the source of truth for workflow state.
10. Small tasks can use lightweight briefs and LLM-auditable review without full large-feature ceremony.

## Open architecture questions

1. Should workflow state live in the existing Hermes state DB, a plugin-owned DB/table set, artifact files, or a hybrid?
2. How should workflow APIs be exposed through the existing dashboard/gateway surfaces?
3. How much of the workflow plugin should be generic plugin infrastructure versus first-party Hermes feature code?
4. What is the exact initial `.hermes/workflow.yaml` schema?
5. What artifact hashing/signing is useful without overengineering?
6. How should WebUI subscribe to runtime updates: polling, SSE, or websocket?
7. How should existing Kanban tasks be migrated or linked into workflows?
8. What is the minimal profile distribution story for the canonical team?
