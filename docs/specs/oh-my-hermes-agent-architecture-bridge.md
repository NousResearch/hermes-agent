# oh-my-hermes-agent Architecture Bridge

**Status:** Draft v0.1  
**Related PRD:** `docs/prd/hermes-workflow-system.md`  
**Related architecture spec:** `docs/specs/hermes-workflow-plugin-architecture.md`  
**Source inspiration:** `eloklam/oh-my-hermes-agent`

## Purpose

This bridge document records which architectural lessons Hermes is adopting from `oh-my-hermes-agent` and how they map into the first-party Hermes Workflow System.

`oh-my-hermes-agent` is treated as a workflow architecture pattern, not as code to vendor. Its value is in its operating model: durable Kanban-backed coordination, named specialist profiles, stable board semantics, deliberate review gates, and a coordinator that routes work instead of doing all work inline.

Hermes Workflow should absorb those ideas into general core primitives rather than copying OMH role names or creating a separate parallel runtime.

## Source characterization

`oh-my-hermes-agent` is best understood as:

- a profile-suite convention
- a Kanban operating model
- a routing policy for multi-agent work
- a set of guardrails for when to delegate to durable profiles instead of using ad hoc assistant/subagent execution

It is explicitly **not** treated as:

- a Hermes plugin
- a Python package
- a new runtime
- an external plugin loader
- a replacement for Hermes Kanban, profiles, gateway dispatch, or workflow state

## Architectural thesis we are adopting

The core lesson is:

> Use LLMs for judgment, but use deterministic Hermes infrastructure for state, routing, dependencies, persistence, and gates.

In OMH terms, the orchestrator delegates through durable named profiles and Kanban tasks when work needs traceability. In Hermes Workflow terms, the workflow system owns the workflow record, DAG, audit events, gates, role/profile mappings, and Kanban materialization; agents only propose and execute bounded artifacts through that substrate.

## OMH ideas adopted into Hermes Workflow

### 1. Named-profile routing

OMH routes work to named specialist profiles instead of treating every delegation as an anonymous subagent call.

Hermes Workflow adopts this as **role/profile mapping**:

```yaml
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
```

The canonical role name remains stable in workflow artifacts. Project/workspace policy maps that role to the installed profile that should execute it.

Why this matters:

- profiles can have durable config, auth, model choices, tools, skills, and memory
- audit logs can record which profile acted
- WebUI can show role and assigned profile separately
- policy can reject DAGs that reference unmapped or unavailable profiles

### 2. Durable Kanban for real work

OMH draws a strong line between quick local reasoning and durable work routed through Kanban.

Hermes Workflow adopts this as:

- small work may remain direct or lightweight
- medium/large work materializes validated workflow nodes into Kanban tasks
- Kanban remains the MVP execution substrate
- workflow DB remains the higher-level source of truth for DAG topology, gates, artifacts, audit events, and Kanban mappings

The workflow system should not ask LLMs to manually create and link large graphs of Kanban tasks. Instead:

```text
DAG artifact → deterministic validation → deterministic Kanban materialization
```

### 3. Coordinator should route, not self-execute serious work

OMH says the orchestrator should decompose and route rather than implement substantial work itself.

Hermes Workflow adopts this with scale-aware policy rather than an absolute ban:

- tiny/small work: direct assistant execution can be acceptable if auditable
- medium work: prefer named profile routing for implementation/review
- large/XL work: require formal workflow artifacts, gates, DAG validation, Kanban materialization, and named profiles

This preserves Hermes agility while preventing large workflows from depending on one long-running chat remembering process rules.

### 4. Parallel discovery lanes before implementation

OMH patterns often split discovery and evidence gathering before implementation, for example explorer/librarian work feeding fixer/reviewer work.

Hermes Workflow adopts the structural idea, but maps it to canonical roles and DAG nodes:

```text
planner / architect / historian evidence lanes
        ↓
approved spec / artifact
        ↓
decomposer DAG
        ↓
engineer nodes
        ↓
integrator, reviewer, publisher
```

For large work, discovery, architecture, and documentation tasks may run in parallel as long as the DAG validator records dependencies and required gates.

### 5. Stable board semantics

OMH avoids creating new timestamped boards for every turn and instead uses stable semantic boards with task/workspace scoping.

Hermes Workflow adopts this rule:

- workflow records include `board`
- workflow records include `workspace_path` where applicable
- Kanban board remains a durable queue boundary
- workflow ID and node ID scope tasks inside the board
- WebUI groups and filters by workflow metadata instead of requiring new boards for every workflow

This avoids board sprawl and keeps workers operating on familiar queues.

### 6. Checkpointing and cost discipline

OMH discourages long blocking polling loops and expensive review cascades after every minor event.

Hermes Workflow adopts this as policy guidance:

- record meaningful checkpoints as audit events
- checkpoint after blockers, materialization, gate decisions, implementation completion, review verdicts, and publish attempts
- avoid automatic repeated review cascades unless policy/user approval allows them
- prefer one deliberate review gate at each major breakpoint
- status reporting should combine deterministic facts with optional labeled LLM synthesis

### 7. Profile-suite concept

OMH effectively acts as a profile suite: a named collection of profiles and routing conventions.

Hermes Workflow should support this concept without making it project-specific. A future policy shape may include:

```yaml
profiles:
  suite: default
  mappings:
    planner: default
    decomposer: default
    architect: architect
    engineer: engineer
    reviewer: reviewer
    adversarial_reviewer: reviewer
    publisher: publisher
    historian: historian
```

An OMH-compatible suite could be represented as:

```yaml
profiles:
  suite: oh-my-hermes-agent
  mappings:
    planner: default
    decomposer: default
    architect: explorer
    engineer: fixer
    reviewer: oracle
    adversarial_reviewer: oracle
    historian: librarian
    designer: designer
    observer: observer
    council: council
```

The workflow system should treat suites as reusable mapping presets, not as hard-coded roles.

## Mapping OMH archetypes to Hermes canonical roles

| OMH archetype | Hermes canonical role or mode | Notes |
|---|---|---|
| `orchestrator` | `planner`, `decomposer` | Coordinates, decomposes, routes; should not self-execute large work. |
| `oracle` | `reviewer` / adversarial-review mode | Adversarial review is a mode/skill, not a separate canonical role. |
| `explorer` | `architect` / discovery node | Performs broad codebase/context discovery before spec or implementation. |
| `librarian` | `historian` | Maintains source-grounded durable evidence, decisions, and status trails. |
| `fixer` | `engineer` | Implements bounded nodes with definition of done. |
| `designer` | UI-specialized `architect` or `reviewer` profile | Optional profile mapping for UI-heavy workflows. |
| `observer` | vision/evidence profile or reviewer support | Optional profile for visual inspection and screenshot evidence. |
| `council` | multi-review gate | Future gate mode for consensus or multi-review workflows. |

Hermes artifacts should use the canonical roles. Profile suites may map those roles to OMH-style profile names.

## Policy implications

The `.hermes/workflow.yaml` schema should evolve toward these concepts:

```yaml
routing:
  default_execution: direct_or_kanban
  non_trivial_work: kanban
  direct_execution_allowed_for:
    - tiny
    - small
  require_named_profiles_for:
    - medium
    - large
    - xl

checkpointing:
  no_long_blocking_polling: true
  checkpoint_after_blocker: true
  checkpoint_after_gate: true
  max_auto_review_cascades: 1

profiles:
  suite: default
  mappings:
    planner: planner
    architect: architect
    reviewer: reviewer
    publisher: publisher
    decomposer: decomposer
    engineer: engineer
    integrator: integrator
    retro: retro
    historian: historian
```

This is intentionally policy-level. The workflow middleware enforces what can be deterministic: known roles, known profiles, valid DAGs, required gates, materialization, state transitions, and audit events.

## Store/API implications

The workflow store and API should keep fields that make OMH-style routing auditable without depending on OMH naming:

- workflow `board`
- workflow `workspace_path`
- workflow `scale`
- workflow `policy_snapshot_json`
- node `role`
- node `profile`
- node `kanban_task_id`
- node `worktree_path`
- node `branch`
- event `actor_type`
- event `actor_id`
- event `data_json` for profile/model/verdict/evidence details

The WebUI should display both role and profile. Example:

```text
Node: Implement workflow store tests
Role: engineer
Profile: fixer / engineer / project-selected profile
Status: running
Kanban task: t_...
Gate: LLM-auditable review pending
```

## What we are not copying

Hermes Workflow should not copy OMH wholesale.

Not adopted:

- OMH role names as first-party canonical roles
- a separate OMH runtime
- timestamped per-turn boards
- prompt-only enforcement of workflow rules
- hard requirement that all work always route through Kanban
- project-specific agent names in core workflow artifacts

The general Hermes workflow system remains project-agnostic.

## MVP adoption plan

### Now / current branch

- Keep PRD and architecture spec canonical around Hermes roles.
- Add workflow store fields that support named-profile routing and auditability.
- Keep Kanban mappings explicit in workflow DB.
- Treat board/workspace/profile as first-class metadata.

### Next implementation slices

1. Store tests for workflow creation, event append/list, and board/workspace/profile metadata.
2. Policy schema extension for routing/profile-suite/checkpointing defaults.
3. DAG schema/validator that rejects unknown roles and unmapped profiles.
4. Kanban materializer that writes workflow/node IDs into task metadata/body and records mappings.
5. API/status serialization that WebUI can render without scraping Kanban prose.

### Later

- Reusable profile-suite presets.
- Multi-review/council gates.
- Visual observer/designer helper profile conventions.
- WebUI controls for gate approval, retry, reassignment, and review cascade decisions.

## Acceptance criteria for this bridge

This bridge is satisfied when:

- workflow PRD/spec readers can see which OMH lessons influenced Hermes Workflow
- Hermes canonical role names remain preserved
- profile-suite routing is documented as an extension point
- durable Kanban usage is clearly separated from workflow DB source-of-truth
- OMH is framed as architectural inspiration, not a dependency
