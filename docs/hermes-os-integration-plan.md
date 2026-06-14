# Hermes OS + Official Hermes Agent Integration Plan

## Direction

Hermes OS remains the control plane and source of truth. Official Hermes Agent is integrated as an optional execution/runtime layer underneath Hermes OS.

Hermes OS also becomes architecture-first by default. It must challenge project ideas, generate specifications, require domain/workflow/dashboard design, and only then allow implementation tasks to reach an agent runtime.

## Guardrails

- Do not replace Hermes OS.
- Do not migrate existing projects.
- Do not overwrite the `hermes` command.
- Use `hermes-agent` or equivalent for the official runtime.
- Do not treat runtime memory as authoritative project/task/report storage.
- Do not begin coding until business system, control plane, domain model, workflows, dashboards, metrics, approvals, and agent boundaries have been reviewed.
- Do not allow agents to own projects, workflows, dashboards, approvals, business logic, or source-of-truth state.

## Command Ownership

- `hermes` belongs to Hermes OS.
- `hermes-agent` belongs to the official Hermes Agent runtime in this repository.
- Shell setup must not alias `hermes` to this repository.
- If `hermes` resolves unexpectedly, check `PATH`, shell aliases, and any symlink under `~/.local/bin`.
- If `hermes-agent` is unavailable, Hermes OS must degrade to direct execution or mark delegation unavailable.

## Runtime Adapter Contract

Hermes OS sends an `AgentRequest`:

- `task_id`
- `project_id`
- `agent_kind`
- `prompt`
- `working_directory`
- `context`
- `tool_policy`
- `runtime_provider`
- `timeout_seconds`
- `dry_run`

Hermes Agent returns an `AgentResponse`:

- `task_id`
- `status`
- `output`
- `artifacts`
- `errors`
- `duration_ms`
- `cost`
- `stdout`
- `stderr`
- `exit_code`

All adapter payloads are validated by `hermes_os_integration.contracts` before Hermes OS stores results.

## Task Batches

### Phase 1 - Standalone Runtime

- `task-001`: Verify standalone Hermes Agent runtime.
- `task-002`: Create `hermes-agent` launcher.
- `task-003`: Document command ownership.

### Phase 2 - Runtime Wrapper

- `task-004`: Define runtime adapter contract.
- `task-005`: Build runtime wrapper prototype.

### Phase 3 - Contracts

- `task-006`: Add validated integration schemas.
- `task-007`: Add adapter error taxonomy.

### Phase 4 - Agent Registry

- `task-008`: Create official-agent registry.
- `task-009`: Map Hermes OS tasks to agent kinds.

### Phase 5 - Delegation

- `task-010`: Design delegation engine.
- `task-011`: Build delegation prototype.

Delegation flow:

```text
Hermes OS task
  -> task type to agent kind mapping
  -> validated AgentRequest
  -> RuntimeWrapper
  -> validated AgentResponse
  -> Hermes OS result storage
```

Retries and escalation are owned by Hermes OS. Runtime memory is never the final record.

### Phase 6 - Memory Boundary

- `task-012`: Define memory boundary.
- `task-013`: Add memory sync guardrails.

Allowed runtime memory:

- Cache.
- Execution-local notes.
- Temporary context.

Disallowed runtime memory:

- Project source of truth.
- Task status source of truth.
- Final report storage.
- Review decision storage.

### Phase 7 - MCP Integration

- `task-014`: Design MCP permission bridge.
- `task-015`: Build MCP adapter prototype.

Hermes OS grants tool permissions. Official runtime consumes only delegated tool capabilities.

Supported MCP categories:

- GitHub.
- Discord.
- Browser.
- Filesystem.
- Documentation.
- Research.
- Market data.
- Broker integrations.

### Phase 8 - Dashboard

- `task-016`: Design dashboard runtime status panels.
- `task-017`: Add runtime health endpoint contract.

Dashboard panels should show:

- Runtime status: running, stopped, unavailable.
- Provider, latency, version, and recent errors.
- Agent usage: tasks executed, success rate, retries, cost, tokens.
- Delegation status: active tasks, queue length, completion time.

Dashboard must continue to work if the runtime is unavailable.

### Phase 9 - Long-Running Workflows

- `task-018`: Design long-running workflow orchestration.
- `task-019`: Build checkpointed workflow prototype.

Workflow pattern:

```text
Research -> Validation -> Review -> Report
```

Hermes OS stores checkpoints after each step and can resume from the latest checkpoint.

### Phase 10 - Market Research

- `task-020`: Define market research agent architecture.

Market research target architecture:

```text
Bucket
  -> Research Agent
  -> Evidence Agent
  -> Validation Agent
  -> Portfolio Agent
  -> Experiment Tracker
  -> Dashboard
```

### Validation

- `task-021`: Create integration smoke tests.
- `task-022`: Create integration rollout plan.

Rollout:

1. Dry-run delegation only.
2. Enable one low-risk project.
3. Enable one agent kind at a time.
4. Monitor runtime health and validation failures.
5. Roll back by disabling runtime delegation.

## Architecture-First Framework

Architecture-first work follows this order:

```text
Idea
  -> Business System
  -> Control Plane
  -> Domain Models
  -> Workflows
  -> Dashboards
  -> Metrics
  -> Approval Gates
  -> Agents
  -> Implementation
  -> Optimization
```

The framework rejects the old path of idea-to-code. Project creation, reviews, and agent delegation must all enforce architecture readiness.

### Constitution

Every Hermes-controlled project loads `.hermes/constitution.md` before work. The constitution defines:

- Business logic before implementation.
- Dashboards before automation.
- Workflows before agents.
- Agents are workers, not owners.
- Persistent state belongs to Hermes OS.
- Agent memory is not source of truth.
- Every project requires a domain model.
- Every project requires measurable outcomes.
- High-risk actions require human approval.
- Coding begins only after architecture review.
- Specifications generate tasks.
- Tasks generate execution.
- Execution generates artifacts.
- Artifacts generate dashboards.
- Dashboards generate feedback loops.

### Architect Command

`hermes architect review <project>` evaluates a project against:

- Business model.
- Control plane.
- Domain model.
- Workflows.
- Dashboards.
- Metrics.
- Approvals.
- Agents.
- Data quality.
- Automation opportunities.
- Scalability.
- Technical debt.

Outputs include missing entities, workflows, metrics, dashboards, approvals, schemas, automation opportunities, architecture score, critical gaps, recommendations, and priority roadmap.

### Grill-Me Skill

`/grill-me` challenges assumptions before work begins. It must cover business, domain, workflow, metrics, dashboard, approvals, automation, agents, scalability, monetization, risk, and data.

### Project Bootstrap

New project creation becomes an architecture pipeline:

1. Architecture interview.
2. Grill-me.
3. Domain modeling.
4. Workflow design.
5. Dashboard design.
6. Specification generation.
7. Task generation.
8. Execution.

Every project receives:

- `PROJECT.md`
- `DOMAIN.md`
- `WORKFLOWS.md`
- `DASHBOARD.md`
- `METRICS.md`
- `APPROVALS.md`
- `AGENTS.md`
- `TASKS.md`
- `DECISIONS.md`
- `ROADMAP.md`
- `ARCHITECTURE.md`

### Domain And Workflow Engines

The domain engine generates entity definitions, relationships, lifecycle definitions, schemas, storage models, and documentation.

The workflow engine defines triggers, inputs, steps, outputs, approvals, metrics, failure states, and escalation rules.

All major entities use validated schemas before AI output can enter source-of-truth state.

### Dashboard-First Development

Before automation is implemented, the project must define daily visibility, success metrics, failure indicators, opportunity indicators, and required reports.

Examples:

- Market research: research throughput, evidence coverage, validation quality, decision readiness, portfolio exposure.
- Media Engine: stories published, coverage mix, brand growth, cross-brand comparison, platform coverage, approval time.
- Investment System: watchlist size, thesis status, portfolio exposure, valuation opportunities, risk metrics.

### Architecture Task Batches

### Phase 11 - Constitution

- `task-023`: Create global Hermes constitution.
- `task-024`: Add constitution loader contract.
- `task-025`: Add architecture order violation warnings.

### Phase 12 - Architect Review

- `task-026`: Design architect review scoring model.
- `task-027`: Build architect review contract.
- `task-028`: Add architect review CLI specification.
- `task-029`: Add architect review report template.

### Phase 13 - Grill-Me

- `task-030`: Define grill-me question bank.
- `task-031`: Build grill-me session schema.
- `task-032`: Add assumption challenge output contract.

### Phase 14 - Project Bootstrap

- `task-033`: Design architecture-first project bootstrap flow.
- `task-034`: Add required project document templates.
- `task-035`: Add specification-to-task generation contract.

### Phase 15 - Domain Modeling

- `task-036`: Design domain modeling engine.
- `task-037`: Add Zod-first entity schema catalog.
- `task-038`: Add domain lifecycle documentation generator contract.

### Phase 16 - Workflow Design

- `task-039`: Design workflow engine contract.
- `task-040`: Add workflow approval and escalation model.
- `task-041`: Add workflow metrics contract.

### Phase 17 - Dashboard-First Gates

- `task-042`: Design dashboard requirements generator.
- `task-043`: Add dashboard readiness gate.
- `task-044`: Add dashboard feedback loop contract.

### Phase 18 - Agent Boundaries

- `task-045`: Add agent ownership prohibition policy.
- `task-046`: Add artifact ingestion validation contract.
- `task-047`: Add runtime delegation readiness gate.

### Phase 19 - Existing Project Reviews

- `task-048`: Review workspace project architecture readiness.
- `task-049`: Review Investment System architecture readiness.
- `task-050`: Review Media Engine and Rinseables architecture readiness.

### Phase 20 - Architect CLI Implementation

- `task-051`: Implement `hermes architect review <project>` CLI entrypoint.
- `task-052`: Add architect CLI JSON and human-readable output modes.
- `task-053`: Add architect CLI exit-code handling and blocked-review behavior.

### Phase 21 - Project Scanners

- `task-054`: Build project discovery scanner.
- `task-055`: Build documentation coverage scanner.
- `task-056`: Build domain/workflow/dashboard/approval scanner.
- `task-057`: Add generic workspace project profiles.

### Phase 22 - Review-Generated Documentation

- `task-058`: Generate missing architecture docs from review output.
- `task-059`: Add safe document write and overwrite policy.
- `task-060`: Add review-to-roadmap generation.
- `task-061`: Add per-project architecture review artifacts.

### Phase 23 - Architecture Dashboard

- `task-062`: Design architecture dashboard data model.
- `task-063`: Add architecture score and gap panels.
- `task-064`: Add approvals and blocked-execution panels.
- `task-065`: Add runtime delegation status panel.

### Phase 24 - Execution Gates

- `task-066`: Connect architecture readiness gate to task execution.
- `task-067`: Add premature coding block behavior.
- `task-068`: Add human override and approval audit trail.
- `task-069`: Add task-generation traceability enforcement.

### Phase 25 - Persistence

- `task-070`: Add persistence model for review reports.
- `task-071`: Add persistence model for grill-me sessions.
- `task-072`: Add persistence model for decisions and approvals.
- `task-073`: Add persistence model for validated agent artifacts.
- `task-074`: Add storage repository interface and local implementation.

### Phase 26 - Real Runtime Invocation

- `task-075`: Replace help-command wrapper behavior with real oneshot invocation.
- `task-076`: Add Official Hermes Agent prompt/context assembly.
- `task-077`: Add runtime artifact capture and ingestion.
- `task-078`: Add runtime timeout, retry, and failure policy.
- `task-079`: Add live delegation integration tests with dry-run fallback.

## Hermes OS v3 Control Plane Roadmap

Hermes OS v3 shifts the center of gravity from architecture review alone to a control plane that compiles architecture into executable work graphs.

North star:

```text
Idea
  -> Architecture
  -> Specification
  -> Work Graph
  -> Execution
  -> Validation
  -> Dashboard
  -> Continuous Improvement
```

### Work Graph Model

The work graph replaces flat task lists. It models:

- Project.
- Epics.
- Workflows.
- Tasks.
- Subtasks.
- Dependencies.
- Approvals.
- Artifacts.
- Metrics.
- Agent assignments.
- Execution results.
- Validation results.

### `hermes plan`

`hermes plan` reads architecture artifacts and emits `workgraph.json`.

Inputs:

- `PROJECT.md`
- `DOMAIN.md`
- `WORKFLOWS.md`
- `DASHBOARD.md`
- `METRICS.md`
- `APPROVALS.md`
- `AGENTS.md`

Responsibilities:

- Read architecture.
- Generate work graph.
- Identify missing work.
- Generate execution order.
- Assign agents.
- Generate validation rules.
- Generate approvals.
- Generate dashboard metrics.

### Execution Evolution

Execution moves from task-by-task execution to:

```text
Work Graph
  -> Dependency Resolution
  -> Execution Queue
  -> Agent Assignment
  -> Validation
  -> Artifacts
  -> Dashboard Update
```

### Dashboard Evolution

Add:

- Architecture dashboard: missing documents, score, workflow completeness, approval coverage, governance status.
- Work graph dashboard: epics, tasks, dependencies, blocked work, execution progress, agent assignments, approval queue.
- Agent dashboard: availability, success, failures, retries, cost, latency, token usage.
- Portfolio dashboard: all projects, all work, all reviews, all approvals, all agents.

### Cross-Project Control Plane

`hermes portfolio` gives one screen for every project and reports:

- Architecture score.
- Task health.
- Execution status.
- Blockers.
- Approvals.
- Runtime usage.

### Phase 27 - Work Graph Schema

- `task-080`: Define work graph core schema.
- `task-081`: Add dependency and execution-order model.
- `task-082`: Add validation result and execution result schemas.
- `task-083`: Add work graph JSON serialization contract.

### Phase 28 - Planning Engine

- `task-084`: Implement architecture artifact reader.
- `task-085`: Build architecture-to-work-graph compiler.
- `task-086`: Add `hermes plan` CLI entrypoint.
- `task-087`: Add missing-work detection.
- `task-088`: Add work graph persistence.

### Phase 29 - Graph Execution Orchestration

- `task-089`: Add dependency resolver.
- `task-090`: Add execution queue builder.
- `task-091`: Add graph-aware execution gate integration.
- `task-092`: Add validation rule generation.
- `task-093`: Add graph execution result ingestion.

### Phase 30 - Intelligent Agent Assignment

- `task-094`: Add agent capability model.
- `task-095`: Add task-to-agent assignment rules.
- `task-096`: Add assignment confidence and fallback behavior.
- `task-097`: Add runtime usage tracking for assignments.

### Phase 31 - Work Graph Dashboards

- `task-098`: Add work graph dashboard data model.
- `task-099`: Add dependency and blocked-work panels.
- `task-100`: Add execution progress and validation panels.
- `task-101`: Add agent assignment dashboard panels.

### Phase 32 - Portfolio Control Plane

- `task-102`: Add portfolio scanner and aggregate model.
- `task-103`: Add `hermes portfolio` CLI entrypoint.
- `task-104`: Add portfolio dashboard contract.
- `task-105`: Add cross-project blocker and approval summaries.

### Phase 33 - Autonomous Review Loops

- `task-106`: Add scheduled architecture review contract.
- `task-107`: Add review loop roadmap update behavior.
- `task-108`: Add continuous architecture score history.
- `task-109`: Add autonomous review safety and approval policy.

### Phase 34 - Market Research Work Graph Expansion

- `task-110`: Model market research-to-portfolio work graph.
- `task-111`: Add evidence quality and validation nodes.
- `task-112`: Add experiment and promotion decision graph nodes.
- `task-113`: Add market research dashboard metrics for work graph execution.
