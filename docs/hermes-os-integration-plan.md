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
- External data.
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

### Phase 10 - Template Engine

- `task-020`: Define template engine architecture.

Template engine target architecture:

```text
Template Definition
  -> Template Registry
  -> Template Loader
  -> Template Compiler
  -> Template Validator
  -> Work Graph
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

- Template engine: template usage, validation pass rate, compile success rate, generated nodes, generated dependencies.
- Workspace projects: architecture score, workflow completeness, approval coverage, execution progress, blocker count.

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
- `task-049`: Review workspace project architecture readiness.
- `task-050`: Review workspace project and workspace project architecture readiness.

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
- Workspace dashboard: all projects, all work, all reviews, all approvals, all agents.

### Cross-Project Control Plane

`hermes workspace` gives one screen for every project and reports:

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

### Phase 32 - Workspace Control Plane

- `task-102`: Add workspace scanner and aggregate model.
- `task-103`: Add `hermes workspace` CLI entrypoint.
- `task-104`: Add workspace dashboard contract.
- `task-105`: Add cross-project blocker and approval summaries.

### Phase 33 - Autonomous Review Loops

- `task-106`: Add scheduled architecture review contract.
- `task-107`: Add review loop roadmap update behavior.
- `task-108`: Add continuous architecture score history.
- `task-109`: Add autonomous review safety and approval policy.

### Phase 34 - Template Engine Expansion

- `task-110`: Model reusable template work graph.
- `task-111`: Add template validation nodes.
- `task-112`: Add template decision graph nodes.
- `task-113`: Add template dashboard metrics for work graph execution.

### Phase 35 - Native Command Hardening

- `task-114`: Add end-to-end CLI tests for `hermes architect review` through the installed console entrypoint.
- `task-115`: Add end-to-end CLI tests for `hermes plan` through the installed console entrypoint.
- `task-116`: Add help text and docs examples for `hermes architect review --persist --db`.
- `task-117`: Add help text and docs examples for `hermes plan --template --persist --db`.
- `task-118`: Add command error envelopes for invalid projects, invalid templates, and persistence failures.

### Phase 36 - Task Generation Engine

- `task-119`: Add a `TaskDefinition` schema with id, title, phase, dependencies, acceptance criteria, risk, and status.
- `task-120`: Add an architecture-review-to-task generator that converts missing docs, schemas, dashboards, and approvals into task definitions.
- `task-121`: Add a work-graph-to-task generator that converts blocked nodes and findings into implementation tasks.
- `task-122`: Add stable task id allocation that continues from the highest existing `task-NNN`.
- `task-123`: Add task artifact writers for `TASKS.md` and `.hermes/tasks.json`.

### Phase 37 - Dashboard Task UI

- `task-124`: Add task summary data to `/api/hermes-os/summary`.
- `task-125`: Add a task backlog panel to the Hermes OS dashboard page.
- `task-126`: Add blocked task and approval-required task filters.
- `task-127`: Add task dependency visualization for work graph-derived tasks.
- `task-128`: Add dashboard actions to regenerate tasks in dry-run mode.

### Phase 38 - Persistence Migrations

- `task-129`: Add SQLite schema versioning for Hermes OS records.
- `task-130`: Add migration runner for future Hermes OS persistence changes.
- `task-131`: Add import from local JSON records into SQLite.
- `task-132`: Add export from SQLite records into portable JSON bundles.
- `task-133`: Add repository integrity checks for malformed payloads and duplicate ids.

### Phase 39 - Scheduled Review Operations

- `task-134`: Add scheduled review job registration through `hermes cron`.
- `task-135`: Add scheduled review dry-run preview output.
- `task-136`: Persist scheduled review run summaries and score deltas.
- `task-137`: Add approval gates for scheduled review writes.
- `task-138`: Add dashboard status for last review, next review, and review failures.

### Phase 40 - Runtime Policy Enforcement

- `task-139`: Wire runtime policy decisions into delegation execution before launching a worker.
- `task-140`: Persist runtime policy audit records.
- `task-141`: Add retry backoff policy and retry exhaustion reporting.
- `task-142`: Add cost budget aggregation per project and per work graph.
- `task-143`: Add approval prompts for policy-blocked high-risk runtime actions.

### Phase 41 - Template Registry

- `task-144`: Add a template registry location under `.hermes/templates`.
- `task-145`: Add template discovery from project, workspace, and user template paths.
- `task-146`: Add template validation diagnostics with line/file context where available.
- `task-147`: Add template version metadata and compatibility checks.
- `task-148`: Add dashboard panels for loaded templates and compile failures.

### Phase 42 - Execution Readiness

- `task-149`: Add a graph execution planner that emits executable batches from dependencies.
- `task-150`: Add dry-run execution reports showing commands, policies, approvals, and expected artifacts.
- `task-151`: Add artifact ingestion from completed runtime tasks into SQLite persistence.
- `task-152`: Add validation result updates after artifact ingestion.
- `task-153`: Add rollout tests that run Architect -> Plan -> Tasks -> Dry-run Execution end to end.

### Phase 43 - Workspace & Project Runtime MVP

- `task-154`: Add project definition schema for `.hermes/projects/<project>/project.yaml`.
- `task-155`: Add workspace project registry loader and validator.
- `task-156`: Add `hermes projects` status output across registered projects.
- `task-157`: Add `hermes switch <project>` command contract and dry-run implementation.
- `task-158`: Add project memory file scaffold for architecture, decisions, progress, experiments, lessons, backlog, and agents.
- `task-159`: Add project memory loader for switch/status context.
- `task-160`: Add project-scoped task registry summary.
- `task-161`: Add project status CLI output for memory, tasks, dashboards, agents, and infrastructure.

### Phase 44 - Workspace Snapshot & Restore

- `task-162`: Add workspace snapshot schema for open files, browser URLs, terminals, services, branch, and open tasks.
- `task-163`: Add `hermes snapshot save` project workspace snapshot command.
- `task-164`: Add `hermes snapshot restore` project workspace restore command.
- `task-165`: Add VS Code workspace restore contract.
- `task-166`: Add browser URL restore contract.
- `task-167`: Add running service restore contract.

### Phase 45 - Project Runtime Manager

- `task-168`: Add runtime service definition schema for project startup commands.
- `task-169`: Add `hermes start <project>` runtime startup command.
- `task-170`: Add runtime process status tracking.
- `task-171`: Add runtime failure and partial-start reporting.
- `task-172`: Add runtime dashboard URL opening contract.
- `task-173`: Add cost and service health hooks for project runtime status.

### Phase 46 - Agent Messaging & Trace Visibility

- `task-174`: Add project-scoped agent registry from project definitions.
- `task-175`: Add project agent message bus record schema.
- `task-176`: Add agent trace record schema with sender, receiver, timestamp, type, content, and correlation ID.
- `task-177`: Add agent trace viewer data contract for dashboard timelines.
- `task-178`: Add agent health summary for dashboard modules.
- `task-179`: Add tests for project-scoped agent messaging isolation.

### Phase 47 - Infrastructure Registry & Unified Dashboard

- `task-180`: Add infrastructure registry schema for project-owned external systems.
- `task-181`: Add vector database registry fields per project.
- `task-182`: Add unified dashboard modules for projects, agent health, costs, experiments, tasks, alerts, infrastructure, queues, and activity feed.
- `task-183`: Add end-to-end workspace runtime MVP test for project switch status within 30 seconds.

### Phase 48 - Real Command Surface Completion

- `task-184`: Audit installed Hermes OS command routing for architect, plan, workspace, projects, switch, start, and snapshot.
- `task-185`: Normalize JSON output envelopes across Hermes OS native commands.
- `task-186`: Normalize non-JSON human output across Hermes OS native commands.
- `task-187`: Add shared command error codes for missing project, invalid registry, unsafe action, and persistence failure.
- `task-188`: Add command-level smoke tests for `hermes workspace` through the installed entrypoint.
- `task-189`: Add command-level smoke tests for `hermes projects` through the installed entrypoint.
- `task-190`: Add command-level smoke tests for `hermes switch` through the installed entrypoint.
- `task-191`: Add command-level smoke tests for `hermes start` through the installed entrypoint.
- `task-192`: Add command-level smoke tests for `hermes snapshot` through the installed entrypoint.
- `task-193`: Add command docs for architecture-to-runtime operating flow.
- `task-194`: Add shell-completion metadata for Hermes OS command arguments.
- `task-195`: Add backward-compatible module CLI redirects for old Hermes OS invocation paths.

### Phase 49 - Guarded Live Runtime Execution

- `task-196`: Add live runtime enablement flag to project definitions.
- `task-197`: Add per-project runtime budget fields for cost and token ceilings.
- `task-198`: Add runtime command allowlist validation before launching services or workers.
- `task-199`: Add human approval requirement for write-capable live runtime actions.
- `task-200`: Add live worker launch adapter around the Official Hermes Agent wrapper.
- `task-201`: Add runtime timeout enforcement for live worker execution.
- `task-202`: Add retry classification for transient runtime failures.
- `task-203`: Add retry exhaustion records for failed live runtime actions.
- `task-204`: Add runtime artifact quarantine before validation and ingestion.
- `task-205`: Add rollback report generation for failed live runtime operations.
- `task-206`: Add live runtime dry-run parity tests.
- `task-207`: Add live runtime opt-in integration test with safe no-op worker command.

### Phase 50 - Workspace Restore Integrations

- `task-208`: Add platform detection for workspace restore launchers.
- `task-209`: Add VS Code workspace reopen implementation behind dry-run/live mode.
- `task-210`: Add editor fallback contract for non-VS Code environments.
- `task-211`: Add browser URL reopen implementation behind dry-run/live mode.
- `task-212`: Add terminal session restore plan for captured terminal commands.
- `task-213`: Add service restart plan for project runtime services.
- `task-214`: Add git branch restore validation before checkout actions.
- `task-215`: Add active task context restore into project memory and dashboard state.
- `task-216`: Add partial restore result schema with skipped, failed, and completed steps.
- `task-217`: Add restore conflict detection for dirty worktrees and running services.
- `task-218`: Add restore approval gate for branch changes and service restarts.
- `task-219`: Add cross-platform restore tests for dry-run contracts.

### Phase 51 - Runtime Dashboard UI

- `task-220`: Add Hermes OS project list panel to the dashboard UI.
- `task-221`: Add project runtime service health panel to the dashboard UI.
- `task-222`: Add workspace snapshot history panel to the dashboard UI.
- `task-223`: Add snapshot restore preview panel to the dashboard UI.
- `task-224`: Add agent trace timeline panel to the dashboard UI.
- `task-225`: Add agent message detail drawer to the dashboard UI.
- `task-226`: Add runtime cost and budget panel to the dashboard UI.
- `task-227`: Add approval queue panel for blocked runtime actions.
- `task-228`: Add infrastructure registry panel to the dashboard UI.
- `task-229`: Add vector database registry panel to the dashboard UI.
- `task-230`: Add task backlog and dependency panel refresh from project runtime state.
- `task-231`: Add dashboard empty, loading, and error states for Hermes OS panels.

### Phase 52 - Durable Project Runtime Persistence

- `task-232`: Add SQLite schema for project definitions and registry cache.
- `task-233`: Add SQLite schema for workspace snapshots.
- `task-234`: Add SQLite schema for snapshot restore attempts.
- `task-235`: Add SQLite schema for project runtime service status.
- `task-236`: Add SQLite schema for agent messages and traces.
- `task-237`: Add SQLite schema for runtime approvals and decisions.
- `task-238`: Add SQLite schema for runtime cost records.
- `task-239`: Add SQLite schema for infrastructure and vector registry records.
- `task-240`: Add persistence repository methods for project runtime reads and writes.
- `task-241`: Add JSON import and export for project runtime persistence records.
- `task-242`: Add integrity checks for orphaned traces, snapshots, approvals, and runtime statuses.
- `task-243`: Add migration tests for project runtime persistence schemas.

### Phase 53 - External Template Packs

- `task-244`: Define external template pack manifest schema.
- `task-245`: Add template pack discovery from workspace and user-level directories.
- `task-246`: Add template pack compatibility checks against Hermes OS schema version.
- `task-247`: Add template pack dependency declaration support.
- `task-248`: Add template pack dry-run install command contract.
- `task-249`: Add template pack live install command with approval gate.
- `task-250`: Add template pack update dry-run and diff output.
- `task-251`: Add template pack uninstall safety checks.
- `task-252`: Add template pack validation diagnostics with file and line context.
- `task-253`: Add template pack dashboard visibility for installed and failing packs.
- `task-254`: Add template pack tests with valid, invalid, incompatible, and missing dependency cases.
- `task-255`: Document template pack authoring conventions and boundary rules.

### Phase 54 - Continuous Workspace Operations

- `task-256`: Add recurring workspace health check scheduler contract.
- `task-257`: Add architecture drift detection against latest project docs and decisions.
- `task-258`: Add stale task detection using task age, blockers, and dependency status.
- `task-259`: Add stale snapshot detection for projects with outdated workspace state.
- `task-260`: Add runtime service drift detection against project startup definitions.
- `task-261`: Add approval queue aging and escalation summary.
- `task-262`: Add cost budget drift detection across runtime records.
- `task-263`: Add agent failure-rate monitoring across project traces.
- `task-264`: Add infrastructure availability check contracts for project-owned systems.
- `task-265`: Add workspace health score calculation and trend history.
- `task-266`: Add dashboard activity feed records for continuous operations.
- `task-267`: Add end-to-end dry-run test for scheduled workspace operations.

### Phase 55 - Production Live Runtime

- `task-268`: Add live runtime execution state machine for queued, running, validating, completed, failed, canceled, and rolled back states.
- `task-269`: Add live runtime process supervisor with pid, exit code, stdout, stderr, and duration capture.
- `task-270`: Add live runtime cancellation command and cancellation audit record.
- `task-271`: Add live runtime resume command for interrupted executions with prior context.
- `task-272`: Add live runtime log tailing contract for dashboard and CLI consumers.
- `task-273`: Add live runtime artifact manifest schema with checksums and validation status.
- `task-274`: Add live runtime validation gate before artifact ingestion into persistence.
- `task-275`: Add live runtime rollback executor for reversible project-runtime actions.
- `task-276`: Add live runtime sandbox profile selection per project and action type.
- `task-277`: Add live runtime environment variable allowlist and redaction policy.
- `task-278`: Add live runtime execution history query API.
- `task-279`: Add production live-runtime integration test using a safe local no-op worker.

### Phase 56 - Approval UX & Governance

- `task-280`: Add approval request schema with requester, reviewer, scope, risk, expiry, and decision fields.
- `task-281`: Add approval queue persistence methods and list/get/update APIs.
- `task-282`: Add `hermes approvals list` command contract.
- `task-283`: Add `hermes approvals approve` command contract with required reason capture.
- `task-284`: Add `hermes approvals reject` command contract with required reason capture.
- `task-285`: Add approval expiry worker contract for stale pending approvals.
- `task-286`: Add policy override record schema for exceptional runtime actions.
- `task-287`: Add approval dashboard actions for approve, reject, and request-more-context flows.
- `task-288`: Add approval notification event records for dashboard activity feed.
- `task-289`: Add approval risk scoring from action type, project, cost, and target resources.
- `task-290`: Add approval audit export for compliance review.
- `task-291`: Add governance tests for approval expiry, override, and audit immutability.

### Phase 57 - Workspace Runtime Automation

- `task-292`: Add project automation workflow schema for switch, restore, start, validate, and notify steps.
- `task-293`: Add `hermes project run <workflow>` command contract.
- `task-294`: Add project switch automation that chains memory load, snapshot restore, service start, and dashboard open.
- `task-295`: Add project shutdown automation for services, snapshots, and final status records.
- `task-296`: Add automation preflight checks for dirty worktree, unavailable tools, blocked approvals, and missing config.
- `task-297`: Add automation step dependency model with skip, retry, and fallback behavior.
- `task-298`: Add automation dry-run diff between current workspace state and target project state.
- `task-299`: Add automation execution history and replay contract.
- `task-300`: Add automation dashboard panel for active and recent project workflows.
- `task-301`: Add automation schedule integration for recurring project warmup flows.
- `task-302`: Add automation failure report with next-action recommendations.
- `task-303`: Add end-to-end dry-run automation test for switch -> restore -> start.

### Phase 58 - Multi-Project Orchestration

- `task-304`: Add cross-project dependency schema with source project, target project, reason, and status.
- `task-305`: Add workspace-level dependency resolver across registered projects.
- `task-306`: Add cross-project blocker registry and blocker aging fields.
- `task-307`: Add shared approval request support for actions spanning multiple projects.
- `task-308`: Add multi-project execution queue planner with per-project isolation.
- `task-309`: Add queue balancing policy for project priority, risk, and readiness.
- `task-310`: Add shared infrastructure dependency mapping across projects.
- `task-311`: Add multi-project dashboard graph panel contract.
- `task-312`: Add cross-project stale dependency detection.
- `task-313`: Add orchestration report export for workspace review.
- `task-314`: Add conflict detection for simultaneous actions against shared resources.
- `task-315`: Add multi-project orchestration tests with dependency and blocker scenarios.

### Phase 59 - Agent Fleet Management

- `task-316`: Add agent fleet registry schema with capability, model, tools, cost, and availability fields.
- `task-317`: Add agent heartbeat record schema and stale-heartbeat detection.
- `task-318`: Add agent capability matching score for task routing.
- `task-319`: Add agent health score from success rate, failure rate, latency, and retry count.
- `task-320`: Add agent cost score from token and runtime usage records.
- `task-321`: Add agent latency score from trace and runtime records.
- `task-322`: Add agent quarantine policy for repeated failures or policy violations.
- `task-323`: Add agent fleet dashboard panels for health, cost, latency, and routing confidence.
- `task-324`: Add agent routing explanation record for every assignment.
- `task-325`: Add agent fallback selection when preferred specialist is unavailable.
- `task-326`: Add agent fleet export for operations review.
- `task-327`: Add fleet routing tests for healthy, degraded, unavailable, and quarantined agents.

### Phase 60 - Observability & Telemetry

- `task-328`: Add Hermes OS event envelope schema with project, phase, severity, source, and correlation id.
- `task-329`: Add structured event writer for command, runtime, approval, restore, and dashboard events.
- `task-330`: Add trace correlation across command invocation, runtime action, agent message, and artifact ingestion.
- `task-331`: Add metric rollup records for daily project health, cost, runtime, and approval activity.
- `task-332`: Add telemetry redaction policy for prompts, env vars, URLs, and credentials.
- `task-333`: Add observability dashboard trend panels for health, cost, approvals, runtime failures, and agent routing.
- `task-334`: Add diagnostics bundle export for a project runtime incident.
- `task-335`: Add event retention and pruning policy.
- `task-336`: Add telemetry import/export compatibility checks.
- `task-337`: Add slow command and slow runtime action detection.
- `task-338`: Add telemetry integrity checks for missing correlation chains.
- `task-339`: Add observability tests for event redaction, rollups, exports, and correlation.

### Phase 61 - Plugin & Connector Boundary

- `task-340`: Add connector manifest schema with permissions, resources, commands, and risk profile.
- `task-341`: Add connector registry discovery from project, workspace, and user paths.
- `task-342`: Add connector permission evaluator for read, write, deploy, purchase, and destructive actions.
- `task-343`: Add connector dry-run contract for external writes.
- `task-344`: Add connector audit record schema with external target and normalized action type.
- `task-345`: Add connector output normalization contract for dashboard and persistence ingestion.
- `task-346`: Add connector health check contract and status dashboard panel.
- `task-347`: Add connector secret reference policy that avoids storing raw secrets.
- `task-348`: Add connector install/update/remove lifecycle contracts.
- `task-349`: Add connector compatibility checks against Hermes OS and project definitions.
- `task-350`: Add connector sandbox tests with fake external systems.
- `task-351`: Document connector boundary rules for domain systems and source-of-truth ownership.

### Phase 62 - Evaluation & Quality Gates

- `task-352`: Add evaluation schema for architecture reviews, work graphs, templates, runtime artifacts, and dashboard contracts.
- `task-353`: Add deterministic validation checks for required fields, dependencies, approvals, and traceability.
- `task-354`: Add rubric-based evaluation contract for quality-sensitive generated documents.
- `task-355`: Add self-review/reflection workflow for generated specs and implementation plans.
- `task-356`: Add evaluation evidence persistence for pass, fail, warning, and waived outcomes.
- `task-357`: Add quality gate runner that blocks execution when required evaluations fail.
- `task-358`: Add human waiver record for failed gates with reason and expiry.
- `task-359`: Add dashboard panel for gate status and recent evaluation failures.
- `task-360`: Add regression evaluation set for Hermes OS planning outputs.
- `task-361`: Add cost-aware evaluation routing for cheap deterministic checks before expensive model review.
- `task-362`: Add evaluation export for release and compliance review.
- `task-363`: Add quality gate tests for pass, fail, warning, waiver, and blocked execution paths.

### Phase 63 - Project Memory Intelligence

- `task-364`: Add project memory index schema with source path, record id, timestamp, and summary fields.
- `task-365`: Add decision retrieval API scoped by project, topic, recency, and confidence.
- `task-366`: Add lessons-learned extraction from completed tasks, failures, and approvals.
- `task-367`: Add project memory summarizer for architecture, decisions, progress, experiments, backlog, and agents.
- `task-368`: Add memory traceability links from summary bullets back to source artifacts.
- `task-369`: Add memory drift detector for stale decisions and conflicting project documents.
- `task-370`: Add memory refresh scheduler contract for long-running projects.
- `task-371`: Add memory query dashboard panel for decisions, lessons, and drift warnings.
- `task-372`: Add memory export/import bundle for project handoff.
- `task-373`: Add guardrail that prevents agent runtime memory from overwriting source-of-truth memory records.
- `task-374`: Add project memory compaction policy for old records and summaries.
- `task-375`: Add memory intelligence tests for retrieval, summarization, drift, and source traceability.

### Phase 64 - Release Hardening

- `task-376`: Add full Hermes OS release checklist covering CLI, dashboard, runtime, persistence, templates, connectors, and docs.
- `task-377`: Add migration compatibility tests from task registry versions 183, 267, and current.
- `task-378`: Add backward compatibility tests for legacy module CLI invocations.
- `task-379`: Add dashboard build verification for Hermes OS runtime pages.
- `task-380`: Add packaging verification for new Hermes OS modules and subcommands.
- `task-381`: Add documentation pass for command examples, safety policies, restore flows, and live runtime rollout.
- `task-382`: Add failure drill for unavailable runtime worker.
- `task-383`: Add failure drill for corrupted project registry.
- `task-384`: Add failure drill for interrupted live execution.
- `task-385`: Add failure drill for failed migration or partial persistence write.
- `task-386`: Add release notes generator from completed Hermes OS task records.
- `task-387`: Add full Hermes OS integration suite target for CI or local release verification.
