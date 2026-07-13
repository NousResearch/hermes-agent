# Hermes OS System Explainer

## What This System Is

Hermes OS is being shaped into an architecture-first control plane. Its job is not to be a giant pile of project logic. Its job is to govern how projects are designed, planned, executed, validated, stored, and improved.

The current system is best understood as:

```text
Hermes OS
  -> architecture control plane
  -> governance layer
  -> work graph compiler
  -> execution orchestrator
  -> dashboard data source

Official Hermes Agent
  -> optional worker runtime
  -> reasoning and tool use
  -> delegated execution
  -> artifact producer
```

The most important design rule is this:

```text
Hermes OS owns state and governance.
Agents produce artifacts.
Templates describe reusable structure.
Projects provide domain specifics.
```

Hermes OS should not know about any specific project domain. It should not know about finance, media, CRM, markets, SaaS, APIs, or any other future business area. Those belong in templates or project artifacts, not in the OS itself.

Workflow design should also use the Agent Pattern Catalog in `AGENT_PATTERNS.md`.
Hermes should classify the task first, then choose the simplest reliable
pattern: single agent/tool use for simple tasks, prompt chaining or planning
for multi-step tasks, parallelization for independent work, router plus
clarification for uncertain routing, human approval for high-risk actions,
reflection plus evaluation for quality-sensitive output, resource-aware routing
for cost-sensitive work, and planning plus memory plus monitoring for
long-running projects.

Hermes OS should also treat projects as first-class runtime units. A project is
not only a folder; it has identity, memory, tasks, agents, infrastructure,
dashboards, documents, runtime, and metrics. The workspace runtime target is:

```text
User
  -> Hermes OS
  -> Project Runtime
  -> Agents
  -> Infrastructure
  -> Outputs
```

The `hermes switch <project>` workflow should eventually load the project
definition, restore the workspace, open dashboard URLs, start services, load
project memory, connect agents, load active tasks, and display project status.
Hermes tracks infrastructure and vector stores, but domain databases remain
project-owned and isolated.

## The Mental Model

The system is designed around this path:

```text
Idea
  -> Grill-Me
  -> Architecture Review
  -> Architecture Artifacts
  -> Work Graph Compilation
  -> Execution
  -> Validation
  -> Dashboard
  -> Continuous Improvement
```

This is intentionally different from:

```text
Idea
  -> Code
  -> Cleanup later
```

Hermes OS tries to prevent premature implementation. Before code should be written, the system wants to know:

- What is the business system?
- What is the control plane?
- What entities exist?
- What workflows exist?
- What dashboards are needed?
- What metrics define success or failure?
- What needs approval?
- What should agents do, and what must they never own?

## Source Of Truth

Hermes OS is the source of truth for:

- Projects.
- Tasks.
- Workflows.
- Architecture reviews.
- Work graphs.
- Approvals.
- Decisions.
- Dashboard state.
- Persistent artifacts.
- Execution history.

Official Hermes Agent is not the source of truth. Runtime memory is not the source of truth. A delegated agent can help research, write, validate, test, and produce artifacts, but Hermes OS decides what gets stored and what becomes authoritative.

## Major Components

### Constitution

File:

```text
.hermes/constitution.md
```

Module:

```text
hermes_os_integration/architecture_first.py
```

The constitution contains the global rules every Hermes-controlled project should follow. The most important rules are:

- Business logic before implementation.
- Dashboards before automation.
- Workflows before agents.
- Agents are workers, not owners.
- Persistent state belongs to Hermes OS.
- Agent memory is not source of truth.
- Coding begins only after architecture review.

Use this as the system's spine. When you are unsure whether Hermes should build something directly, ask whether the constitution allows it.

### Architecture-First Contracts

Module:

```text
hermes_os_integration/architecture_first.py
```

This module defines the architecture-first primitives:

- Required project documents.
- Architecture order.
- Review categories.
- Grill-me question categories.
- Review request and report structures.
- Dashboard readiness checks.
- Agent ownership rules.
- Artifact ingestion checks.
- Runtime delegation readiness.

The required project documents are:

```text
PROJECT.md
DOMAIN.md
WORKFLOWS.md
DASHBOARD.md
METRICS.md
APPROVALS.md
AGENTS.md
TASKS.md
DECISIONS.md
ROADMAP.md
ARCHITECTURE.md
```

These documents are the inputs that let Hermes OS reason about a project without hardcoding the project's domain.

### Architect Review CLI

Module:

```text
hermes_os_integration/architect_cli.py
```

Conceptual command:

```bash
hermes architect review <project>
```

Current module entrypoint:

```bash
python -m hermes_os_integration.architect_cli review <project> --projects-root /path/to/projects
```

What it does:

1. Resolves a project path.
2. Scans for required architecture documents.
3. Builds an architecture review request.
4. Produces an architecture review report.
5. Optionally generates missing docs.
6. Optionally writes a review artifact.
7. Returns stable exit codes.

Useful flags:

```bash
--json
--block-on-critical
--write-report
--generate-docs
--generate-tasks
--persist
--db /path/to/hermes-os.sqlite3
--overwrite
```

Native Hermes command examples:

```bash
hermes architect review . --json
hermes architect review . --write-report --persist --db .hermes/hermes-os.sqlite3
hermes architect review . --generate-docs --generate-tasks
```

Use this when you want Hermes OS to answer:

```text
Is this project architecturally ready?
What is missing?
What should be done next?
Should execution be blocked?
```

### Plan CLI

Module:

```text
hermes_os_integration/plan_cli.py
```

Native command:

```bash
hermes plan <project>
```

Useful examples:

```bash
hermes plan . --json
hermes plan . --write --generate-tasks
hermes plan . --template .hermes/templates/base-project.yaml --persist --db .hermes/hermes-os.sqlite3
```

`--template` accepts a JSON/YAML template file or a directory of template files.
`--persist` stores the compiled work graph in the Hermes OS SQLite repository.

### Project Scanners

Module:

```text
hermes_os_integration/scanners.py
```

The scanner discovers projects and checks architecture coverage. It looks for project docs in:

```text
project/
project/docs/
```

It intentionally does not ship with domain-specific project aliases or profiles. That was a key correction. The OS should not know that a particular workspace project represents a domain. If a project needs domain-specific treatment later, that should come from:

- Project docs.
- A template.
- A user-provided configuration.
- A plugin or connector.

The scanner should remain generic.

### Document Generation

Module:

```text
hermes_os_integration/doc_generation.py
```

This component generates missing architecture documents from review output.

It is deliberately conservative:

- It creates missing docs by default.
- It does not overwrite existing docs unless explicitly allowed.
- It writes traceability back to the architecture review.
- It can store review artifacts under a project-local Hermes path.

Use this after an architecture review when a project is missing source-of-truth documents.

### Runtime Contracts

Module:

```text
hermes_os_integration/contracts.py
```

The runtime contract defines what Hermes OS sends to a worker and what it expects back.

Hermes OS sends an `AgentRequest`:

```text
task_id
project_id
agent_kind
prompt
working_directory
context
tool_policy
runtime_provider
timeout_seconds
dry_run
```

The runtime returns an `AgentResponse`:

```text
task_id
status
output
artifacts
errors
duration_ms
cost
stdout
stderr
exit_code
```

The point of this contract is to make runtime delegation boring and inspectable. Hermes OS does not let arbitrary worker output become source-of-truth state without validation.

### Error Taxonomy

Module:

```text
hermes_os_integration/errors.py
```

This defines shared error codes such as:

```text
runtime_unavailable
runtime_timeout
validation_error
permission_denied
tool_failure
process_error
state_conflict
```

The system uses these codes so CLI commands, dashboards, persistence, and orchestration can understand failures consistently.

### Runtime Wrapper

Module:

```text
hermes_os_integration/wrapper.py
```

The runtime wrapper invokes Official Hermes Agent through the `hermes-agent` launcher.

Current behavior:

- Dry-run mode returns a deterministic accepted response without invoking the runtime.
- Live mode builds a `hermes-agent --oneshot` command.
- The prompt includes:
  - task objective,
  - project ID,
  - agent kind,
  - constitution rules,
  - source-of-truth boundary,
  - tool policy,
  - context.
- It captures stdout, stderr, exit code, errors, timeout, and duration.

The wrapper exists so Hermes OS can use Official Hermes Agent as infrastructure without surrendering control.

### Delegation Engine

Module:

```text
hermes_os_integration/delegation.py
```

Delegation turns a Hermes OS task into a worker request.

Flow:

```text
Task payload
  -> task type
  -> agent kind
  -> AgentRequest
  -> RuntimeWrapper
  -> AgentResponse
  -> persisted output reference
```

Use dry-run mode for early rollout. Live delegation should come later, after architecture gates and persistence are trusted.

### Agent Registry

Module:

```text
hermes_os_integration/registry.py
```

The registry maps task types to generic agent kinds:

- research
- coding
- testing
- review
- documentation
- deployment
- template
- experiment

The registry should stay generic. Do not add project-specific agent kinds here. If a domain needs specialized behavior later, prefer a template, plugin, or project-level configuration.

### Memory Boundary

Module:

```text
hermes_os_integration/memory_boundary.py
```

This module protects Hermes OS source-of-truth state from runtime memory writes.

Allowed runtime memory:

- cache,
- execution-local notes,
- temporary context.

Disallowed runtime memory:

- project source of truth,
- task status source of truth,
- final reports,
- review decisions,
- approval decisions.

This is one of the most important safety boundaries in the system.

### MCP Bridge

Module:

```text
hermes_os_integration/mcp_bridge.py
```

The MCP bridge defines tool permission grants. Hermes OS is the permission authority. Runtime workers receive only delegated capabilities.

Supported categories are generic:

- GitHub.
- Discord.
- Browser.
- Filesystem.
- Documentation.
- Research.
- External data.
- Custom integrations.

The bridge should never become a project-specific integration catalog.

### Health Checks

Module:

```text
hermes_os_integration/health.py
```

Health checks tell Hermes OS whether the worker runtime is available.

The dashboard and delegation system should degrade cleanly if the runtime is missing, stopped, or unavailable.

### Checkpointed Workflows

Module:

```text
hermes_os_integration/workflows.py
```

This prototype supports long-running workflows with checkpoints.

Example pattern:

```text
Research
  -> Validation
  -> Review
  -> Report
```

Hermes OS stores checkpoints after steps. That makes workflows resumable and auditable.

### Execution Gates

Module:

```text
hermes_os_integration/gates.py
```

Execution gates decide whether work is allowed.

They can:

- block coding before architecture is ready,
- allow safe non-coding work,
- record human overrides,
- enforce traceability from task to specification and review.

This is how Hermes OS prevents premature implementation.

### Persistence

Module:

```text
hermes_os_integration/persistence.py
```

The local repository writes deterministic JSON records under:

```text
project/.hermes/records/
```

It supports:

- review reports,
- grill-me sessions,
- decisions,
- approvals,
- agent artifacts,
- work graphs,
- score history,
- runtime usage.

This is intentionally simple today. It gives Hermes OS durable records without needing a full database yet.

### Dashboard Contracts

Module:

```text
hermes_os_integration/dashboard.py
```

The dashboard layer currently produces data contracts, not a full UI.

Panels include:

- architecture score,
- architecture gaps,
- approvals and blocked executions,
- runtime delegation,
- work graph summary,
- dependency blocks,
- execution and validation,
- agent assignments.

This lets a future UI render state without inventing its own logic.

### Template Engine

Module:

```text
hermes_os_integration/templates.py
```

This component is the replacement for domain-specific work graph code.

It includes:

- `TemplateDefinition`
- `TemplateRegistry`
- `TemplateLoader`
- `TemplateValidator`
- `TemplateCompiler`
- `base_project_template`

Templates are where repeatable patterns belong. Hermes OS should provide the engine, not domain assumptions.

Example template shape:

```python
{
    "template_id": "custom",
    "name": "Custom",
    "nodes": [
        {"id": "architecture", "type": "epic", "title": "Architecture"},
        {"id": "workflow-design", "type": "workflow", "title": "Workflow Design"}
    ],
    "dependencies": [
        {
            "source_id": "architecture",
            "target_id": "workflow-design",
            "reason": "architecture before workflows"
        }
    ],
    "metrics": [
        {"id": "template-completeness", "title": "Template Completeness"}
    ]
}
```

Later, domain templates can exist outside the OS:

```text
templates/
  investment/
  media/
  saas/
  crm/
  research/
  custom/
```

But the OS should only know how to load, validate, compile, and track templates.

### Work Graph Compiler

Module:

```text
hermes_os_integration/work_graph.py
```

The work graph is the bridge between architecture and execution.

It models:

- project,
- epic,
- workflow,
- task,
- subtask,
- dependency,
- approval,
- artifact,
- metric,
- agent assignment,
- execution result,
- validation result.

The compiler reads architecture artifacts and generates a `WorkGraph`.

Core functions:

- `read_architecture_artifacts`
- `compile_work_graph`
- `detect_missing_work`
- `assign_agents`
- `generate_validation_rules`
- `resolve_dependencies`
- `build_execution_queue`
- `ingest_execution_result`
- `serialize_work_graph`
- `deserialize_work_graph`
- `save_work_graph`

The work graph is what lets Hermes OS move from:

```text
we have docs and tasks
```

to:

```text
we know what can run, what is blocked, what needs approval, what validates success, and what dashboard state should change
```

### Plan CLI

Module:

```text
hermes_os_integration/plan_cli.py
```

Conceptual command:

```bash
hermes plan <project>
```

Current module entrypoint:

```bash
python -m hermes_os_integration.plan_cli <project> --projects-root /path/to/projects
```

Useful flags:

```bash
--json
--write
```

What it does:

1. Scans the project.
2. Reads architecture artifacts.
3. Compiles a work graph.
4. Reports missing work.
5. Optionally writes `workgraph.json`.

Use this after architecture review and document generation.

### Workspace Control Plane

Module:

```text
hermes_os_integration/workspace_control.py
```

Conceptual command:

```bash
hermes workspace
```

Current module entrypoint:

```bash
python -m hermes_os_integration.workspace_control --projects-root /path/to/projects
```

This summarizes all discovered projects in a workspace:

- architecture score,
- blockers,
- approvals,
- runtime usage placeholder,
- aggregate dashboard data.

This is the cross-project control-plane view. It should stay domain-neutral.

### Autonomous Review Loops

Module:

```text
hermes_os_integration/review_loops.py
```

This component defines scheduled architecture review behavior.

It supports:

- scheduled review contracts,
- roadmap proposals,
- score history records,
- autonomous review safety policy.

The safety policy distinguishes:

- read-only,
- proposal,
- write.

High-risk autonomous writes require approval.

## How The Main Workflows Fit Together

### Workflow 1: Review A Project

Use when you want to understand architecture readiness.

```bash
python -m hermes_os_integration.architect_cli review my-project \
  --projects-root /Users/hq/Workspace/projects \
  --json
```

Result:

- architecture score,
- missing documents,
- missing schemas,
- missing dashboards,
- missing approvals,
- recommendations,
- priority roadmap,
- blocked status.

Best use:

- Run this before asking for implementation.
- Use `--block-on-critical` when you want the command to fail if architecture is not ready.
- Use `--write-report` when you want a durable review artifact.

### Workflow 2: Generate Missing Docs

Use when a project is missing source-of-truth architecture documents.

```bash
python -m hermes_os_integration.architect_cli review my-project \
  --projects-root /Users/hq/Workspace/projects \
  --generate-docs \
  --write-report
```

By default, existing docs are not overwritten.

Best use:

- Generate structure first.
- Then fill the docs with real business/domain/workflow decisions.
- Do not treat generated docs as final truth without review.

### Workflow 3: Compile A Work Graph

Use when architecture docs exist and you want executable structure.

```bash
python -m hermes_os_integration.plan_cli my-project \
  --projects-root /Users/hq/Workspace/projects \
  --json
```

To write:

```bash
python -m hermes_os_integration.plan_cli my-project \
  --projects-root /Users/hq/Workspace/projects \
  --write
```

Result:

```text
project/workgraph.json
```

Best use:

- Compile after architecture review.
- Treat findings as missing work.
- Use the graph to understand dependencies and blocked nodes before execution.

### Workflow 4: Use A Template

Use templates for repeatable structure.

Good template content:

- generic node patterns,
- dependencies,
- approval gates,
- metrics,
- validation rules.

Bad template content inside Hermes OS:

- specific project logic,
- domain assumptions,
- business-specific rules,
- hardcoded project names.

Templates can become domain-specific outside the OS, but the OS should only load and compile them.

### Workflow 5: Delegate Work To Official Hermes Agent

Use delegation only after architecture readiness is understood.

Dry-run first:

```python
from hermes_os_integration.delegation import delegate_task

result = delegate_task({
    "task_id": "task-1",
    "project_id": "my-project",
    "task_type": "docs",
    "prompt": "Draft the missing workflow document.",
    "working_directory": "/path/to/project",
    "dry_run": True,
})
```

Live runtime uses `hermes-agent --oneshot` through the wrapper. Hermes OS still owns the response, artifact ingestion, and state update.

Best use:

- Keep runtime delegation opt-in.
- Prefer dry-run until gates and persistence are trusted.
- Never let runtime memory become authoritative state.

### Workflow 6: View Workspace Health

Use workspace control to see cross-project readiness.

```bash
python -m hermes_os_integration.workspace_control \
  --projects-root /Users/hq/Workspace/projects \
  --json
```

Result:

- project count,
- architecture scores,
- blockers,
- approval gaps,
- dashboard-ready summary data.

Best use:

- Run periodically.
- Use it to decide which project needs architecture work next.
- Do not encode project domains into the workspace control plane.

## How To Get The Most Out Of The System

### 1. Start With Documents, Not Code

For any project, create or generate:

```text
PROJECT.md
DOMAIN.md
WORKFLOWS.md
DASHBOARD.md
METRICS.md
APPROVALS.md
AGENTS.md
TASKS.md
DECISIONS.md
ROADMAP.md
ARCHITECTURE.md
```

Then run architecture review.

### 2. Use Grill-Me Before Task Generation

The grill-me question bank exists to challenge assumptions. Use it when a project idea feels exciting but still fuzzy.

The goal is not to slow you down. The goal is to avoid building the wrong thing quickly.

### 3. Use Reviews As Governance, Not As Decoration

An architecture review should decide:

- Is this project ready for execution?
- What is missing?
- What should be blocked?
- What can safely be delegated?
- What needs approval?

If the review says the project is blocked, respect that unless a human override is recorded.

### 4. Compile Work Graphs Before Execution

Tasks alone are too flat. The work graph gives you:

- dependencies,
- execution order,
- blocked work,
- validation rules,
- agent assignments,
- dashboard state.

Use `hermes plan` behavior before running implementation work.

### 5. Keep Hermes OS Domain-Neutral

This is the big design lesson.

Do not add code like:

```text
domain_graph.py
vertical_pipeline.py
business_workflow.py
project_specific_profile.py
```

inside the OS.

Instead, add:

```text
TemplateRegistry
TemplateLoader
TemplateCompiler
TemplateValidator
```

Then place domain logic in:

```text
templates/
project docs/
plugins/
connectors/
project-level configuration/
```

Hermes OS should become more powerful by becoming more general.

### 6. Keep Agents As Workers

Agents can:

- research,
- analyze,
- validate,
- generate,
- review,
- document,
- test.

Agents should not:

- own project state,
- own workflows,
- own dashboards,
- own approvals,
- own business logic,
- directly mutate source-of-truth records.

### 7. Use Persistence For Institutional Memory

Persist:

- architecture reviews,
- grill-me sessions,
- decisions,
- approvals,
- work graphs,
- artifacts,
- score history,
- runtime usage.

This gives Hermes OS memory without relying on agent memory.

### 8. Treat Dashboards As Feedback Loops

A dashboard is not just a screen. It is the feedback loop that tells Hermes OS what needs improvement.

Dashboard panels should answer:

- What is ready?
- What is blocked?
- What is missing?
- What changed?
- What failed validation?
- What needs approval?
- Which worker runtime is healthy?

### 9. Use Dry-Run As The Default Rollout Mode

Dry-run mode lets you validate:

- contracts,
- prompts,
- gates,
- work graph compilation,
- dashboard data,
- persistence,
- runtime availability.

Use live runtime only when the path is clear.

## What Is Real Today

Real and test-covered:

- Architecture-first contracts.
- Constitution rules.
- Architecture review reports.
- Project scanning.
- Missing document detection.
- Safe document generation.
- Runtime contracts.
- Runtime wrapper command assembly.
- Dry-run delegation.
- Agent registry.
- Memory boundary checks.
- MCP permission bridge.
- Execution gates.
- Local persistence.
- Dashboard data contracts.
- Work graph schema and compiler.
- Dependency resolution.
- Execution queue building.
- Validation rule generation.
- Work graph serialization.
- Plan CLI module entrypoint.
- Workspace control module entrypoint.
- Template engine.
- Autonomous review loop contracts.

Prototype or integration layer:

- Full mounted `hermes architect`, `hermes plan`, and `hermes workspace` commands inside the actual Hermes OS command surface.
- Full live Official Hermes Agent runtime execution in production workflows.
- Full dashboard UI.
- Full database-backed persistence.
- External template marketplace or template directory loader.
- Continuous scheduled jobs.

## Practical Command Cheat Sheet

Run tests:

```bash
.venv/bin/python -m pytest tests/hermes_os_integration -q
```

Review a project:

```bash
python -m hermes_os_integration.architect_cli review my-project \
  --projects-root /Users/hq/Workspace/projects \
  --json
```

Generate missing docs:

```bash
python -m hermes_os_integration.architect_cli review my-project \
  --projects-root /Users/hq/Workspace/projects \
  --generate-docs \
  --write-report
```

Compile a work graph:

```bash
python -m hermes_os_integration.plan_cli my-project \
  --projects-root /Users/hq/Workspace/projects \
  --write
```

View workspace control-plane status:

```bash
python -m hermes_os_integration.workspace_control \
  --projects-root /Users/hq/Workspace/projects \
  --json
```

Check Official Hermes Agent launcher:

```bash
./hermes-agent --help
```

## Recommended Operating Rhythm

For a new project:

1. Create the project folder.
2. Run architecture review.
3. Generate missing docs.
4. Fill in the docs with real project decisions.
5. Run architecture review again.
6. Compile the work graph.
7. Resolve blockers and approvals.
8. Use dry-run delegation.
9. Persist artifacts and decisions.
10. Enable live runtime only when gates pass.

For an existing project:

1. Scan it.
2. Review architecture.
3. Generate missing docs carefully.
4. Compile a work graph.
5. Use dashboard data to find blockers.
6. Add templates only when a pattern repeats across projects.

For Hermes OS itself:

1. Keep core generic.
2. Move domain logic into templates.
3. Move integrations into plugins/connectors.
4. Keep source-of-truth state in Hermes OS.
5. Keep agents as replaceable workers.

## Design Principles To Preserve

### Generality Over Convenience

If adding a domain-specific shortcut makes one project easier but the OS more specific, do not put it in the OS.

### Artifacts Over Memory

If something matters, persist it as a record or artifact. Do not rely on agent memory.

### Gates Over Trust

Trusting the agent is not a governance model. Use gates, approvals, validation, and dashboards.

### Work Graphs Over Task Lists

Task lists are useful, but they do not encode dependency, validation, approval, or dashboard feedback. Work graphs do.

### Templates Over Hardcoding

Templates let the system support many project types without embedding domain assumptions into the OS.

## Where To Extend Next

The strongest next build steps are:

1. Mount the module CLIs into the real Hermes OS command surface.
2. Add a template directory loader for external templates.
3. Add a real dashboard UI consuming the dashboard panel contracts.
4. Replace local JSON persistence with a durable database layer when scale requires it.
5. Add scheduled autonomous review jobs.
6. Add live runtime execution policies around cost, retries, approval, and audit.
7. Add plugin boundaries for external systems.

The system is now at the point where the foundation is useful, but the next gains come from turning the contracts into an everyday command surface and UI.
